from src.models import DinoVLM
import torch
from icecream import ic
import json
from transformers import pipeline
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from src.dataset_utils import *
from torch.utils.data import DataLoader
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import numpy as np
from tqdm import tqdm
import wandb
import time
import random
import warnings


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def get_logger(log_path):
    # ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


class Gemma():
    def __init__(self):
        self.model_base = DinoVLM()
        self.processor = self.model_base.processor
        self.model_base.freeze_lm()
        self.model_base.freeze_vision()
        # dataset = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train")
        dataset = load_dataset("HuggingFaceM4/the_cauldron", "localized_narratives")
        # dataset['train'][0]
        dataset_split = dataset['train'].train_test_split(test_size = 0.1)

        self.train_dataloader = DataLoader(dataset_split['train'], batch_size=1, shuffle = True, collate_fn=self.collate_fn)
        self.test_dataset = DataLoader(dataset_split['test'], batch_size=1, shuffle = True, collate_fn=self.collate_fn)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=r"language_model\.layers\.\d+\.self_attn\.(q|k|v)_proj",
            lora_dropout=0.05,
            bias="none",
        )

        # ! the base model consists of the freezed vison encoder & language model with the connector head unfreezed.
        # and we generate the lora model from here.     
        self.model = get_peft_model(self.model_base, lora_config)
        self.model.print_trainable_parameters()
        lora_params = [(n, p.shape) for n, p in self.model.named_parameters() if 'lora' in n and p.requires_grad]
        print(f"LoRA layers found: {len(lora_params)}")
        print(lora_params[:4])  # print first 4


        # i guess lora internally freezes everything so, added thi
        for p in self.model.connector.parameters():
            p.requires_grad = True

        self.NUM_EPOCHS      = 4
        self.GRAD_ACCUM_STEPS = 4
        LR              = 2e-4
        self.MAX_GRAD_NORM   = 1.0
        self.SAVE_DIR        = f"/media/system/ZERBUIS_EXT_STOR/temp/exp/experiment/depth_tuning/experiments_gemma/checkpoints_{time.time()}"
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.logger = get_logger(self.SAVE_DIR + '/logger.txt')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LR,
            weight_decay=0.01,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=len(self.train_dataloader) * self.NUM_EPOCHS)
        self.best_eval_loss = float("inf")
        total_trainable_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.logger.info(f'[INFO] Total trainable parameters: {total_trainable_params}')


        wandb.init(
            project = 'gemma_finetuning_fit3d',
            name = self.SAVE_DIR + '/logger.txt',
            mode = "online",
            config = {
                "learning_rate": LR,
                "epochs": self.NUM_EPOCHS,
                "batch_size": 1,
                # "weight_decay": WEIGHT_DECAY,
                # "warmup_ratio": WARMUP_RATIO,
                "model": "gemma",
                "dataset": "the_cauldron"
            }
        )

        logging.getLogger("transformers.models.paligemma.processing_paligemma").setLevel(logging.ERROR)

    def collate_fn(self, examples):
        texts = []
        images = []

        for ex in examples:
            image = ex["images"][0]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((384, 384))

            qa = random.choice(ex["texts"])
            texts.append((qa["user"], qa["assistant"]))
            images.append(image)

        prompt_texts = [f"{u}\n" for u, _ in texts]
        full_texts   = [f"{u}\n{a}" for u, a in texts]

        batch_input = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        prompt_only = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch_input["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # mask image token positions
        labels[labels == 257152] = -100

        for i in range(len(texts)):
            # count real (non-pad) tokens in the prompt — robust to padding
            prompt_ids = prompt_only["input_ids"][i]
            prompt_real_len = (prompt_ids != self.processor.tokenizer.pad_token_id).sum().item()

            # the full sequence also has padding on the left potentially,
            # so find the actual start of content in batch_input
            full_ids = batch_input["input_ids"][i]
            full_real_len = (full_ids != self.processor.tokenizer.pad_token_id).sum().item()

            # how many tokens at the END are the answer
            # mask everything except the answer tokens
            answer_len = full_real_len - prompt_real_len
            mask_until = batch_input["input_ids"].shape[1] - answer_len

            labels[i, :mask_until] = -100

        batch_input["labels"] = labels
        return batch_input
    
    def train(self):

        for epoch in tqdm(range(self.NUM_EPOCHS)):
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad()

            for step, batch in tqdm(enumerate(self.train_dataloader), leave=True, total=len(self.train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                warnings.filterwarnings(
                    "ignore",
                    message="You are passing both `text` and `images` to `PaliGemmaProcessor`.*"
                )

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.GRAD_ACCUM_STEPS

                loss.backward()
                train_loss += outputs.loss.item()

                if (step + 1) % self.GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if step % 10 == 0:
                    avg = train_loss / (step + 1)
                    wandb.log({'average loss': avg, 'step': step})

            # handle leftover steps
            if (step + 1) % self.GRAD_ACCUM_STEPS != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            avg_train_loss = train_loss / len(self.train_dataloader)
            wandb.log({"average train loss": avg_train_loss, "epoch": epoch})
            self.logger.info(f"Epoch {epoch+1} | Avg train loss: {avg_train_loss:.4f}")

            # eval
            self.model.eval()
            eval_loss = 0.0

            try:
                with torch.no_grad():
                    for batch in self.test_dataset:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outputs = self.model(**batch)
                        eval_loss += outputs.loss.item()

                avg_eval_loss = eval_loss / len(self.test_dataset)
                self.logger.info(f"Epoch {epoch+1} | Eval loss: {avg_eval_loss:.4f}")
                wandb.log({"avg eval loss": avg_eval_loss, "epoch": epoch})

                if avg_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = avg_eval_loss
                    self.model.save_pretrained(os.path.join(self.SAVE_DIR, "best"))
                    self.logger.info(f"  ↳ New best saved (eval_loss={avg_eval_loss:.4f})")

            except Exception as err:
                self.logger.info(err)

            self.model.save_pretrained(os.path.join(self.SAVE_DIR, f"epoch_{epoch+1}"))

        self.logger.info(f"\nTraining done. Best eval loss: {self.best_eval_loss:.4f}")

if __name__ == "__main__":
    inst = Gemma()
    inst.train()