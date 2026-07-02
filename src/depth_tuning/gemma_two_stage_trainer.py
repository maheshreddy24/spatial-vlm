from src.models import DinoVLM, SigLipVLM
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
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
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger



"""
    1. this is two stage training, where the first stage is training the connector network from scratch, with high lr
    2. and in stage two we add lora to the llm and then train both the connector and llm with a low learning rate.
    3. the model is in self.model_base. in first stage we use without lora and then after the connector trainig we add the lora params to the lm and then train 
    4. the grad accumilate is the number of gradient steps.
"""
class Gemma():
    def __init__(self):
        # self.model_base = DinoVLM()
        self.model_base = SigLipVLM()
        self.processor  = self.model_base.processor
        self.model_base.freeze_lm()
        self.model_base.freeze_vision()

        dataset       = load_dataset("HuggingFaceM4/the_cauldron", "localized_narratives")
        dataset_split = dataset['train'].train_test_split(test_size=0.1)

        self.train_dataloader = DataLoader(dataset_split['train'], batch_size=1, shuffle=True,  collate_fn=self.collate_fn)
        self.test_dataloader  = DataLoader(dataset_split['test'],  batch_size=1, shuffle=False, collate_fn=self.collate_fn)

        self.NUM_EPOCHS       = 3   # per stage
        self.GRAD_ACCUM_STEPS = 16
        self.LR               = 2e-4
        self.MAX_GRAD_NORM    = 1.0
        self.SAVE_DIR         = f"/media/system/ZERBUIS_EXT_STOR/temp/exp/experiment/depth_tuning/experiments_gemma/checkpoints_{time.time()}"
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        self.logger = get_logger(os.path.join(self.SAVE_DIR, 'logger.txt'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_base = self.model_base.to(self.device)

        logging.getLogger("transformers.models.paligemma.processing_paligemma").setLevel(logging.ERROR)


    def _apply_lora(self):
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model_base = get_peft_model(self.model_base, lora_config)
        self.model_base.print_trainable_parameters()

        # re-enable connector since PEFT freezes all non-LoRA params
        for p in self.model_base.connector.parameters():
            p.requires_grad = True

        lora_params = [(n, p.shape) for n, p in self.model_base.named_parameters() if 'lora' in n and p.requires_grad]
        self.logger.info(f"LoRA layers found: {len(lora_params)}")

    def _build_optimizer(self, stage: str, lr = None):
        if wandb.run is not None:
            wandb.finish()

        if lr is None:
            lr = self.LR
            
        self.best_eval_loss = float("inf")
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model_base.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_dataloader) * self.NUM_EPOCHS
        )

        total = sum(p.numel() for p in self.model_base.parameters() if p.requires_grad)
        self.logger.info(f"[{stage}] Trainable parameters: {total:,}")

        wandb.init(
            project="gemma_finetuning_fit3d",
            name=f"{os.path.basename(self.SAVE_DIR)}_{stage}",
            mode="online",
            config={
                "stage": stage,
                "learning_rate": lr,
                "epochs": self.NUM_EPOCHS,
                "grad_accum": self.GRAD_ACCUM_STEPS,
                "dataset": "localized_narratives",
            }
        )

    def collate_fn(self, examples):
        texts  = []
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

        batch_input  = self.processor(text=full_texts,   images=images, return_tensors="pt", padding=True)
        prompt_only  = self.processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)

        labels = batch_input["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == 257152] = -100

        # for i in range(len(texts)):
        #     prompt_real_len = (prompt_only["input_ids"][i] != self.processor.tokenizer.pad_token_id).sum().item()
        #     full_real_len   = (batch_input["input_ids"][i] != self.processor.tokenizer.pad_token_id).sum().item()
        #     answer_len      = full_real_len - prompt_real_len
        #     mask_until      = batch_input["input_ids"].shape[1] - answer_len
        #     labels[i, :mask_until] = -100

        for i in range(len(texts)):
            prompt_real_len = (prompt_only["input_ids"][i] != self.processor.tokenizer.pad_token_id).sum().item()
            full_real_len   = (batch_input["input_ids"][i] != self.processor.tokenizer.pad_token_id).sum().item()
            answer_len      = full_real_len - prompt_real_len

            # sanity guard
            if answer_len <= 0:
                self.logger.warning(f"Sample {i} has answer_len={answer_len}, skipping label assignment")
                labels[i, :] = -100
                continue

            mask_until = batch_input["input_ids"].shape[1] - answer_len
            labels[i, :mask_until] = -100

        batch_input["labels"] = labels
        return batch_input

    def _fit(self, stage: str, epochs = None):
        save_dir = os.path.join(self.SAVE_DIR, stage)
        os.makedirs(save_dir, exist_ok=True)

        for epoch in tqdm(range(epochs), desc=f"{stage} epochs"):
            self.model_base.train()
            train_loss = 0.0
            self.optimizer.zero_grad()

            for step, batch in tqdm(enumerate(self.train_dataloader), leave=True, total=len(self.train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                warnings.filterwarnings("ignore", message="You are passing both `text` and `images` to `PaliGemmaProcessor`.*")

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model_base(**batch)
                    loss    = outputs.loss / self.GRAD_ACCUM_STEPS

                loss.backward()
                train_loss += outputs.loss.item()

                if (step + 1) % self.GRAD_ACCUM_STEPS == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model_base.parameters(), self.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    wandb.log({"grad_norm": grad_norm.item(), "step": step})

                if step % 10 == 0:
                    wandb.log({"loss": train_loss / (step + 1), "step": step, "stage": stage})

            # leftover steps
            if (step + 1) % self.GRAD_ACCUM_STEPS != 0:
                torch.nn.utils.clip_grad_norm_(self.model_base.parameters(), self.MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            avg_train_loss = train_loss / len(self.train_dataloader)
            self.logger.info(f"[{stage}] Epoch {epoch+1} | Train loss: {avg_train_loss:.4f}")
            wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})

            # eval
            self.model_base.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_dataloader, desc="eval", leave=False):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model_base(**batch)
                    eval_loss += outputs.loss.item()

            avg_eval_loss = eval_loss / len(self.test_dataloader)
            self.logger.info(f"[{stage}] Epoch {epoch+1} | Eval loss: {avg_eval_loss:.4f}")
            wandb.log({"avg_eval_loss": avg_eval_loss, "epoch": epoch})

            # if avg_eval_loss < self.best_eval_loss:
            #     self.best_eval_loss = avg_eval_loss
            #     self.model_base.save_pretrained(os.path.join(save_dir, "best"))
            #     self.logger.info(f"  ↳ New best saved (eval_loss={avg_eval_loss:.4f})")

            # self.model_base.save_pretrained(os.path.join(save_dir, f"epoch_{epoch+1}"))

            if avg_eval_loss < self.best_eval_loss:
                self.best_eval_loss = avg_eval_loss
                best_dir = os.path.join(save_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                if stage == "stage1":
                    try:
                        torch.save(self.model_base.connector.state_dict(), os.path.join(best_dir, "connector.pt"))
                        self.logger.info("Connector model pretrained weight saved!!")
                    except Exception as err:
                        self.logger.info(f"erro while saving the model {err}")

                else:
                    self.model_base.save_pretrained(best_dir)
                self.logger.info(f"  ↳ New best saved (eval_loss={avg_eval_loss:.4f})")

            epoch_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            if stage == "stage1":
                try:
                    torch.save(self.model_base.connector.state_dict(), os.path.join(epoch_dir, "connector.pt"))
                except Exception as err:
                    self.logger.info(f"erro while saving the model {err}")
            else:
                self.model_base.save_pretrained(epoch_dir)


        self.logger.info(f"[{stage}] Done. Best eval loss: {self.best_eval_loss:.4f}")
        wandb.finish()

    # def train(self):
    #     # ── Stage 1: connector only ───────────────────────────────────────────
    #     self.logger.info("=" * 40)
    #     self.logger.info("STAGE 1 — connector alignment")
    #     self.logger.info("=" * 40)
    #     self._build_optimizer(stage="stage1", lr=2e-3) # we use a high learning rate to allign the model properly
    #     self._fit(stage="stage1", epochs=1)

    #     # # load best connector weights from stage 1 #!! why do we even neeed this, we are trainingonly for one epoch so just continue with the same
    #     # best_stage1 = os.path.join(self.SAVE_DIR, "stage1", "best")
    #     # self.logger.info(f"Loading best stage1 checkpoint from {best_stage1}")
    #     # self.model_base.load_adapter(best_stage1)

    #     # ── Stage 2: connector + LoRA ─────────────────────────────────────────
    #     self.logger.info("=" * 40)
    #     self.logger.info("STAGE 2 — LoRA + connector fine-tuning")
    #     self.logger.info("=" * 40)
    #     self._apply_lora()
    #     self._build_optimizer(stage="stage2", lr = 2e-5) # low learnign rate for better allignment
    #     self._fit(stage="stage2", epochs=self.NUM_EPOCHS)

    def train(self):
        self.logger.info("=" * 40)
        self.logger.info("STAGE 1 — connector alignment")
        self.logger.info("=" * 40)
        self._build_optimizer(stage="stage1", lr=5e-4)
        self._fit(stage="stage1", epochs=3)

        best_connector_path = os.path.join(self.SAVE_DIR, "stage1", "best", "connector.pt")
        if os.path.exists(best_connector_path):
            state_dict = torch.load(best_connector_path, map_location=self.device)
            self.model_base.connector.load_state_dict(state_dict)
            self.logger.info(f"Loaded best stage1 connector from {best_connector_path}")
        else:
            self.logger.warning(f"Best stage1 connector not found at {best_connector_path}, continuing with c urrent weights")

        self.logger.info("=" * 40)
        self.logger.info("STAGE 2 — LoRA + connector fine-tuning")
        self.logger.info("=" * 40)
        self._apply_lora()
        self._build_optimizer(stage="stage2", lr=2e-5)
        self._fit(stage="stage2", epochs=self.NUM_EPOCHS)


if __name__ == '__main__':
    inst = Gemma()
    inst.train()