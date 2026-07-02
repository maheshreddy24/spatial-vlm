# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from torch.utils.data import DataLoader
from datasets import load_dataset
from icecream import ic
import time
import os
from transformers import get_cosine_schedule_with_warmup
from tqdm.autonotebook import tqdm
import warnings
import wandb
import random

"""
Option A (baseline): Freeze vision encoder, LoRA on LM,         full-finetune connector
Option B            : LoRA on vision encoder + LM,              full-finetune connector
Option C (gentle)   : LoRA on vision encoder (very low lr) + LM, full-finetune connector
"""


class GemmaHF():
    def __init__(self):
        MODEL_NAME = "google/paligemma2-3b-mix-224"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            use_fast=True,  # [FIX] typo: use_fact → use_fast
        )

        dataset = load_dataset("array/SAT-v2", split="train")
        dataset = dataset.train_test_split(0.2)
        self.train_loader = DataLoader(dataset['train'], collate_fn=self.collate_fn, batch_size=1)
        self.test_loader  = DataLoader(dataset['test'],  collate_fn=self.collate_fn, batch_size=1)

        self.GRAD_ACCUM_STEPS = 8
        self.paligemma = None

        self.output_dir = f'experiments_gemma_hf/{time.time()}'
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_model(self, MODEL_NAME="google/paligemma2-3b-mix-224"):
        print("Model reinitialization!")
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,   
            attn_implementation="sdpa",
        ).to(self.device)                
    
    def lora_lm(self):
        """
        User concern: q_proj/k_proj/v_proj/o_proj exist in BOTH the vision
        encoder and the language model. Answer: by passing only
        self.paligemma.language_model to get_peft_model, LoRA is scoped
        strictly to that submodule. The vision encoder's attention layers are
        completely untouched.

        However, from_pretrained loads all weights with requires_grad=True, so
        we must explicitly freeze the vision tower ourselves.
        """
       
        target_modules = []

        for name, module in self.paligemma.named_modules():
            if (
                "language_model" in name
                and any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
            ):
                target_modules.append(name)

        for p in self.paligemma.vision_tower.parameters():
            p.requires_grad = False

        # LoRA scoped only to the language model submodule
        lora_config = LoraConfig(
            r=32,
            lora_alpha=128,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        self.paligemma = get_peft_model(
            self.paligemma, lora_config
        )

        # Unfreeze connector (it lives outside language_model so it was never
        # frozen, but be explicit)
        for name, p in self.paligemma.named_parameters():   # [FIX] named_parameters(), not parameters()
            if 'multi_modal_projector' in name:              #       p is a tensor, not a string
                print(f'[UNFREEZING] {name}')
                p.requires_grad = True

        # [FIX] print_trainable_parameters() is on the PeftModel wrapping
        #        language_model, not on self.paligemma itself
        self.paligemma.print_trainable_parameters()


    def lora_full(self):
        """
        Applies LoRA to every q/k/v/o_proj in the model — both vision encoder
        and language model. The connector has no such projections, so
        get_peft_model freezes it as a base weight; we unfreeze it below.

        lr_v in build_optimizer controls how aggressively the vision encoder
        is updated (Option B: 1e-4, Option C: 1e-6).
        """

        lora_config = LoraConfig(
            r=32,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.paligemma = get_peft_model(self.paligemma, lora_config)

        # Unfreeze connector — get_peft_model froze everything not covered by LoRA
        for name, p in self.paligemma.named_parameters():
            if 'multi_modal_projector' in name:
                print(f'[UNFREEZING] {name}')
                p.requires_grad = True

        self.paligemma.print_trainable_parameters()

    def build_optimizer(self, lr_v, lr_l, epochs, stage):
        """
        [FIX] Build AdamW with parameter groups so vision encoder and
        language model can have independent learning rates.

          Option A: lr_v=0  — vision frozen, group not created
          Option B: lr_v=1e-4, lr_l=1e-5
          Option C: lr_v=1e-6, lr_l=1e-5  (gentle nudge for vision)
        """
        weight_decay  = 0.01
        warmup_ratio  = 0.05

        vision_params    = []
        connector_params = []
        language_params  = []

        for name, p in self.paligemma.named_parameters():
            if not p.requires_grad:
                continue
            if 'vision_tower' in name:
                vision_params.append(p)
            elif 'multi_modal_projector' in name:
                connector_params.append(p)
            else:
                language_params.append(p)

        param_groups = []
        if vision_params:                # empty for Option A (vision is frozen)
            param_groups.append({'params': vision_params,    'lr': lr_v, 'weight_decay': weight_decay})
        if connector_params:
            param_groups.append({'params': connector_params, 'lr': lr_l, 'weight_decay': weight_decay})
        if language_params:
            param_groups.append({'params': language_params,  'lr': lr_l, 'weight_decay': weight_decay})

        self.optimizer = torch.optim.AdamW(param_groups) 

        # [FIX] scheduler counts optimizer steps, not raw batch steps
        num_training_steps = (epochs * len(self.train_loader)) // self.GRAD_ACCUM_STEPS
        num_warmup_steps   = int(warmup_ratio * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        wandb.init(
            project="gemma_finetuning_fit3d",
            name=f"gemma_hf_finetuning_{stage}",
            mode="online",
            config={
                "stage":      stage,
                "lr_v":       lr_v,
                "lr_l":       lr_l,
                "epochs":     epochs,
                "grad_accum": self.GRAD_ACCUM_STEPS,
                "dataset":    "array/SAT-v2",
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

            question = ex["question"].strip()
            options  = list(ex["answers"])   # copy so we don't mutate the original/cached list
            random.shuffle(options)          # FIX: randomize order so answer isn't always index 0 -> 'A'

            # correct_answer is the raw value → find its (post-shuffle) index → convert to letter
            answer_idx = options.index(ex["correct_answer"])
            answer     = chr(65 + answer_idx)                # 'A', 'B', 'C', ...

            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])

            prompt = (
                f"<image> Question: {question}\n"
                f"Options:\n{options_text}\n"
                f"Answer:"
            )
            texts.append((prompt, answer))
            images.append(image)

        eos        = self.processor.tokenizer.eos_token
        full_texts = [f"{q}\n{a}{eos}" for q, a in texts]   # FIX 2: append EOS so model learns to stop

        # FIX: only one processor call needed now
        batch_input = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True,
        )

        labels = batch_input["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == 257152] = -100  # image token id

        for i in range(len(texts)):
            full_len   = (batch_input["attention_mask"][i] == 1).sum().item()

            # Detect if processor added a trailing \n after EOS
            newline_id  = self.processor.tokenizer.encode("\n", add_special_tokens=False)[-1]  # 108
            last_tok    = batch_input["input_ids"][i][full_len - 1].item()
            has_trailing = (last_tok == newline_id)

            a_ids       = self.processor.tokenizer(texts[i][1], add_special_tokens=False)["input_ids"]
            answer_len  = len(a_ids) + 1 + (1 if has_trailing else 0)  # A + EOS + maybe \n
            mask_until  = full_len - answer_len

            if mask_until <= 0:
                labels[i, :] = -100
                continue

            labels[i, :mask_until] = -100

        batch_input["labels"] = labels
        return batch_input

    def _fit(self, stage, epochs=1):
        for epoch in range(epochs):
            self.paligemma.train()
            total_loss = 0.0

            # [FIX] zero_grad once before the loop, not inside every step
            self.optimizer.zero_grad()

            for step, batch in tqdm(
                enumerate(self.train_loader), total=len(self.train_loader), leave=True
            ):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="You are passing both `text` and `images` to `PaliGemmaProcessor`.*"
                    )
                    outputs = self.paligemma(**batch)

                # [FIX] scale the loss for gradient accumulation before backward
                loss = outputs.loss / self.GRAD_ACCUM_STEPS
                loss.backward()

                # Track the true (unscaled) per-batch loss for logging
                total_loss += outputs.loss.item()

                is_accum_step = (step + 1) % self.GRAD_ACCUM_STEPS == 0
                is_last_step  = (step + 1) == len(self.train_loader)

                # [FIX] only call optimizer.step when gradients are fully accumulated
                if is_accum_step or is_last_step:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    wandb.log({
                        "train_loss_step": total_loss / (step + 1),
                        "step": step,
                    })

            avg_train_loss = total_loss / len(self.train_loader)

            self.paligemma.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in self.test_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="You are passing both `text` and `images` to `PaliGemmaProcessor`.*"
                        )
                        outputs = self.paligemma(**batch)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(self.test_loader)

            wandb.log({
                "val_loss":   avg_val_loss,
                "train_loss": avg_train_loss,
                "epoch":      epoch + 1,
            })
            print(
                f"Epoch [{epoch+1}/{epochs}]  "
                f"Train Loss: {avg_train_loss:.4f}  "
                f"Val Loss:   {avg_val_loss:.4f}"
            )

        # [FIX] use the stage variable — was hardcoded as the string "stage"
        self.paligemma.save_pretrained(f"{self.output_dir}/{stage}")
        wandb.finish()  


    def train(self):

        print("=" * 60)
        print("Option C — Vision: LoRA (low lr) | LM: LoRA | Connector: full")
        self._init_model()
        try:
            self.lora_full()
            self.build_optimizer(lr_v=1e-6, lr_l=1e-5, epochs=1, stage="OptionC")
            self._fit(stage="OptionC", epochs=1)
        except Exception as err:
            print(err)


        print("=" * 60)
        print("Option A — Vision: FROZEN  | LM: LoRA       | Connector: full")
        self._init_model()
        try:
            self.lora_lm()
            self.build_optimizer(lr_v=0,    lr_l=1e-5, epochs=1, stage="OptionA")
            print('Unfreezed vision encoder params')
            for name, p in self.paligemma.named_parameters():
                if 'vision' in name and p.requires_grad:
                    print(name)
                    break
            self._fit(stage="OptionA", epochs=1)
        except Exception as err:
            print(err)

        print("=" * 60)
        print("Option B — Vision: LoRA    | LM: LoRA       | Connector: full")
        self._init_model()
        try:
            self.lora_full()
            self.build_optimizer(lr_v=1e-4, lr_l=1e-5, epochs=1, stage="OptionB")
            self._fit(stage="OptionB", epochs=1)
        except Exception as err:
            print(err)



if __name__ == '__main__':
    inst = GemmaHF()
    inst.train()