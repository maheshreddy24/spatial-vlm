import warnings
import torch
from PIL import Image
import re
from datasets import load_dataset
import csv
import os
from collections import defaultdict
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from src.models import DinoVLM

logs = 'dino_vlm_results'
os.makedirs(logs, exist_ok=True)

csv_file = f'{logs}/dinovlm_results.csv'

cv_bench_2d = load_dataset("nyu-visionx/CV-Bench", "2D")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")

model = DinoVLM(dino_dim=384, gemma_dim=2304)
model.freeze_vision()
model.freeze_lm() 
lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=r"language_model\.layers\.\d+\.self_attn\.(q|k|v)_proj",
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

checkpoint_dir = "/media/system/ZERBUIS_EXT_STOR/temp/exp/experiment/depth_tuning/experiments_gemma/checkpoints_1780482936.8283668/stage2/best"
model.load_adapter(checkpoint_dir, adapter_name="default")
# model = model.merge_and_unload()
model.eval().to("cuda")

print("Model loaded!")

# ── Eval loop ─────────────────────────────────────────────────────────────────
correct = 0
total = 0
task_correct = defaultdict(int)
task_total = defaultdict(int)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["task", "prompt", "gt", "pred", "raw_response", "correct", "accuracy"])

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)

    for i, sample in tqdm(
        enumerate(cv_bench_2d['test']),
        desc='Samples',
        total=len(cv_bench_2d['test'])
    ):
        image   = sample['image'].resize((384, 384))
        question = sample['prompt']
        sample_task = sample['task']
        gt = sample['answer']

        prompt = (
            "answer en "
            "Answer only with the option letter "
            "(A, B, C, or D).\n"
            f"{question}"
        )

        warnings.filterwarnings(
            "ignore",
            message="You are passing both `text` and `images` to `PaliGemmaProcessor`.*"
        )

        # DinoVLM.generate() takes images + prompts directly
        with torch.no_grad():
            raw_response = model.generate_output(
                images=[image],
                prompts=[prompt],
                max_new_tokens=10,
            )[0]


        pred = raw_response.strip().upper()
        match = re.search(r'[A-D]', pred)
        pred = match.group(0) if match else pred

        gt_match = re.search(r'[A-D]', gt.upper())
        gt_clean = gt_match.group(0) if gt_match else gt.strip().upper()

        is_correct = pred == gt_clean
        total += 1
        if is_correct:
            correct += 1

        task_total[sample_task] += 1
        if is_correct:
            task_correct[sample_task] += 1

        accuracy = (correct / total) * 100

        writer.writerow([
            sample_task, question, gt_clean,
            pred, raw_response, is_correct, accuracy
        ])

        if i % 100 == 0 and i > 0:
            print(f"Overall Accuracy till now: {accuracy:.2f}%")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n==============================")
print(f"Final Overall Accuracy: {(correct / total) * 100:.2f}%")
print("==============================")

print("\nPer-task Accuracy:")
for t in sorted(task_total.keys()):
    acc = (task_correct[t] / task_total[t]) * 100
    print(f"  {t}: {acc:.2f}% ({task_correct[t]}/{task_total[t]})")

print(f'\nLogs saved in {csv_file}')