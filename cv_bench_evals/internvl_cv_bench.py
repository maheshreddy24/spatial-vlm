import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import re
from datasets import load_dataset
import csv
import os
from collections import defaultdict

logs = 'intern_vl_res'
os.makedirs(logs, exist_ok=True)

MODEL_NAME = "OpenGVLab/InternVL2_5-2B"
csv_file = f'{logs}/{MODEL_NAME.split("/")[-1]}_results.csv'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 2D tasks only
cv_bench_2d = load_dataset("nyu-visionx/CV-Bench", "2D")

transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def pil_to_pixel_values(image):
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values.to(torch.bfloat16).cuda()

print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True
)

print("Model loaded!")

generation_config = {
    "max_new_tokens": 32,
    "do_sample": False
}

correct = 0
total = 0

# Per-task stats
task_correct = defaultdict(int)
task_total = defaultdict(int)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "task",
        "prompt",
        "gt",
        "pred",
        "raw_response",
        "correct",
        "accuracy"
    ])

with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)

    for i, sample in tqdm(
        enumerate(cv_bench_2d['test']),
        desc='Samples',
        total=len(cv_bench_2d['test'])
    ):

        image = sample['image']
        prompt = sample['prompt']
        sample_task = sample['task']
        gt = sample['answer']

        prompt = (
            "Please answer only the option as (<option>), "
            "without any extra explanation.\n" + prompt
        )

        pixel_values = pil_to_pixel_values(image)

        question = f"<image>\n{prompt}"

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config
        )

        # Extract predicted option
        match = re.search(r'\(?\b([A-D])\b\)?', response)
        pred = match.group(1) if match else response.strip()

        # Clean GT
        gt_match = re.search(r'([A-D])', gt)
        gt_clean = gt_match.group(1) if gt_match else gt.strip()

        is_correct = pred == gt_clean

        # Overall stats
        total += 1
        if is_correct:
            correct += 1

        # Per-task stats
        task_total[sample_task] += 1
        if is_correct:
            task_correct[sample_task] += 1

        accuracy = (correct / total) * 100

        writer.writerow([
            sample_task,
            prompt,
            gt_clean,
            pred,
            response,
            is_correct,
            accuracy
        ])

        if i % 100 == 0 and i > 0:
            print(f"Overall Accuracy till now: {accuracy:.2f}%")

print("\n==============================")
print(f"Final Overall Accuracy: {(correct / total) * 100:.2f}%")
print("==============================")

print("\nPer-task Accuracy:")
for t in sorted(task_total.keys()):
    acc = (task_correct[t] / task_total[t]) * 100
    print(
        f"{t}: "
        f"{acc:.2f}% "
        f"({task_correct[t]}/{task_total[t]})"
    )

print(f'\nLogs saved in {csv_file}')