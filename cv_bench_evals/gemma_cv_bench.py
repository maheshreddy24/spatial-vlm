
import warnings
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import re
from datasets import load_dataset
import csv
import os
from collections import defaultdict
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from icecream import ic


logs = 'gemma_results'
os.makedirs(logs, exist_ok=True)

MODEL_NAME = "google/paligemma2-3b-mix-224"
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


# ! gemma model
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    attn_implementation="sdpa",
    # use_fast = True
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    use_fact = True
    
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
        image = image.resize((224, 224))
        question = sample['prompt']
        sample_task = sample['task']
        gt = sample['answer']

        prompt = (
            "answer en "
            "Answer only with the option letter "
            "(A, B, C, or D).\n"
            f"{question}"
        )

        inputs = processor(
            image,
            prompt,
            return_tensors="pt"
        ).to(model.device)

        # ic(inputs)

        warnings.filterwarnings(
            "ignore",
            message="You are passing both `text` and `images`.*",
            category=UserWarning
        )
        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                cache_implementation="static"
            )

        
        predicted_label = processor.decode(
            output[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        response = predicted_label

    
        pred = predicted_label.upper()

        match = re.search(r'[A-D]', pred)

        pred = match.group(0) if match else pred

        gt_match = re.search(r'[A-D]', gt.upper())

        gt_clean = (
            gt_match.group(0)
            if gt_match
            else gt.strip().upper()
        )

        is_correct = pred == gt_clean

        total += 1

        if is_correct:
            correct += 1

        task_total[sample_task] += 1

        if is_correct:
            task_correct[sample_task] += 1

        accuracy = (correct / total) * 100

        writer.writerow([
            sample_task,
            question,
            gt_clean,
            pred,
            response,
            is_correct,
            accuracy
        ])

        if i % 100 == 0 and i > 0:

            print(
                f"Overall Accuracy till now: "
                f"{accuracy:.2f}%"
            )

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