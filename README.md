# spatial-vlm

Experiments to improve spatial understanding in Vision Language Models (VLMs). Most experiments are conducted on [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), a dataset with two task types: object counting and relative positioning. We evaluate InternVL and PaliGemma, and improve their vision encoders through 3D-aware finetuning.

Evaluation scripts and results are in `/cv_bench_evals`. Finetuning uses LoRA or full finetuning depending on model size.

## Results

| Model | Counting | Relative Position |
|---|---|---|
| PaliGemma2-3B | 60.66 | 76.00 |
| InternVL2-2B | 63.4 | 66.8 |
| InternVL2-4B | 64.2 | 70.1 |
| InternVL2-5-2B | 63.8 | 67.5 |
| InternVL2-5-4B | 64.9 | 72.3 |
| InternVL2-8B | 65.7 | 76.4 |
| **PaliGemma2-3B (Ours)** | **72.0** | **86.0** |

> **Note:** Additional results and analysis will be added soon.

## References

Vision encoder finetuning resources:
- [FiT3D](https://github.com/ywyue/FiT3D)
- [3D-VLM-GD](https://github.com/kaist-cvml/3d-vlm-gd)