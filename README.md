# spatial-vlm

Experiments to improve spatial understanding in Vision Language Models (VLMs). Most experiments are conducted on [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), a dataset with two task types: object counting and relative positioning. We evaluate InternVL and PaliGemma, and improve their vision encoders through 3D-aware finetuning.

Evaluation scripts and results are in `/cv_bench_evals`. Finetuning uses LoRA or full finetuning depending on model size.

> **Note:** Additional results and analysis will be added soon.

## References

Vision encoder finetuning resources:
- [FiT3D](https://github.com/ywyue/FiT3D)
- [3D-VLM-GD](https://github.com/kaist-cvml/3d-vlm-gd)