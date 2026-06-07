# spatial-vlm

Experiments to improve spatial understanding in Vision Language Models (VLMs). Most experiments are conducted on [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), a dataset with two task types: object counting and relative positioning. We evaluate InternVL and PaliGemma, and improve their vision encoders through 3D-aware finetuning.

Evaluation scripts and results are in `/cv_bench_evals`. Finetuning uses LoRA or full finetuning depending on model size.

---

## Results

| Model | Counting (%) | Relative Position (%) |
|---|---|---|
| PaliGemma2-3B | 60.66 | 76.00 |
| InternVL2-2B | 63.4 | 66.8 |
| InternVL2-4B | 64.2 | 70.1 |
| InternVL2-5-2B | 63.8 | 67.5 |
| InternVL2-5-4B | 64.9 | 72.3 |
| InternVL2-8B | 65.7 | 76.4 |
| **PaliGemma2-3B (Ours — 3D Finetuned Encoder)** | **72.0** | **86.0** |

> Additional results and analysis will be added soon.

---

## Training

### PaliGemma — Standard Finetuning

```bash
python3 src/depth_tuning/base_hf_gemma_trainer.py
```

Three finetuning variants are supported:

- **Option A (baseline):** Freeze vision encoder, LoRA on LM, full-finetune connector
- **Option B:** LoRA on vision encoder + LM, full-finetune connector
- **Option C (gentle):** LoRA on vision encoder (very low lr) + LM, full-finetune connector

Dataset: [SAT-v2](https://huggingface.co/datasets/array/SAT-v2)

### PaliGemma — Two-Stage Training with 3D Finetuned Encoder

```bash
python3 src/depth_tuning/gemma_two_stage_trainer.py
```

This pipeline couples a 3D-finetuned vision encoder with the PaliGemma LM via a connector layer. Training follows the [LLaVA 1.5](https://arxiv.org/abs/2310.03744) strategy:

1. **Stage 1:** Connector training to align modalities
2. **Stage 2:** Language model finetuning with LoRA

Dataset: [`HuggingFaceM4/the_cauldron`](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), `localized_narratives` split

### Early Fusion with Depth Anything v2

See `src/depth_tuning/fusion_models`.

This module implements early fusion combining SigLIP (the default PaliGemma vision encoder) with [Depth Anything v2](https://arxiv.org/abs/2406.09414). A cross-attention layer is applied before the connector, allowing depth features to be incorporated prior to language model input.

---

## References

**Vision encoder finetuning:**
- [FiT3D](https://github.com/ywyue/FiT3D)
- [3D-VLM-GD](https://github.com/kaist-cvml/3d-vlm-gd)

**Citations:**
- LLaVA 1.5 — [arxiv.org/abs/2310.03744](https://arxiv.org/abs/2310.03744)
- Depth Anything v2 — [arxiv.org/abs/2406.09414](https://arxiv.org/abs/2406.09414)
- Understanding the Impact of Geometric Foundation Models on Vision-Language-Action Models — [arxiv.org/abs/2605.24642](https://arxiv.org/abs/2605.24642)