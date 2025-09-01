# Adaptive Noise Scheduling (ANS) — Minimal Prototype

This repo is a **low-VRAM (4 GB) runnable prototype** to develop and test Adaptive Noise Scheduling for Stable Diffusion.
It uses SD 1.5 at 256–448px with memory-saving flags. Train the ANS (tiny MLP) to predict **step-count** per prompt.

## Quickstart

```bash
# (Recommended) Python 3.10+
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# Optional: verify CUDA / torch
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 1) Generate a small baseline and a few alternative schedules (labels)
python label_generation/label_generation_small.py --prompts data/prompts.txt --out data/labels_small.jsonl

# 2) Build CLIP text embeddings for prompts
python utils/build_embeddings.py --prompts data/prompts.txt --out data/embeddings.npy --ids_out data/prompt_ids.json

# 3) Train the ANS (tiny MLP) to predict step count
python ans_model/train_ans.py --labels data/labels_small.jsonl --embeddings data/embeddings.npy --ids data/prompt_ids.json --out ans_model/ans_steps.pt

# 4) Use ANS to predict steps and generate images
python evaluation/predict_and_generate.py --prompts data/test_prompts.txt --ans ans_model/ans_steps.pt --out_dir outputs_ans

# 5) Compare CLIPScore vs baseline
python evaluation/compute_clipscore.py --prompts data/test_prompts.txt --images_dir outputs_ans --out results_ans.json
```

## Files/Dirs
- `label_generation/label_generation_small.py` — runs SD with a few step counts and saves the best per prompt (tiny set).
- `utils/build_embeddings.py` — CLIP text embeddings saver (precompute to keep training tiny).
- `ans_model/train_ans.py` — trains a tiny MLP to predict **step count** (classification over bins).
- `evaluation/predict_and_generate.py` — uses ANS-predicted steps for generation.
- `evaluation/compute_clipscore.py` — computes CLIPScore (text–image) for quick quality checks.
- `utils/schedules.py`, `utils/metrics.py` — helpers.

## Notes
- Default model: `runwayml/stable-diffusion-v1-5` (lighter than SDXL).
- Resolution defaults to **384×384** on 4GB. Lower to **256×256** if needed.
- Enable: `pipe.enable_attention_slicing()`, `pipe.enable_vae_slicing()`, `pipe.enable_model_cpu_offload()`, and `pipe.enable_xformers_memory_efficient_attention()`.
- Start with **three step bins**: {15, 25, 35}. You can expand once you have access to V100/T4.
