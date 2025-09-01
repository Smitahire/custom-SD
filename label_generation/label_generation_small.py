import argparse, os, json, time
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from utils.schedules import STEP_BINS
from utils.metrics import load_clip, clipscore_for_image_text

def setup_pipe(height, width):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()  # aggressive but fits 4GB
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

def main(args):
    os.makedirs(args.tmp_dir, exist_ok=True)
    pipe = setup_pipe(args.height, args.width)
    clip_model, clip_proc, device = load_clip()

    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    out = open(args.out, 'w', encoding='utf-8')

    for idx, prompt in enumerate(tqdm(prompts, desc="Labeling")):
        best = None

        for steps in STEP_BINS:  # small set for 4GB
            t0 = time.time()
            image = pipe(prompt, num_inference_steps=int(steps), guidance_scale=7.0,
                         height=args.height, width=args.width).images[0]
            fn = os.path.join(args.tmp_dir, f"p{idx}_s{steps}.png")
            image.save(fn)
            dt = time.time() - t0

            score = clipscore_for_image_text(clip_model, clip_proc, device, fn, prompt)

            rec = {'prompt_id': idx, 'prompt': prompt, 'steps': int(steps), 'clipscore': score, 'latency_s': dt, 'image': fn}
            if (best is None) or (score > best['clipscore'] - 1e-6 ): #and dt < best['latency_s']
                best = rec

        out.write(json.dumps(best) + "\n")
        out.flush()

    out.close()
    print(f"Saved labels to {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tmp_dir', default='tmp_images')
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=384)
    args = parser.parse_args()
    main(args)
