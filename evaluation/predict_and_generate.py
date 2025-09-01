import argparse, os, json
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np

from utils.schedules import STEP_BINS
from ans_model.train_ans import TinyMLP

def setup_pipe(height, width):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        #pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

def load_text_encoder():
    tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True)
    enc = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = enc.to(device).eval()
    return tok, enc, device

def embed_text(tokenizer, encoder, device, text):
    with torch.no_grad():
        tokens = tokenizer([text], padding=True, return_tensors='pt').to(device)
        out = encoder(**tokens)
        pooled = out.pooler_output.detach().cpu().numpy()[0]
    return pooled

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = setup_pipe(args.height, args.width)
    tokenizer, text_encoder, device = load_text_encoder()

    # Load ANS
    ckpt = torch.load(args.ans, map_location=device)
    ans = TinyMLP(dim_in=512, dim_hidden=512, num_classes=len(STEP_BINS)).to(device)
    ans.load_state_dict(ckpt['model'])
    ans.eval()

    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    for i, ptxt in enumerate(prompts):
        emb = embed_text(tokenizer, text_encoder, device, ptxt)
        x = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = ans(x)
            pred_idx = int(logits.argmax(dim=1).item())
            steps = STEP_BINS[pred_idx]

        image = pipe(ptxt, num_inference_steps=int(steps), guidance_scale=7.0,
                     height=args.height, width=args.width).images[0]
        image.save(os.path.join(args.out_dir, f"p{i}_steps{steps}.png"))

    print(f"Saved images to {args.out_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prompts', required=True)
    p.add_argument('--ans', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--width', type=int, default=384)
    args = p.parse_args()
    main(args)
