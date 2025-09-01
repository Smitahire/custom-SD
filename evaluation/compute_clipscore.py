import argparse, os, json, glob
from utils.metrics import load_clip, clipscore_for_image_text

def main(args):
    clip_model, clip_proc, device = load_clip()
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = []
    for i, p in enumerate(prompts):
        # pick the first image matching pattern for prompt i
        pattern = os.path.join(args.images_dir, f"p{i}_*.png")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        img = files[0]
        score = clipscore_for_image_text(clip_model, clip_proc, device, img, p)
        results.append({'prompt_id': i, 'prompt': p, 'image': img, 'clipscore': score})

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} results to {args.out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prompts', required=True)
    p.add_argument('--images_dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    main(args)
