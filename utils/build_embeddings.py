import argparse, json, numpy as np, torch
from transformers import CLIPTokenizer, CLIPTextModel

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    text_encoder.eval()

    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    ids = list(range(len(prompts)))
    embs = []

    with torch.no_grad():
        for p in prompts:
            tokens = tokenizer([p], padding=True, return_tensors='pt').to(device)
            out = text_encoder(**tokens)
            pooled = out.pooler_output.detach().cpu().numpy()[0]
            embs.append(pooled)

    embs = np.stack(embs, axis=0)
    np.save(args.out, embs)

    with open(args.ids_out, 'w', encoding='utf-8') as f:
        json.dump({'ids': ids, 'prompts': prompts}, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ids_out', required=True)
    args = parser.parse_args()
    main(args)
