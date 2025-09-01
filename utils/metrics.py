import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import os, glob, json
from typing import List, Dict

def load_clip(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True)#need to update torch to remove use safetensor
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = model.to(device)
    model.eval()
    return model, processor, device

@torch.no_grad()
def clipscore_for_image_text(model, processor, device, image_path: str, text: str) -> float:
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=[text], images=image, return_tensors='pt', padding=True).to(device)
    outputs = model(**inputs)
    # cosine similarity between image and text embeddings
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    sim = (image_embeds @ text_embeds.t()).squeeze().item()
    return float(sim)
