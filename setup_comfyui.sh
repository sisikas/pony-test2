#!/bin/bash

# ComfyUI Setup Script for Hugging Face Spaces
# This script sets up ComfyUI for deployment on Hugging Face Spaces

echo "Setting up ComfyUI for Hugging Face Spaces..."

# Create ComfyUI directory
mkdir -p comfyui
cd comfyui

# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git .

# Install ComfyUI requirements
pip install -r requirements.txt

# Install additional requirements for your use case
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install huggingface_hub

# Create custom nodes directory
mkdir -p custom_nodes

# Download your models to ComfyUI
echo "Setting up models..."

# Create models directory structure
mkdir -p models/checkpoints
mkdir -p models/loras
mkdir -p models/embeddings

# Download your custom models
echo "Downloading custom models from Hugging Face..."

# Download checkpoint
python -c "
from huggingface_hub import hf_hub_download
import os

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id='skas12/illustrious-test1',
    filename='realismIllustriousBy_v50FP16.safetensors',
    local_dir='models/checkpoints'
)
print(f'Checkpoint downloaded to: {checkpoint_path}')

# Download LoRAs
loras = [
    'Pony Realism Slider.safetensors',
    'RealSkin_slider.safetensors', 
    'insta baddie PN.safetensors',
    'Real_Beauty.safetensors',
    'Pony_DetailV2.0.safetensors',
    'perfect ass sliderV1.safetensors',
    'Detail_Tweaker_Illustrious_BSY_V3.safetensors'
]

for lora in loras:
    try:
        lora_path = hf_hub_download(
            repo_id='skas12/illustrious-test1',
            filename=lora,
            local_dir='models/loras'
        )
        print(f'LoRA downloaded: {lora}')
    except Exception as e:
        print(f'Failed to download {lora}: {e}')

# Download embeddings
embeddings = [
    'Stable_Yogis_Realism_Positives_V1.safetensors',
    'Stable_Yogis_Anatomy_Negatives_V1-neg.safetensors',
    'Stable_Yogis_General_Negatives_V1-neg.safetensors',
    'Stable_Yogis_Realism_Negatives_V1-neg.safetensors'
]

for embedding in embeddings:
    try:
        embedding_path = hf_hub_download(
            repo_id='skas12/illustrious-test1',
            filename=embedding,
            local_dir='models/embeddings'
        )
        print(f'Embedding downloaded: {embedding}')
    except Exception as e:
        print(f'Failed to download {embedding}: {e}')
"

echo "ComfyUI setup complete!"
echo "To run ComfyUI: python main.py --listen 0.0.0.0 --port 8188"
