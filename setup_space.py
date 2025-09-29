#!/usr/bin/env python3
"""
Setup script for Hugging Face Spaces
This script installs ComfyUI and downloads models when the space starts
"""

import os
import subprocess
import sys
from huggingface_hub import hf_hub_download

def setup_comfyui():
    """Setup ComfyUI for Hugging Face Spaces"""
    print("Setting up ComfyUI for Hugging Face Spaces...")
    
    # Create ComfyUI directory
    os.makedirs("comfyui", exist_ok=True)
    
    # Clone ComfyUI if not exists
    if not os.path.exists("comfyui/main.py"):
        print("Cloning ComfyUI repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/comfyanonymous/ComfyUI.git", 
            "comfyui"
        ], check=True)
    
    # Install ComfyUI requirements
    print("Installing ComfyUI requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "-r", "comfyui/requirements.txt"
    ], check=True)
    
    # Create models directory structure
    os.makedirs("comfyui/models/checkpoints", exist_ok=True)
    os.makedirs("comfyui/models/loras", exist_ok=True)
    os.makedirs("comfyui/models/embeddings", exist_ok=True)
    
    # Download models
    print("Downloading custom models...")
    
    try:
        # Download checkpoint
        checkpoint_path = hf_hub_download(
            repo_id="skas12/illustrious-test1",
            filename="realismIllustriousBy_v50FP16.safetensors",
            local_dir="comfyui/models/checkpoints"
        )
        print(f"Checkpoint downloaded: {checkpoint_path}")
        
        # Download LoRAs
        loras = [
            "Pony Realism Slider.safetensors",
            "RealSkin_slider.safetensors", 
            "insta baddie PN.safetensors",
            "Real_Beauty.safetensors",
            "Pony_DetailV2.0.safetensors",
            "perfect ass sliderV1.safetensors",
            "Detail_Tweaker_Illustrious_BSY_V3.safetensors"
        ]
        
        for lora in loras:
            try:
                lora_path = hf_hub_download(
                    repo_id="skas12/illustrious-test1",
                    filename=lora,
                    local_dir="comfyui/models/loras"
                )
                print(f"LoRA downloaded: {lora}")
            except Exception as e:
                print(f"Failed to download {lora}: {e}")
        
        # Download embeddings
        embeddings = [
            "Stable_Yogis_Realism_Positives_V1.safetensors",
            "Stable_Yogis_Anatomy_Negatives_V1-neg.safetensors",
            "Stable_Yogis_General_Negatives_V1-neg.safetensors",
            "Stable_Yogis_Realism_Negatives_V1-neg.safetensors"
        ]
        
        for embedding in embeddings:
            try:
                embedding_path = hf_hub_download(
                    repo_id="skas12/illustrious-test1",
                    filename=embedding,
                    local_dir="comfyui/models/embeddings"
                )
                print(f"Embedding downloaded: {embedding}")
            except Exception as e:
                print(f"Failed to download {embedding}: {e}")
        
        print("ComfyUI setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up ComfyUI: {e}")
        return False

if __name__ == "__main__":
    setup_comfyui()
