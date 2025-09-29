# -*- coding: utf-8 -*-
"""
ComfyUI Interface for Hugging Face Spaces
This provides the full ComfyUI web interface with node editor
"""

import os
import subprocess
import time
import threading
import requests
from huggingface_hub import hf_hub_download

class ComfyUISetup:
    def __init__(self):
        self.comfyui_process = None
        self.setup_comfyui()
    
    def setup_comfyui(self):
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
        try:
            subprocess.run([
                "python", "-m", "pip", "install", 
                "-r", "comfyui/requirements.txt"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing ComfyUI requirements: {e}")
            # Try installing missing packages individually
            missing_packages = ["tqdm", "psutil", "aiohttp", "websockets", "pyyaml"]
            for package in missing_packages:
                try:
                    subprocess.run(["python", "-m", "pip", "install", package], check=True)
                    print(f"Installed {package}")
                except:
                    print(f"Failed to install {package}")
        
        # Create models directory structure
        os.makedirs("comfyui/models/checkpoints", exist_ok=True)
        os.makedirs("comfyui/models/loras", exist_ok=True)
        os.makedirs("comfyui/models/embeddings", exist_ok=True)
        
        # Download your custom models
        self.download_models()
        
        # Start ComfyUI server
        self.start_comfyui_server()
    
    def download_models(self):
        """Download your custom models"""
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
            
            print("All models downloaded successfully!")
            
        except Exception as e:
            print(f"Error downloading models: {e}")
    
    def start_comfyui_server(self):
        """Start ComfyUI server"""
        print("Starting ComfyUI server...")
        
        try:
            # Start ComfyUI server
            cmd = [
                "python", "comfyui/main.py", 
                "--listen", "0.0.0.0", 
                "--port", "7860",  # Use 7860 for Hugging Face Spaces
                "--cpu"  # Use CPU for Hugging Face Spaces
            ]
            
            self.comfyui_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get("http://127.0.0.1:7860", timeout=1)
                    if response.status_code == 200:
                        print("ComfyUI server started successfully!")
                        print("Access ComfyUI at: http://127.0.0.1:7860")
                        return True
                except:
                    time.sleep(1)
            
            print("ComfyUI server started (may take a moment to be ready)")
            return True
            
        except Exception as e:
            print(f"Error starting ComfyUI: {e}")
            return False

# Initialize ComfyUI setup
if __name__ == "__main__":
    print("Starting ComfyUI setup...")
    comfyui_setup = ComfyUISetup()
    
    # Keep the process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down ComfyUI...")
        if comfyui_setup.comfyui_process:
            comfyui_setup.comfyui_process.terminate()