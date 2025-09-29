#!/usr/bin/env python3
"""
ComfyUI API Client for Pony Generation
This script provides an interface to run ComfyUI workflows programmatically.
"""

import json
import requests
import websocket
import uuid
import base64
import io
from PIL import Image
from typing import Dict, Any, Optional

class ComfyUIAPI:
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        
    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Queue a prompt for execution and return the prompt ID"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        
        response = requests.post(f"{self.server_url}/prompt", data=data)
        return response.json()['prompt_id']
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        """Get an image from ComfyUI server"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = requests.get(f"{self.server_url}/view", params=data)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise Exception(f"Failed to get image: {response.status_code}")
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        response = requests.get(f"{self.server_url}/history/{prompt_id}")
        return response.json()
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for workflow completion and return results"""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            
            if prompt_id in history:
                status = history[prompt_id].get('status', {})
                if status.get('status_str') == 'success':
                    return history[prompt_id]
                elif status.get('status_str') == 'error':
                    raise Exception(f"Workflow failed: {status.get('messages', 'Unknown error')}")
            
            time.sleep(1)
        
        raise Exception("Workflow timed out")

class PonyComfyUIWorkflow:
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        self.api = ComfyUIAPI(server_url)
        
    def create_pony_workflow(self, 
                           prompt: str,
                           negative_prompt: str = "",
                           width: int = 1024,
                           height: int = 1024,
                           steps: int = 18,
                           cfg: float = 7.0,
                           seed: int = None) -> Dict[str, Any]:
        """Create a ComfyUI workflow for pony generation"""
        
        if seed is None:
            seed = 3891560175039
            
        # This is a simplified workflow - you'd need to create the actual ComfyUI workflow JSON
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "realismIllustriousBy_v50FP16.safetensors"
                }
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": "Pony Realism Slider.safetensors",
                    "strength_model": 1.0,
                    "strength_clip": 1.0
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["2", 1]
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["2", 1]
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_sde",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["6", 0]
                }
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "pony",
                    "images": ["7", 0]
                }
            }
        }
        
        return workflow
    
    def generate_pony(self, 
                     prompt: str,
                     negative_prompt: str = "",
                     width: int = 1024,
                     height: int = 1024,
                     steps: int = 18,
                     cfg: float = 7.0,
                     seed: int = None) -> Image.Image:
        """Generate a pony image using ComfyUI workflow"""
        
        # Create workflow
        workflow = self.create_pony_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        
        # Queue the workflow
        prompt_id = self.api.queue_prompt(workflow)
        print(f"Queued workflow with ID: {prompt_id}")
        
        # Wait for completion
        result = self.api.wait_for_completion(prompt_id)
        
        # Get the generated image
        outputs = result.get('outputs', {})
        if '8' in outputs and 'images' in outputs['8']:
            image_info = outputs['8']['images'][0]
            filename = image_info['filename']
            return self.api.get_image(filename)
        else:
            raise Exception("No image generated")

# Example usage
if __name__ == "__main__":
    # Initialize ComfyUI client
    pony_workflow = PonyComfyUIWorkflow()
    
    # Generate a pony
    try:
        image = pony_workflow.generate_pony(
            prompt="Stable_Yogis_Realism_Positives_V1, photorealistic, realistic skin texture, a young woman, brown long hair, looking at viewer, outdoors, beach background",
            negative_prompt="Stable_Yogis_Anatomy_Negatives_V1, Stable_Yogis_Realism_Negatives_V1, Stable_Yogis_General_Negatives_V1",
            width=1024,
            height=1024,
            steps=18,
            cfg=7.0,
            seed=3891560175039
        )
        
        # Save the result
        image.save("generated_pony.png")
        print("Pony generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
