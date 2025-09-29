# -*- coding: utf-8 -*-
"""
ComfyUI-based Pony Generator for Hugging Face Spaces
This app uses ComfyUI workflows instead of direct Diffusers
"""

import gradio as gr
import json
import requests
import websocket
import uuid
import time
import os
import subprocess
import threading
from typing import Dict, Any, Optional, List
from PIL import Image
import io

class ComfyUIManager:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8188"
        self.client_id = str(uuid.uuid4())
        self.comfyui_process = None
        self.is_running = False
        
    def start_comfyui(self):
        """Start ComfyUI server in background"""
        if not self.is_running:
            try:
                # Start ComfyUI server
                cmd = [
                    "python", "comfyui/main.py", 
                    "--listen", "0.0.0.0", 
                    "--port", "8188",
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
                        response = requests.get(f"{self.server_url}/system_stats", timeout=1)
                        if response.status_code == 200:
                            self.is_running = True
                            print("ComfyUI server started successfully!")
                            return True
                    except:
                        time.sleep(1)
                
                print("Failed to start ComfyUI server")
                return False
                
            except Exception as e:
                print(f"Error starting ComfyUI: {e}")
                return False
        return True
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for execution"""
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        
        try:
            response = requests.post(f"{self.server_url}/prompt", data=data, timeout=10)
            if response.status_code == 200:
                return response.json()['prompt_id']
            else:
                raise Exception(f"Failed to queue prompt: {response.status_code}")
        except Exception as e:
            raise Exception(f"ComfyUI server not responding: {e}")
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        """Get generated image from ComfyUI"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        
        try:
            response = requests.get(f"{self.server_url}/view", params=data, timeout=10)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            else:
                raise Exception(f"Failed to get image: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to retrieve image: {e}")
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history"""
        try:
            response = requests.get(f"{self.server_url}/history/{prompt_id}", timeout=10)
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get history: {e}")
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for workflow completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                history = self.get_history(prompt_id)
                
                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('status_str') == 'success':
                        return history[prompt_id]
                    elif status.get('status_str') == 'error':
                        raise Exception(f"Workflow failed: {status.get('messages', 'Unknown error')}")
                
                time.sleep(1)
            except Exception as e:
                if "not responding" in str(e):
                    raise Exception("ComfyUI server is not running. Please check the server status.")
                raise e
        
        raise Exception("Workflow timed out")

class PonyComfyUIWorkflow:
    def __init__(self):
        self.comfyui = ComfyUIManager()
        
    def create_workflow(self, 
                       prompt: str,
                       negative_prompt: str = "",
                       width: int = 1024,
                       height: int = 1024,
                       steps: int = 18,
                       cfg: float = 7.0,
                       seed: int = None,
                       lora_weights: List[float] = None) -> Dict[str, Any]:
        """Create ComfyUI workflow for pony generation"""
        
        if seed is None:
            seed = 3891560175039
            
        if lora_weights is None:
            lora_weights = [1.0, 1.0, 0.94, 0.9, 3.0, 0.34, 0.0]
        
        # ComfyUI workflow JSON
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
                    "strength_model": lora_weights[0],
                    "strength_clip": lora_weights[0]
                }
            },
            "3": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["2", 0],
                    "clip": ["2", 1],
                    "lora_name": "RealSkin_slider.safetensors",
                    "strength_model": lora_weights[1],
                    "strength_clip": lora_weights[1]
                }
            },
            "4": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["3", 0],
                    "clip": ["3", 1],
                    "lora_name": "insta baddie PN.safetensors",
                    "strength_model": lora_weights[2],
                    "strength_clip": lora_weights[2]
                }
            },
            "5": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": "Real_Beauty.safetensors",
                    "strength_model": lora_weights[3],
                    "strength_clip": lora_weights[3]
                }
            },
            "6": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["5", 0],
                    "clip": ["5", 1],
                    "lora_name": "Pony_DetailV2.0.safetensors",
                    "strength_model": lora_weights[4],
                    "strength_clip": lora_weights[4]
                }
            },
            "7": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["6", 0],
                    "clip": ["6", 1],
                    "lora_name": "perfect ass sliderV1.safetensors",
                    "strength_model": lora_weights[5],
                    "strength_clip": lora_weights[5]
                }
            },
            "8": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["7", 1]
                }
            },
            "9": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["7", 1]
                }
            },
            "10": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_sde",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["7", 0],
                    "positive": ["8", 0],
                    "negative": ["9", 0],
                    "latent_image": ["10", 0]
                }
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["11", 0],
                    "vae": ["1", 2]
                }
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "pony",
                    "images": ["12", 0]
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
                     seed: int = None,
                     lora_weights: List[float] = None) -> tuple[Image.Image, str]:
        """Generate pony image using ComfyUI workflow"""
        
        try:
            # Ensure ComfyUI is running
            if not self.comfyui.is_running:
                if not self.comfyui.start_comfyui():
                    return None, "Failed to start ComfyUI server"
            
            # Create workflow
            workflow = self.create_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                lora_weights=lora_weights
            )
            
            # Queue workflow
            prompt_id = self.comfyui.queue_prompt(workflow)
            print(f"Queued workflow with ID: {prompt_id}")
            
            # Wait for completion
            result = self.comfyui.wait_for_completion(prompt_id)
            
            # Get generated image
            outputs = result.get('outputs', {})
            if '13' in outputs and 'images' in outputs['13']:
                image_info = outputs['13']['images'][0]
                filename = image_info['filename']
                image = self.comfyui.get_image(filename)
                return image, "Pony generated successfully with ComfyUI!"
            else:
                return None, "No image generated"
                
        except Exception as e:
            return None, f"Error: {str(e)}"

# Initialize the workflow manager
try:
    pony_workflow = PonyComfyUIWorkflow()
    model_loaded = True
    model_error = None
except Exception as e:
    pony_workflow = None
    model_loaded = False
    model_error = str(e)

def create_interface():
    with gr.Blocks(title="ComfyUI Pony Generator", theme=gr.themes.Soft()) as demo:
        if not model_loaded:
            gr.Markdown(f"""
            # ComfyUI Setup Failed
            
            **Error**: {model_error}
            
            Please ensure ComfyUI is properly installed and configured.
            """)
        else:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="Stable_Yogis_Realism_Positives_V1, photorealistic, realistic skin texture, a young woman, brown long hair, looking at viewer, outdoors, beach background"
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="Stable_Yogis_Anatomy_Negatives_V1, Stable_Yogis_Realism_Negatives_V1, Stable_Yogis_General_Negatives_V1"
                    )
                    
                    # LoRA weight controls
                    lora_controls = []
                    lora_names = [
                        "Pony Realism Slider",
                        "RealSkin_slider", 
                        "insta baddie PN",
                        "Real_Beauty",
                        "Pony_DetailV2.0",
                        "perfect ass sliderV1",
                        "Detail_Tweaker_Illustrious_BSY_V3"
                    ]
                    default_weights = [1.0, 1.0, 0.94, 0.9, 3.0, 0.34, 0.0]
                    
                    for i, (name, default) in enumerate(zip(lora_names, default_weights)):
                        control = gr.Slider(
                            minimum=-5.0,
                            maximum=5.0,
                            value=default,
                            step=0.01,
                            label=name,
                            interactive=True
                        )
                        lora_controls.append(control)
                    
                    with gr.Row():
                        width = gr.Slider(512, 1536, 1024, step=64, label="Width")
                        height = gr.Slider(512, 1536, 1024, step=64, label="Height")
                    
                    with gr.Row():
                        steps = gr.Slider(10, 50, 18, step=1, label="Steps")
                        cfg = gr.Slider(1.0, 20.0, 7.0, step=0.1, label="CFG Scale")
                        seed = gr.Number(label="Seed", value=3891560175039, precision=0)
                    
                    generate_btn = gr.Button("Generate with ComfyUI", variant="primary", size="lg")
                    
                with gr.Column():
                    output_image = gr.Image(label="Generated Image", type="pil", format="png")
                    status = gr.Textbox(label="Status", interactive=False)
            
            # Event handler
            def generate_image(prompt, negative_prompt, width, height, steps, cfg, seed, *lora_weights):
                if pony_workflow is None:
                    return None, "ComfyUI not available"
                
                return pony_workflow.generate_pony(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    lora_weights=list(lora_weights)
                )
            
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt, negative_prompt, width, height, steps, cfg, seed] + lora_controls,
                outputs=[output_image, status]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
