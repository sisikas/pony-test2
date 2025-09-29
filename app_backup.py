import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
import tempfile
import os

class PonyGenerator:
    def __init__(self):
        self.pipe = None
        self.load_model()
    
    def load_model(self):
        """Load the custom pony model with Hugging Face Hub integration"""
        print("🦄 Loading custom pony model...")
        
        try:
            # Method 1: Try direct Hugging Face Hub download first
            try:
                print("📥 Downloading checkpoint from Hugging Face Hub...")
                checkpoint_path = hf_hub_download(
                    repo_id="skas12/illustrious-test1",
                    filename="realismIllustriousBy_v50FP16.safetensors"
                )
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                print("✅ Checkpoint loaded from Hugging Face Hub!")
            except Exception as hf_error:
                print(f"❌ Hugging Face Hub download failed: {hf_error}")
                # Method 2: Try loading as a proper HF repository
                print("🔄 Trying to load as HF repository...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "skas12/illustrious-test1",
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
                print("✅ Checkpoint loaded as HF repository!")
            
            # Load the LoRA
            print("📥 Loading Pony Realism Slider LoRA...")
            try:
                # Try downloading LoRA from Hugging Face Hub
                lora_path = hf_hub_download(
                    repo_id="skas12/illustrious-test1",
                    filename="Pony Realism Slider.safetensors"
                )
                self.pipe.load_lora_weights(lora_path, adapter_name="pony_realism")
            except Exception as lora_error:
                print(f"❌ LoRA download failed: {lora_error}")
                # Fallback: try loading as HF repository
                self.pipe.load_lora_weights(
                    "skas12/illustrious-test1", 
                    weight_name="Pony Realism Slider.safetensors",
                    adapter_name="pony_realism"
                )
            
            self.pipe.set_adapters(["pony_realism"], adapter_weights=[1.0])
            print("✅ LoRA loaded successfully!")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("🚀 Custom pony model loaded on GPU")
            else:
                print("💻 Custom pony model loaded on CPU")
            
            print("🦄 Your custom pony model loaded successfully!")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load custom model: {e}")
            print("🚨 CUSTOM MODEL IS REQUIRED - NO FALLBACK TO BASE SDXL!")
            raise Exception(f"Failed to load custom model: {e}")

    def generate_image(self, prompt, negative_prompt, width, height, steps, guidance_scale, seed):
        """Generate pony image with custom model"""
        if self.pipe is None:
            return None, "❌ Model not loaded properly"
        
        try:
            print(f"🎨 Generating pony image with prompt: {prompt}")
            
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            else:
                generator = None
            
            # Generate image
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            print("✅ Image generated successfully!")
            return result.images[0], "🦄 Image generated successfully!"
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None, f"❌ Error: {str(e)}"

# Initialize the model
try:
    pony_gen = PonyGenerator()
    model_loaded = True
    model_error = None
except Exception as e:
    pony_gen = None
    model_loaded = False
    model_error = str(e)

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="🦄 Custom Pony Generator", theme=gr.themes.Soft()) as demo:
        if not model_loaded:
            gr.Markdown(f"""
            # 🚨 Model Loading Failed
            
            **Error**: {model_error}
            
            The custom model failed to load. Please check:
            1. Your Hugging Face repository `skas12/illustrious-test1` exists
            2. The checkpoint file `realismIllustriousBy_v50FP16.safetensors` is available
            3. The LoRA file `Pony Realism Slider.safetensors` is available
            
            **NO FALLBACK TO BASE SDXL - CUSTOM MODEL REQUIRED!**
            """)
        else:
            gr.Markdown("""
            # 🦄 Custom Pony Generator
            
            Generate beautiful pony images using your custom CivitAI models:
            - **Base Model**: Realism Illustrious
            - **LoRA**: Pony Realism Slider
            
            Powered by Hugging Face Spaces! 🚀
            """)
        
        if model_loaded:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="🎨 Prompt",
                        placeholder="A majestic pony with rainbow mane, high quality, detailed",
                        value="A beautiful pony with rainbow mane, high quality, detailed"
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="🚫 Negative Prompt",
                        value="blurry, low quality, distorted, bad anatomy, nsfw"
                    )
                    
                    with gr.Row():
                        width = gr.Slider(512, 1536, 1024, step=64, label="📐 Width")
                        height = gr.Slider(512, 1536, 1024, step=64, label="📐 Height")
                    
                    with gr.Row():
                        steps = gr.Slider(10, 50, 25, step=1, label="🎯 Steps")
                        guidance_scale = gr.Slider(1.0, 20.0, 7.5, step=0.1, label="🎛️ Guidance Scale")
                        seed = gr.Number(label="🌱 Seed (optional)", precision=0)
                    
                    generate_btn = gr.Button("🦄 Generate Pony", variant="primary", size="lg")
                    
                with gr.Column():
                    output_image = gr.Image(label="🖼️ Generated Image", type="pil")
                    status = gr.Textbox(label="📊 Status", interactive=False)
            
            # Event handlers
            generate_btn.click(
                fn=pony_gen.generate_image,
                inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed],
                outputs=[output_image, status]
            )
            
            # Example prompts
            gr.Examples(
                examples=[
                    ["A majestic pony with rainbow mane, high quality, detailed"],
                    ["A cute pony in a magical forest, fantasy art"],
                    ["A realistic pony portrait, professional photography"],
                    ["A pony with butterfly wings, magical, ethereal"]
                ],
                inputs=prompt
            )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
