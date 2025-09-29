import os
import tempfile
from typing import List
from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        print("Loading your custom pony models from Hugging Face...")
        
        # Load your custom models from Hugging Face Hub
        # This is much more reliable than CivitAI downloads
        try:
            # Load the main checkpoint (illustrious)
            print("Loading Realism Illustrious checkpoint...")
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                "skas12/illustrious-test1/realismIllustriousBy_v50FP16.safetensors",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            print("✅ Checkpoint loaded successfully!")
            
            # Load the LoRA
            print("Loading Pony Realism Slider LoRA...")
            self.pipe.load_lora_weights(
                "skas12/illustrious-test1/Pony Realism Slider.safetensors", 
                adapter_name="pony_realism"
            )
            self.pipe.set_adapters(["pony_realism"], adapter_weights=[1.0])
            print("✅ LoRA loaded successfully!")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("✅ Custom pony model loaded on GPU")
            else:
                print("✅ Custom pony model loaded on CPU")
            
            print("🦄 Your custom pony model loaded successfully!")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load custom model: {e}")
            print("🚨 CUSTOM MODEL IS REQUIRED - NO FALLBACK TO BASE SDXL!")
            raise Exception(f"Failed to load custom model: {e}")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for pony image generation using your custom CivitAI models"),
        negative_prompt: str = Input(description="Negative prompt", default="blurry, low quality, distorted, bad anatomy, nsfw"),
        width: int = Input(description="Image width", default=1024, ge=512, le=1536),
        height: int = Input(description="Image height", default=1024, ge=512, le=1536),
        num_inference_steps: int = Input(description="Number of inference steps", default=25, ge=10, le=50),
        guidance_scale: float = Input(description="Guidance scale", default=7.5, ge=1.0, le=20.0),
        seed: int = Input(description="Random seed for reproducibility", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        
        print(f"Generating pony image with prompt: {prompt}")
        
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
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        # Save the generated image
        image = result.images[0]
        
        # Save to temporary file
        output_path = Path(tempfile.mktemp(suffix=".png"))
        image.save(output_path)
        
        print("✅ Image generated successfully!")
        return output_path

def main():
    """Main function for local testing"""
    predictor = Predictor()
    predictor.setup()
    
    # Test generation
    result = predictor.predict(
        prompt="A majestic pony with rainbow mane, high quality, detailed",
        negative_prompt="blurry, low quality",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=42
    )
    
    print(f"Generated image saved to: {result}")

if __name__ == "__main__":
    main()
