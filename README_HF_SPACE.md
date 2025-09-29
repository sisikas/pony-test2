---
title: ComfyUI Pony Generator
emoji: 🦄
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Generate beautiful pony images using ComfyUI workflows with custom CivitAI models
---

# 🦄 ComfyUI Pony Generator

Generate beautiful pony images using ComfyUI workflows with your custom CivitAI models on Hugging Face Spaces!

## 🎯 Features

- **ComfyUI Workflows**: Advanced node-based generation pipeline
- **Custom Models**: Uses your exact CivitAI models
  - Base Model: Realism Illustrious (ID: 2091367)
  - Multiple LoRAs: Pony Realism Slider, RealSkin, insta baddie, Real Beauty, Pony Detail, perfect ass slider, Detail Tweaker
- **Advanced LoRA Stacking**: Visual workflow with individual LoRA weights
- **High Quality**: SDXL-based generation with optimized memory management
- **Interactive UI**: Easy-to-use Gradio interface
- **Free Hosting**: Powered by Hugging Face Spaces

## 🚀 How to Use

1. Enter your prompt in the text box
2. Adjust LoRA weights using the sliders
3. Configure generation settings (width, height, steps, CFG scale)
4. Click "Generate with ComfyUI"
5. Download your generated image!

## 🎨 Example Prompts

- "Stable_Yogis_Realism_Positives_V1, photorealistic, realistic skin texture, a young woman, brown long hair, looking at viewer, outdoors, beach background"
- "A majestic pony with rainbow mane, high quality, detailed"
- "A cute pony in a magical forest, fantasy art"
- "A realistic pony portrait, professional photography"

## 🔧 Technical Details

- **Framework**: Gradio + ComfyUI
- **Model**: Stable Diffusion XL with custom checkpoint
- **LoRAs**: 7 different LoRAs with individual weight control
- **Embeddings**: Automatic textual inversion loading
- **Hardware**: GPU acceleration with optimized memory management
- **Workflow**: Visual node-based ComfyUI pipeline

## 📝 Model Credits

- Base Model: [Realism Illustrious](https://civitai.com/models/2091367)
- LoRAs: Multiple LoRAs from your custom collection
- Embeddings: Stable Yogis textual inversions

## 🆕 What's New

- **ComfyUI Integration**: Now uses ComfyUI workflows instead of direct Diffusers
- **Better Memory Management**: Optimized for Hugging Face Spaces
- **Visual Workflow**: Clear understanding of the generation pipeline
- **Advanced Features**: Better LoRA stacking and parameter control

Enjoy generating beautiful pony images with ComfyUI! 🦄✨
