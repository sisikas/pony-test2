# 🦄 Pony Generator - Replicate Deployment

## 🎯 **Custom CivitAI Models on Replicate**

This deployment uses your exact custom models:
- **Base Model**: Realism Illustrious (ID: 2091367)
- **LoRA**: Pony Realism Slider (ID: 1253021)
- **Weight**: 1.0 (full strength)

## 🚀 **Deployment Steps**

### **Step 1: Install Cog**
```bash
pip install cog
```

### **Step 2: Login to Replicate**
```bash
cog login
```

### **Step 3: Deploy Your Model**
```bash
cog push r8.im/your-username/pony-generator
```

### **Step 4: Test Your Model**
```python
import replicate

# Your deployed model
model = "your-username/pony-generator:latest"

# Test generation
output = replicate.run(
    model,
    input={
        "prompt": "A beautiful pony with rainbow mane, high quality",
        "negative_prompt": "blurry, low quality",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": 42
    }
)

print("Generated image:", output)
```

## 💰 **Pricing**

- **Per generation**: ~$0.01-0.05 per image
- **No idle charges**
- **Pay only when generating**

## 🎯 **Features**

- ✅ **Custom CivitAI models** (your exact checkpoint + LoRA)
- ✅ **Pay-per-generation** pricing
- ✅ **GPU acceleration**
- ✅ **Professional API**

## 🔧 **Backend Integration**

Update your backend to use Replicate:

```python
import replicate

# Your model URL
MODEL_URL = "your-username/pony-generator:latest"

# Generate image
result = replicate.run(
    MODEL_URL,
    input={
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed
    }
)
```

## ✅ **Benefits**

- ✅ **Cheaper than Banana.dev**
- ✅ **Custom models supported**
- ✅ **Pay-per-generation**
- ✅ **Professional API**
- ✅ **No idle charges**

**Perfect for commercial use with your custom pony models!** 🦄💰
