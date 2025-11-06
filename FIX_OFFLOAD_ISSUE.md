# 🔧 Fix: Model Offloading to Disk

## Problem
The model is trying to offload to disk because `device_map="auto"` detects insufficient GPU memory.

## Solution Options

### Option 1: Use CPU (Works but Slow)
If you don't have enough GPU memory, use CPU:

```python
# In your script, modify load_pretrained_model call:
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cpu"  # Force CPU
)
```

### Option 2: Use 4-bit Quantization (Recommended for Limited GPU)
This reduces memory usage significantly:

```python
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True  # Use 4-bit quantization
)
```

### Option 3: Provide Offload Folder
If you want to allow disk offloading:

```python
import os
offload_folder = "./offload_cache"
os.makedirs(offload_folder, exist_ok=True)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    offload_folder=offload_folder
)
```

### Option 4: Force GPU (If You Have Enough Memory)
```python
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cuda"  # Force GPU
)
```

---

## Recommended: Use 4-bit Quantization

For a laptop GPU, 4-bit quantization is best:

```python
# In test_visionzip_single.py, change line 30-34:
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True  # Add this line
)
```

This reduces memory from ~14GB to ~4GB.

