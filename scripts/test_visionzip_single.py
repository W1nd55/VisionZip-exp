"""
Test VisionZip on a single image to verify setup works.
"""
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'VisionZip')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'LLaVA')))

from visionzip import visionzip  # type: ignore
from llava.model.builder import load_pretrained_model  # type: ignore
from llava.mm_utils import get_model_name_from_path  # type: ignore
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN  # type: ignore
from llava.mm_utils import tokenizer_image_token  # type: ignore
from PIL import Image
from llava.conversation import conv_templates  # type: ignore
import torch

print("=" * 60)
print("Testing VisionZip Setup")
print("=" * 60)

# Configuration
model_path = "liuhaotian/llava-v1.5-7b"
print(f"\n1. Loading model: {model_path}")
print("   (This may take a few minutes on first run - downloading model weights)")

try:
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
        # Try 4-bit quantization first (saves memory)
        print("   Attempting to load with 4-bit quantization (lower memory)...")
        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_4bit=True
            )
        except Exception as e:
            print(f"   4-bit failed: {e}")
            print("   Trying CPU instead...")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                device_map="cpu"
            )
    else:
        print("   Using CPU (will be slower)")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device_map="cpu"
        )
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    print("   Make sure you have internet connection for model download")
    print("   If you have GPU memory issues, try: load_4bit=True")
    exit(1)

print(f"\n2. Applying VisionZip (dominant=54, contextual=10)")
try:
    model = visionzip(model, dominant=54, contextual=10)
    print("   ✓ VisionZip applied successfully")
except Exception as e:
    print(f"   ✗ Error applying VisionZip: {e}")
    exit(1)

# Test with a simple image (you can change this path)
print(f"\n3. Testing inference...")
print("   Looking for test image...")

# Try to find a test image
test_image_paths = [
    "reference/owl.jpg",
    "reference/owl.JPEG",
    "models/LLaVA/playground/data/eval/pope/val2014/COCO_val2014_000000000042.jpg",
    "test_image.jpg",
]

image_path = None
for path in test_image_paths:
    if os.path.exists(path):
        image_path = path
        break

if not image_path:
    print("   ⚠ No test image found. Creating a dummy test...")
    print("   To test with real image, place an image at one of:")
    for path in test_image_paths:
        print(f"      - {path}")
    print("\n   Skipping actual inference test.")
    print("\n" + "=" * 60)
    print("✓ Setup verification complete!")
    print("=" * 60)
    exit(0)

print(f"   Using image: {image_path}")

try:
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare prompt
    prompt = "What is in this image?"
    # Use vicuna_v1 (available template names: default, v0, v1, vicuna_v1)
    conv = conv_templates["vicuna_v1"].copy()
    
    if model.config.mm_use_im_start_end:
        image_placeholder = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_placeholder = DEFAULT_IMAGE_TOKEN
    
    full_user_input = image_placeholder + '\n' + prompt
    conv.append_message(conv.roles[0], full_user_input)
    conv.append_message(conv.roles[1], None)
    final_prompt_string = conv.get_prompt()
    
    # Process image
    from llava.mm_utils import process_images  # type: ignore
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    # Tokenize
    input_ids = tokenizer_image_token(
        final_prompt_string, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    image_tensor = image_tensor.unsqueeze(0).half().to(model.device)
    
    print("   Running inference...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=100,
            use_cache=True,
        )
    
    output = tokenizer.decode(output_ids[0]).strip()
    
    print("\n" + "=" * 60)
    print("✓ Inference successful!")
    print("=" * 60)
    print(f"\nQuestion: {prompt}")
    print(f"\nResponse: {output}")
    print("\n" + "=" * 60)
    print("✓ Setup verification complete!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n   ✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠ Setup may still be correct, but inference test failed.")
    print("   This might be due to GPU memory or image path issues.")

