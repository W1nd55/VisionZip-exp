#!/usr/bin/env python3
"""
Download and prepare MME benchmark dataset
"""
import os
import json
from datasets import load_dataset
from PIL import Image
import shutil

def download_mme():
    print("=" * 60)
    print("Downloading MME Benchmark Dataset")
    print("=" * 60)
    
    # Set output directory
    output_dir = "/u/m/a/maheenabooba/cs769/models/LLaVA/playground/eval/MME/MME_Benchmark_release_version"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n✓ Output directory: {output_dir}")
    
    # Download MME dataset from Hugging Face
    print("\n1. Downloading MME dataset from Hugging Face...")
    print("   (This may take a few minutes - dataset is ~864MB)")
    
    try:
        dataset = load_dataset("lmms-lab/MME", split="test")
        print(f"✓ Downloaded {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nTrying alternative approach...")
        # If Hugging Face download fails, provide manual instructions
        print("\nPlease download manually from:")
        print("  https://huggingface.co/datasets/lmms-lab/MME")
        return False
    
    # Extract and organize images by category
    print("\n2. Extracting and organizing images...")
    categories = set()
    image_count = 0
    
    for idx, item in enumerate(dataset):
        try:
            # Get category from the image path
            image_path = item['image_path'] if 'image_path' in item else item['question_id']
            category = os.path.dirname(image_path)
            categories.add(category)
            
            # Create category directory
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Save image
            image_filename = os.path.basename(image_path)
            output_path = os.path.join(category_dir, image_filename)
            
            # Get image from dataset
            if 'image' in item:
                img = item['image']
                if not os.path.exists(output_path):
                    img.save(output_path)
                    image_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(dataset)} images...")
                
        except Exception as e:
            print(f"   Warning: Failed to process item {idx}: {e}")
            continue
    
    print(f"\n✓ Extracted {image_count} images across {len(categories)} categories")
    print(f"  Categories: {sorted(categories)}")
    
    # Verify the structure matches llava_mme.jsonl
    jsonl_path = "/u/m/a/maheenabooba/cs769/models/LLaVA/playground/eval/MME/llava_mme.jsonl"
    print(f"\n3. Verifying against {jsonl_path}...")
    
    with open(jsonl_path, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    missing = []
    for q in questions[:10]:  # Check first 10
        img_path = os.path.join(output_dir, q['image'])
        if not os.path.exists(img_path):
            missing.append(q['image'])
    
    if missing:
        print(f"✗ Warning: {len(missing)} images not found:")
        for m in missing[:5]:
            print(f"   - {m}")
    else:
        print("✓ All checked images found!")
    
    print("\n" + "=" * 60)
    print("✓ MME dataset download complete!")
    print(f"  Location: {output_dir}")
    print(f"  Questions: {len(questions)} in llava_mme.jsonl")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    download_mme()

