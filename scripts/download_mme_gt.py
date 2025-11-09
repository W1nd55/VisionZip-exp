#!/usr/bin/env python3
"""
Download MME Ground Truth files
"""
import os
import requests
import zipfile
import tempfile

def download_mme_gt():
    """Download MME benchmark with GT files"""
    
    mme_dir = "/u/m/a/maheenabooba/cs769/models/LLaVA/playground/eval/MME"
    gt_url = "https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/releases/download/v0.0.0/MME_Benchmark_release_version.zip"
    
    print("Downloading MME Benchmark with GT files...")
    
    # Try direct download
    try:
        response = requests.get(gt_url, stream=True, timeout=30)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            print(f"Extracting to {mme_dir}...")
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                zip_ref.extractall(mme_dir)
            
            os.unlink(tmp_file_path)
            print("✓ GT files downloaded successfully!")
            return True
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    # Alternative: Try to get from Hugging Face dataset
    print("\nTrying alternative: Extracting GT from Hugging Face dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("lmms-lab/MME", split="test")
        
        # Create GT structure
        gt_base = os.path.join(mme_dir, "MME_Benchmark_release_version")
        
        # Group by category
        categories = {}
        for item in dataset:
            cat = item.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # Create GT files
        for category, items in categories.items():
            cat_dir = os.path.join(gt_base, category)
            qa_dir = os.path.join(cat_dir, "questions_answers_YN")
            os.makedirs(qa_dir, exist_ok=True)
            
            # Group by image file
            by_image = {}
            for item in items:
                img_file = item.get('image_path', item.get('question_id', '')).split('/')[-1]
                base_name = img_file.replace('.png', '').replace('.jpg', '')
                if base_name not in by_image:
                    by_image[base_name] = []
                by_image[base_name].append(item)
            
            # Write GT files
            for base_name, items_list in by_image.items():
                gt_file = os.path.join(qa_dir, f"{base_name}.txt")
                with open(gt_file, 'w') as f:
                    for item in items_list:
                        question = item.get('question', item.get('text', ''))
                        answer = item.get('answer', item.get('gt_answer', 'yes'))  # Default fallback
                        # Clean question
                        question = question.replace('Answer the question using a single word or phrase.', '').strip()
                        if 'Please answer yes or no.' not in question:
                            question = question + ' Please answer yes or no.'
                        f.write(f"{question}\t{answer}\n")
        
        print(f"✓ Created GT files from Hugging Face dataset!")
        return True
        
    except Exception as e:
        print(f"Alternative method failed: {e}")
        print("\nPlease download manually from:")
        print("  https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models")
        print("  Look for MME_Benchmark_release_version.zip")
        return False

if __name__ == "__main__":
    download_mme_gt()

