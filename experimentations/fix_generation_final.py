import os

file_path = "/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py"

# The exact block to find and replace
target_block = """                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )"""

# The replacement block
replacement_block = """                # Simplified EOS check
                is_eos = torch.isin(next_tokens, eos_token_id_tensor)
                unfinished_sequences = unfinished_sequences.mul((~is_eos).long())"""

with open(file_path, 'r') as f:
    content = f.read()

if target_block in content:
    new_content = content.replace(target_block, replacement_block)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✅ Successfully replaced the problematic code block!")
else:
    print("⚠️ Target block not found. It might have been modified already.")
    # Fallback: check if it's partially modified and fix it
    if "is_eos = torch.isin" in content and "SyntaxError" not in content:
         print("Looks like the fix is already applied.")

