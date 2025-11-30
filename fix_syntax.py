import os

path = "/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py"

print(f"🔧 Repairing {path}...")

if not os.path.exists(path):
    print("❌ File not found!")
    exit(1)

with open(path, "r") as f:
    content = f.read()

lines = content.splitlines()
new_lines = []
skip = False
fixed = False

for i, line in enumerate(lines):
    # 1. Find the start of the block
    if "if eos_token_id_tensor is not None:" in line:
        new_lines.append(line)
        
        # 2. Insert the CORRECT logic
        indent = line[:line.find("if")] + "    "
        new_lines.append(indent + "# Simplified EOS check (Fixed)")
        new_lines.append(indent + "is_eos = torch.isin(next_tokens, eos_token_id_tensor)")
        new_lines.append(indent + "unfinished_sequences = unfinished_sequences.mul((~is_eos).long())")
        
        # 3. Enable skipping to discard the old/broken lines
        skip = True
        fixed = True
        print("   -> Found target block. Replacing...")
        
    elif skip:
        # 4. Stop skipping when we hit the NEXT block
        if "if unfinished_sequences.max() == 0:" in line:
            new_lines.append(line)
            skip = False
        # else: We are discarding the broken lines (e.g. 'unfinished_sequences.mul(')
        
    else:
        new_lines.append(line)

if fixed:
    with open(path, "w") as f:
        f.write("\n".join(new_lines))
    print("✅ File repaired successfully!")
else:
    print("⚠️ Could not find the target block to fix. The file might be very different than expected.")
