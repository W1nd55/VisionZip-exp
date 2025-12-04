import torch
file_path = "/home/w1nd519994824/project/mittalshivam003/VisionZip-exp/models/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)" in line:
        # Replace with a simpler check: verify if next_tokens is NOT in eos_token_id_tensor
        indent = line[:line.find("next_tokens")]
        new_lines.append(indent + "# Simplified EOS check to avoid CUDA errors\n")
        new_lines.append(indent + "is_eos = torch.isin(next_tokens, eos_token_id_tensor)\n")
        new_lines.append(indent + "unfinished_sequences = unfinished_sequences.mul((~is_eos).long())\n")
    else:
        new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("✅ Applied generation fix!")
