from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pathlib
import torch
import warnings



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_offset_path", type=str, default="")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--sparse_save_path", type=str, default="")

    return parser.parse_args()

args = parse_args()

sparse_model = AutoModelForCausalLM.from_pretrained(args.sparse_offset_path)
sparse = sparse_model.state_dict()
tokenizer = AutoTokenizer.from_pretrained(args.sparse_offset_path)


base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
base = base_model.state_dict()

for key in tqdm(sparse):
    base[key].to(sparse[key].dtype)
    if sparse[key].shape != base[key].shape:
        warnings.warn(f"{key} {sparse[key].shape} , {base[key].shape}")
        sparse[key] = sparse[key][:base[key].shape[0]]
    sparse[key] += base[key]

print("BEGIN")
pathlib.Path(args.sparse_save_path).mkdir(parents=True, exist_ok=True) 
sparse_model.save_pretrained(args.sparse_save_path)
tokenizer.save_pretrained(args.sparse_save_path)
print("finish save")
