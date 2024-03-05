from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import warnings
import pathlib
import torch

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned_model_path", type=str, default="")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--offset_save_path", type=str, default="")
    return parser.parse_args()

args = parse_args()

tuned_model = AutoModelForCausalLM.from_pretrained(args.tuned_model_path, torch_dtype="auto")
tuned = tuned_model.state_dict()
tokenizer = AutoTokenizer.from_pretrained(args.tuned_model_path, torch_dtype="auto")


base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype="auto")
base = base_model.state_dict()


for key in tqdm(tuned):
    base[key].to(tuned[key].dtype)
    if tuned[key].shape != base[key].shape:
        warnings.warn(f"{key} {tuned[key].shape} , {base[key].shape}")
        tuned[key] = tuned[key][:base[key].shape[0]]
    tuned[key] -= base[key]

pathlib.Path(args.offset_save_path).mkdir(parents=True, exist_ok=True) 
tuned_model.save_pretrained(args.offset_save_path) 
tokenizer.save_pretrained(args.offset_save_path)

print("Finish")




