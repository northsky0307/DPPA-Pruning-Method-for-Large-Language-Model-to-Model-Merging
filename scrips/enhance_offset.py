from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import torch
import os
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eight_sparse_model", type=str, default='')
    parser.add_argument("--nine_sparse_model", type=str, default='')

    parser.add_argument("--eight_ratio_key", type=float, default=0.0)
    parser.add_argument("--nine_ratio_key", type=float, default=0.0)

    parser.add_argument("--offset_save_path", type=str, default='')
    return parser.parse_args()

args = parse_args()

eight_model= AutoModelForCausalLM.from_pretrained(args.eight_sparse_model)
eight = eight_model.state_dict()

nine_model= AutoModelForCausalLM.from_pretrained(args.nine_sparse_model)
nine = nine_model.state_dict()
tokenizer = AutoTokenizer.from_pretrained(args.nine_sparse_model)

def my_max(x, y):
    x_mask = x != 0
    y_mask = y != 0
    if y_mask.sum() == 0:
        return x
    x_abs = torch.abs(x)
    y_abs = torch.abs(y)
    update = y_abs > x_abs
    update += (y_mask.long() - x_mask.long()) > 0
    x[update] *= 0
    x[update] += y[update]
    return x


for key in tqdm(nine):

    add = torch.zeros(nine[key].shape)
    add = my_max(add, eight[key] * args.eight_ratio_key)
    add = my_max(add, nine[key] * args.nine_ratio_key)

    nine[key] *= 0
    nine[key] += add



tokenizer.save_pretrained(args.offset_save_path)
nine_model.save_pretrained(args.offset_save_path)
print("enhance offset finished")