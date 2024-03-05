import subprocess
import re
import math

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--program_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--task1", type=str, default="")
    parser.add_argument("--task2", type=str, default="")
    parser.add_argument("--task1_dense", type=float, default=1)
    parser.add_argument("--task2_dense", type=float, default=1)
    parser.add_argument("--nine_ratio", type=float, default=1)
    parser.add_argument("--eight_ratio", type=float, default=1)
    parser.add_argument("--eval_sh", type=str, default="")

    return parser.parse_args()

args = parse_args()



program_path = args.program_path
cache = args.cache
model_name = args.model_name
base_model_path = args.base_model_path

task1 = args.task1
task1_dense = args.task1_dense
task2 = args.task2
task2_dense = args.task2_dense

alpha = [args.nine_ratio, args.eight_ratio]
alpha_idx = 0
perfermence = 0

eight_sparse_model = f"{cache}/{model_name}/0.8_magnitude_owl_layer_wei_{model_name}"
nine_sparse_model = f"{cache}/{model_name}/0.9_magnitude_owl_layer_wei_{model_name}"



def get_model(alpha, add_base=True):
    eight_ratio_key = f'{alpha[1]:.1f}'
    nine_ratio_key = f'{alpha[0]:.1f}'

    model_save_path = f'{cache}/{model_name}_model/{eight_ratio_key}_{nine_ratio_key}_{model_name}'

    return_code = subprocess.run(['python', f'{program_path}/enhance_offset.py',
    '--eight_ratio_key', f'{eight_ratio_key}',
    '--nine_ratio_key', f'{nine_ratio_key}',

    '--eight_sparse_model', f'{eight_sparse_model}',
    '--nine_sparse_model', f'{nine_sparse_model}',

    '--offset_save_path', f'{model_save_path}'
    ])
    if add_base:
        return_code = subprocess.run([
        'python', f'{program_path}/offset_add_base_model.py',
            '--sparse_offset_path', f'{model_save_path}',
            '--base_model_path',f'{base_model_path}',
            '--sparse_save_path', f'{model_save_path}'
        ])
    return model_save_path



def math_eval_result(task, model_path):
    return_code = subprocess.run(['bash', args.eval_sh, task, model_path], text=True, stdout=subprocess.PIPE)
    retrun_txt = return_code.stdout

    retrun_txt = retrun_txt[retrun_txt.index("!!!!print result!!!!"):]
    split_txt = re.split('\n| ',retrun_txt)
    def deal(list_ori,p):   
        list_new=[]				
        list_short=[]			
        for i in list_ori:
            if i!=p:		
                list_short.append(i)
            else:
                list_short = list(filter(lambda x: x != '', list_short))
                list_new.append(list_short)
                list_short=[]  
        return list_new

    split_txt = deal(split_txt[2:], 'done')
    print(split_txt)
    txt = split_txt[0]
    # return eval score
    return float(txt[txt.index('ratio') + 1])

# Modify Function to Other SFT Model 
Func = None
if args.model_name=="Abel-7B-001":
    Func = math_eval_result



score = dict()
while(alpha_idx < len(alpha)):
    print(alpha)
    model_path = get_model(alpha)
    print(model_path)

    task1_result = Func(task1, model_path)
    task2_result = Func(task2, model_path)
    subprocess.run([f'rm -r {model_path}'], shell=True)
    
    temp_perfermence = math.sqrt((task1_result / task1_dense) * (task2_result / task2_dense))
    score[tuple(alpha)] = (temp_perfermence, task1_result, task2_result)
    print((temp_perfermence, task1_result, task2_result))
    

    if temp_perfermence > perfermence:
        perfermence = temp_perfermence
        alpha[alpha_idx] += 0.1
    else:
        if alpha[alpha_idx] > 1:
            alpha[alpha_idx] -= 0.1
        get_model(alpha, add_base=False)
        alpha_idx += 1
        if alpha_idx < len(alpha):
            if alpha[alpha_idx] < 0.1:
                alpha[alpha_idx] = 1
            else:
                alpha[alpha_idx] += 0.1

print(score)