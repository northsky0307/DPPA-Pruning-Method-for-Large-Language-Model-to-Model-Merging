gpu_start=5
gpu_end=6

function wait_gpu() {
    mem=1000000
    until [ $mem -le 100 ]
    do
        gpu_mem=$(nvidia-smi | \
            grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
            sed "s/^|//" | \
            awk '{print ($8" "$10)}' | \
            sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
            awk '{print $1}')
        i=0
        mem=0
        for s in $gpu_mem; do
            if [ $i -ge $gpu_start ]
            then
                if [ $i -le $gpu_end ]
                then
                    mem=`expr $s + $mem`
                fi
            fi
            i=`expr $i + 1`
        done
    done
    echo all clear
}

cache=/data/yczhu/yczhu_DPPA/DPPA
program_path=${cache}/scrips
model_path=${cache}/model
deal_model_path=${cache}/deal_model
base_model=${model_path}/llama2-7B 

# Model Setting
model_name="Abel-7B-001"
offset_model_name=${deal_model_path}/offset-llama2-abel
# Shell to eval SFT model
eval_sh="gair_abel/math_eval.sh"
# DPA Setting 
task1='gsm8k'
task2='math'
task1_dense=0.5981
task2_dense=0.13

# For DPA Method 1 and 2, just set different ratio to init.
nine_ratio=1.8
eight_ratio=0.0


task_list=($task1 $task2)
prun_path=${cache}/${model_name}
finetune_model=${model_path}/${model_name}

# Get offset
echo "Begin offset"
python ${program_path}/get_offset.py \
    --tuned_model_path ${finetune_model} \
    --base_model_path ${base_model} \
    --offset_save_path ${offset_model_name}


# DP
# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
ratio=(0.9 0.8)
# magnitude_owl_layer_wei magnitude_owl magnitude mario
method_list=(magnitude_owl_layer_wei)

len_ratio=${#ratio[*]}
len_method_list=${#method_list[*]}

range=$(($len_method_list*$len_ratio-1))





deal_model_list=()

cnt=$gpu_start
cmd=''
add_base_cmd=''
for idx in $(seq 0 $range)
do
    remainder_ratio=$(($idx%$len_ratio))
    remainder_method_list=$(($idx/$len_ratio))
    method=${method_list[$remainder_method_list]} 
    rate=${ratio[$remainder_ratio]} 

    echo $rate $method

        pruning_offset=${prun_path}/${rate}_${method}_${model_name}
        lamda=0.08
        pruning_python="CUDA_VISIBLE_DEVICES=$cnt python ${program_path}/pruning/main.py --model_name_or_path ${offset_model_name} --Lamda ${lamda} --Hyper_m 5 --model ${offset_model_name} --prune_method $method --sparsity_ratio $rate --sparsity_type unstructured --save_model ${pruning_offset}"


        pruning_add_base_model=${pruning_offset}_add
        add_base_python="python ${program_path}/offset_add_base_model.py --sparse_offset_path ${pruning_offset} --base_model_path ${base_model} --sparse_save_path ${pruning_add_base_model}"
        deal_model_list+=( $pruning_add_base_model)

        if [ $cnt -ne $gpu_start ];
        then
            add_base_cmd+=' & '
            cmd+=' & '
        fi
        cmd+=$pruning_python
        add_base_cmd+=$add_base_python
        cnt=`expr $cnt + 1`

        if [ $cnt -gt $gpu_end ];
        then
            cnt=$gpu_start
            wait_gpu
            eval $cmd
            wait_gpu
            eval $add_base_cmd
            wait_gpu
            cmd=''
            add_base_cmd=''
        fi
done
wait_gpu 
eval $cmd
wait_gpu
eval $add_base_cmd
wait_gpu

# Eval DP Performence
for task in $task_list 
do
    cnt=$gpu_start
    cmd=''
    add_base_cmd=''
    for deal_model in ${deal_model_list[@]}
    do

            eval_python="bash $eval_sh $task $deal_model $cnt"
            if [ $cnt -ne $gpu_start ];
            then
                cmd+=' & '
            fi
            cmd+=$eval_python
            cnt=`expr $cnt + 1`

            if [ $cnt -gt $gpu_end ];
            then
                cnt=$gpu_start
                wait_gpu
                eval $cmd
                wait_gpu
                cmd=''
            fi
    done
    wait_gpu 
    eval $cmd
    wait_gpu

done

#DPA
python ${program_path}/python_contral.py --cache ${cache}  --base_model_path ${base_model} --program_path ${program_path} --model_name ${model_name} \
    --task1 ${task1} --task2 ${task2} --task1_dense ${task1_dense} --task2_dense ${task2_dense} \
    --nine_ratio ${nine_ratio} --eight_ratio ${eight_ratio} --eval_sh ${eval_sh}