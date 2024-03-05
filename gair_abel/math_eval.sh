
gpu_start=0
gpu_end=7


function choose_gpu() {
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
        for s in $gpu_mem; do
            if [ $i -ge $gpu_start ]
            then
                if [ $i -le $gpu_end ]
                then
                    if [ $s -le 100 ]
                    then
                        return $i
                    fi
                fi
            fi
            i=`expr $i + 1`
        done
    done
}
free_gpu=$3

if [ ! $free_gpu ]
then
    choose_gpu
    free_gpu=$?
fi



task=$1
model_path=$2
CRTDIR=$(pwd)
cd ./gair_abel

txt="CUDA_VISIBLE_DEVICES=$free_gpu python -m evaluation.inference --model_dir $model_path --temperature 0.0 --top_p 1.0 --output_file ./outputs/${task}/prun$model_path.jsonl --dev_set ${task}  --prompt_type math-single"
eval $txt
echo !!!!!!!!!!!!!!!!!!!!print result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
eval $txt "--eval_only True"
cd $CRTDIR