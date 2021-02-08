#!/bin/bash
# set path to dataset here
postname=latentCopy_z5_ikp
dataroot=data_modify/add_stop
params_pre=baseline_modify/configs/generation
if [[ $1 == "train" ]]; then
  echo "start training: "${postname}
  CUDA_VISIBLE_DEVICES=0 python baseline_modify.py \
    --params_file ${params_pre}/params_latentCopy_z5_ikp.json \
    --dataroot ${dataroot} \
    --exp_name ${postname}
elif [[ $1 == "generate" ]]; then
  postsample=no_sample_greedy
  result=gdrg_${postname}_${postsample}_debug

  echo "start evaluation "${result}
  CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${postname} \
        --generation_params_file ${params_pre}//generation_params_greedy.json \
        --eval_dataset val \
        --p_debug \
        --dataroot ${dataroot} \
        --labels_file data_modify/add_stop/val/labels.json \
        --scorefile result/${result}.score.json \
        --output_file result/${result}.json
  CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot ${dataroot} \
    --outfile result/${result}.json \
    --scorefile result/${result}.score.json
fi