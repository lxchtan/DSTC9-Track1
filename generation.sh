#!/bin/bash

# This script demonstrates how to train baseline models with this repo
# response generation
# And we show how to generate responses for test dataset without labels.json at the end

# set path to dataset here

# --- 20200803 ----
prename=rg-hml128-kml128
if  [[ $1 == 1 ]]; then
  postname=baseline_gpt2_debug
  if [[ x$2 == x ]]; then
    CUDA_VISIBLE_DEVICES=0 python3 baseline.py \
    --params_file baseline/configs/generation/params.json \
    --dataroot data \
    --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_beam_g2s8
    result=gdrg_${postname}_${postsample}
    CUDA_VISIBLE_DEVICES=1 python3 baseline.py --generate runs/${prename}-${postname} \
        --generation_params_file baseline/configs/generation/generation_params.json \
        --eval_dataset val \
        --dataroot data \
        --labels_file data/val/labels.json \
        --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
        --outfile result/${result}.json \
        --scorefile result/${result}.score.json
  fi

# PLATO fourstep
elif [[ $1 == 2 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_beam_g8s2
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 3 ]]; then
    postsample=no_sample_beam_g4s4
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 4 ]]; then
    postsample=sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_sample_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 5 ]]; then
    postsample=sample_beam_g4s4
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_sample_beam.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 7 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_fourstep_priorbehind
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 9 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep_second
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_beam_g4s4
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 3 ]]; then
    postsample=sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_sample_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 30 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z128_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --z_hidden_size 128 \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 31 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_nobow
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --loss_stage nobow \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 37 ]]; then
  postname=baseline_modify_plato_real_base_bz4_gas32_ep10_z5_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --dataroot data \
      --plato_modify real \
      --exp_name ${prename}-${postname}
#  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 38 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_kld_epsilon
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --loss_stage kld_epsilon \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 39 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_MaskKnowledge_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --plato_dataset_modify MaskKnowledge \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 40 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_kld_epsilon_normal
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --loss_stage kld_epsilon_normal \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 41 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_kld_epsilon_normal_real
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --loss_stage kld_epsilon_normal \
      --plato_modify real \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 42 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z19_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --z_hidden_size 19 \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 43 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_kld_epsilon_normal_02
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --loss_stage kld_epsilon_normal \
      --kld_epsilon 0.2 \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 44 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_kld_epsilon_normal_real_1
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --loss_stage kld_epsilon_normal \
      --plato_modify real \
      --kld_epsilon 1.0 \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi

# PLATO_COPY
elif [[ $1 == 45 ]]; then
  postname=baseline_modify_platoCopy_base_bz4_gas32_ep10_z5_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_platoCopy_z5.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  # elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi

elif [[ $1 == 10 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z64_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z64.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 11 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z1_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z1.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 12 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z2_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=2 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z2.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 13 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z3_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z2.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 14 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z4_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=2 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z4.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 15 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z6_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z6.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 16 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z8_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z8.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 17 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z10_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=2 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z10.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 18 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z12_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z12.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 19 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z14_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z14.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 20 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z14_fourstep_debug
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z14.json \
      --dataroot data \
      --persona_dataset \
      --exp_name ${prename}-${postname}
  fi
elif [[ $1 == 25 ]]; then
  postname=baseline_modify_plato_large_bz1_gas128_ep10_z5_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_large_z5.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_beam_g4s4
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 27 ]]; then
  postname=baseline_modify_plato_large_bz1_gas128_ep10_z5_fourstep_after_ep10
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_large_z5_after_ep10.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_beam_g4s4
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 29 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep20_z5_fourstep_lr1e4
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5_ep20.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 34 ]]; then
  postname=baseline_modify_plato_base_bz4_gas64_ep5_z5_fourstep_percentage
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=2 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5_bz256.json \
      --loss_stage fourstep_percentage \
      --dataroot data \
      --exp_name ${prename}-${postname}
#  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 35 ]]; then
  postname=baseline_modify_plato_base_bz4_gas16_ep20_z5_fourstep_percentage
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5_bz64.json \
      --loss_stage fourstep_percentage \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi

# PLATO-NS
elif [[ $1 == 21 ]]; then
  postname=baseline_modify_platoNS_base_bz2_gas64_ep10_z5_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_platoNS_z5.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}_checkpoint-149
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname}/checkpoint-149 \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 22 ]]; then
  postname=baseline_modify_platoNS_base_bz2_gas64_ep10_z5_normal
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_platoNS_z5.json \
      --dataroot data \
      --loss_stage normal \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}_checkpoint-149
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname}/checkpoint-149 \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 24 ]]; then
  postname=baseline_modify_platoNS_base_bz2_gas64_ep10_z20_normal
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_platoNS_z20.json \
      --dataroot data \
      --loss_stage normal \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi

# DataModify: Wrong Selection
# PLATO
elif [[ $1 == 23 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep_noisy
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --dataroot data_modify/noisy_modify \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data_modify/noisy_modify \
          --labels_file data_modify/noisy_modify/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data_modify/noisy_modify \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  # After Selection
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_greedy
    result=select_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file result/baseline_modify_large.ks.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 3 ]]; then
    postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep
    postsample=no_sample_greedy
    result=select_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file result/baseline_modify_large.ks.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 4 ]]; then
    postsample=no_sample_greedy
    result=select_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python scripts/scores2.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --outfile2 result/${result}.json \
      --scorefile result/${result}.tp_score.json
    postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep
    result=select_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python scripts/scores2.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --outfile2 result/${result}.json \
      --scorefile result/${result}.tp_score.json
  elif [[ $2 == 5 ]]; then
    postsample=no_sample_greedy
    result=select_sdes_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file result/separate_domain_elec_seq.ks.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 6 ]]; then
    postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep
    postsample=no_sample_greedy
    result=select_sdgs_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file result/separate_domain.ks.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 7 ]]; then
    postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep
    postsample=sample_greedy
    result=select_${postname}_${postsample}
    echo $result
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_sample_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file result/baseline_modify_large.ks.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 26 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep_noisy_add
  dataroot=data_modify/noisy_add
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=3 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5.json \
      --dataroot ${dataroot} \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot ${dataroot} \
          --labels_file data_modify/noisy_modify/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot ${dataroot} \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  # After Selection
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_greedy
    result=select_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file result/baseline_modify_large.ks.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 3 ]]; then
    postsample=no_sample_greedy
    result=select_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python scripts/scores2.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --outfile2 result/${result}.json \
      --scorefile result/${result}.tp_score.json
  fi

# After persona pretrained
# change "params_roberta.json" to "params_roberta_after_persona.json"
elif [[ $1 == 3 ]]; then
  postname=baseline_modify_roberta_base_persona_bz8_gas16_ep12_roberta_base_bz8_gas16_ep3
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_roberta_after_persona.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  fi
# change "params_roberta.json" to "params_roberta_after_persona.json"
elif [[ $1 == 5 ]]; then
  postname=baseline_modify_roberta_base_persona_bz8_gas16_ep12_roberta_base_bz8_gas16_ep7
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_roberta_after_persona.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
      postsample=no_sample_greedy
      result=gdrg_${postname}_${postsample}
      echo "start evaluation: "${postsample}
      CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
            --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
            --eval_dataset val \
            --dataroot data \
            --labels_file data/val/labels.json \
            --output_file result/${result}.json
      CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
        --outfile result/${result}.json \
        --scorefile result/${result}.score.json
  fi
# pipeline persona
elif [[ $1 == 28 ]]; then
  postname=baseline_modify_roberta_large_add_persona_bz1_gas128_ep10
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_roberta_large.json \
      --dataroot data \
      --persona_dataset \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
      postsample=no_sample_greedy
      result=gdrg_${postname}_${postsample}
      echo "start evaluation: "${postsample}
      CUDA_VISIBLE_DEVICES=0 python baseline_modify.py --generate runs/${prename}-${postname} \
            --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
            --eval_dataset val \
            --dataroot data \
            --labels_file data/val/labels.json \
            --scorefile result/${result}.score.json \
            --output_file result/${result}.json
      CUDA_VISIBLE_DEVICES=0 python scripts/scores.py --dataset val --dataroot data/ \
        --outfile result/${result}.json \
        --scorefile result/${result}.score.json
  fi

# PLATO after persona
elif [[ $1 == 6 ]]; then
  postname=baseline_modify_roberta_base_persona_bz8_gas16_ep12_plato_base_bz4_gas32_ep10_fourstep
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=0 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_after_persona.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
      postsample=no_sample_greedy
      result=gdrg_${postname}_${postsample}
      echo "start evaluation: "${postsample}
      CUDA_VISIBLE_DEVICES=3 python baseline_modify.py --generate runs/${prename}-${postname} \
            --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
            --eval_dataset val \
            --dataroot data \
            --labels_file data/val/labels.json \
            --output_file result/${result}.json
      CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
        --outfile result/${result}.json \
        --scorefile result/${result}.score.json
  fi

# Normal GPT
elif [[ $1 == 4 ]]; then
  postname=baseline
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=2 python3 baseline.py \
      --params_file baseline/configs/generation/params.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=select_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python3 baseline.py --generate runs/${prename}-${postname} \
      --generation_params_file baseline/configs/generation/generation_params.json \
      --eval_dataset val \
      --dataroot data \
      --labels_file result/baseline_modify_large.ks.json \
      --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 2 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python3 baseline.py --generate runs/${prename}-${postname} \
     --generation_params_file baseline/configs/generation/generation_params.json \
     --eval_dataset val \
     --dataroot data \
     --labels_file data/val/labels.json \
     --output_file result/${result}.json \
     --scorefile result/${result}.score.json
    CUDA_VISIBLE_DEVICES=3 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  elif [[ $2 == 3 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    result2=select_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python scripts/scores2.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --outfile2 result/${result2}.json \
      --scorefile result/${result}.tp_score.json
  elif [[ $2 == 4 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation: "${postsample}
    CUDA_VISIBLE_DEVICES=3 python scripts/scores2.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --outfile2 result/${result}.json \
      --scorefile result/${result}.tp_score2.json
  fi

# Prefinetune
elif [[ $1 == 32 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep_after_prefinetune
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5_after_prefinetune.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
elif [[ $1 == 36 ]]; then
  postname=baseline_modify_plato_base_bz4_gas32_ep10_z5_fourstep_after_prefinetune1
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=2 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_plato_z5_after_prefinetune1.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
#   elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=2 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=2 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi
# Roberta
elif [[ $1 == 33 ]]; then
  postname=baseline_modify_roberta_base_bz8_gas16_ep10_after_prefinetune
  echo $postname
  if [[ x$2 == x ]]; then
    echo "start training"
    CUDA_VISIBLE_DEVICES=1 python3 baseline_modify.py \
      --params_file baseline_modify/configs/generation/params_roberta.json \
      --dataroot data \
      --exp_name ${prename}-${postname}
  elif [[ $2 == 1 ]]; then
    postsample=no_sample_greedy
    result=gdrg_${postname}_${postsample}
    echo "start evaluation"
    CUDA_VISIBLE_DEVICES=1 python baseline_modify.py --generate runs/${prename}-${postname} \
          --generation_params_file baseline_modify/configs/generation/generation_params_greedy.json \
          --eval_dataset val \
          --dataroot data \
          --labels_file data/val/labels.json \
          --scorefile result/${result}.score.json \
          --output_file result/${result}.json
    CUDA_VISIBLE_DEVICES=1 python scripts/scores.py --dataset val --dataroot data/ \
      --outfile result/${result}.json \
      --scorefile result/${result}.score.json
  fi

# please start with 41
fi
