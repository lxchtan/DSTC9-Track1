#!/bin/bash

version="baseline_modify"
num_gpus=4

# End-to-end evaluation:
# This is a demonstration on how to generate responses with the trained models
# The input files are knowledge.json and logs.json. labels.json is not required
# We use the validation data in this example.

# Prepare directories for intermediate results of each subtask
mkdir -p pred2/val

# First we do knowledge-seeking turn detection on the test dataset
# Use --eval_dataset to specify the name of the dataset, in this case, val.
# Use --output_file to generate labels.json with predictions
# Specify --no_labels since there's no labels.json to read
python3 baseline_modify.py --eval_only --checkpoint runs/ktd-${version}/ \
   --eval_dataset val \
   --dataroot data \
   --no_labels \
   --output_file pred2/val/baseline_modify.ktd.json

# Next we do knowledge selection based on the predictions generated previously
# Use --labels_file to take the results from the previous task
# Use --output_file to generate labels.json with predictions
#python3 baseline_modify.py --eval_only --checkpoint runs/ks-all-${version} \
#   --eval_all_snippets \
#   --dataroot data \
#   --eval_dataset val \
#   --labels_file pred2/val/baseline_modify.ktd.json \
#   --output_file pred2/val/baseline_modify.ks.json
# To speed up the validation, can use distributed model
python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} baseline_modify.py --eval_only \
   --checkpoint runs/ks-all-${version} \
   --eval_all_snippets \
   --dataroot data \
   --eval_dataset val \
   --labels_file pred2/val/baseline_modify.ktd.json \
   --output_file pred2/val/baseline_modify.ks.json

# Finally we do response generation based on the selected knowledge
python3 baseline_modify.py --generate runs/rg-hml128-kml128-${version} \
        --generation_params_file baseline/configs/generation/generation_params.json \
        --eval_dataset val \
        --dataroot data \
        --labels_file pred2/val/baseline_modify.ks.json \
        --output_file baseline_modify_val.json
