#!/bin/bash

export TRAIN_SCRIPT_PATH="$(realpath "$0")"


EXPERIMENTS="90-1-1-1 91-1-1-1"

MODEL_NAME='Scratch'
GPU_NUM=0
DDP=1
NUM_EPOCH=25
BATCH_SIZE=4
LR=0.0006
LR_SCHEDULING=1
LR_SCHEDULER_TYPE='CosineAnnealing'
OBS_SEC=0.5
NUM_CORES=6

# DDP configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8
HOST_NODE_ADDR=10000
BOOL_MIXED_PRECISION=1
BOOL_ONE2MANY=1

# Iterate over experiments
for exp_config in $EXPERIMENTS; do

    EXP_ID=$(echo "$exp_config" | cut -d'-' -f1)  
    BOOL_DEPTH_AUX=$(echo "$exp_config" | cut -d'-' -f2)  
    BOOL_BEVSEG_AUX=$(echo "$exp_config" | cut -d'-' -f3)  
    BOOL_PVSEG_AUX=$(echo "$exp_config" | cut -d'-' -f4)  

    
    echo "========================================="
    echo "Starting experiment ID: $EXP_ID"
    echo "BOOL_DEPTH_AUX: $BOOL_DEPTH_AUX"
    echo "BOOL_BEVSEG_AUX: $BOOL_BEVSEG_AUX"
    echo "BOOL_PVSEG_AUX: $BOOL_PVSEG_AUX"
    echo "========================================="

    # Training
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nnodes=1 \
        --nproc_per_node=$NPROC_PER_NODE \
        --max_restarts=0 \
        --rdzv_id=$GPU_NUM \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:$HOST_NODE_ADDR \
        train.py \
        --model_name $MODEL_NAME \
        --exp_id $EXP_ID \
        --gpu_num $GPU_NUM \
        --ddp $DDP \
        --num_cores $NUM_CORES \
        --num_epochs $NUM_EPOCH \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --apply_lr_scheduling $LR_SCHEDULING \
        --lr_schd_type $LR_SCHEDULER_TYPE \
        --past_horizon_seconds $OBS_SEC \
        --bool_mixed_precision $BOOL_MIXED_PRECISION \
        --bool_depth_aux $BOOL_DEPTH_AUX \
        --bool_pvseg_aux $BOOL_PVSEG_AUX \
        --bool_bevseg_aux $BOOL_BEVSEG_AUX \
        --bool_one2many $BOOL_ONE2MANY

    # Evaluation
: '    
    python test_3Dlane.py \
        --model_name $MODEL_NAME \
        --exp_id $EXP_ID \
        --gpu_num $GPU_NUM \
        --visualization 0 \
        --is_test_all 1
'    
    echo "Completed experiment ID: $EXP_ID"
    echo ""
done

echo "All experiments completed!"


