#!/bin/bash
set -x
GPUS_PER_NODE=16
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(($PHILLY_CONTAINER_PORT_RANGE_START + 23))
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#DATA_PATH=<Specify path and file prefix>_text_sentence
DATA_PATH=/philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/Megatron-LM/my-bert_text_sentence
#CHECKPOINT_PATH=<Specify path>

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --data-path $DATA_PATH \
       --vocab-file /philly/rr3/msrhyperprojvc2_scratch/saemal/amir/data/wikidata/raid/Megatron-LM/bert-large-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
#       --fp16
#       --save $CHECKPOINT_PATH \
#       --load $CHECKPOINT_PATH \
