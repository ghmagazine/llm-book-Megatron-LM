#!/bin/bash
#$ -l rt_C.large=1
#$ -l h_rt=0:30:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /bb/llm/gaf51275/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/8.9.7
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.4.0

# python virtualenv
source .env/bin/activate

INPUT_FILE=/groups/gag51395/datasets/raw/llm-book-ja-wiki/ja_wiki.jsonl
OUTPUT_DIR=/groups/gag51395/datasets/binarized/llm-book

mkdir -p ${OUTPUT_DIR}

python tools/preprocess_data.py \
  --input ${INPUT_FILE} \
  --output-prefix ${OUTPUT_DIR}/ja_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /groups/gag51395/hf-checkpoints/Llama-2-7b-hf/tokenizer.model \
  --append-eod \
  --workers 64
