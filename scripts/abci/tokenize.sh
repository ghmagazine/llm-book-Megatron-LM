#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:30:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use path/to/modules/modulefiles/
module use /apps/modules-abci-2.0-2022/modulefiles/rhel8/compilers

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/8.9.7
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.2.0

# python virtualenv
source .env/bin/activate

INPUT_FILE=
OUTPUT_DIR=

mkdir -p ${OUTPUT_DIR}

python tools/preprocess_data.py \
  --input ${INPUT_FILE} \
  --output-prefix ${OUTPUT_DIR}/ja_wiki_text_document \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model  \
  --append-eod \
  --workers 64
