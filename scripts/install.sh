#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
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

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja

pip install -r requirements.txt
pip install zarr tensorstore

# apex install
cd ..
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.6

# flash attention install
pip uninstall flash-attn

cd ..
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.4.2
pip install -e .
