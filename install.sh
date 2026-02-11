#!/bin/bash

conda create -n t2v python==3.10
conda activate t2v

pip install --upgrade pip
pip install "transformers>=4.57.0" accelerate
pip install flash-attn --no-build-isolation --no-cache-dir

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda

cd GSA
python -m pip install -e segment_anything
pip install --no-build-isolation -r requirements.txt
# GroundingDINO weights — eval scripts expect GSA/groundingdino_swint_ogc.pth
wget -nc https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# SAM weights — eval scripts expect GSA/sam_vit_h_4b8939.pth
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

cd dot
wget -nc -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -nc -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -nc -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -nc -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -nc -O checkpoints/movi_f_cotracker3_wind_60.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget -nc -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -nc -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth
cd ..

# Qwen3-VL model weights are auto-downloaded on first run via HuggingFace.
# To pre-download:
#   huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir weights/Qwen3-VL-32B-Instruct
mkdir -p weights
