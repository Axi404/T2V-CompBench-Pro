#!/bin/bash

conda create -n t2v python==3.10
conda activate t2v

cd LLaVA
pip install --upgrade pip # enable PEP 660 support 
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
cd ..

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

# LLaVA model weights — eval scripts expect ./weights/llava-v1.6-34b
mkdir -p weights
huggingface-cli download liuhaotian/llava-v1.6-34b --local-dir weights/llava-v1.6-34b

# ── Optional: Video-LLM backend (replaces LLaVA for MLLM categories) ──
# Usage: bash install.sh --videolm
if [[ "$1" == "--videolm" ]]; then
    echo "Installing Video-LLM dependencies (Qwen-VL)..."
    pip install "transformers>=4.57.0" accelerate
    pip install flash-attn --no-build-isolation --no-cache-dir
    pip install "qwen-vl-utils[decord]"  # only needed for Qwen2.5-VL fallback
    # Default model: Qwen3-VL-32B-Instruct (~66GB BF16, fits on 80GB GPU)
    # Weights are auto-downloaded on first run via HuggingFace.
    # To pre-download:
    #   huggingface-cli download Qwen/Qwen3-VL-32B-Instruct
fi