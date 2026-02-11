#!/bin/bash

set -e

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # Make `conda activate` work in non-interactive shells (no `conda init` needed).
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda.sh under ${CONDA_BASE}/etc/profile.d/."
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "t2v"; then
    echo "[INFO] Conda env 't2v' already exists, reusing it."
else
    conda create -n t2v python==3.10 -y
fi
conda activate t2v

detect_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        echo "${CUDA_HOME}"
        return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"
        return 0
    fi

    for candidate in /usr/local/cuda /usr/local/cuda-12.4 /opt/cuda; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

if CUDA_HOME_DETECTED="$(detect_cuda_home)"; then
    export CUDA_HOME="${CUDA_HOME_DETECTED}"
else
    echo "[ERROR] CUDA toolkit not found (nvcc missing)."
    exit 1
fi

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "[INFO] CUDA_HOME=${CUDA_HOME}"
nvcc --version

pip install --upgrade pip setuptools wheel

# Install CUDA-enabled torch first (required before flash-attn build).
# Pin by default to a combo that has stable flash-attn wheel support.
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"

pip install --upgrade --force-reinstall \
    torch=="${TORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] torch.cuda:", torch.version.cuda)
print("[INFO] cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
if torch.version.cuda is None:
    raise SystemExit("[ERROR] Installed torch is CPU-only; flash-attn requires CUDA torch.")
PY

pip install "transformers>=4.57.0" accelerate ninja packaging

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
PY_TAG="${PY_TAG:-cp310-cp310}"
ABI_TAG="$(python - <<'PY'
import torch
print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
PY
)"
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}%2Bcu12torch2.5cxx11abi${ABI_TAG}-${PY_TAG}-linux_x86_64.whl"

echo "[INFO] Try flash-attn wheel: ${FLASH_ATTN_WHEEL_URL}"
if ! pip install "${FLASH_ATTN_WHEEL_URL}"; then
    echo "[WARN] Prebuilt wheel install failed, fallback to source build."
    MAX_JOBS="${MAX_JOBS:-$(nproc)}" FLASH_ATTN_FORCE_BUILD=TRUE \
        pip install flash-attn=="${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir --verbose
fi

cd GSA
python -m pip install -e segment_anything
# Avoid downgrading the Qwen3 runtime stack by excluding pinned torch/transformers deps.
GSA_FILTERED_REQ="$(mktemp)"
grep -Ev '^(torch|torchaudio|torchvision|transformers|tokenizers|huggingface-hub)==.*$' \
    requirements.txt > "${GSA_FILTERED_REQ}"
pip install --no-build-isolation -r "${GSA_FILTERED_REQ}"
rm -f "${GSA_FILTERED_REQ}"
# GroundingDINO weights — eval scripts expect GSA/groundingdino_swint_ogc.pth
wget -nc https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# SAM weights — eval scripts expect GSA/sam_vit_h_4b8939.pth
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

cd dot
wget -nc -P checkpoints https://hf-mirror.com/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -nc -P checkpoints https://hf-mirror.com/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -nc -P checkpoints https://hf-mirror.com/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -nc -P checkpoints https://hf-mirror.com/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -nc -O checkpoints/movi_f_cotracker3_wind_60.pth https://hf-mirror.com/facebook/cotracker3/resolve/main/scaled_offline.pth
wget -nc -P checkpoints https://hf-mirror.com/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -nc -P checkpoints https://hf-mirror.com/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth
cd ..

# Qwen3-VL model weights are auto-downloaded on first run via HuggingFace.
# To pre-download:
#   huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir weights/Qwen3-VL-32B-Instruct
mkdir -p weights

huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir weights/Qwen3-VL-32B-Instruct

# Re-assert the Qwen3 runtime stack in case other deps modified it.
pip install --upgrade --force-reinstall \
    torch=="${TORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade "transformers>=4.57.0" "tokenizers>=0.21,<0.23" "huggingface-hub>=0.34.0"

# Keep ABI-compatible scientific stack for OpenCV/Gradio.
pip install --upgrade --force-reinstall \
    "numpy==1.26.4" \
    "pillow<11,>=10.0.0" \
    "opencv-python==4.9.0.80"
