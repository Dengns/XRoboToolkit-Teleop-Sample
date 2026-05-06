#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-ts_pico_teleop}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "未找到 conda 初始化脚本，请先安装 Miniconda 或 Anaconda。"
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] 检测到环境 $ENV_NAME 已存在，开始更新。"
    conda env update -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml" --prune
else
    echo "[INFO] 开始创建环境 $ENV_NAME。"
    conda env create -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml"
fi

conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt"

echo "[INFO] 环境安装完成。"
echo "[INFO] 激活命令: conda activate $ENV_NAME"
echo "[INFO] 启动命令: python test/realman_contrl_steamvr_tracker.py"
