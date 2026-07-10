#!/bin/bash
# Geo_sim 一键安装脚本
# 支持 macOS / Linux / Windows (WSL)

set -e

echo "╔══════════════════════════════════════════╗"
echo "║   Geo_sim — 地幔对流仿真框架 安装程序      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 检查Python
PYTHON=$(which python3 || which python)
echo "→ Python: $($PYTHON --version)"

# 安装核心依赖
echo "→ 安装核心依赖..."
$PYTHON -m pip install --quiet numpy scipy matplotlib pyyaml tqdm

# 检测平台并安装加速包
OS=$(uname -s)
ARCH=$(uname -m)

echo "→ 检测平台: $OS / $ARCH"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "  ✓ Apple Silicon (M1/M2/M3) — MPS GPU可用"
    echo "→ 安装PyTorch (MPS后端)..."
    $PYTHON -m pip install --quiet torch torchvision
    echo "→ 安装Numba (ARM64 JIT)..."
    $PYTHON -m pip install --quiet numba
elif [[ "$OS" == "Linux" ]]; then
    echo "  ✓ Linux — CUDA/MPS检测中"
    if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  CUDA GPU 可用"
    fi
    $PYTHON -m pip install --quiet numba
elif [[ "$OS" == MINGW* ]] || [[ "$OS" == MSYS* ]]; then
    echo "  ✓ Windows — CPU模式"
    $PYTHON -m pip install --quiet numba
fi

# 安装可选可视化
echo "→ 安装可视化依赖 (可选)..."
$PYTHON -m pip install --quiet meshio 2>/dev/null || echo "  (meshio跳过)"

# 安装项目
echo "→ 安装Geo_sim..."
$PYTHON -m pip install -e . --quiet 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ✓ 安装完成!                             ║"
echo "║                                          ║"
echo "║   快速开始:                               ║"
echo "║     geo-sim check    检查环境             ║"
echo "║     geo-sim demo     运行演示             ║"
echo "║     geo-sim benchmark 运行基准测试         ║"
echo "╚══════════════════════════════════════════╝"
