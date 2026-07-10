#!/usr/bin/env python3
"""Geo_sim 完整演示 — 安装→仿真→画图→保存"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║          Geo_sim — 完整演示                        ║")
    print("║   地幔对流仿真框架 (对标Underworld, 更易用)         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # ── 环境 ──
    print("【1/4】环境检查")
    deps_ok = True
    for mod in ['numpy', 'scipy', 'matplotlib', 'tqdm']:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except ImportError:
            print(f"  ✗ {mod} — 请运行: pip install {mod}")
            deps_ok = False

    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✓ Apple MPS GPU (加速ML训练)")
        elif torch.cuda.is_available():
            print("  ✓ CUDA GPU (加速ML训练)")
    except ImportError:
        print("  - PyTorch未安装 (ML加速不可用)")

    try:
        import numba
        print("  ✓ Numba JIT (加速FEM装配)")
    except ImportError:
        print("  - Numba未安装 (pip install numba)")

    if not deps_ok:
        print("\n请先安装缺失依赖:")
        print("  pip install numpy scipy matplotlib tqdm")
        return 1

    # ── 仿真 ──
    print("\n【2/4】运行地幔对流仿真")
    from core.stokes_solver import StokesConfig, PicardStokesSolver

    config = StokesConfig(
        nx=32, ny=32, rayleigh=1e5,
        max_picard_iterations=30, picard_tolerance=1e-3,
        max_time_steps=80, dt=1e-3,
        output_dir='./demo_output',
    )

    solver = PicardStokesSolver(config)
    history = solver.run(n_steps=80, verbose=False, show_progress=True)

    nu_final = history['nusselt'][-1]
    print(f"\n  最终 Nusselt 数: {nu_final:.3f}")
    print(f"  平均 Picard 迭代: {np.mean(history['picard_iterations']):.1f}")

    # ── 画图 ──
    print("\n【3/4】生成可视化图表")
    solver.plot(field='all', save=True, show=False)

    # 时间序列
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history['time'], history['nusselt'], 'b-', lw=2)
        ax.axhline(y=10.53, color='r', ls='--', alpha=0.7, label='Blankenbach ref (Ra=10⁵)')
        ax.set_xlabel('Time'); ax.set_ylabel('Nusselt Number')
        ax.set_title('Nusselt Number vs Time')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('./demo_output/nusselt_timeseries.png', dpi=150)
        plt.close()
        print("  ✓ 时间序列图已保存")
    except Exception as e:
        print(f"  ! {e}")

    # ── 保存 ──
    print("\n【4/4】保存结果")
    solver.save('./demo_output/convection_result')
    print("  ✓ 结果已保存到 ./demo_output/")

    print("\n" + "=" * 50)
    print("演示完成!")
    print("查看结果: open demo_output/")
    print("=" * 50)
    return 0


if __name__ == '__main__':
    sys.exit(main())
