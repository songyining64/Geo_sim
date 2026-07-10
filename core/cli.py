"""
CLI入口 — pip install geo_sim 后可直接使用命令
"""

import argparse
import sys
from pathlib import Path


def ensure_path():
    sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_demo():
    """运行演示仿真"""
    ensure_path()
    from core.stokes_solver import StokesConfig, PicardStokesSolver
    import numpy as np

    print("╔══════════════════════════════════════════╗")
    print("║   Geo_sim — 地幔对流演示                  ║")
    print("╚══════════════════════════════════════════╝")
    print()

    config = StokesConfig(nx=32, ny=32, rayleigh=1e5,
                          max_picard_iterations=30, picard_tolerance=1e-3,
                          max_time_steps=100, dt=1e-3,
                          output_dir='./demo_output')

    solver = PicardStokesSolver(config)
    history = solver.run(n_steps=100, verbose=True, show_progress=True)

    solver.save()
    solver.plot()
    solver.plot(field='viscosity')
    print(f"\n✓ 演示完成! 图表保存在 {config.output_dir}/")
    return 0


def cmd_benchmark():
    """运行Blankenbach基准"""
    ensure_path()
    from core.numba_accelerator import blankenbach_benchmark
    blankenbach_benchmark(max_ra=1e6, nx_base=16, n_steps=100)
    return 0


def cmd_install_check():
    """检查安装环境"""
    print("Geo_sim 环境检查")
    print("=" * 40)
    deps = {
        'numpy': '核心计算', 'scipy': '稀疏矩阵',
        'matplotlib': '可视化', 'yaml': '配置文件',
        'tqdm': '进度条'
    }
    for mod, desc in deps.items():
        try:
            __import__(mod)
            print(f"  ✓ {mod:15s} — {desc}")
        except ImportError:
            print(f"  ✗ {mod:15s} — {desc} (缺失!)")

    opt = {'torch': 'GPU/ML训练', 'numba': 'JIT加速',
           'torch_geometric': '图神经网络', 'meshio': 'VTK输出'}
    print()
    for mod, desc in opt.items():
        try:
            __import__(mod)
            print(f"  ✓ {mod:15s} — {desc}")
        except ImportError:
            print(f"  - {mod:15s} — {desc} (可选)")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU: CUDA ✓")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"\n  GPU: Apple MPS ✓")
        else:
            print(f"\n  GPU: 未检测到 (CPU模式)")
    except ImportError:
        print(f"\n  GPU: PyTorch未安装")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Geo_sim — 地质动力学仿真框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 示例:
  geo-sim demo         运行地幔对流演示
  geo-sim run config.yaml  从YAML配置文件运行
  geo-sim benchmark    运行Blankenbach基准测试
  geo-sim check        检查安装环境
        """
    )
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('demo', help='运行地幔对流演示 (含自动画图)')
    run_p = sub.add_parser('run', help='从YAML配置文件运行仿真')
    run_p.add_argument('config', help='YAML配置文件路径')
    run_p.add_argument('--steps', type=int, default=None)
    run_p.add_argument('--resume', action='store_true')
    run_p.add_argument('--checkpoint', type=int, default=50)
    sub.add_parser('benchmark', help='运行Blankenbach 1989基准测试')
    sub.add_parser('check', help='检查安装环境')

    args = parser.parse_args()

    if args.cmd == 'demo':
        return cmd_demo()
    elif args.cmd == 'run':
        return cmd_run_config(args)
    elif args.cmd == 'benchmark':
        return cmd_benchmark()
    elif args.cmd == 'check':
        return cmd_install_check()
    else:
        parser.print_help()
        return 0


def cmd_run_config(args):
    """从YAML配置运行仿真"""
    from core.stokes_solver import StokesConfig, PicardStokesSolver

    config = StokesConfig.from_yaml(args.config)
    if args.steps is not None:
        config.max_time_steps = args.steps

    solver = PicardStokesSolver(config)
    history = solver.run(
        n_steps=config.max_time_steps,
        verbose=True,
        show_progress=True,
        checkpoint_every=args.checkpoint,
        resume=args.resume,
    )
    solver.plot(field='all', save=True, show=False)
    solver.save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
