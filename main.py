#!/usr/bin/env python3
"""
Geo_sim - Geodynamic Simulation Framework

主入口文件，提供命令行接口。
"""

import argparse
import sys
from pathlib import Path


def ensure_package_path():
    """确保项目根目录在PYTHONPATH中"""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def cmd_benchmark(args):
    """运行基准测试"""
    print("=" * 60)
    print("Geo_sim Benchmark")
    print("=" * 60)
    try:
        import numpy as np
        from core.geodynamic_simulation import GeodynamicSimulation, GeodynamicConfig

        config = GeodynamicConfig(
            name="benchmark",
            description="基准测试",
            numerical_params={
                'time_steps': args.time_steps or 10,
                'dt': 0.001,
                'tolerance': 1e-6,
                'max_iterations': 1000
            }
        )
        
        sim = GeodynamicSimulation(config)
        sim.create_mesh("rectangular", nx=args.nx or 20, ny=args.ny or 20)
        
        from materials import create_mantle_material
        sim.add_material(create_mantle_material())
        
        sim.setup_solver(solver_type=args.solver or "auto")
        sim.setup()
        
        import time
        start = time.time()
        result = sim.run(time_steps=config.numerical_params['time_steps'])
        elapsed = time.time() - start
        
        print(f"\nBenchmark Results:")
        print(f"  Time steps: {config.numerical_params['time_steps']}")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Average per step: {elapsed / config.numerical_params['time_steps']:.4f}s")
        
        perf = sim.get_performance_summary()
        if perf:
            print(f"  Performance: {perf}")
            
        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_test(args):
    """运行测试"""
    print("Running tests...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=Path(__file__).parent
    )
    return result.returncode


def cmd_demo(args):
    """运行演示"""
    try:
        from core.complete_interface_demo import main as run_demo
        run_demo()
        return 0
    except ImportError:
        print("Demo module not available.")
        return 1


def cmd_info(args):
    """显示系统信息"""
    print("=" * 60)
    print("Geo_sim System Information")
    print("=" * 60)
    
    # Python info
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Core dependencies
    for lib in ["numpy", "scipy", "matplotlib", "pandas", "yaml", "h5py"]:
        try:
            mod = __import__(lib.replace("-", "_"))
            version = getattr(mod, "__version__", "unknown")
            print(f"  {lib}: {version}")
        except ImportError:
            print(f"  {lib}: not installed")
    
    # Optional dependencies
    print("\nOptional:")
    for lib in ["torch", "cupy", "numba", "mpi4py", "stable_baselines3", "torch_geometric"]:
        try:
            mod = __import__(lib)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {lib}: {version}")
        except ImportError:
            print(f"  {lib}: not installed")
    
    print(f"\nProject path: {Path(__file__).parent}")
    return 0


def main():
    ensure_package_path()
    
    parser = argparse.ArgumentParser(
        description="Geo_sim - Geodynamic Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py benchmark --nx 20 --ny 20 --time-steps 50
  python main.py test
  python main.py demo
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--nx", type=int, help="Number of x elements")
    bench_parser.add_argument("--ny", type=int, help="Number of y elements") 
    bench_parser.add_argument("--time-steps", type=int, help="Number of time steps")
    bench_parser.add_argument("--solver", choices=["auto", "multigrid", "multiphysics", "parallel"], help="Solver type")
    
    # test command
    subparsers.add_parser("test", help="Run tests")
    
    # demo command
    subparsers.add_parser("demo", help="Run demonstration")
    
    # info command
    subparsers.add_parser("info", help="Show system information")
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
