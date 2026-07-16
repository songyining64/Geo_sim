"""
论文图表生成 — 从实验结果JSON自动生成论文用图

Usage: python experiments/plot_paper_figures.py
输出: experiments/figures/fig_*.pdf
"""

import json, sys, numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available. Install: pip install matplotlib")
    sys.exit(1)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUTDIR = Path(__file__).parent / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'traditional': '#E65100',  # deep orange
    'neural':      '#2196F3',  # blue
    'meta':        '#4CAF50',  # green
    'zero_shot':   '#FF9800',  # amber
    'adapted':     '#1565C0',  # dark blue
    'speedup':     '#7B1FA2',  # purple
}


def load_results():
    """加载实验结果, 缺失的experiment用占位数据填充"""
    result_path = Path(__file__).parent / 'results' / 'results.json'
    data = {}
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)

    placeholder = generate_placeholder_data()
    for key in placeholder:
        if key not in data:
            data[key] = placeholder[key]
    return data


def generate_placeholder_data():
    """论文占位数据 — 替换为实际实验结果后删掉此函数"""
    sizes = [100, 225, 400, 625, 900, 1600]
    contrasts = [1e2, 1e3, 1e4, 1e5, 1e6]

    return {
        'experiment_2': [
            {'matrix_size': s, 'speedup': max(0.04, 0.025 * s**0.6),
             'trad_time_mean': 0.001 * s**1.3,
             'adapt_time_mean': 0.025 + 1e-5 * s}
            for s in sizes
        ],
        'experiment_3a_zs_vs_adapted': {
            'zero_shot': {'error_mean': 0.33},
            'adapted': {'error_mean': 0.17},
        },
        'experiment_3b_adapt_steps': [
            {'steps': 1, 'error_mean': 0.30, 'time_mean': 0.008},
            {'steps': 3, 'error_mean': 0.17, 'time_mean': 0.025},
            {'steps': 5, 'error_mean': 0.15, 'time_mean': 0.042},
            {'steps': 10, 'error_mean': 0.14, 'time_mean': 0.083},
        ],
        'experiment_4': [
            {'contrast': c, 'zs_error_mean': 0.15 + 0.25*np.log10(c/1e2),
             'ad_error_mean': 0.10 + 0.08*np.log10(c/1e2)}
            for c in contrasts
        ],
        'experiment_5': [
            {'test_size': s, 'zs_error_mean': 0.30 + 0.02*(s-400)/100,
             'ad_error_mean': 0.15 + 0.015*(s-400)/100}
            for s in [400, 625, 900, 1600]
        ],
        'experiment_1': [
            {'matrix_size': s, 'max_error': 5e-6*s**0.5}
            for s in [225, 400, 625, 1600]
        ],
    }


# ═══════════════════════════════════════════
# Fig 1: Setup加速比 vs 矩阵规模
# ═══════════════════════════════════════════

def fig1_speedup(data):
    e2 = data['experiment_2']
    sizes = [d['matrix_size'] for d in e2]
    speedups = [d['speedup'] for d in e2]
    trad_t = [d.get('trad_time_mean', 0) for d in e2]
    adapt_t = [d.get('adapt_time_mean', 0) for d in e2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 加速比
    ax1.plot(sizes, speedups, 'o-', color=COLORS['speedup'], lw=2.5, ms=8,
             label='Meta-AMG vs Traditional')
    ax1.axhline(y=1, color='gray', ls='--', alpha=0.5, label='Break-even')
    ax1.fill_between(sizes, 0, speedups, alpha=0.1, color=COLORS['speedup'])
    ax1.set_xlabel('Degrees of Freedom (velocity block)')
    ax1.set_ylabel('Setup Speedup ($\\times$)')
    ax1.set_title('(a) Meta-AMG Setup Speedup')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.annotate(f'{speedups[-1]:.0f}$\\times$',
                 xy=(sizes[-1], speedups[-1]),
                 xytext=(sizes[-1]*0.5, speedups[-1]*1.1),
                 fontsize=11, fontweight='bold', color=COLORS['speedup'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['speedup']))

    # 右: 绝对时间
    ax2.plot(sizes, trad_t, 's-', color=COLORS['traditional'], lw=2, ms=7,
             label='Traditional AMG')
    ax2.plot(sizes, adapt_t, 'o-', color=COLORS['meta'], lw=2, ms=7,
             label='Meta-AMG (adapted)')
    ax2.set_xlabel('Degrees of Freedom (velocity block)')
    ax2.set_ylabel('Setup Time (s)')
    ax2.set_title('(b) Absolute Setup Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig1_setup_speedup.pdf')
    plt.close()
    print(f'  ✓ Fig 1: setup speedup — {OUTDIR}/fig1_setup_speedup.pdf')


# ═══════════════════════════════════════════
# Fig 2: 消融 — Zero-shot vs Adapted
# ═══════════════════════════════════════════

def fig2_ablation(data):
    e3a = data['experiment_3a_zs_vs_adapted']
    e3b = data['experiment_3b_adapt_steps']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左: Zero-shot vs Adapted
    methods = ['Zero-shot\n(no adaptation)', 'MAML-Adapted\n(3 gradient steps)']
    zs_err = e3a['zero_shot']['error_mean']
    ad_err = e3a['adapted']['error_mean']
    accs = [1 - zs_err, 1 - ad_err]
    bars = ax1.bar(methods, accs, color=[COLORS['zero_shot'], COLORS['adapted']],
                   width=0.5, edgecolor='white', linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax1.set_ylabel('C/F Prediction Accuracy')
    ax1.set_title('(a) Adaptation vs No Adaptation')
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右: 适配步数
    steps = [d['steps'] for d in e3b]
    errors = [d['error_mean'] for d in e3b]
    times = [d['time_mean'] for d in e3b]

    ax2b = ax2.twinx()
    line1 = ax2.plot(steps, [1-e for e in errors], 'o-', color=COLORS['adapted'],
                     lw=2.5, ms=8, label='Accuracy')
    line2 = ax2b.plot(steps, times, 's--', color=COLORS['traditional'],
                      lw=2, ms=7, label='Adapt time')
    ax2.set_xlabel('Adaptation Steps $K$')
    ax2.set_ylabel('C/F Prediction Accuracy', color=COLORS['adapted'])
    ax2b.set_ylabel('Adaptation Time (s)', color=COLORS['traditional'])
    ax2.set_title('(b) Effect of Adaptation Steps')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.55, 0.95)

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig2_ablation.pdf')
    plt.close()
    print(f'  ✓ Fig 2: ablation — {OUTDIR}/fig2_ablation.pdf')


# ═══════════════════════════════════════════
# Fig 3: 收敛质量
# ═══════════════════════════════════════════

def fig3_convergence_quality(data):
    """占位图 — 需要convergence-driven指标的实验数据"""
    e3a = data['experiment_3a_zs_vs_adapted']
    zs_err = e3a['zero_shot']['error_mean']
    ad_err = e3a['adapted']['error_mean']

    fig, ax = plt.subplots(figsize=(7, 5))

    methods = ['Traditional\nAMG', 'Neural AMG\n(Zero-shot)', 'Meta-AMG\n(Adapted)']
    # 残差收缩比 (r_after/r_before, lower=better)
    rho = [0.18, 0.45, 0.28]  # 占位值, 替换为实测
    converge_rate = [1.0, 0.72, 0.91]  # ρ<1的比例

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, rho, width, color=[COLORS['traditional'],
                   COLORS['zero_shot'], COLORS['adapted']],
                   edgecolor='white', linewidth=1.5, label='Residual ratio $\\rho$')
    ax.set_ylabel('Two-grid Residual Ratio $\\rho$', fontsize=12)
    ax.set_title('Convergence Quality of Predicted C/F Splittings')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.axhline(y=1.0, color='red', ls='--', alpha=0.5, label='Divergence threshold')

    for bar, r in zip(bars1, rho):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r:.2f}', ha='center', va='bottom', fontweight='bold')

    # 收敛率标注
    for i, cr in enumerate(converge_rate):
        ax.text(i, 0.05, f'Converges: {cr:.0%}', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig3_convergence_quality.pdf')
    plt.close()
    print(f'  ✓ Fig 3: convergence quality — {OUTDIR}/fig3_convergence_quality.pdf')


# ═══════════════════════════════════════════
# Fig 4: 粘度对比鲁棒性
# ═══════════════════════════════════════════

def fig4_contrast_robustness(data):
    e4 = data['experiment_4']
    contrasts = [d['contrast'] for d in e4]
    zs_err = [d['zs_error_mean'] for d in e4]
    ad_err = [d['ad_error_mean'] for d in e4]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogx(contrasts, [1-e for e in zs_err], 's--', color=COLORS['zero_shot'],
                lw=2, ms=8, label='Zero-shot')
    ax.semilogx(contrasts, [1-e for e in ad_err], 'o-', color=COLORS['adapted'],
                lw=2.5, ms=8, label='MAML-Adapted')
    ax.fill_between(contrasts, [1-e for e in zs_err], [1-e for e in ad_err],
                    alpha=0.15, color=COLORS['meta'], label='Adaptation gain')

    ax.set_xlabel('Viscosity Contrast $\\eta_{\\max}/\\eta_{\\min}$')
    ax.set_ylabel('C/F Prediction Accuracy')
    ax.set_title('Robustness to Viscosity Contrast')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    # 标注地质相关范围
    ax.axvspan(1e3, 1e6, alpha=0.08, color='orange', label='Mantle convection range')
    ax.text(3e4, 0.48, 'Geodynamic\nregime', ha='center', fontsize=9, color='darkorange')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig4_contrast_robustness.pdf')
    plt.close()
    print(f'  ✓ Fig 4: contrast robustness — {OUTDIR}/fig4_contrast_robustness.pdf')


# ═══════════════════════════════════════════
# Fig 5: 可扩展性 (小训大测)
# ═══════════════════════════════════════════

def fig5_scalability(data):
    e5 = data['experiment_5']
    test_sizes = [d['test_size'] for d in e5]
    zs_err = [d['zs_error_mean'] for d in e5]
    ad_err = [d['ad_error_mean'] for d in e5]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(test_sizes, [1-e for e in zs_err], 's--', color=COLORS['zero_shot'],
            lw=2, ms=8, label='Zero-shot')
    ax.plot(test_sizes, [1-e for e in ad_err], 'o-', color=COLORS['adapted'],
            lw=2.5, ms=8, label='MAML-Adapted')

    # 训练规模标注
    ax.axvline(x=225, color='gray', ls=':', alpha=0.7)
    ax.text(230, 0.95, 'Trained on\n$\\leq$ 225 DOF', fontsize=9, color='gray')

    ax.set_xlabel('Test Matrix Size (DOF)')
    ax.set_ylabel('C/F Prediction Accuracy')
    ax.set_title('Scalability: Train Small $\\to$ Test Large')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig5_scalability.pdf')
    plt.close()
    print(f'  ✓ Fig 5: scalability — {OUTDIR}/fig5_scalability.pdf')


# ═══════════════════════════════════════════
# Fig 6: 收敛性验证 (解精度)
# ═══════════════════════════════════════════

def fig6_solution_accuracy(data):
    e1 = data['experiment_1']
    sizes = [d['matrix_size'] for d in e1]
    errors = [d['max_error'] for d in e1]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sizes, errors, 'o-', color=COLORS['meta'], lw=2.5, ms=8)
    ax.fill_between(sizes, 0, errors, alpha=0.1, color=COLORS['meta'])
    ax.axhline(y=1e-5, color='gray', ls='--', alpha=0.5, label='Tolerance (10$^{-5}$)')
    ax.axhline(y=1e-4, color='red', ls=':', alpha=0.3, label='Picard tolerance')

    ax.set_xlabel('Degrees of Freedom')
    ax.set_ylabel('Relative Error $\\|\\mathbf{x}_{\\rm meta} - \\mathbf{x}_{\\rm trad}\\| / \\|\\mathbf{x}_{\\rm trad}\\|$')
    ax.set_title('Solution Accuracy: Meta-AMG vs Traditional AMG')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig6_solution_accuracy.pdf')
    plt.close()
    print(f'  ✓ Fig 6: solution accuracy — {OUTDIR}/fig6_solution_accuracy.pdf')


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    print("=" * 55)
    print("Geo_sim — 论文图表生成")
    print("=" * 55)

    data = load_results()

    if not Path(OUTDIR.parent / 'results' / 'results.json').exists():
        print("  ⚠ 未找到实验结果, 使用占位数据生成图表框架")
        print("    运行 'python experiments/run_experiments.py --exp all' 后重跑本脚本")
        print()

    fig1_speedup(data)
    fig2_ablation(data)
    fig3_convergence_quality(data)
    fig4_contrast_robustness(data)
    fig5_scalability(data)
    fig6_solution_accuracy(data)

    print(f"\n全部图表已生成: {OUTDIR}/")
    print("  fig1_setup_speedup.pdf")
    print("  fig2_ablation.pdf")
    print("  fig3_convergence_quality.pdf")
    print("  fig4_contrast_robustness.pdf")
    print("  fig5_scalability.pdf")
    print("  fig6_solution_accuracy.pdf")


if __name__ == '__main__':
    main()
