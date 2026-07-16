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
    """备用占位 — 现在用实际实验数据, 仅当results.json缺失某实验时回退"""
    return {}


# ═══════════════════════════════════════════
# Fig 1: Setup加速比 vs 矩阵规模
# ═══════════════════════════════════════════

def fig1_speedup(data):
    """使用实验实测数据: traditional vs reuse (Meta-AMG)"""
    e2 = data.get('experiment_2', [])
    if not e2:
        print('  ⚠ No experiment_2 data, skipping Fig 1')
        return

    sizes = []
    trad_t = []
    meta_t = []
    speedups = []

    for d in e2:
        n = d['matrix_size']
        if 'traditional' in d and 'reuse' in d:
            sizes.append(n)
            tt = d['traditional'].get('setup_time_mean_mean', 0)
            mt = d['reuse'].get('setup_time_mean_mean', 0)
            trad_t.append(tt)
            meta_t.append(mt)
            speedups.append(tt / max(mt, 1e-12))

    if not sizes:
        print('  ⚠ No timing data in experiment_2')
        return

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
    for s, sp in zip(sizes, speedups):
        ax1.annotate(f'{sp:.1f}$\\times$', (s, sp), textcoords='offset points',
                     xytext=(0, 10), fontsize=9, ha='center', color=COLORS['speedup'])

    # 右: 绝对时间
    ax2.plot(sizes, trad_t, 's-', color=COLORS['traditional'], lw=2, ms=7,
             label='Traditional AMG')
    ax2.plot(sizes, meta_t, 'o-', color=COLORS['meta'], lw=2, ms=7,
             label='Meta-AMG')
    ax2.set_xlabel('Degrees of Freedom (velocity block)')
    ax2.set_ylabel('Setup Time (s)')
    ax2.set_title('(b) Absolute Setup Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig1_setup_speedup.pdf')
    plt.close()
    print(f'  ✓ Fig 1: {len(sizes)} data points, speedup range '
          f'{min(speedups):.1f}x–{max(speedups):.1f}x')


# ═══════════════════════════════════════════
# Fig 2: 消融 — Zero-shot vs Adapted
# ═══════════════════════════════════════════

def fig2_ablation(data):
    e3a = data.get('experiment_3a_zs_vs_adapted', {})
    e3b = data.get('experiment_3b_adapt_steps', [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左: Zero-shot vs Adapted (用实测误差)
    if e3a:
        zs_err = e3a.get('zero_shot', {}).get('error_mean', 0.5)
        ad_err = e3a.get('adapted', {}).get('error_mean', 0.5)
    else:
        zs_err, ad_err = 0.5, 0.5

    methods = ['Zero-shot\n(no adaptation)', 'MAML-Adapted\n(3 gradient steps)']
    accs = [1 - zs_err, 1 - ad_err]
    bars = ax1.bar(methods, accs, color=[COLORS['zero_shot'], COLORS['adapted']],
                   width=0.5, edgecolor='white', linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax1.set_ylabel('C/F Prediction Accuracy')
    ax1.set_title('(a) Adaptation vs No Adaptation')
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右: 适配步数 (实测数据)
    if e3b:
        steps = [d['steps'] for d in e3b]
        errors = [d['error_mean'] for d in e3b]
        times = [d['time_mean'] for d in e3b]
    else:
        steps, errors, times = [1,3,5,10], [0.5]*4, [0.03]*4

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

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig2_ablation.pdf')
    plt.close()
    print(f'  ✓ Fig 2: ablation — zs={1-zs_err:.1%}, ad={1-ad_err:.1%}, '
          f'{len(steps)} adapt steps')


# ═══════════════════════════════════════════
# Fig 3: 收敛质量
# ═══════════════════════════════════════════

def fig3_convergence_quality(data):
    """使用实测Stokes replay数据 + 训练conv metrics"""
    e2s = data.get('experiment_2_stokes_picard_replay', {})
    e2sf = data.get('experiment_2_stokes_picard_full', {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: Setup时间对比 (Stokes replay实测)
    ax1 = axes[0]
    methods_display = ['Traditional', 'Reuse\n(Meta-AMG)', 'Zero-shot\n(no adapt)']
    colors_method = [COLORS['traditional'], COLORS['meta'], COLORS['zero_shot']]
    times = []
    labels = []
    for method_key, display_name in [('traditional','Traditional'),
                                      ('reuse','Reuse (Meta)'),
                                      ('zero_shot','Zero-shot'),
                                      ('adapted','Adapted')]:
        if method_key in e2s:
            t = e2s[method_key].get('setup_time_mean_mean', 0)
            times.append(t)
            labels.append(display_name)

    if times:
        xs = np.arange(len(times))
        bars = ax1.bar(xs, times, color=colors_method[:len(times)],
                       edgecolor='white', linewidth=1.5)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Setup Time (s)')
        ax1.set_title('(a) Setup Time on Stokes Sequence')
        for bar, t in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{t*1000:.1f}ms', ha='center', va='bottom', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

    # 右: 物理指标 (Nu, Vrms — 实测)
    ax2 = axes[1]
    if 'reference_direct' in e2sf and 'meta_blocked' in e2sf:
        ref = e2sf['reference_direct']
        meta = e2sf['meta_blocked']
        metrics = ['Nusselt', 'RMS velocity', 'Wall time']
        ref_vals = [ref.get('nusselt_mean', 0), 0, ref.get('wall_time_mean', 0)]
        meta_vals = [meta.get('nusselt_mean', 0), 0, meta.get('wall_time_mean', 0)]
        # Vrms from experiment data if available
        if 'rms_velocity_mean' in ref:
            ref_vals[1] = ref['rms_velocity_mean']
        if 'rms_velocity_mean' in meta:
            meta_vals[1] = meta['rms_velocity_mean']

        x = np.arange(len(metrics))
        w = 0.35
        ax2.bar(x - w/2, ref_vals, w, color=COLORS['traditional'],
                label='Traditional', edgecolor='white')
        ax2.bar(x + w/2, meta_vals, w, color=COLORS['meta'],
                label='Meta-AMG', edgecolor='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.set_title('(b) Physical Accuracy (vs Direct Solve)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig3_convergence_quality.pdf')
    plt.close()
    print(f'  ✓ Fig 3: Stokes convergence — {len(times)} methods compared')


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
    """使用E1实测数据: 各方法的求解误差"""
    e1 = data.get('experiment_1', [])
    if not e1:
        print('  ⚠ No experiment_1 data, skipping Fig 6')
        return

    sizes = []
    trad_errs = []
    reuse_errs = []
    zs_errs = []
    ad_errs = []

    for d in e1:
        sizes.append(d['matrix_size'])
        for method, store in [('traditional', trad_errs), ('reuse', reuse_errs),
                               ('zero_shot', zs_errs), ('adapted', ad_errs)]:
            if method in d and 'residual_norm_mean' in d[method]:
                store.append(d[method]['residual_norm_mean'])
            else:
                store.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))

    methods_plot = [
        (trad_errs, COLORS['traditional'], 'Traditional AMG'),
        (reuse_errs, COLORS['meta'], 'Meta-AMG (reuse)'),
        (ad_errs, COLORS['adapted'], 'Meta-AMG (adapted)'),
    ]

    for errs, color, label in methods_plot:
        if any(e > 0 for e in errs):
            ax.plot(sizes, errs, 'o-', color=color, lw=2, ms=7, label=label)

    ax.set_xlabel('Degrees of Freedom')
    ax.set_ylabel('Residual Norm $\\|\\mathbf{b} - A\\mathbf{x}\\|$')
    ax.set_title('Solver Accuracy Across Matrix Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(OUTDIR / 'fig6_solution_accuracy.pdf')
    plt.close()
    print(f'  ✓ Fig 6: {len(sizes)} data points from real experiment')


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
