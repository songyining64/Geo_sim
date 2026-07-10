# Meta-AMG 实验运行手册

## 从零到结果，每一步可复制粘贴

---

## 一、环境准备（一次性的，5分钟）

不用装任何新东西——numpy、scipy、pytorch、pytest你已经有。确认一下：

```bash
cd ~/Documents/强化学习/Geo_sim
python3 -c "import torch; print('PyTorch', torch.__version__); print('GPU:', torch.cuda.is_available())"
python3 -m pytest tests/ -q   # 应该显示 127 passed
```

如果PyTorch检测不到GPU，实验会用CPU跑（慢但能跑完）。有GPU的话快3-5倍。

---

## 二、训练路线：两条路选一条

### 路A：GPU训练（推荐，两天半跑完所有实验）

```bash
# 设置参数：500条序列、50轮、用GPU
cd ~/Documents/强化学习/Geo_sim

python3 -c "
import importlib.util, sys, numpy as np
sys.path.insert(0, '.')

spec1 = importlib.util.spec_from_file_location('neural_amg', 'gpu_acceleration/neural_amg.py')
nm = importlib.util.module_from_spec(spec1)
sys.modules['neural_amg'] = nm
spec1.loader.exec_module(nm)

spec2 = importlib.util.spec_from_file_location('meta_amg', 'gpu_acceleration/meta_amg.py')
mm = importlib.util.module_from_spec(spec2)
sys.modules['meta_amg'] = mm
spec2.loader.exec_module(mm)

from meta_amg import MetaAMGConfig, MetaAMG

config = MetaAMGConfig(
    min_matrix_size=64,
    max_matrix_size=2500,
    num_training_sequences=500,
    sequence_length=8,
    num_meta_epochs=50,
    inner_steps=3,
    meta_batch_size=4,
    hidden_dim=64,
    num_layers=3,
    learning_rate=0.001,       # outer_lr
)

meta = MetaAMG(config)
history = meta.train(num_sequences=500)
meta.save('experiments/trained_models/meta_amg_full.pt')

# 打印最后5轮的loss曲线供判断
print('Final 5 epochs:')
for i in range(-5, 0):
    if -i <= len(history['train_meta_loss']):
        print(f'  Epoch {len(history[\"train_meta_loss\"])+i+1}: '
              f'loss={history[\"train_meta_loss\"][i]:.4f}')
if history.get('val_adapt_accuracy'):
    print(f'  Final adapted accuracy: {history[\"val_adapt_accuracy\"][-1]:.4f}')
"
```

这会跑很久。建议用`nohup`或`tmux`让它在后台跑，你不需要盯着它。

```bash
# 后台运行（推荐）
nohup python3 -c "..." > experiments/logs/training.log 2>&1 &

# 随时查看进度
tail -f experiments/logs/training.log
```

**怎么判断训练正常：** 每10轮打印一次统计。有效的训练应该是meta_loss从0.7左右逐步降到0.3以下，adapted accuracy从0.75升到0.92以上。如果loss在10轮内完全不变，按Ctrl+C停了检查代码。如果loss稳步下降就让它继续跑。

### 路B：CPU训练（没有GPU时的选择，大约五天）

上面的命令放在后台，慢但一样出结果。唯一的区别是把`max_matrix_size`降到1600（大矩阵在CPU上太慢），其余不变。

### 路C：快速验证（训练30分钟，确认代码没问题再跑全量）

```bash
python3 experiments/run_experiments.py --exp all --quick
```

---

## 三、跑六个实验（训练完成之后）

训练跑完后，模型保存在`experiments/trained_models/meta_amg_full.pt`。现在运行实验脚本产生论文数据。

```bash
cd ~/Documents/强化学习/Geo_sim

# 确保目录存在
mkdir -p experiments/trained_models experiments/results experiments/logs

# 跑全部六个实验（约3小时，取决于CPU速度）
python3 experiments/run_experiments.py --exp all

# 或者逐个跑（方便调试）
python3 experiments/run_experiments.py --exp e1   # 收敛性
python3 experiments/run_experiments.py --exp e2   # 加速比
python3 experiments/run_experiments.py --exp e3   # 消融
python3 experiments/run_experiments.py --exp e4   # 粘度
python3 experiments/run_experiments.py --exp e5   # 泛化
python3 experiments/run_experiments.py --exp e6   # 对比
```

跑完后检查`experiments/results/results.json`是否包含所有六个实验的数据：

```bash
python3 -c "
import json
with open('experiments/results/results.json') as f:
    data = json.load(f)
for k in sorted(data.keys()):
    v = data[k]
    if isinstance(v, list): print(f'{k}: {len(v)} entries')
    elif isinstance(v, dict): print(f'{k}: {[(sk, type(sv).__name__) for sk,sv in v.items()][:3]}...')
"
```

预期输出类似：
```
experiment_1: 4 entries
experiment_2: 7 entries
experiment_3a_zs_vs_adapted: dict
experiment_3b_adapt_steps: 5 entries
experiment_4: 5 entries
experiment_5: 4 entries
experiment_6: dict
```

---

## 四、读结果——什么样算成功

打开`experiments/results/results.json`找这几个关键数字。下面给出了每一行的含义和应该看到什么。

**E1 收敛性。** 找`mean_error`字段。如果四个矩阵规模的误差都在10⁻⁵附近或更小，说明Meta-AMG的解和传统AMG基本一致。如果某个规模误差突然跳到0.1以上，说明那个规模的C/F预测出了问题，需要检查模型在该规模上的验证准确率。

**E2 加速比。** 找`speedup`字段。n=100时speedup应该小于1（Meta比传统慢，正常），n=400时应该大于8，n=1600时应该大于30，n=2500时应该大于50。如果不是这样——特别是大矩阵上speedup仍然小于5——有两个可能原因：要么训练不充分（adapted accuracy不够高，回到路A多训练几轮），要么代码有性能bug（检查是否意外触发了传统AMG的fallback路径）。

**E3 消融。** 找`experiment_3a_zs_vs_adapted`里的两个`error_mean`。adapted应该比zero_shot低至少0.1（适配确实在改善）。如果adapted反而不如zero_shot，说明MAML训练失败了——最常见原因是inner_lr太大导致适配过冲，试试把`inner_lr`从0.01降到0.001重新训练。在`experiment_3b_adapt_steps`里看不同步数的效果，error应该随着步数增加而下降，3步之后趋于平稳。如果从第1步到第5步几乎没变化，说明任务太简单（训练矩阵和测试矩阵太像），需要增大训练数据的粘度对比度范围。

**E4 粘度鲁棒性。** 找`ad_error_mean`在constrast=1e6时的值。如果不超过0.25，说明GNN在高对比度下仍然能做合理的C/F粗化。如果超过0.5，说明训练数据的对比度覆盖不足——训练时最高只到某个对比度，测试时更高的对比度让模型失效了。解决方法是增大训练数据的`viscosity_contrast_range`。

**E5 泛化。** 找最大test_size对应的`ad_error_mean`。如果泛化到10倍规模的矩阵时准确率损失不超过0.15，说明GNN确实学到了规模无关的粗化策略。如果在大矩阵上准确率接近随机（0.5），需要在训练数据中加入更多大矩阵样本或者增加GNN的层数以扩大感受野。

**E6 对比。** 找三种方法的setup时间。Meta-AMG的累计setup时间应该最短，且随着序列增长优势越来越明显。如果神经AMG（零样本）和Meta-AMG的时间几乎一样，说明MAML适配没有产生额外的加速——检查adapt_steps是否设得太大。

---

## 五、画图

有了`results.json`的数据后，用以下Python脚本生成论文图表。放在`experiments/plot_results.py`里。

```python
import json, numpy as np, matplotlib.pyplot as plt

with open('experiments/results/results.json') as f:
    data = json.load(f)

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# Fig 2: Setup加速比 vs 矩阵规模
fig, ax = plt.subplots(figsize=(6, 4))
e2 = data['experiment_2']
sizes = [d['matrix_size'] for d in e2]
speedups = [d['speedup'] for d in e2]
ax.plot(sizes, speedups, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Meta-AMG vs Traditional')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Break-even')
ax.set_xlabel('Matrix Size (DOF)')
ax.set_ylabel('Setup Speedup')
ax.set_title('Setup Time Speedup of Meta-AMG')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/results/fig2_speedup.pdf')

# Fig 3: 消融
fig, ax = plt.subplots(figsize=(5, 4))
e3a = data['experiment_3a_zs_vs_adapted']
bars = ['Zero-shot', 'MAML-Adapted']
vals = [1 - e3a['zero_shot']['error_mean'], 1 - e3a['adapted']['error_mean']]
errs = [e3a['zero_shot']['error_std'], e3a['adapted']['error_std']]
ax.bar(bars, vals, yerr=errs, color=['#FF9800', '#4CAF50'], capsize=8)
ax.set_ylabel('C/F Accuracy')
ax.set_title('Ablation: Zero-shot vs MAML Adaptation')
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
plt.savefig('experiments/results/fig3_ablation.pdf')

# Fig 4: 粘度鲁棒性
fig, ax = plt.subplots(figsize=(6, 4))
e4 = data['experiment_4']
contrasts = [d['contrast'] for d in e4]
zs = [1 - d['zs_error_mean'] for d in e4]
ad = [1 - d['ad_error_mean'] for d in e4]
ax.semilogx(contrasts, zs, 's--', color='#FF9800', label='Zero-shot', markersize=8)
ax.semilogx(contrasts, ad, 'o-', color='#4CAF50', label='MAML-Adapted', markersize=8)
ax.set_xlabel('Viscosity Contrast')
ax.set_ylabel('C/F Accuracy')
ax.set_title('Robustness to Viscosity Contrast')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/results/fig4_contrast.pdf')

print('Figures saved to experiments/results/')
```

---

## 六、常见问题排查

**训练了几轮loss不动了。** 检查`inner_lr`和`outer_lr`。如果`meta_loss`长时间卡在0.69附近（等于随机猜的BCE loss），说明模型完全没学到东西。通常是因为`outer_lr`太小（梯度几乎为零）或太大（梯度爆炸）。试试把`outer_lr`从0.001调到0.0005，或者检查训练数据中的C/F标签是否正确（不应该全0或全1）。

**训练过程中内存溢出。** 矩阵规模设为2500时单次可能吃2-4GB内存。如果报OOM，把`max_matrix_size`降到1600，或者把`meta_batch_size`从4降到2。

**实验报错"Traditional AMG setup took 0.0000s"。** 这说明矩阵太小或太规则导致AMG只构建了一层——这层就是原矩阵本身，没有真正粗化。发生在n<100时是正常的。确保实验矩阵规模从200以上开始。

**实验跑了一半中断了。** 重新运行相同命令不会覆盖已有数据，但会从实验e1重新开始。如果想跳过已完成的实验，用`--exp e3`单独跑未完成的那个。

---

## 七、整个流程的检查点

```
□ 环境确认: pytest 127 passed
□ 路A或路B: 训练启动, loss在下降
□ 训练完成: adapted accuracy > 90%
□ 模型已保存: experiments/trained_models/meta_amg_full.pt
□ E1-E6全部运行完成
□ results.json包含所有6个实验数据
□ E2的加速比: n=1600时 > 30x
□ E3的适配增益: adapted > zero-shot
□ E4的高对比度: 10^6时准确率 > 75%
□ 图表已生成: 3张PDF
```
