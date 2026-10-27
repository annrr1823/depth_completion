# 作业十：RGB 引导的稀疏深度补全

基于 NYU Depth V2，从真实深度中随机采样 **500** 个有效点，结合 RGB 恢复稠密深度图。  
参考论文：Ma & Karaman, *Sparse-to-Dense* (ICRA 2018)。

> **当前阶段约定**：已完成思路梳理与代码实现；**先不要下载数据集，也不要正式训练**。  
> 可用无数据通路自检：`python scripts/sanity_check.py`

---

## 1. 任务与数据流

| 项目 | 内容 |
|------|------|
| 输入 | RGB `[H,W,3]` + 稀疏深度 `[H,W,1]`（500 个有效点，其余为 0） |
| 输出 | 稠密深度 `[H,W,1]` |
| 数据 | [sayakpaul/nyu_depth_v2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) |
| 指标 | RMSE、Abs Relative Difference |

**稀疏构造**：在 GT 有效像素（`0 < d ≤ max_depth`）上无放回随机采样 500 点；无效像素在 loss/指标中被 mask 掉。

---

## 2. 整体思路

```
RGB ──┐
      ├─ concat(RGB, sparse, mask) ─► Encoder ─► Decoder(+多尺度稀疏注入)
Sparse┤                                      │
Mask ─┘                                      ▼
                         粗深度 D_coarse(稀疏最近邻填充)
                                      + 残差 Δ(RGB 边界注意力)
                                      ▼
                               D = softplus(D_coarse + Δ)
```

1. **数据处理**（评分 15）：读取 RGB/Depth → resize 对齐 → 采样 500 点 + validity mask。  
2. **融合模型**（评分 20）：早期融合 5 通道输入，U-Net 式编解码输出同分辨率深度。  
3. **损失与训练**（评分 20）：有效像素上的 L1/MSE/BerHu；稀疏点一致性；RGB 边缘引导平滑。  
4. **指标**（评分 15）：仅在有效 GT 上累积 RMSE / AbsRel。  
5. **可视化**（评分 10）：RGB | Sparse | Pred | GT，并附误差图分析边界/无纹理区。  
6. **新颖性**（评分 20）：见下一节 SAGRNet。

---

## 3. 模型新颖性：SAGRNet

相对经典 Sparse-to-Dense「RGBD 拼接 + 直接回归」，本仓库主模型 **SAGRNet**（`models/sagrnet.py`）有三点设计：

1. **显式 validity mask**  
   稀疏图中的 0 表示“无测量”而非深度为 0，额外通道让网络可区分。

2. **多尺度稀疏引导注入**  
   Decoder 每个尺度拼接下采样后的 sparse/mask，使已知深度在上采样过程中持续锚定，减轻墙面/地板等无纹理区域漂移。

3. **稀疏锚定残差预测**  
   先对稀疏点做最近邻式填充得到 `D_coarse`，网络只预测残差；再用 RGB 特征生成空间注意力，强化物体边界处的修正。

另提供基线 `SparseToDenseNet`（`models/sparse_to_dense.py`），可在配置里切换对比。

---

## 4. 目录结构

```
depth_completion/
├── configs/default.yaml      # 超参
├── datasets/                 # NYU 读取 + 稀疏采样
├── models/                   # SAGRNet / Sparse-to-Dense
├── losses/                   # 掩码 L1/MSE/BerHu + 边缘平滑
├── metrics/                  # RMSE / AbsRel
├── utils/                    # 采样、可视化、配置
├── scripts/
│   ├── download_nyu.py       # 以后再跑：下载数据
│   └── sanity_check.py       # 现在可跑：无数据自检
├── train.py                  # 训练（支持 --dry-run）
├── eval.py                   # 评估
└── visualize_results.py      # 结果与误差分析
```

---

## 5. 环境

```bash
cd /data/turuirui/turuirui/depth_completion
pip install -r requirements.txt
```

---

## 6. 推荐使用顺序（暂不训练）

```bash
# ① 通路自检（不下载、不训练）
python scripts/sanity_check.py

# ② 仅构建模型做假前向
python train.py --dry-run

# ---------- 以下步骤等你确认后再执行 ----------
# ③ 下载 NYU Depth V2
# python scripts/download_nyu.py --cache-dir data/nyu_depth_v2

# ④ 训练
# python train.py --config configs/default.yaml

# ⑤ 评估 + 可视化
# python eval.py --checkpoint outputs/checkpoints/sagrnet_nyu_500/best.pth
# python visualize_results.py --checkpoint outputs/checkpoints/sagrnet_nyu_500/best.pth
```

切换基线模型：改 `configs/default.yaml` 中 `model.name` 为 `sparse_to_dense`。

---

## 7. 与评分表的对应

| 评分项 | 实现位置 |
|--------|----------|
| 深度数据处理 | `datasets/nyu_dataset.py`, `utils/sparse_sample.py` |
| 融合模型设计 | `models/sagrnet.py`, `models/sparse_to_dense.py` |
| 损失函数与训练 | `losses/depth_loss.py`, `train.py` |
| 指标计算 | `metrics/depth_metrics.py`, `eval.py` |
| 深度可视化 | `utils/visualize.py`, `visualize_results.py` |
| 模型新颖性 | SAGRNet：mask + 多尺度注入 + 残差锚定 + RGB 注意力 |

---

## 8. 指标定义（作业要求）

在有效深度像素集合 \(\Omega\) 上：

\[
\mathrm{RMSE}=\sqrt{\frac{1}{|\Omega|}\sum_{i\in\Omega}(d_i-\hat{d}_i)^2}
\]

\[
\mathrm{AbsRel}=\frac{1}{|\Omega|}\sum_{i\in\Omega}\frac{|d_i-\hat{d}_i|}{d_i}
\]
