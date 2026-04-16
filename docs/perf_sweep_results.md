# StreamPETR Video Training Performance Sweep

实验日期：2026-04-15 ~ 2026-04-16
硬件：8× A100（单节点）
数据集：NuScenes-mini ×8（~2560 train samples）
统计方法：max_iters=200，跳过前 10 步 warmup，取剩余 190 步 **median**

**五种方案**：
- **Image**：标准 JPEG pipeline，DataLoader worker 做 CPU decode
- **OpenCV**：video OpenCV pipeline，DataLoader worker 用 cv2.VideoCapture 做 CPU decode
- **Stream**：video stream access，DataLoader worker 内 GPU decode
- **Prefetch**：demuxer-free，主进程后台线程预取 + NVDEC decode，与 backward 重叠
- **Prefetch+Pool**：Prefetch + SharedGOPPool，跨 worker 共享 GOP 缓存，主进程零拷贝读取

---

## 实验一：Batch Size 扫描（固定 8 GPU）

### 加速比 vs Image Baseline（加速比 = image_time / variant_time）

| BS | Image (s) | OpenCV (s) | Stream (s) | Prefetch (s) | Pf+Pool (s) | OpenCV | Stream | Prefetch | **Pf+Pool** |
|----|-----------|------------|------------|--------------|-------------|--------|--------|----------|-------------|
| 1  | 0.263     | 0.350      | 0.277      | 0.247        | 0.235       | 0.752x | 0.952x | 1.065x   | **1.119x**  |
| 2  | 0.356     | 0.515      | 0.353      | 0.316        | 0.292       | 0.691x | 1.008x | 1.127x   | **1.219x**  |
| 4  | 0.516     | 0.964      | 0.626      | 0.476        | 0.459       | 0.535x | 0.825x | 1.085x   | **1.124x**  |
| 8  | 0.875     | 1.452      | 1.022      | 0.849        | 0.804       | 0.602x | 0.856x | 1.031x   | **1.088x**  |
| 16 | 1.671     | 3.536      | —          | 1.676        | 1.462       | 0.473x | —      | 0.997x   | **1.143x**  |

> Stream bs=16 未测试；Stream 最大支持 per-GPU bs=8（显存限制）

### 吞吐量（samples/sec = total_bs / median_time）

| BS | total_bs | Image | OpenCV | Stream | Prefetch | Pf+Pool | Pf+Pool/Img |
|----|----------|-------|--------|--------|----------|---------|-------------|
| 1  | 8        | 30.4  | 22.9   | 28.9   | 32.4     | 34.0    | **1.119x**  |
| 2  | 16       | 45.0  | 31.0   | 45.4   | 50.6     | 54.8    | **1.219x**  |
| 4  | 32       | 62.0  | 33.2   | 51.1   | 67.2     | 69.7    | **1.124x**  |
| 8  | 64       | 73.2  | 44.1   | 62.6   | 75.4     | 79.6    | **1.088x**  |
| 16 | 128      | 76.6  | 36.2   | —      | 76.4     | 87.6    | **1.143x**  |

### data_time（median s）

| BS | Image  | OpenCV | Stream | Prefetch | Pf+Pool     |
|----|--------|--------|--------|----------|-------------|
| 1  | 0.017  | 0.025  | 0.017  | 0.008    | **0.009**   |
| 2  | 0.023  | 0.035  | 0.023  | 0.006    | **0.006**   |
| 4  | 0.041  | 0.064  | 0.039  | 0.013    | **0.007**   |
| 8  | 0.069  | 0.091  | 0.063  | 0.023    | **0.009**   |
| 16 | 0.123  | 0.181  | —      | 0.047    | **0.012**   |

### 观察

1. **Prefetch+Pool 在所有 bs 下均为最快方案**，加速比 1.09x ~ 1.22x vs Image。
2. Prefetch+Pool vs Prefetch 的额外加速在 bs=16 时最显著（1.146x），因为高 bs 下 Prefetch 的 data_time 毛刺积累更多。
3. **Prefetch+Pool 的 data_time 几乎不随 bs 增长**（0.006 ~ 0.012s），而 Prefetch 从 0.006s 涨到 0.047s，Image 从 0.017s 涨到 0.123s。SharedGOPPool 消除了 per-worker cache miss 导致的数据加载抖动。
4. bs=16 处 Prefetch 已与 Image 持平（0.997x），但 Prefetch+Pool 仍有 14.3% 加速——说明 Pool 在高负载下的收益更大。
5. OpenCV 全面最慢，随 bs 增大劣势加剧：0.75x → 0.47x。
6. 吞吐量峰值：Prefetch+Pool bs=16 达到 87.6 samples/sec，比 Image 的 76.6 高 14.3%。

---

## 实验二：GPU 数量扩展（固定 per-GPU bs=8，弱扩展）

### 加速比 vs Image Baseline

| GPUs | Image (s) | OpenCV (s) | Stream (s) | Prefetch (s) | Pf+Pool (s) | OpenCV | Stream | Prefetch | **Pf+Pool** |
|------|-----------|------------|------------|--------------|-------------|--------|--------|----------|-------------|
| 1    | 0.847     | 0.971      | 0.918      | 0.770        | 0.760       | 0.872x | 0.922x | 1.100x   | **1.114x**  |
| 2    | 0.772     | 0.914      | 0.869      | 0.754        | 0.725       | 0.844x | 0.888x | 1.024x   | **1.065x**  |
| 4    | 0.823     | 1.056      | 0.943      | 0.789        | 0.768       | 0.779x | 0.873x | 1.043x   | **1.072x**  |
| 8    | 0.875     | 1.452      | 1.022      | 0.859        | 0.804       | 0.602x | 0.856x | 1.019x   | **1.088x**  |

### 每 GPU 吞吐量（samples/sec/GPU = 8 / median_time）

| GPUs | Image | OpenCV | Stream | Prefetch | Pf+Pool | Pf+Pool/Img |
|------|-------|--------|--------|----------|---------|-------------|
| 1    | 9.45  | 8.24   | 8.71   | 10.39    | 10.53   | **1.114x**  |
| 2    | 10.36 | 8.75   | 9.21   | 10.61    | 11.03   | **1.065x**  |
| 4    | 9.72  | 7.57   | 8.48   | 10.14    | 10.42   | **1.072x**  |
| 8    | 9.14  | 5.51   | 7.83   | 9.31     | 9.95    | **1.088x**  |

### data_time（median s）

| GPUs | Image | OpenCV | Stream | Prefetch | Pf+Pool     |
|------|-------|--------|--------|----------|-------------|
| 1    | 0.055 | 0.060  | 0.053  | 0.006    | **0.006**   |
| 2    | 0.056 | 0.062  | 0.057  | 0.008    | **0.006**   |
| 4    | 0.061 | 0.064  | 0.058  | 0.014    | **0.010**   |
| 8    | 0.069 | 0.091  | 0.063  | 0.022    | **0.009**   |

### 观察

1. **Prefetch+Pool 在所有 GPU 数下均为最快方案**，加速比 1.07x ~ 1.11x vs Image。
2. Prefetch+Pool 的扩展性优于 Prefetch：8 GPU 时 Pool 加速 1.088x vs Prefetch 的 1.019x，差距从 1 GPU 的 1.3% 扩大到 8 GPU 的 6.8%。
3. **SharedGOPPool 的 data_time 不随 GPU 数增长**（0.006 ~ 0.010s），而 Prefetch 从 0.006s 涨到 0.022s。多 GPU 下 worker 数更多，per-worker cache 命中率下降，而 SharedGOPPool 是跨 worker 共享的，命中率不受影响。
4. OpenCV 扩展性最差：1 GPU 0.87x → 8 GPU 0.60x，per-GPU 吞吐从 8.24 降至 5.51（下降 33%）。
5. Prefetch+Pool 8 GPU per-GPU 吞吐（9.95）甚至高于 Image 4 GPU（9.72），说明数据管线优化在多卡场景下释放了更多计算资源。
