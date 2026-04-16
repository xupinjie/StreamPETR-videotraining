# StreamPETR Video Training Performance Sweep

实验日期：2026-04-15
硬件：8× A100（单节点）
数据集：NuScenes-mini ×8（~2560 train samples）
统计方法：max_iters=200，跳过前 10 步 warmup，取剩余 190 步 **median**

**四种方案**：
- **Image**：标准 JPEG pipeline，DataLoader worker 做 CPU decode
- **OpenCV**：video OpenCV pipeline，DataLoader worker 用 cv2.VideoCapture 做 CPU decode
- **Stream**：video stream access，DataLoader worker 内 GPU decode
- **Prefetch**：demuxer-free，主进程后台线程预取 + NVDEC decode，与 backward 重叠

---

## 实验一：Batch Size 扫描（固定 8 GPU）

### 加速比 vs Image Baseline（加速比 = image_time / variant_time）

| BS | Image (s) | OpenCV (s) | Stream (s) | Prefetch (s) | OpenCV 加速比 | Stream 加速比 | Prefetch 加速比 |
|----|-----------|------------|------------|--------------|--------------|--------------|----------------|
| 1  | 0.263     | 0.350      | 0.277      | 0.247        | 0.752x       | 0.952x       | **1.065x**     |
| 2  | 0.356     | 0.515      | 0.353      | 0.316        | 0.691x       | 1.008x       | **1.127x**     |
| 4  | 0.516     | 0.964      | 0.626      | 0.476        | 0.535x       | 0.825x       | **1.085x**     |
| 8  | 0.875     | 1.452      | 1.022      | 0.849        | 0.602x       | 0.856x       | **1.031x**     |
| 16 | 1.671     | 3.536      | —          | 1.676        | 0.473x       | —            | 0.997x         |

> Stream bs=16 未测试；Stream 最大支持 per-GPU bs=8（显存限制）

### 吞吐量（samples/sec = total_bs / median_time）

| BS | total_bs | Image | OpenCV | Stream | Prefetch | Pf/Img 加速比 |
|----|----------|-------|--------|--------|----------|---------------|
| 1  | 8        | 30.4  | 22.9   | 28.9   | 32.4     | **1.065x**    |
| 2  | 16       | 45.0  | 31.0   | 45.4   | 50.6     | **1.127x**    |
| 4  | 32       | 62.0  | 33.2   | 51.1   | 67.2     | **1.085x**    |
| 8  | 64       | 73.2  | 44.1   | 62.6   | 75.4     | **1.031x**    |
| 16 | 128      | 76.6  | 36.2   | —      | 76.4     | 0.997x        |

### data_time（median s）

| BS | Image  | OpenCV | Stream | Prefetch |
|----|--------|--------|--------|----------|
| 1  | 0.017  | 0.025  | 0.017  | **0.008** |
| 2  | 0.023  | 0.035  | 0.023  | **0.006** |
| 4  | 0.041  | 0.064  | 0.039  | **0.013** |
| 8  | 0.069  | 0.091  | 0.063  | **0.023** |
| 16 | 0.123  | 0.181  | —      | **0.047** |

### 观察

1. Prefetch 加速比在 bs=2 时最大（1.127x），随 bs 增大递减，bs=16 时与 image 持平（0.997x）。
2. Stream 在所有 bs 下均慢于或持平 image（0.83x ~ 1.01x）。
3. **OpenCV 全面最慢**，随 bs 增大劣势加剧：0.75x (bs=1) → 0.47x (bs=16)，CPU decode 在高负载下饱和。
4. OpenCV bs=16 吞吐量（36.2）反而低于 bs=8（44.1），CPU 已成为严重瓶颈。
5. 吞吐量在 bs=8→16 增长极小：image +4.6%，prefetch +1.3%。
6. Prefetch data_time 始终低于 image 60-75%。

---

## 实验二：GPU 数量扩展（固定 per-GPU bs=8，弱扩展）

### 加速比 vs Image Baseline

| GPUs | Image (s) | OpenCV (s) | Stream (s) | Prefetch (s) | OpenCV 加速比 | Stream 加速比 | Prefetch 加速比 |
|------|-----------|------------|------------|--------------|--------------|--------------|----------------|
| 1    | 0.847     | 0.971      | 0.918      | 0.770        | 0.872x       | 0.922x       | **1.100x**     |
| 2    | 0.772     | 0.914      | 0.869      | 0.754        | 0.844x       | 0.888x       | **1.024x**     |
| 4    | 0.823     | 1.056      | 0.943      | 0.789        | 0.779x       | 0.873x       | **1.043x**     |
| 8    | 0.875     | 1.452      | 1.022      | 0.859        | 0.602x       | 0.856x       | **1.019x**     |

### 每 GPU 吞吐量（samples/sec/GPU = 8 / median_time）

| GPUs | Image | OpenCV | Stream | Prefetch | Pf/Img 加速比 |
|------|-------|--------|--------|----------|---------------|
| 1    | 9.45  | 8.24   | 8.71   | 10.39    | **1.100x**    |
| 2    | 10.36 | 8.75   | 9.21   | 10.61    | **1.024x**    |
| 4    | 9.72  | 7.57   | 8.48   | 10.14    | **1.043x**    |
| 8    | 9.14  | 5.51   | 7.83   | 9.31     | **1.019x**    |

### data_time（median s）

| GPUs | Image | OpenCV | Stream | Prefetch |
|------|-------|--------|--------|----------|
| 1    | 0.055 | 0.060  | 0.053  | **0.006** |
| 2    | 0.056 | 0.062  | 0.057  | **0.008** |
| 4    | 0.061 | 0.064  | 0.058  | **0.014** |
| 8    | 0.069 | 0.091  | 0.063  | **0.022** |

### 观察

1. Prefetch 在所有 GPU 数下均快于 Image，1 GPU 时加速比最大（1.100x），8 GPU 时收窄至 1.019x。
2. Stream 加速比在 0.86~0.92x 范围，始终慢于 image。
3. **OpenCV 扩展性最差**：1 GPU 时 0.87x 差距尚可，8 GPU 时恶化至 0.60x。per-GPU 吞吐从 8.24 降至 5.51（下降 33%），而 image 仅从 9.45 降至 9.14（下降 3%）。CPU decode 作为固定共享资源，在多卡下成为瓶颈。
4. Prefetch data_time 为 6-22ms，image/stream 为 53-69ms。
