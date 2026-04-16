# nuscenes-mini-x8 Dataset

nuscenes-mini-x8 是将 nuscenes-mini（10 个场景）物理复制 8 份得到的训练数据集（80 个场景），
用于在不下载完整 nuscenes（~300 GB）的前提下满足 `InfiniteGroupEachSampleInBatchSampler` 的约束：

```
groups_num = unique_scenes × seq_split_num ≥ global_batch_size
80 × 2 = 160 ≥ 128  (8 GPU × bs=16)
```

---

## 目录结构

```
/data/nuscenes-mini-x8/
├── v1.0-mini/              # 扩增后的 nuScenes metadata JSON
├── samples/                # 图像帧（每个原始文件复制 8 份，filename 加 _x{n} 后缀）
├── sweeps/                 # 同上
├── video_samples/          # HEVC 视频（每个原始视频复制 8 份，scene_token 加 _x{n}）
├── video_samples_packets/  # GOP 二进制包（每个源目录复制 8 份，scene_token 加 _x{n}）
├── maps/                   # 符号链接 → nuscenes-mini/maps/
└── stream_petr/            # pkl 输出目录
    ├── nuscenes2d_temporal_infos_train.pkl
    ├── nuscenes2d_temporal_infos_val.pkl
    ├── nuscenes2d_video_temporal_infos_train.pkl
    └── nuscenes2d_video_temporal_infos_val.pkl
```

---

## Step 1：生成 x8 数据集

```bash
python tools/create_nusc_mini_x8_dataset.py \
    --src  /data/nuscenes-mini \
    --out  /data/nuscenes-mini-x8 \
    --multiplier 8 \
    --workers 8
```

脚本做了以下事情：
- 将 `scene / sample / sample_data / ego_pose / sample_annotation / instance` 表各复制 8 份，token 加 `_x{n}` 后缀
- 为每份副本增加 timestamp 偏移（`n × 1e13 µs`），确保按时间戳排序后同一副本的帧连续，不发生交叉
- 将 `samples/` 和 `sweeps/` 下的图像文件复制 8 份（文件名加 `_x{n}`）
- 将 `video_samples/` 下的视频复制 8 份（scene_token 部分加 `_x{n}`）
- `maps/` 和 `can_bus/` 用符号链接共享，不复制

> **注意**：`video_samples_packets/`（GOP 二进制包）是目录结构，
> `create_nusc_mini_x8_dataset.py` 只处理扁平文件，不会复制 packets。
> 需要单独运行 Step 1.5 生成。

只重新生成 metadata（不复制文件）：

```bash
python tools/create_nusc_mini_x8_dataset.py \
    --src /data/nuscenes-mini \
    --out /data/nuscenes-mini-x8 \
    --metadata-only
```

---

## Step 1.5：生成 x8 GOP 二进制包（demuxer-free 训练需要）

`video_samples_packets/` 下每个视频对应一个子目录，内含 `gop.{first_frame_id}.{gop_len}.bin` 文件。
由于是目录结构，Step 1 的 `create_nusc_mini_x8_dataset.py` 不会处理（只处理扁平文件），
需要单独运行：

```bash
python tools/create_gop_packets_x8.py \
    --src /data/nuscenes-mini \
    --dst /data/nuscenes-mini-x8 \
    --multiplier 8 \
    --workers 8
```

脚本逻辑：
- 扫描源 `video_samples_packets/` 下的 `{scene_token}__CAM_*.mp4/` 目录
- 为每个源目录物理复制 8 份，命名为 `{scene_token}_x{n}__CAM_*.mp4/`
- 目录内的 `gop.*.bin` 文件直接复制，文件名不变（与 scene_token 无关）
- 使用物理复制（非 symlink）以反映真实 I/O 行为

> **前提**：源 `/data/nuscenes-mini/video_samples_packets/` 必须已存在。
> 如果没有，需要先对源数据集运行 GOP 提取工具生成 packets。

---

## Step 2：生成图像 pkl

```bash
cd tools
python create_data_nusc.py \
    --root-path /data/nuscenes-mini-x8 \
    --out-dir   /data/nuscenes-mini-x8/stream_petr \
    --extra-tag nuscenes2d \
    --version   v1.0-mini
```

输出：
- `stream_petr/nuscenes2d_temporal_infos_train.pkl`
- `stream_petr/nuscenes2d_temporal_infos_val.pkl`

---

## Step 3：生成视频 pkl

`video_samples/` 已由 Step 1 准备好（HEVC 格式），跳过重新编码，直接生成 pkl：

```bash
cd tools
python create_video_data_nusc.py \
    --root-path /data/nuscenes-mini-x8 \
    --out-dir   /data/nuscenes-mini-x8/stream_petr \
    --extra-tag nuscenes2d_video \
    --version   v1.0-mini
```

默认不加 `--encode-mjpeg`，脚本直接使用 `video_samples/` 中已有的视频文件，
只遍历帧以赋值 `video_frame_id`，然后写 pkl。

输出：
- `stream_petr/nuscenes2d_video_temporal_infos_train.pkl`
- `stream_petr/nuscenes2d_video_temporal_infos_val.pkl`

如需从 jpg 重新编码视频（完整流程）：

```bash
python create_video_data_nusc.py ... --encode-mjpeg
```

> **注意**：如果重新生成了 `v1.0-mini/` 下的 metadata JSON（例如重跑 Step 1），
> 必须先删除 `tools/nusc.pkl` 缓存，否则视频 pkl 会使用旧的时间戳导致序列顺序错误：
> ```bash
> rm tools/nusc.pkl
> rm /data/nuscenes-mini-x8/stream_petr/nuscenes2d_video_temporal_infos_*.pkl
> ```

---

## 关键设计说明

### Timestamp 偏移

nuScenes 数据集在 `load_annotations` 中按 timestamp 排序帧。若 8 份副本的 timestamp 完全相同，
排序后各副本的帧会交叉混合，导致 `_set_sequence_group_flag` 断言失败：

```
AssertionError: len(new_flags) != num_scenes × seq_split_num
```

解决方案：副本 n 的所有记录加偏移 `n × TIMESTAMP_OFFSET_US`（= 1e13 µs ≈ 115 天），
使排序结果为：副本 0 的全部帧 → 副本 1 的全部帧 → … → 副本 7 的全部帧。

### Scene name 匹配

扩增后场景名为 `scene-xxxx_x{n}`，不再匹配 `nuscenes.utils.splits` 中的原始名称。
`nuscenes_converter.py` 和 `video_nuscenes_converter.py` 中的 `_match_scenes_by_split()`
在匹配前去掉 `_x\d+$` 后缀，使扩增场景能正确归入 train/val split。
