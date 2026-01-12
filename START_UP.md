## prepare docker env
```bash
nvcr.io/nvidia/pytorch:25.04-py3

docker run --gpus all -it --name streampetr_opt_torch102634 --shm-size=16g -v /home/charlie/pinjiex:/ws  -v /home/charlie/pinjiex/datas:/data --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:25.04-py3  bash
```

## prepare code
git clone ssh://git@gitlab-master.nvidia.com:12051/pinjiex/StreamPETR.git
git checkout pinjie/video_training

## prepare tset data
download test data nusense-mini from google driver which contain images, videos, GOP bin, pkl


## install mmcv
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/charliech/mmcv.git
cd mmcv && git checkout v1.6.0_blackwell
export MMCV_WITH_OPS=1 MMCV_CUDA_ARGS=-std=c++17
pip install -v -e . --user
```

## install mmdetection3d
```bash
git clone --recursive --branch=v1.6.0_blackwell ssh://git@gitlab-master.nvidia.com:12051/charliech/mmdetection3d.git
cd mmdetection3d && pip install -v -e . --user

pip install mmdet==2.28.2 mmsegmentation==0.29.1 --user
```

## install nuscenes-devkit
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/charliech/nuscenes-devkit.git
cd nuscenes-devkit
# git checkout -b master_blackwell remotes/origin/master_blackwell
cd setup && pip install -v -e . --user
```

## install pycd
```bash
git clone https://github.com/dimatura/pypcd.git
cd pypcd && python setup.py install --user
```

## other python package
```bash
# #if you want to use opencv based video training
pip install fvcore==0.1.5.post20221221 --user
pip uninstall -y yapf && pip install yapf==0.40.1 --user
# pip install opencv-python
# rm -rf /usr/local/lib/python3.12/dist-packages/cv2/

apt update && apt install -y libgl1-mesa-dev #https://github.com/NVIDIA/DALI/issues/5052
# apt update && apt install -y libgl1 libglx-mesa0

# #if you want to use accvlab based video training
# pip uninstall opencv-python -y
# pip install opencv-python-headless==4.11.0.86 ##version depend on your original version
```

## install ffmpeg
```bash
wget https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz
tar -xf ffmpeg-4.4.6.tar.xz
cd ffmpeg-4.4.6
LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\'  ./configure --enable-shared --enable-decoder=vp9  --arch=x86_64 --prefix=/ws/test/streampetr-videotraining/deps/ffmpeg-installed
make -j
make install
export FFMPEG_DIR=/ws/test/streampetr-videotraining/deps/ffmpeg-installed
```

## install accvlab for video decoding
```bash
git clone --recursive git@github.com:xupinjie/ACCV-Lab.git

pip install scikit-build --user
cd ./ACCV-Lab/build_config && pip install --no-build-isolation . --user
cd ../packages/on_demand_video_decoder && pip install --no-build-isolation . --user
```

# run

put your data in `/data`

## run image training
```bash
tools/dist_train.sh projects/configs/StreamPETR_video/stream_petr_r50_flash_704_bs2_seq_24e.py 1 --work-dir work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/
```

## run video training
```bash
tools/dist_train.sh projects/configs/StreamPETR_video/stream_petr_r50_flash_704_bs2_seq_24e_video_stream_access.py 1 --work-dir work_dirs/stream_petr_r50_flash_704_bs2_seq_24e_video_stream_access/
```

## run auto labeling
TODO:
