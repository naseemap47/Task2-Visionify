# Task2-Visionify
Implement a program to detect and track the movement of a single object (e.g., a ball) across multiple frames of a video sequence. The object may change in size, orientation, or color over time.

## About
Track person using yolov8 detection model with `botsort` and `bytetrack` tracker.<br>
For PoC purposes we can use simple OpenCV and YOLOv8.

For the production or deployment purposes we can use [Deepstream pipline](https://developer.nvidia.com/deepstream-sdk) or [jetson-inference pipeline](https://github.com/dusty-nv/jetson-inference).<br>
In this way we will get best and optimised performance.

## Installation Setup
### 1. Clone the repo
```bash
git clone https://github.com/naseemap47/Task2-Visionify
```
### 2. Install Dependencies
```bash
cd Task2-Visionify
conda create -n torch2 python=3.9 -y
conda activate torch2
# If GPU is availabe
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# Otherwise
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch -y
pip install -r requirements.txt
```
## Inference
- `-m`, `--model`: path to yolov8 model.
- `-s`, `--source`: path to video (*.mp4).
- `-c`, `--conf`: Prediction confidence (0<conf<1).
- `-t`, `--tracker`: choose tracker `botsort` or `bytetrack`.
- `--save`: To save as a video.

Example:

```bash
python3 main.py -s videos/5738755-hd_1920_1080_24fps.mp4

# To Save
python3 main.py -s videos/11022597-hd_1920_1080_25fps.mp4 --save
```
