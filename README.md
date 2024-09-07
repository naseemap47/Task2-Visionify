# Task2-Visionify
Implement a program to detect and track the movement of a single object (e.g., a ball) across multiple frames of a video sequence. The object may change in size, orientation, or color over time.

## About
Track person using yolov8 detection model with `botsort` and `bytetrack` tracker.<br>
For PoC purposes we can use simple OpenCV and YOLOv8.

For the production or deployment purposes we can use [Deepstream pipline](https://developer.nvidia.com/deepstream-sdk) or [jetson-inference pipeline](https://github.com/dusty-nv/jetson-inference).<br>
In this way we will get best and optimised performance.

### Limitations
- Detection model performance
- Botsort assigns the track IDs to the object early, but it sometimes misses the tracking in a few frames for a specific object.
- On the other side, Bytetrack tries to assign IDs after more calculations over several frames, but then the tracking ID is much more stable over frames.
-  Due to the immediate assigning of tracking ID in botsort, sometimes it considers a 2-3 frames detection as part of the track and assigns them tracking ID which can cause issues because later objects do not exist in coming frames.
- Bytetrack on the other side does not consider that object as a track. 
- Bytetrack is slow as compared to Botsort in processing, which means low FPS, but better accuracy.

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
### 3. Prepare sample video data
```bash
bash prepare_data.sh
```
## Inference
- `-m`, `--model`: path to yolov8 model.<br>
**If YOLOv8 COCO model not present, It will automatically download.**
- `-s`, `--source`: path to video (*.mp4).
- `-c`, `--conf`: Prediction confidence (0<conf<1).
- `-t`, `--tracker`: choose tracker `botsort` or `bytetrack`.
- `--save`: To save as a video.

Example:

```bash
python3 main.py -s videos/5738755-hd_1920_1080_24fps.mp4 -m yolov8s.pt -c 0.25

# To Save
python3 main.py -s videos/11022597-hd_1920_1080_25fps.mp4 -m yolov8n.pt -c 0.25 --save
```
