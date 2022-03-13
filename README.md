# driver_monitoring_system

This is a project aimed to monitor a driver's status and actions, such as yawn, phonecall, etc.

## Architecture

The driver monitoring system consists of two parts.

* Facial tracking: an API based on [Mediapipe](https://github.com/google/mediapipe) to track facial status, which predicts the eye status (open, close), if open, the gazing direction (left, right, center), and yawn.
* Action detection: a deep learning model (MobileNet) to predcit driver's behavior (phonecall, textting). Yolov5 is further used to detect phones to enhance performance.

## Requirements

```
python=3.8
tensorflow=2.8.0
torch=1.11.0
opencv-python=4.5.5
mediapipe=0.8.9.1
matplotlib=3.5.1
numpy=1.22.3
scikit-learn=1.0.2
```

## Usage

```bash
$ git clone https://github.com/jhan15/driver_monitoring.git
$ cd driver_monitoring

# driver monitorting system
python3 dms.py --video <path_to_video> --checkpoint models/model_split.h5
               --webcam <cam_id>

# play with only facial tracking
$ python3 facial.py
```

## Dataset

The dataset used to train action detection model is [DMD](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset).
