# Indian Urban Traffic Vehicle and Pedestrian Counter

Detects and counts **cars**, **buses**, **trucks**, **motorcycles**, and **pedestrians** in Indian urban traffic videos using YOLOv8 and SORT tracker.

## Requirements

```bash
ultralytics
opencv-python 
cvzone 
numpy
filterpy
imageio
lap
```

##  Setup

```bash
pip install -r requirements.txt
```

## Run script
``` bash
python vehicle_pedestrian_counter.py \
  --source sample_videos/<your video_name here; default is demo.mp4>.mp4 \
  --weights Yolo-Weights/yolov8n.pt \
  --output counts.csv
```
## Project structure
```
Basic-Vehicle-and-Pedestrian-Counter-from-Video/
├── mask.png                         # Region of interest image
├── sort.py                          # SORT tracker logic
├── vehicle_pedestrian_counter.py    # Main script with counter logic
├── requirements.txt                 # Python dependencies
├── counts.csv                       # Output file (generated after run)
├── sample_videos/
│   └── cars.mp4                     # Place your input videos here
├── Yolo-Weights/
│   └── yolov8n.pt                   # Pretrained Ultralytics YOLOv8 model
├── README.md                        # Project description & usage
└── output/                          # Folder to store screenshots/demos
```