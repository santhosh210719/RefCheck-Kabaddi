# RefCheck: Kabaddi Decision Review Assistant

Third-year ECE project by Santhosh R.

## What it does
Detects players and referees in kabaddi footage, finds the raider and
counts defenders, and (in progress) flags touch and out-of-bounds moments.

## Notebooks
1. 01_resnet50_player_vs_referee.ipynb: ResNet50 transfer learning on cropped
   Player / referee images. Test accuracy 92.89%, macro F1 0.907.
2. 02_yolo_detection_and_rules.ipynb: YOLOv8s trained on the Roboflow
   "kabaddi-player-detection" dataset (test mAP50 0.78; Player 0.93, referee 0.63),
   plus rule-based raider and defender counting on 10 sample clips.

## Data
- Roboflow: vivek-gangurde / kabaddi-player-detection, version 1 (YOLOv8 format).
- 10 sample match clips (not included in this repository).

## How to run
Open the notebooks on Kaggle with a GPU, add your own Roboflow API key as a
Kaggle Secret named ROBOFLOW_API_KEY, then run the cells in order.

## Known limits
- Referee detection is weaker than player detection.
- Defender count can read 1 low when players overlap.
- Touch and out-of-bounds are review flags, not rulings.
