import os
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM2_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_GROUNDINGDINO_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

import warnings
warnings.filterwarnings('ignore')

from inference import get_model
import supervision as sv
import numpy as np
import torch
from collections import deque
from libs.func import (
    resolve_goalkeepers_team_id, 
    draw_pitch, 
    draw_pitch_voronoi_diagram_2, 
    replace_outliers_based_on_distance )
from libs.team import TeamClassifier
from libs.view import ViewTransformer
from libs.configs import SoccerPitchConfiguration
from libs.annotators import (
    draw_points_on_pitch, 
    draw_pitch_voronoi_diagram, 
    draw_paths_on_pitch)
import cv2

# Environment variables
from dotenv import load_dotenv
load_dotenv()
import importlib
_inf = importlib.import_module('inference')
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") or getattr(_inf, 'API_KEY', None)

# Load model YOLO detect
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
# Load model YOLO pose
FIELD_DETECTION_MODEL_ID = 'football-field-detection-f07vi/14'
FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

SOURCE_VIDEO_PATH = '/home/vphuc/Project/AI/Football_Analysis/input/121364_0.mp4'
BALL_TRACKING_OUTPUT_IMG_PATH = '/home/vphuc/Project/AI/Football_Analysis/ball_tracking_121364_0.jpg'

# Constants
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
STRIDE = 30
MAXLEN = 5
CONFIG = SoccerPitchConfiguration()
CAP = cv2.VideoCapture(SOURCE_VIDEO_PATH)
WIDTH  = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
PITCH_WIDTH = 1300
PITCH_HEIGHT = 800
FPS    = int(CAP.get(cv2.CAP_PROP_FPS))
VIDEO_LENGTH = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT) / FPS)
MAX_DISTANCE_THRESHOLD = 500


frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

path_raw = []
M = deque(maxlen=MAXLEN)
count = 0
for frame in frame_generator:
    count += 1
    print(f'Frame {int(count % FPS)} in {int(count / FPS)}/{VIDEO_LENGTH}s')
    if count == 5:
        break
    # ball, goalkeeper, player, referee detection    

    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)


    # detect pitch key points

    result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    # project ball, players and referies on pitch

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )


    # ball tracking path

    M.append(transformer.m)
    transformer.m = np.mean(np.array(M), axis=0)

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    path_raw.append(pitch_ball_xy)

# post-process ball tracking path
path = [
    np.empty((0, 2), dtype=np.float32) if coorinates.shape[0] >= 2 else coorinates
    for coorinates
    in path_raw
]

path = [coorinates.flatten() for coorinates in path]

path = replace_outliers_based_on_distance(path, MAX_DISTANCE_THRESHOLD)

# draw ball tracking path
annotated_frame = draw_pitch(CONFIG)
annotated_frame = draw_paths_on_pitch(
    config=CONFIG,
    paths=[path],
    color=sv.Color.WHITE,
    pitch=annotated_frame)

cv2.imwrite(BALL_TRACKING_OUTPUT_IMG_PATH, annotated_frame)


CAP.release()