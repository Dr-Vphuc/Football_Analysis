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
VORONOI_OUTPUT_VIDEO_PATH = '/home/vphuc/Project/AI/Football_Analysis/voronoi_121364_0.mp4'

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

# Open video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_voronoi = cv2.VideoWriter(VORONOI_OUTPUT_VIDEO_PATH, fourcc, FPS, (PITCH_WIDTH, PITCH_HEIGHT))

if not out_voronoi.isOpened(): print("voronoi writer failed")


tracker = sv.ByteTrack()
tracker.reset()

# Fit team classifier using KMeans
kmeans_fitting_frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE)

team_classifier = TeamClassifier(DEVICE, BATCH_SIZE)
crops = []

# Crop player images for KMeans fitting
print('Fitting cluster...')
for frame in kmeans_fitting_frame_generator:
    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]

    crops += players_crops

# Fit the KMeans cluster
team_classifier.fit(crops)

print('Cluster fitted\n')
# Generate video with annotations
print('Starting generate video...')

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)


count = 0
for frame in frame_generator:
    count += 1
    print(f'Frame {int(count % FPS)} in {int(count / FPS)}/{VIDEO_LENGTH}s')
    # ball, goalkeeper, player, referee detection    

    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)

    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

    # team assignment
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections)

    referees_detections.class_id -= 1

    all_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections, referees_detections])


    # frame visualization

    labels = [
        f"#{tracker_id}"
        for tracker_id
        in all_detections.tracker_id
    ]

    all_detections.class_id = all_detections.class_id.astype(int)


    players_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections
    ])

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

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = transformer.transform_points(points=referees_xy)


    # visualize voronoi diagram

    annotated_frame = draw_pitch(CONFIG)
    annotated_frame = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=pitch_players_xy[players_detections.class_id == 0],
        team_2_xy=pitch_players_xy[players_detections.class_id == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=annotated_frame)

    out_voronoi.write(annotated_frame)

# Release video writers
out_voronoi.release()

CAP.release()