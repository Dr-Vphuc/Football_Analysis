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

from dotenv import load_dotenv
load_dotenv()
import importlib
_inf = importlib.import_module('inference')
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") or getattr(_inf, 'API_KEY', None)

PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

SOURCE_VIDEO_PATH = '/home/vphuc/Project/AI/Football_Analysis/121364_0.mp4'


box_annotator = sv.BoxAnnotator(
    color = sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color = sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
    text_color = sv.Color.from_hex('#000000')
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=.3)[0]
detections = sv.Detections.from_inference(result)

labels = [
    f'{class_name} {confidence:.2f}'
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(
    scene = annotated_frame,
    detections = detections
)
annotated_frame = label_annotator.annotate(
    scene = annotated_frame,
    detections = detections,
    labels = labels
)

sv.plot_image(annotated_frame)