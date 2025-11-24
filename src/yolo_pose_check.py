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

FIELD_DETECTION_MODEL_ID = 'football-field-detection-f07vi/14'
FIELD_DETECTION_MODEL = get_model(
    model_id = FIELD_DETECTION_MODEL_ID,
    api_key = ROBOFLOW_API_KEY
)
SOURCE_VIDEO_PATH = '/home/vphuc/Project/AI/Football_Analysis/121364_0.mp4'


vertex_annotator = sv.VertexAnnotator(
    color = sv.Color.from_hex('#FF1493'),
    radius = 10
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = FIELD_DETECTION_MODEL.infer(frame, confidence = .3)[0]
key_points = sv.KeyPoints.from_inference(result)

annotated_frame = frame.copy()
annotated_frame = vertex_annotator.annotate(
    scene = annotated_frame,
    key_points = key_points
)

sv.plot_image(annotated_frame)