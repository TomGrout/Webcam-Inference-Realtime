# config.py

#MQTT Settings  
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "dalby/traffic"

# Model & Vision Settings  
MODEL_PATH = "models/yolov8m.onnx"
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5

# COCO Class mapping for built environment tracking
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Camera & Village   
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
EXPECTED_CAMS = 2
LOCATION_NAMES = ["BurroughEnd", "WoodgateHill"]