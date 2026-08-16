# mqtt_handler.py
import json
import paho.mqtt.client as mqtt
import config

def init_mqtt():
    """Initializes and starts the MQTT background network loop."""
    print("[INFO] Connecting to MQTT broker...")
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()  # Fallback for older paho-mqtt versions

    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_start()
        print(f"[INFO] MQTT Connected to {config.MQTT_BROKER}:{config.MQTT_PORT}")
        return client
    except Exception as e:
        print(f"[WARN] MQTT connection failed ({e}). Running in local-only mode.")
        return None

def publish_detections(client, stream_name: str, fps: float, detected_objects: list, timestamp: float):
    """Safely publishes detection payloads to the designated topic."""
    if not client or not detected_objects:
        return

    payload = {
        "source": stream_name,
        "timestamp": timestamp,
        "fps": round(fps, 1),
        "count": len(detected_objects),
        "objects": detected_objects
    }
    
    topic = f"{config.MQTT_TOPIC_PREFIX}/{stream_name}"
    try:
        client.publish(topic, json.dumps(payload))
    except Exception as e:
        print(f"[WARN] Failed to publish MQTT message: {e}")