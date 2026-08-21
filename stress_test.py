#!/usr/bin/env python3
# stress_test.py

import os
# SILENCE ONNXRUNTIME C++ LOGS (Must happen before any ONNX imports)
os.environ["ORT_LOGGING_LEVEL"] = "3" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import argparse
import config
from mqtt_handler import init_mqtt, publish_detections
import random

# Realistic Data Pools
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
DIRECTIONS = ["inbound", "outbound"]
CAMERAS = ["BurroughEnd", "WoodgateHill", "AcademyWay", "AccessRoad", 
           "BlakeAve", "HyntonRoad", "InskipRoad", "LilibetClose", 
           "SelinasLane", "WagstaffGardens"]

# Keep a running total to simulate the real edge counter
simulated_totals = {
    "BurroughEnd": {"inbound": 0, "outbound": 0},
    "WoodgateHill": {"inbound": 0, "outbound": 0},
    "AcademyWay": {"inbound": 0, "outbound": 0},
    "AccessRoad": {"inbound": 0, "outbound": 0},
    "BlakeAve": {"inbound": 0, "outbound": 0},
    "HyntonRoad": {"inbound": 0, "outbound": 0},
    "InskipRoad": {"inbound": 0, "outbound": 0},
    "LilibetClose": {"inbound": 0, "outbound": 0},
    "SelinasLane": {"inbound": 0, "outbound": 0},
    "WagstaffGardens": {"inbound": 0, "outbound": 0}   
}

def generate_random_batch(batch_size):
    """Generates a realistic batch of crossing events for a random camera."""
    camera = random.choice(CAMERAS)
    events = []
    
    for _ in range(batch_size):
        direction = random.choice(DIRECTIONS)
        simulated_totals[camera][direction] += 1
        
        events.append({
            "track_id": random.randint(10000, 99999),
            "class": random.choice(CLASSES),
            "direction": direction,
            "confidence": round(random.uniform(0.55, 0.99), 2)
        })
        
    return camera, events, simulated_totals[camera].copy()


def graph_results():
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        # Read the CSV (Node creates the headers automatically now)
        df = pd.read_csv('graph_db_metrics.csv')
        
        # Set 'Time' as the X-axis, and plot the three metrics
        df.set_index('Time').plot(y=['Received', 'Written', 'Buffer'], figsize=(10, 5))
        
        plt.title("System Backpressure Regression")
        plt.xlabel("Elapsed Time (Seconds)")
        plt.ylabel("Message Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[WARN] pandas or matplotlib not installed. Skipping graph.")
        print("Run: pip install pandas matplotlib")
        
        
def stress_test(duration_seconds):
    mqtt_client = init_mqtt()
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    messages_sent = 0
    
    while time.time() < end_time:
        camera, events_batch, totals = generate_random_batch(500)
        fps = round(random.uniform(24.0, 60.0), 1)
        
        # 2. Publish the batch
        publish_detections(
            mqtt_client, 
            camera, 
            fps, 
            events_batch,
            time.time(), 
            totals,
        )
        
        messages_sent += 1
            
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    
    actual_duration = time.time() - start_time
    rate = messages_sent / actual_duration
    
    print("\n\nTest Complete")
    print("Duration specified: ", duration_seconds, "   Actual duration: ", actual_duration)
    print("Messages sent: ", messages_sent)
    print("Throughput rate: ", rate, " msgs/sec")
    print("\n\n")
    
    #graph_results()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mqtt time in secs")
    parser.add_argument("--time", type=int, default=10)
    args = parser.parse_args()
    
stress_test(args.time)