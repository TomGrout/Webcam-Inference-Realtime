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

def stress_test(duration_seconds):
    mqtt_client = init_mqtt()
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    messages_sent = 0
    
    while time.time() < end_time:
        publish_detections(
            mqtt_client, "test", 0.0, 
            [{"track_id": 99, "class": "test", "direction": "outbound", "confidence": 0.00}],
            time.time(), {"inbound": 0, "outbound": 0})
        
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
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mqtt time in secs")
    parser.add_argument("--time", type=int, default=10)
    args = parser.parse_args()
    
stress_test(args.time)