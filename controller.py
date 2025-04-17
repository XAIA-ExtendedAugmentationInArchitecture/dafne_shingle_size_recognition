import time
import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# MQTT topics
ACTION_TOPIC = "/dafne/material_registration/actions"
RESULT_TOPIC = "/dafne/material_registration/result"

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code:", rc)
    client.subscribe(RESULT_TOPIC)

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode('utf-8')}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

time.sleep(4)  # Ensure connection is established

def send_command(cmd, obj_ids, attempt=0):
    message = dict("action": cmd, "ids": obj_ids, "attempt": attempt)
    message_str = json.dumps(message)
    client.publish(ACTION_TOPIC, message_str)
    print(f"Sent command: {message_str}")


# Interactive loop:
while True:
    new_cmd = input("Enter command and id (e.g., 'compute 2', 'accept 2', 'reject 2') or 'exit': ").strip().lower()
    if new_cmd == "exit":
        break
    try:
        parts = new_cmd.split()
        if len(parts) == 2:
            command = parts[0]
            obj_id = int(parts[1])
            send_command(command, obj_id)
        else:
            print("Please enter in the format: <command> <id>")
    except Exception as e:
        print("Error in command format:", e)
    time.sleep(3)

client.loop_stop()
client.disconnect()
