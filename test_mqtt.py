import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime

# MQTT Configuration
MQTT_BROKER = "23a63ad1f9db4bd68c6a1dd8b8afba7.s1.eu.hivemq.cloud"
MQTT_PORT = 8883 
MQTT_USERNAME = "server"
MQTT_PASSWORD = "Italy2001"
MQTT_TOPIC_COMMANDS = "commands/to-mac"
MQTT_TOPIC_DIRECTIONS = "directions/from-mac"

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print("on_connect called with rc:", rc)
    if rc == 0:
        print("✅ Connected!")
        # Subscribe to commands topic
        client.subscribe(MQTT_TOPIC_COMMANDS)
        print(f"✅ Subscribed to {MQTT_TOPIC_COMMANDS}")
    else:
        print("❌ Failed to connect, rc =", rc)
        print("Common return codes:")
        print("0: Connection successful")
        print("1: Connection refused - incorrect protocol version")
        print("2: Connection refused - invalid client identifier")
        print("3: Connection refused - server unavailable")
        print("4: Connection refused - bad username or password")
        print("5: Connection refused - not authorized")

# Callback when a message is received
def on_message(client, userdata, msg):
    print(f"\n📥 Received message on topic {msg.topic}: {msg.payload.decode()}")

# Create MQTT client with protocol version 5
client = mqtt.Client()

# Set username and password
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Set callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
print("Connecting...")
client.tls_set()
client.connect(MQTT_BROKER, MQTT_PORT, 60)  # Increased timeout to 60 seconds
client.loop_forever()

# Wait for connection to establish
time.sleep(2)

# Test publishing a message
test_message = {
    "direction": "test_direction",
    "timestamp": datetime.now().isoformat()
}

try:
    print("\n📤 Publishing test message...")
    client.publish(MQTT_TOPIC_DIRECTIONS, json.dumps(test_message))
except Exception as e:
    print(f"❌ Error publishing message: {e}")

# Keep the script running for a while to receive any messages
print("\n⏳ Waiting for messages (press Ctrl+C to exit)...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Exiting...")
    client.loop_stop()
    client.disconnect() 