import cv2
import torch
import numpy as np
import mss
import warnings
import os
import pygame
import paho.mqtt.client as mqtt
import json
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning)

# MQTT Configuration
MQTT_BROKER = "23a63ad1f9db4bd68c6a1dd8b8afba7.s1.eu.hivemq.cloud"
MQTT_PORT = 8884

# Server (Mac) credentials
MQTT_SERVER_USERNAME = "server"  # Your current server username
MQTT_SERVER_PASSWORD = "Italy2001"  # Your current server password



# Topics
MQTT_TOPIC_COMMANDS = "commands/to-mac"
MQTT_TOPIC_DIRECTIONS = "directions/from-mac"

# MQTT Client setup
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_SERVER_USERNAME, MQTT_SERVER_PASSWORD)  # Using server credentials for the Mac

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        print("Connected to HiveMQ Cloud!")
        client.subscribe(MQTT_TOPIC_COMMANDS)
        print(f"Subscribed to {MQTT_TOPIC_COMMANDS}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for when a message is received"""
    try:
        payload = msg.payload.decode()
        print(f"Received command: {payload}")
        # Handle any commands from the app here
    except Exception as e:
        print(f"Error processing message: {e}")

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects"""
    print(f"Disconnected with result code: {rc}")
    # Attempt to reconnect
    client.reconnect()

# Set up MQTT callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

# Connect to MQTT broker
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")

# Load the default YOLov5s model for testing
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# For testing, use all classes
FOCUS_CLASSES = model.names  # Show all classes

CENTER_ZONE_START_X = 398
CENTER_ZONE_END_X = 796
ALERT_ZONE_TOP_Y = 1492
ALERT_ZONE_BOTTOM_Y = 2240

def initialize_audio():
    """Initialize pygame and audio system"""
    try:
        pygame.init()
        print("Pygame initialized successfully")
        
        # Initialize the mixer with specific parameters
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        print("Audio mixer initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing audio: {e}")
        return False

def load_sounds():
    """Load the sound effects"""
    try:
        # Get the absolute path to the sound file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sound_path = os.path.join(current_dir, 'SurroundTest', 'Assets', 'Audio', 'soundEffect.wav')
        
        print(f"Attempting to load sound from: {sound_path}")
        print(f"File exists: {os.path.exists(sound_path)}")
        
        if not os.path.exists(sound_path):
            print("Sound file not found!")
            return None
            
        sound = pygame.mixer.Sound(sound_path)
        print("Sound loaded successfully")
        return sound
    except Exception as e:
        print(f"Error loading sound: {e}")
        return None

def play_sound(sound, channel, side):
    """Play sound on specified channel with stereo effect"""
    try:
        if sound:
            # Stop any currently playing sounds
            pygame.mixer.stop()
            print(f"[DEBUG] About to play sound on side: {side}")
            # Print channel state and volume
            print(f"[DEBUG] Channel busy before play: {channel.get_busy()}")
            print(f"[DEBUG] Channel volume before play: {channel.get_volume()}")
            # For debugging, use default channel
            sound.play()
            print(f"[DEBUG] Used sound.play() (default channel)")
            # If you want to test the original code, comment out the above and uncomment below:
            # if side == 'left':
            #     channel.set_volume(1.0, 0.0)
            # else:  # right
            #     channel.set_volume(0.0, 1.0)
            # channel.play(sound)
            # print(f"Playing sound on {side} side")
        else:
            print("No sound loaded to play")
    except Exception as e:
        print(f"Error playing sound: {e}")

def analyze_objects(objects, capture_width, capture_height):
    """Analyze objects and determine the most important alert"""
    if not objects:
        print("No objects detected")
        return None, None

    # Calculate zones relative to the right third capture area
    # Center zone: middle third of the right third (so it's centered in the capture area)
    center_zone_start_x = capture_width / 3  # Start at 1/3 of capture width
    center_zone_end_x = 2 * capture_width / 3  # End at 2/3 of capture width
    
    # Alert zone: bottom third of the capture area
    alert_zone_top_y = 2 * capture_height / 3  # Start at 2/3 of capture height
    alert_zone_bottom_y = capture_height  # End at bottom of capture area
    
    zone_counts = {"left": [], "center": [], "right": []}

    # Debug: Print all detected objects
    print("\nAll detected objects:")
    for obj in objects:
        print(f"Object: {obj['class']} at x={obj['center_x']}, y={obj['center_y']}, pixel_distance={obj['pixel_distance']:.0f}px, normalized={obj['distance']:.2f}, confidence={obj['confidence']}, in_alert_zone={obj['in_alert_zone']}")

    for obj in objects:
        x = obj['center_x']
        y = obj['center_y']
        # Use the pre-calculated alert zone status from bounding box intersection
        if center_zone_start_x <= x <= center_zone_end_x:
            if obj['in_alert_zone']:
                zone_counts["center"].append(obj)
        elif x < center_zone_start_x:
            zone_counts["left"].append(obj)
        else:
            zone_counts["right"].append(obj)

    print(f"\nObjects in zones - Left: {len(zone_counts['left'])}, Center: {len(zone_counts['center'])}, Right: {len(zone_counts['right'])}")

    # Find closest objects in each zone
    closest_left = min(zone_counts["left"], key=lambda x: x['pixel_distance']) if zone_counts["left"] else None
    closest_right = min(zone_counts["right"], key=lambda x: x['pixel_distance']) if zone_counts["right"] else None

    # First, check if there are any objects in the center
    if zone_counts["center"]:
        # Find the closest object in the center that's in the alert zone
        center_objects_in_alert_zone = [obj for obj in zone_counts["center"] if obj['in_alert_zone']]
        if center_objects_in_alert_zone:
            center_object = min(center_objects_in_alert_zone, key=lambda x: x['pixel_distance'])
            print(f"\nCenter object details:")
            print(f"Class: {center_object['class']}")
            print(f"Pixel distance: {center_object['pixel_distance']:.0f}px")
            print(f"Normalized distance: {center_object['distance']:.2f}")
            print(f"Confidence: {center_object['confidence']}")
            print(f"Position: x={center_object['center_x']}, y={center_object['center_y']}")
            print(f"Bounding box: {center_object['box_coords']}")
            
            # If the center object is close enough and has good confidence
            if center_object['confidence'] > 0.5:
                # Check closest objects on each side
                left_distance = closest_left['pixel_distance'] if closest_left else float('inf')
                right_distance = closest_right['pixel_distance'] if closest_right else float('inf')
                
                print(f"\nClosest objects:")
                if closest_left:
                    print(f"Left: {closest_left['class']} at {closest_left['pixel_distance']:.0f}px")
                if closest_right:
                    print(f"Right: {closest_right['class']} at {closest_right['pixel_distance']:.0f}px")
                
                # Determine which side has more space (further closest object)
                if left_distance > right_distance:
                    print("Left side has more space, suggesting move LEFT")
                    return "left", center_object
                else:
                    print("Right side has more space, suggesting move RIGHT")
                    return "right", center_object
            else:
                print("\nCenter object rejected because:")
                print(f"- Confidence {center_object['confidence']} <= 0.5")
        else:
            print("No objects in alert zone")
    else:
        print("No objects in center zone")
    
    return None, None

def write_direction(direction):
    """Write direction to file and publish to MQTT"""
    with open("last_direction.txt", "w") as f:
        f.write(direction)
    
    # Publish direction to MQTT
    try:
        payload = json.dumps({
            "direction": direction,
            "timestamp": datetime.now().isoformat()
        })
        mqtt_client.publish(MQTT_TOPIC_DIRECTIONS, payload)
        print(f"Published direction to MQTT: {direction}")
    except Exception as e:
        print(f"Error publishing to MQTT: {e}")

def main():
    # Initialize audio system
    if not initialize_audio():
        print("Failed to initialize audio system")
        return

    # Create channels for left and right
    try:
        left_channel = pygame.mixer.Channel(0)
        right_channel = pygame.mixer.Channel(1)
        print("Audio channels created successfully")
    except Exception as e:
        print(f"Error creating audio channels: {e}")
        return

    # Load sound
    sound = load_sounds()
    if not sound:
        print("Failed to load sound file")
        return

    # Test sound playback immediately after loading
    print("Testing sound playback...")
    sound.play()
    pygame.time.wait(1000)  # Wait 1 second to hear the sound
    print("Sound test complete.")

    with mss.mss() as sct:
        # Get the primary monitor (usually monitor[0])
        screen = sct.monitors[0]  # Changed from monitor[1] to monitor[0] for primary display
        screen_width = 1792  # Fixed width for user's display
        screen_height = 1120  # Fixed height for user's display

        # Calculate relative dimensions
        capture_width = int(screen_width / 3)  # One third of screen width (597 pixels)
        capture_height = screen_height  # Full height
        capture_left = int(screen_width * 2 / 3)  # Start from right third
        capture_top = 0  # Start from top of screen

        # Define capture area with relative dimensions
        monitor = {
            "top": capture_top,
            "left": capture_left,
            "width": capture_width,
            "height": capture_height
        }

        print(f"Screen dimensions: {screen_width}x{screen_height}")
        print(f"Capture area: {monitor}")

        # Calculate zones relative to the right third capture area
        # Center zone: middle third of the right third (centered in capture area)
        center_zone_start_x = capture_width / 3  # Start at 1/3 of capture width
        center_zone_end_x = 2 * capture_width / 3  # End at 2/3 of capture width
        
        # Alert zone: bottom third of the capture area
        alert_zone_top_y = 2 * capture_height / 3  # Start at 2/3 of capture height
        alert_zone_bottom_y = capture_height  # End at bottom of capture area
        
        # Calculate reference points for distance calculations
        center_zone_mid_x = capture_width / 2  # Center of capture area
        center_zone_mid_y = alert_zone_top_y + (capture_height - alert_zone_top_y) / 2  # Middle of alert zone
        
        # Calculate thirds for visualization
        third_w = capture_width / 3
        third_h = capture_height / 3

        print(f"Center zone: x={center_zone_start_x:.0f} to {center_zone_end_x:.0f}")
        print(f"Alert zone: x={center_zone_start_x:.0f} to {center_zone_end_x:.0f}, y={alert_zone_top_y:.0f} to {alert_zone_bottom_y:.0f}")
        print(f"Reference point: ({center_zone_mid_x:.0f}, {center_zone_mid_y:.0f})")

        # Add a longer cooldown for sound
        last_sound_time = 0
        sound_cooldown = 2.5  # 2.5 seconds cooldown
        last_direction = None

        while True:
            # Grab frame from right third
            frame = np.array(sct.grab(monitor))[:, :, :3]

            # Run detection
            results = model(frame)

            # Create a clean output frame (copy of original)
            output = frame.copy()
            
            # Draw center points and print object info
            focus_detections = []
            detected_objects = []
            
            for *box, conf, cls in results.xyxy[0]:
                if conf >= 0.3:  # Increased confidence threshold for more accurate detection
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate center point with higher precision
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Calculate pixel distance from center zone reference point
                    dx = center_x - center_zone_mid_x
                    dy = center_y - center_zone_mid_y
                    pixel_distance = (dx**2 + dy**2)**0.5
                    
                    # Normalize distance to 0-1 range using capture area diagonal as max
                    max_distance = (capture_width**2 + capture_height**2)**0.5
                    normalized_distance = min(1.0, pixel_distance / max_distance)
                    
                    # Get class name
                    class_name = model.names[int(cls)]
                    
                    # Check if bounding box intersects with alert zone
                    # Alert zone: center_zone_start_x to center_zone_end_x, alert_zone_top_y to alert_zone_bottom_y
                    box_in_alert_zone = (
                        x1 < center_zone_end_x and x2 > center_zone_start_x and  # Horizontal overlap
                        y1 < alert_zone_bottom_y and y2 > alert_zone_top_y       # Vertical overlap
                    )
                    
                    # Store object info
                    object_info = {
                        'class': class_name,
                        'confidence': float(conf),
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'distance': normalized_distance,
                        'pixel_distance': pixel_distance,
                        'in_alert_zone': box_in_alert_zone,
                        'box_coords': (x1, y1, x2, y2)  # Store bounding box coordinates
                    }
                    detected_objects.append(object_info)
                    
                    # Draw larger bounding box with thicker lines
                    box_thickness = max(3, int(capture_width / 200))  # Thicker lines for better visibility
                    
                    # Color the bounding box based on alert zone status
                    if box_in_alert_zone:
                        box_color = (0, 0, 255)  # Red for objects in alert zone
                    else:
                        box_color = (0, 255, 0)  # Green for objects outside alert zone
                    
                    cv2.rectangle(output, (x1, y1), (x2, y2), box_color, box_thickness)
                    
                    # Create label with distance info
                    distance_text = f"{pixel_distance:.0f}px"  # Show pixel distance
                    label = f"{class_name} ({distance_text})"
                    
                    # Draw larger text with background for better visibility
                    font_scale = capture_width / 800  # Scale font size with capture width (increased from 1000)
                    font_thickness = max(2, int(capture_width / 400))  # Scale thickness with capture width (increased from 500)
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Draw text background
                    cv2.rectangle(output, 
                                (x1, y1 - text_height - 10),
                                (x1 + text_width + 10, y1),
                                (0, 0, 0),
                                -1)
                    
                    # Draw text
                    cv2.putText(output, label, (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                    
                    focus_detections.append(object_info)

            # Draw center zone reference point
            cv2.circle(output, 
                      (int(center_zone_mid_x), int(center_zone_mid_y)), 
                      10, (0, 255, 0), -1)  # Green dot for center zone reference

            # Draw alert zone boundary (top of alert zone)
            cv2.line(output,
                    (0, int(alert_zone_top_y)),
                    (capture_width, int(alert_zone_top_y)),
                    (0, 255, 0), 2)

            # Draw center zone boundaries (vertical lines only in alert zone)
            cv2.line(output, 
                    (int(center_zone_start_x), int(alert_zone_top_y)),
                    (int(center_zone_start_x), capture_height),
                    (0, 255, 0), 2)
            cv2.line(output, 
                    (int(center_zone_end_x), int(alert_zone_top_y)),
                    (int(center_zone_end_x), capture_height),
                    (0, 255, 0), 2)

            # Draw vertical thirds (for reference)
            cv2.line(output, (int(third_w), 0), (int(third_w), capture_height), (255, 255, 0), 1)
            cv2.line(output, (int(2*third_w), 0), (int(2*third_w), capture_height), (255, 255, 0), 1)
            # Draw horizontal thirds (for reference)
            cv2.line(output, (0, int(third_h)), (capture_width, int(third_h)), (255, 255, 0), 1)
            cv2.line(output, (0, int(2*third_h)), (capture_width, int(2*third_h)), (255, 255, 0), 1)
            # Draw center zone (middle vertical third)
            cv2.rectangle(output, (int(center_zone_start_x), 0), (int(center_zone_end_x), capture_height), (0, 255, 0), 2)
            # Draw alert zone (intersection)
            cv2.rectangle(output, (int(center_zone_start_x), int(alert_zone_top_y)), (int(center_zone_end_x), int(alert_zone_bottom_y)), (0, 0, 255), 2)

            # Analyze objects and determine alert
            direction_to_play, closest_object = analyze_objects(detected_objects, capture_width, capture_height)
            
            # Check if enough time has passed since last sound
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            time_since_last_sound = current_time - last_sound_time
            
            # Only play sound if cooldown has passed
            if time_since_last_sound > sound_cooldown and direction_to_play:
                write_direction(direction_to_play)
                print(f"Playing sound in direction: {direction_to_play}")
                if direction_to_play == "left":
                    play_sound(sound, left_channel, "left")
                elif direction_to_play == "right":
                    play_sound(sound, right_channel, "right")
                last_sound_time = current_time
                last_direction = direction_to_play
            elif direction_to_play:
                print(f"Alert suppressed - {sound_cooldown - time_since_last_sound:.1f}s remaining on cooldown")

            # Debug: Print the shape of the captured frame and output image
            print("Frame shape:", frame.shape)
            print("Output shape:", output.shape)

            # Show result
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imshow("Focused Detection - Right Third", output)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        pygame.quit()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()