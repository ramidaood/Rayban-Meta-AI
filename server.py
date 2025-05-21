import cv2
import torch
import numpy as np
import mss
import warnings
import os
import pygame
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the default YOLov5s model for testing
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# For testing, use all classes
FOCUS_CLASSES = model.names  # Show all classes

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
            
            if side == 'left':
                # Full volume on left, no volume on right
                channel.set_volume(1.0, 0.0)
            else:  # right
                # No volume on left, full volume on right
                channel.set_volume(0.0, 1.0)
            channel.play(sound)
            print(f"Playing sound on {side} side")
        else:
            print("No sound loaded to play")
    except Exception as e:
        print(f"Error playing sound: {e}")

def analyze_objects(objects, screen_width, screen_height):
    """Analyze objects and determine the most important alert"""
    if not objects:
        print("No objects detected")
        return None, None

    # Get the actual display window dimensions
    display_width = screen_width
    display_height = screen_height

    # Move center point one fifth to the left
    center_point = (display_width / 2) - (display_width / 5)
    center_zone_width = display_width * 0.15  # 15% of screen width
    center_start = center_point - (center_zone_width / 2)
    center_end = center_point + (center_zone_width / 2)

    print(f"Display window size: {display_width}x{display_height}")
    print(f"Center point: {center_point}")
    print(f"Center zone: {center_start} to {center_end}")

    zone_counts = {
        "left": [],
        "center": [],
        "right": []
    }

    # Debug: Print all detected objects
    print("\nAll detected objects:")
    for obj in objects:
        print(f"Object: {obj['class']} at x={obj['center_x']}, pixel_distance={obj['pixel_distance']:.0f}px, normalized={obj['distance']:.2f}, confidence={obj['confidence']}, in_alert_zone={obj['in_alert_zone']}")

    for obj in objects:
        x = obj['center_x']
        # Temporarily removed bottom half restriction for testing
        if center_start <= x < center_end:
            zone_counts["center"].append(obj)
            print(f"Object in center zone: {obj['class']} at x={x}")
        elif x < center_start:
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
            print(f"Position: x={center_object['center_x']}")
            
            # If the center object is close enough and has good confidence
            # Using pixel distance threshold of 300 pixels
            if center_object['pixel_distance'] < 300 and center_object['confidence'] > 0.5:
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
                if center_object['pixel_distance'] >= 300:
                    print(f"- Pixel distance {center_object['pixel_distance']:.0f}px >= 300px")
                if center_object['confidence'] <= 0.5:
                    print(f"- Confidence {center_object['confidence']} <= 0.5")
        else:
            print("No objects in alert zone")
    else:
        print("No objects in center zone")
    
    return None, None

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

    with mss.mss() as sct:
        screen = sct.monitors[1]
        screen_width = screen['width']
        screen_height = screen['height']

        # Define right third of screen, but only lower half
        monitor = {
            "top": int(screen_height / 2),  # Start from middle of screen
            "left": int(screen_width * 2 / 3),  # Start from right third
            "width": int(screen_width / 3),     # One third of screen width
            "height": int(screen_height / 2)    # Half height
        }

        print(f"Monitor capture area: {monitor}")

        # Calculate screen center relative to the captured area
        screen_center_x = monitor["width"] / 2
        screen_center_y = monitor["height"] / 2

        # Calculate center zone reference point (bottom half of center zone)
        center_zone_width = screen_width * 0.15  # 15% of screen width
        center_point = (screen_width / 2) - (screen_width / 5)  # One fifth to the left
        center_zone_x = center_point
        center_zone_y = monitor["height"]  # Bottom of the monitor window

        # Calculate center zone boundaries
        center_zone_start = center_zone_x - (center_zone_width / 2)
        center_zone_end = center_zone_x + (center_zone_width / 2)
        center_zone_mid_y = monitor["height"] / 2  # Middle of the monitor window

        print(f"Screen center: ({screen_center_x}, {screen_center_y})")
        print(f"Center zone reference: ({center_zone_x}, {center_zone_y})")
        print(f"Center zone boundaries: x={center_zone_start} to {center_zone_end}, y=0 to {center_zone_mid_y}")

        # Add a longer cooldown for sound
        last_sound_time = 0
        sound_cooldown = 2.5  # 3 seconds cooldown
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
                    dx = center_x - center_zone_x
                    dy = center_y - center_zone_y
                    pixel_distance = (dx**2 + dy**2)**0.5
                    
                    # Normalize distance to 0-1 range using screen diagonal as max
                    max_distance = (monitor["width"]**2 + monitor["height"]**2)**0.5
                    normalized_distance = min(1.0, pixel_distance / max_distance)
                    
                    # Get class name
                    class_name = model.names[int(cls)]
                    
                    # Store object info
                    object_info = {
                        'class': class_name,
                        'confidence': float(conf),
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'distance': normalized_distance,
                        'pixel_distance': pixel_distance,  # Store raw pixel distance for debugging
                        'in_alert_zone': (center_zone_start <= center_x <= center_zone_end and 
                                        center_y >= center_zone_mid_y)  # Flag for objects in alert zone
                    }
                    detected_objects.append(object_info)
                    
                    # Draw larger center point (red circle)
                    cv2.circle(output, (int(center_x), int(center_y)), 10, (0, 0, 255), -1)
                    
                    # Draw larger crosshairs for better precision
                    crosshair_size = 20
                    cv2.line(output, 
                            (int(center_x - crosshair_size), int(center_y)),
                            (int(center_x + crosshair_size), int(center_y)),
                            (0, 0, 255), 2)
                    cv2.line(output, 
                            (int(center_x), int(center_y - crosshair_size)),
                            (int(center_x), int(center_y + crosshair_size)),
                            (0, 0, 255), 2)
                    
                    # Create label with distance info
                    distance_text = f"{pixel_distance:.0f}px"  # Show pixel distance
                    label = f"{class_name} ({distance_text})"
                    
                    # Draw larger text with background for better visibility
                    font_scale = 1.0
                    font_thickness = 2
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
                      (int(center_zone_x), int(center_zone_y)), 
                      10, (0, 255, 0), -1)  # Green dot for center zone reference

            # Draw center zone boundaries
            cv2.line(output, 
                    (int(center_zone_start), 0),
                    (int(center_zone_start), monitor["height"]),
                    (0, 255, 0), 2)
            cv2.line(output, 
                    (int(center_zone_end), 0),
                    (int(center_zone_end), monitor["height"]),
                    (0, 255, 0), 2)
            
            # Draw horizontal line for bottom half
            cv2.line(output,
                    (int(center_zone_start), int(center_zone_mid_y)),
                    (int(center_zone_end), int(center_zone_mid_y)),
                    (0, 255, 0), 2)

            # Analyze objects and determine alert
            direction_to_play, closest_object = analyze_objects(detected_objects, screen_width, screen_height)
            
            # Check if enough time has passed since last sound
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            time_since_last_sound = current_time - last_sound_time
            
            # Only play sound if cooldown has passed
            if time_since_last_sound > sound_cooldown and direction_to_play:
                print(f"Playing sound in direction: {direction_to_play}")
                if direction_to_play == "left":
                    play_sound(sound, left_channel, "left")
                elif direction_to_play == "right":
                    play_sound(sound, right_channel, "right")
                last_sound_time = current_time
                last_direction = direction_to_play
            elif direction_to_play:
                print(f"Alert suppressed - {sound_cooldown - time_since_last_sound:.1f}s remaining on cooldown")

            # Show result
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imshow("Focused Detection - Right Third", output)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()