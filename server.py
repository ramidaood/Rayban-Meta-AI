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

def get_navigation_instruction(center_x, center_y, screen_center_x, screen_center_y, screen_width):
    """Calculate navigation instructions based on object position relative to screen center."""
    # Calculate relative position
    dx = center_x - screen_center_x
    dy = center_y - screen_center_y
    
    # Calculate center zone threshold (1/5 of screen width)
    center_threshold = screen_width / 5
    
    # Determine horizontal direction with a larger deadzone
    if abs(dx) < center_threshold * 1.2:  # Increased deadzone for more stability
        horizontal = "center"
    else:
        horizontal = "right" if dx > 0 else "left"
    
    # Determine vertical direction
    if abs(dy) < 50:  # Within 50 pixels of center
        vertical = "center"
    else:
        vertical = "down" if dy > 0 else "up"
    
    # Calculate distance (normalized to 0-1 range)
    distance = min(1.0, (dx**2 + dy**2)**0.5 / (screen_center_x * 2))
    
    return horizontal, vertical, distance

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
            "left": int(screen_width * 2 / 3),
            "width": int(screen_width / 3),
            "height": int(screen_height / 2)  # Only capture lower half
        }

        # Calculate screen center
        screen_center_x = monitor["width"] / 2
        screen_center_y = monitor["height"] / 2

        # Add a small cooldown for sound
        last_sound_time = 0
        sound_cooldown = 0.3  # 300ms cooldown
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
            objects_on_sides = {"left": [], "right": []}  # Store objects with their vertical position
            
            for *box, conf, cls in results.xyxy[0]:
                if conf >= 0.3:  # Increased confidence threshold for more accurate detection
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate center point with higher precision
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Get navigation instructions
                    horizontal, vertical, distance = get_navigation_instruction(
                        center_x, center_y, screen_center_x, screen_center_y, monitor["width"]
                    )
                    
                    # Only track objects that are significantly off-center
                    if horizontal != "center":
                        # Track objects on sides with their vertical position and confidence
                        if horizontal == "left":
                            objects_on_sides["left"].append((center_y, distance, conf))
                        elif horizontal == "right":
                            objects_on_sides["right"].append((center_y, distance, conf))
                    
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
                    
                    # Get class name
                    class_name = model.names[int(cls)]
                    
                    # Create label with navigation info
                    distance_text = "very close" if distance < 0.2 else "close" if distance < 0.4 else "far"
                    label = f"{class_name} - {horizontal} {vertical} ({distance_text})"
                    
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
                    
                    # Store detection info for logging
                    detection_info = {
                        'class': class_name,
                        'confidence': float(conf),
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'direction': f"{horizontal} {vertical}",
                        'distance': distance_text
                    }
                    focus_detections.append(detection_info)

            # Sort objects by vertical position (lower objects first), distance, and confidence
            for side in objects_on_sides:
                objects_on_sides[side].sort(key=lambda x: (-x[0], x[1], -x[2]))  # Sort by y position (descending), distance, and confidence (descending)
            
            # Check if enough time has passed since last sound
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_sound_time > sound_cooldown:
                # Determine which direction to play (prioritize lower objects)
                direction_to_play = None
                
                # Check left side first (objects on left, play right sound)
                if objects_on_sides["left"]:
                    direction_to_play = "right"  # Play on opposite side
                # Then check right side
                elif objects_on_sides["right"]:
                    direction_to_play = "left"  # Play on opposite side
                
                # Only play if direction changed or no previous direction
                if direction_to_play and direction_to_play != last_direction:
                    if direction_to_play == "left":
                        play_sound(sound, left_channel, "left")
                    else:
                        play_sound(sound, right_channel, "right")
                    last_sound_time = current_time
                    last_direction = direction_to_play

            # Log detections with navigation info
            if focus_detections:
                print("✅ Detected:", ", ".join(
                    f"{d['class']} ({d['confidence']:.2f}) - {d['direction']} ({d['distance']})"
                    for d in focus_detections
                ))

            # Show result
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imshow("Focused Detection - Lower Half", output)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()