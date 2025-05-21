import cv2
import torch
import numpy as np
import mss
from navigation import NavigationSystem

def main():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Initialize screen capture
    with mss.mss() as sct:
        screen = sct.monitors[1]
        screen_width = screen['width']
        screen_height = screen['height']
        
        # Initialize navigation system
        nav = NavigationSystem(screen_width, screen_height)
        
        # Define capture area (right third of screen, lower half)
        monitor = {
            "top": int(screen_height / 2),
            "left": int(screen_width * 2 / 3),
            "width": int(screen_width / 3),
            "height": int(screen_height / 2)
        }
        
        # Add a small cooldown for sound
        last_sound_time = 0
        sound_cooldown = 0.3  # 300ms cooldown
        
        while True:
            # Capture frame
            frame = np.array(sct.grab(monitor))[:, :, :3]
            
            # Run detection
            results = model(frame)
            
            # Process detections
            detected_objects = []
            for *box, conf, cls in results.xyxy[0]:
                if conf >= 0.3:
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Calculate normalized distance
                    max_distance = min(monitor['width'], monitor['height'])
                    dx = (center_x - monitor['width']/2) / max_distance
                    dy = (center_y - monitor['height']/2) / max_distance
                    distance = min(1.0, (dx**2 + dy**2)**0.5)
                    
                    object_info = {
                        'class': model.names[int(cls)],
                        'confidence': float(conf),
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'distance': distance
                    }
                    detected_objects.append(object_info)
                    
                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                    
                    # Add label
                    label = f"{object_info['class']} ({distance:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Analyze objects and get navigation direction
            direction, closest_object = nav.analyze_objects(detected_objects)
            
            # Play sound if needed
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_sound_time > sound_cooldown and direction:
                nav.play_sound(direction)
                last_sound_time = current_time
                
                # Add direction indicator
                cv2.putText(frame, f"Move {direction.upper()}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw debug information
            frame = nav.draw_debug_info(frame)
            
            # Show result
            cv2.imshow("Navigation Test", frame)
            
            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 