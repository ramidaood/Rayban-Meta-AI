import cv2
import torch
import numpy as np
import mss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the default YOLOv5s model for testing
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# For testing, use all classes
FOCUS_CLASSES = model.names  # Show all classes

def main():
    with mss.mss() as sct:
        screen = sct.monitors[1]
        screen_width = screen['width']
        screen_height = screen['height']

        # Define right third of screen
        monitor = {
            "top": 0,
            "left": int(screen_width * 2 / 3),
            "width": int(screen_width / 3),
            "height": screen_height
        }

        while True:
            # Grab frame from right third
            frame = np.array(sct.grab(monitor))[:, :, :3]

            # Run detection
            results = model(frame)

            # Debug: print detection tensor shape
            print("Detections tensor shape:", results.xyxy[0].shape)

            # Create a clean output frame (copy of original)
            output = frame.copy()
            
            # Draw center points and print object info
            focus_detections = []
            for *box, conf, cls in results.xyxy[0]:
                if conf >= 0.1:  # Confidence threshold
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate center point with higher precision
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Draw larger center point (red circle)
                    cv2.circle(output, (int(center_x), int(center_y)), 10, (0, 0, 255), -1)  # Increased radius to 10
                    
                    # Draw larger crosshairs for better precision
                    crosshair_size = 20  # Increased crosshair size
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
                    
                    # Create label with precise coordinates
                    label = f"{class_name} ({center_x:.1f}, {center_y:.1f})"
                    
                    # Draw larger text with background for better visibility
                    font_scale = 1.0  # Increased font size
                    font_thickness = 2  # Increased thickness
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
                    
                    # Print detection info with precise coordinates
                    print(f"Detected {class_name} at center position: ({center_x:.1f}, {center_y:.1f}) pixels")
                    
                    # Store detection info for logging
                    detection_info = {
                        'class': class_name,
                        'confidence': float(conf),
                        'center_x': float(center_x),
                        'center_y': float(center_y)
                    }
                    focus_detections.append(detection_info)
            # Log detections
            if focus_detections:
                print("✅ Detected:", ", ".join(f"{d['class']} ({d['confidence']:.2f}) at center: ({d['center_x']:.1f}, {d['center_y']:.1f})" for d in focus_detections))

            # Show result
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imshow("Focused Detection - Right Third", output)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()