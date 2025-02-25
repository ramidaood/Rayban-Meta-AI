import cv2
import numpy as np
import mss
import time
import torch
import warnings

def main():
    # Suppress FutureWarnings from YOLOv5
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load the YOLOv5 small model from PyTorch Hub
    # This downloads the model on first run; later runs will use the cache.
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.5  # Confidence threshold

    # Define obstacle classes (using COCO classes as an example)
    obstacle_classes = {'person', 'bicycle', 'car', 'bus', 'motorbike', 'chair', 'diningtable', 'sofa'}

    # Use mss to manage screen captures
    with mss.mss() as sct:
        # Check that we have at least two physical monitors:
        # sct.monitors[0] = virtual full screen, [1] = primary, [2] = second, etc.
        if len(sct.monitors) < 3:
            print("No second monitor found. Make sure two monitors are connected.")
            return

        # main_monitor is typically sct.monitors[1] (primary display)
        main_monitor = sct.monitors[1]

        # second_monitor is sct.monitors[2]
        second_monitor = sct.monitors[2]
        # second_monitor might look like {'left': 1920, 'top': 0, 'width': 1920, 'height': 1080} on a side-by-side setup

        # Extract the top-left coordinates of the second monitor
        second_left = second_monitor['left']
        second_top = second_monitor['top']

        # Create a single OpenCV window and place it on the second monitor
        window_name = "YOLOv5 Detection (Second Monitor)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Example window size (adjust as needed)
        window_width, window_height = 800, 600
        cv2.resizeWindow(window_name, window_width, window_height)
        # Move the window to the second monitor’s coordinate space
        cv2.moveWindow(window_name, second_left, second_top)

        while True:
            # Capture from the main monitor
            sct_img = sct.grab(main_monitor)
            frame = np.array(sct_img)
            # Convert from BGRA (mss format) to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLOv5 detection on the captured frame
            results = model(frame)

            # Retrieve detections as a Pandas DataFrame
            detections = results.pandas().xyxy[0]

            # Filter detections to only obstacle classes
            obstacles = detections[detections['name'].isin(obstacle_classes)]
            if obstacles.empty:
                print("No obstacles detected.")
            else:
                print("Obstacles detected:")
                print(obstacles[['name', 'confidence']])

            # Render the detection results on the image
            annotated_frame = results.render()[0]

            # Display the annotated frame on the second monitor
            cv2.imshow(window_name, annotated_frame)

            # Break the loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional: Sleep to reduce CPU usage
            time.sleep(0.1)

        # Clean up
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()