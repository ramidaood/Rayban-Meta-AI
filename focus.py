import cv2
import torch
import numpy as np
import mss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp_fasttest_subset/weights/best.pt')
FOCUS_CLASSES = ['sidewalk', 'street', 'hole', 'obstacle','person','tree','traffic light','traffic sign','lamp','bench']

def main():
    with mss.mss() as sct:
        screen = sct.monitors[1]
        screen_width = screen['width']
        screen_height = screen['height']
        monitor = {
            "top": 0,
            "left": int(screen_width * 2 / 3),
            "width": int(screen_width / 3),
            "height": screen_height
        }

        while True:
            frame = np.array(sct.grab(monitor))[:, :, :3]
            results = model(frame)
            # Filter detections to only focus classes
            focus_boxes = []
            for *box, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                if label in FOCUS_CLASSES:
                    focus_boxes.append((box, conf, label))
            # Draw only focus class boxes
            output = frame.copy()
            for (x1, y1, x2, y2), conf, label in focus_boxes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(output, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Focused Detection - Right Third", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()