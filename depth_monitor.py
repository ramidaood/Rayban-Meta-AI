import torch
import cv2
import numpy as np
import mss

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3  # Confidence threshold

# Load MiDaS model and transform
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

with mss.mss() as sct:
    screen_width = 1792
    screen_height = 1120
    capture_width = int(screen_width / 3)
    capture_height = screen_height
    capture_left = int(screen_width * 2 / 3)
    capture_top = 0

    monitor = {
        "top": capture_top,
        "left": capture_left,
        "width": capture_width,
        "height": capture_height
    }

    while True:
        frame = np.array(sct.grab(monitor))[:, :, :3]  # RGB

        # Depth
        input_tensor = midas_transforms(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to("cpu")
        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # Detection
        results = model(frame)
        detect_frame = frame.copy()
        for *box, conf, cls in results.xyxy[0]:
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(detect_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detect_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Resize and stack views
        vis_original = cv2.resize(frame, (capture_width, capture_height))
        vis_depth = cv2.resize(depth_vis_color, (capture_width, capture_height))
        vis_detect = cv2.resize(detect_frame, (capture_width, capture_height))
        combined = np.hstack((vis_original, vis_depth, vis_detect))

        # Show
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imshow("Normal | Depth | Detection", combined_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()