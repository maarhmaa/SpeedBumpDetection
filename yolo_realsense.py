import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
import time

# Load class names
CLASSES = yaml_load(check_yaml("data.yaml"))["names"]

def draw_bounding_box(img, depth_img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]}: {confidence:.2f}"
    color = (0, 0, 255)

    # Calculate width and height of the label box.
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # Draw filled rectangle for label background.
    cv2.rectangle(img, (x, y - label_height - baseline), (x + label_width, y), color, thickness=cv2.FILLED)
    
    # Draw label text.
    cv2.putText(img, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw bounding box.
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # Distance Estimation
    cx = int((x + x_plus_w) / 2)
    cy = int((y + y_plus_h) / 2)
    distance = depth_img.get_distance(cx, cy)
    #print(f"Box coordinates: x={x}, y={y}, x_plus_w={x_plus_w}, y_plus_h={y_plus_h}")
    #print(f"Calculated center: cx={cx}, cy={cy}")
    #print(f"Distance measured: {distance} meters")
    dist_label = f"{distance:.2f} m"

    # Draw distance text below the bounding box.
    distance_text_position = (x, y_plus_h)
    cv2.putText(img, dist_label, distance_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main(onnx_model):
    model = cv2.dnn.readNetFromONNX(onnx_model)
    
    # Set preferable backend and target to CUDA
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    last_time = time.time()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            original_image = np.asanyarray(color_frame.get_data())

            [height, width, _] = original_image.shape
            length = max(height, width)
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = original_image

            scale = length / 640
            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
            model.setInput(blob)

            outputs = model.forward()
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.25:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2],
                        outputs[0][i][3],
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]

                x = round(box[0] * scale)
                y = round(box[1] * scale)
                w = round(box[2] * scale)
                h = round(box[3] * scale)

                x_plus_w = x + w
                y_plus_h = y + h
                draw_bounding_box(
                    original_image,
                    depth_frame,
                    class_ids[index],
                    scores[index],
                    x,
                    y,
                    x_plus_w,
                    y_plus_h,
                )

            # Menghitung FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time)  # Hitung FPS
            last_time = current_time

            # Tampilkan FPS di layar
            cv2.putText(original_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.imshow("Speed Bump Detection", original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main("dnn/yolov8n.onnx")
