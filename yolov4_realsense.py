import pyrealsense2 as rs
import cv2
import numpy as np
import time

def draw_bounding_box(img, depth_img, confidence, x, y, x_plus_w, y_plus_h, class_name):
    color = (0, 0, 255)
    label = f"{class_name}: {confidence:.2f}"

    if confidence > 0.7:
        # Calculate width and height of the label box
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw filled rectangle for label background
        cv2.rectangle(img, (x, y - label_height - baseline), (x + label_width, y), color, thickness=cv2.FILLED)
        
        # Draw label text
        cv2.putText(img, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + x_plus_w, y + y_plus_h), color, 2)

        # Distance estimation
        cx = int((x + x_plus_w) / 2)
        cy = int((y + y_plus_h) / 2)
        distance = depth_img.get_distance(cx, cy)  # Get distance from depth frame
        dist_label = f"{distance / 1000:.2f} m"

        # Draw distance text below the bounding box
        distance_text_position = (x, y_plus_h)
        cv2.putText(img, dist_label, distance_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    # Load YOLOv4 Tiny pretrained weights and cfg
    model = cv2.dnn.readNetFromDarknet('dnn/yolov4-tiny-custom.cfg', 'dnn/yolov4-tiny-custom_best.weights')

    # Set preferable backend and target to CUDA
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    # Load class names
    class_name = "speedbump"

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

            original_image = np.asanyarray(color_frame.get_data())
            blob = cv2.dnn.blobFromImage(original_image, 1/255.0, (416, 416), swapRB=True, crop=False)
            model.setInput(blob)
            outs = model.forward(model.getUnconnectedOutLayersNames())

            frame_height, frame_width = original_image.shape[:2]
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    confidence = scores[0]  # Assuming only one class
                    if confidence > 0.5:
                        center_x, center_y, width, height = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box
                    draw_bounding_box(original_image, depth_frame, confidences[i], x, y, x+w, y+h, class_name)
                        
            # Menghitung FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time)  # Hitung FPS
            last_time = current_time

            # Tampilkan FPS di layar
            cv2.putText(original_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.imshow('Speed Bump Detection', original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
