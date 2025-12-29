import argparse
from pathlib import Path
import numpy as np
import cv2
import time
import onnxruntime as ort


class YOLOv11PoseDetector:
    """
    YOLOv8/YOLOv11 Pose Detection - Hỗ trợ ảnh/video/camera
    """
    def __init__(self, model_path: Path):
        print(f"Đang tải model: {model_path}")
        
        # Tạo session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )
        
        # Lấy thông tin input/output
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Lấy input shape
        input_shape = self.session.get_inputs()[0].shape
        print(f"Input shape: {input_shape}")
        
        # Xử lý dynamic shape
        if isinstance(input_shape[2], str) or input_shape[2] == -1:
            self.input_height = 640
            self.input_width = 640
            print(f"⚠️ Dynamic shape detected, using default: 640x640")
        else:
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            print(f"✅ Fixed shape: {self.input_height}x{self.input_width}")
        
        # Output shape info
        output_shape = self.session.get_outputs()[0].shape
        print(f"Output shape: {output_shape}")
        
        # Skeleton connections (COCO 17 keypoints)
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # Màu sắc cho skeleton
        self.colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
            (255, 0, 170), (255, 0, 85), (255, 0, 0)
        ]
        
        print("✅ Model loaded successfully!")
    
    def preprocess(self, frame):
        """
        Preprocess frame cho YOLOv11
        """
        # Lưu kích thước gốc
        orig_h, orig_w = frame.shape[:2]
        
        # Tính tỷ lệ scale
        scale = min(self.input_height / orig_h, self.input_width / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Letterbox padding
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        offset_y = (self.input_height - new_h) // 2
        offset_x = (self.input_width - new_w) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        
        # Normalize: BGR to RGB, HWC to CHW, [0-255] to [0-1]
        image = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        return image, scale, offset_x, offset_y
    
    def postprocess(self, output, orig_shape, scale, offset_x, offset_y, 
                    conf_threshold=0.25, iou_threshold=0.7):
        """
        Post-process YOLOv11 output
        Output shape: [batch, 56, anchors] -> [1, 56, 8400]
        """
        # Remove batch dimension
        output = output[0]  # [56, 8400]
        
        # Transpose to [8400, 56]
        output = output.transpose(1, 0)
        
        results = []
        
        for detection in output:
            # YOLOv11 format: [x, y, w, h, score, kp1_x, kp1_y, kp1_conf, ..., kp17_x, kp17_y, kp17_conf]
            # Total: 4 + 1 + 51 = 56
            box = detection[:4]  # x, y, w, h (center format)
            score = detection[4]
            keypoints = detection[5:].reshape(17, 3)  # 17 keypoints x (x, y, conf)
            
            # Filter by confidence
            if score < conf_threshold:
                continue
            
            # Scale box back to original image
            cx, cy, w, h = box
            cx = (cx - offset_x) / scale
            cy = (cy - offset_y) / scale
            w = w / scale
            h = h / scale
            
            # Scale keypoints back to original image
            scaled_keypoints = keypoints.copy()
            scaled_keypoints[:, 0] = (keypoints[:, 0] - offset_x) / scale
            scaled_keypoints[:, 1] = (keypoints[:, 1] - offset_y) / scale
            
            results.append({
                'box': [cx, cy, w, h],
                'score': score,
                'keypoints': scaled_keypoints
            })
        
        # Apply NMS
        if len(results) > 0:
            results = self.nms(results, iou_threshold)
        
        return results
    
    def nms(self, detections, iou_threshold):
        """
        Non-Maximum Suppression
        """
        if len(detections) == 0:
            return []
        
        # Convert to numpy for faster computation
        boxes = np.array([d['box'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        
        # Convert center format to corner format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def predict(self, frame):
        """
        Run inference on frame
        """
        # Preprocess
        input_tensor, scale, offset_x, offset_y = self.preprocess(frame)
        
        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # Postprocess
        results = self.postprocess(outputs[0], frame.shape[:2], scale, offset_x, offset_y)
        
        return results
    
    def draw_results(self, frame, results, kp_threshold=0.5):
        """
        Draw bounding boxes and skeleton on frame
        """
        for result in results:
            box = result['box']
            score = result['score']
            keypoints = result['keypoints']
            
            # Draw bounding box
            cx, cy, w, h = box
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f'Person {score:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw skeleton
            for i, bone in enumerate(self.skeleton):
                kp1 = keypoints[bone[0] - 1]
                kp2 = keypoints[bone[1] - 1]
                
                if kp1[2] > kp_threshold and kp2[2] > kp_threshold:
                    x1, y1 = int(kp1[0]), int(kp1[1])
                    x2, y2 = int(kp2[0]), int(kp2[1])
                    
                    if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                        0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                        color = self.colors[i % len(self.colors)]
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw keypoints
            for kp in keypoints:
                if kp[2] > kp_threshold:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        
        return frame


def process_image(detector, image_path: Path, output_path: Path = None):
    """
    Process single image
    """
    print(f"Processing image: {image_path}")
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Error: Cannot read image {image_path}")
        return
    
    # Predict
    results = detector.predict(frame)
    print(f"✅ Detected {len(results)} person(s)")
    
    # Draw results
    output_frame = detector.draw_results(frame.copy(), results)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), output_frame)
        print(f"✅ Saved to: {output_path}")
    
    cv2.imshow('YOLOv11 Pose Detection - Press any key to exit', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector, video_path: Path, output_path: Path = None):
    """
    Process video file
    """
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("Processing... Press 'q' to quit")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict
        results = detector.predict(frame)
        
        # Draw results
        output_frame = detector.draw_results(frame, results)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Display info
        cv2.putText(output_frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, f'Persons: {len(results)}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, f'Frame: {frame_count}/{total_frames}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv11 Pose Detection - Press q to quit', output_frame)
        
        # Save if needed
        if out:
            out.write(output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
        print(f"✅ Saved video to: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"✅ Processed {frame_count} frames, Average FPS: {frame_count / elapsed:.1f}")


def process_camera(detector, camera_id: int = 0, output_path: Path = None):
    """
    Process camera stream in real-time
    """
    print(f"Opening camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera {camera_id}")
        return
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Create video writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 20, (width, height))
    
    print("✅ Camera ready! Press 'q' to quit")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame from camera")
            break
        
        # Predict
        results = detector.predict(frame)
        
        # Draw results
        output_frame = detector.draw_results(frame, results)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Display info
        cv2.putText(output_frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, f'Persons: {len(results)}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv11 Camera Pose Detection - Press q to quit', output_frame)
        
        # Save if needed
        if out:
            out.write(output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
        print(f"✅ Saved recording to: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"✅ Processed {frame_count} frames, Average FPS: {frame_count / elapsed:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLOv11 Pose Detection - Support Image/Video/Camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Process image
python script.py --model yolo11n-pose.onnx --source image.jpg

# Process video
python script.py --model yolo11n-pose.onnx --source video.mp4 --save output.mp4

# Process camera (default camera 0)
python script.py --model yolo11n-pose.onnx --source camera

# Process specific camera
python script.py --model yolo11n-pose.onnx --source camera --camera-id 1

# Record camera
python script.py --model yolo11n-pose.onnx --source camera --save recording.mp4
        """
    )
    
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--source', type=str, required=True,
                        help='Input source: image path, video path, or "camera"')
    parser.add_argument('--save', type=Path,
                        help='Path to save output (image/video)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS IOU threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOv11PoseDetector(args.model)
    
    # Process based on source type
    if args.source.lower() == 'camera':
        process_camera(detector, args.camera_id, args.save)
    elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(detector, Path(args.source), args.save)
    elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(detector, Path(args.source), args.save)
    else:
        print(f"❌ Error: Unsupported file format: {args.source}")
        parser.print_help()