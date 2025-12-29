import argparse
from pathlib import Path
import numpy as np
import cv2
import time
import onnxruntime as ort


class YOLOv11Detector:
    """
    Universal YOLO Object Detection (v8, v10, v11, v12...)
    Support Image/Video/Camera với real-time inference
    """
    
    # COCO 80 classes
    CLASSES = [
        'blue', 'none', 'red', 'white', 'yellow'
    ]
    
    # Colors cho mỗi class (random nhưng consistent)
    COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
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
        
        # Auto-detect output format
        self._detect_output_format(output_shape)
        
        print("✅ Model loaded successfully!")
    
    def _detect_output_format(self, output_shape):
        """
        Tự động phát hiện format output
        YOLOv8/v11: [batch, 84, anchors] - 80 classes COCO
        84 = 4 (box) + 80 (classes)
        """
        if isinstance(output_shape[1], str) or isinstance(output_shape[2], str):
            self.num_classes = 80
            self.format_type = "v8/v11"
            print(f"⚠️ Dynamic output, assuming: 80 classes (COCO)")
        else:
            dim1 = output_shape[1] if not isinstance(output_shape[1], str) else 0
            dim2 = output_shape[2] if not isinstance(output_shape[2], str) else 0
            
            if dim1 == 84 or dim2 == 84:
                self.num_classes = 80
                self.format_type = "v8/v11"
            else:
                # Try to calculate: features = 4 + num_classes
                features = max(dim1, dim2)
                self.num_classes = features - 4
                self.format_type = "auto-detected"
            
            print(f"✅ Detected: {self.num_classes} classes ({self.format_type})")
    
    def preprocess(self, frame):
        """
        Preprocess frame cho YOLO
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Tính scale để resize
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
        image = np.expand_dims(image, axis=0)
        
        return image, scale, offset_x, offset_y
    
    def postprocess(self, output, orig_shape, scale, offset_x, offset_y,
                    conf_threshold=0.25, iou_threshold=0.45):
        """
        Post-process YOLO output
        Output: [batch, 84, anchors] hoặc [batch, anchors, 84]
        """
        # Remove batch dimension
        output = output[0]
        
        # Transpose nếu cần: đảm bảo shape là [anchors, 84]
        if output.shape[0] < output.shape[1]:
            output = output.transpose(1, 0)
        
        # output shape: [8400, 84]
        # 84 = 4 (box: cx, cy, w, h) + 80 (class scores)
        
        boxes = output[:, :4]  # [8400, 4]
        scores = output[:, 4:]  # [8400, 80]
        
        # Lấy class có score cao nhất cho mỗi detection
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        # Filter theo confidence threshold
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Scale boxes về original image
        boxes[:, 0] = (boxes[:, 0] - offset_x) / scale  # cx
        boxes[:, 1] = (boxes[:, 1] - offset_y) / scale  # cy
        boxes[:, 2] = boxes[:, 2] / scale  # w
        boxes[:, 3] = boxes[:, 3] / scale  # h
        
        # Convert từ center format (cx, cy, w, h) sang corner format (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        indices = self.nms(boxes_xyxy, confidences, iou_threshold)
        
        # Format results
        results = []
        for idx in indices:
            results.append({
                'box': boxes_xyxy[idx],
                'score': confidences[idx],
                'class_id': class_ids[idx],
                'class_name': self.CLASSES[class_ids[idx]] if class_ids[idx] < len(self.CLASSES) else f'class_{class_ids[idx]}'
            })
        
        return results
    
    def nms(self, boxes, scores, iou_threshold):
        """
        Non-Maximum Suppression
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
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
        
        return keep
    
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
    
    def draw_results(self, frame, results):
        """
        Draw bounding boxes và labels lên frame
        """
        for result in results:
            box = result['box']
            score = result['score']
            class_id = result['class_id']
            class_name = result['class_name']
            
            # Get coordinates
            x1, y1, x2, y2 = box.astype(int)
            
            # Get color cho class này
            color = tuple(map(int, self.COLORS[class_id % len(self.COLORS)]))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f'{class_name} {score:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
    print(f"✅ Detected {len(results)} object(s)")
    
    # Print detections
    for result in results:
        print(f"  - {result['class_name']}: {result['score']:.2f}")
    
    # Draw results
    output_frame = detector.draw_results(frame.copy(), results)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), output_frame)
        print(f"✅ Saved to: {output_path}")
    
    cv2.imshow('YOLOv11 Detection - Press any key to exit', output_frame)
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
        cv2.putText(output_frame, f'Objects: {len(results)}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_frame, f'Frame: {frame_count}/{total_frames}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv11 Detection - Press q to quit', output_frame)
        
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
        cv2.putText(output_frame, f'Objects: {len(results)}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv11 Camera Detection - Press q to quit', output_frame)
        
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
        description='Universal YOLO Object Detection (v8/v10/v11/v12...) - Support Image/Video/Camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Process image
python script.py --model yolo11n.onnx --source image.jpg

# Process video with save
python script.py --model yolo11n.onnx --source video.mp4 --save output.mp4

# Process camera (default camera 0)
python script.py --model yolo11n.onnx --source camera

# Custom thresholds
python script.py --model yolo11n.onnx --source camera --conf 0.3 --iou 0.5

# Filter specific classes only (chỉ detect người và xe)
python script.py --model yolo11n.onnx --source camera --filter "person,car,truck"

# Use custom classes file
python script.py --model custom.onnx --source camera --classes classes.txt

# Use inline custom classes
python script.py --model custom.onnx --source camera --classes "cat,dog,bird"

Supported models:
- YOLOv8 (all sizes: n, s, m, l, x)
- YOLOv10 (if available)
- YOLOv11 (all sizes: n, s, m, l, x)
- YOLOv12 (future versions)
- Custom YOLO models with any number of classes
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
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IOU threshold (default: 0.45)')
    parser.add_argument('--classes', type=str,
                        help='Path to custom classes file (one class per line) or comma-separated class names')
    parser.add_argument('--filter', type=str,
                        help='Only detect specific classes (comma-separated), e.g., "person,car,dog"')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOv11Detector(args.model)
    
    # Load custom classes nếu có
    if args.classes:
        if Path(args.classes).exists():
            # Đọc từ file
            with open(args.classes, 'r', encoding='utf-8') as f:
                custom_classes = [line.strip() for line in f if line.strip()]
            detector.CLASSES = custom_classes
            print(f"✅ Loaded {len(custom_classes)} custom classes from {args.classes}")
        else:
            # Parse từ command line (comma-separated)
            custom_classes = [c.strip() for c in args.classes.split(',')]
            detector.CLASSES = custom_classes
            print(f"✅ Using {len(custom_classes)} custom classes: {', '.join(custom_classes[:5])}...")
    
    # Setup class filter nếu có
    class_filter = None
    if args.filter:
        filter_names = [c.strip().lower() for c in args.filter.split(',')]
        class_filter = [i for i, name in enumerate(detector.CLASSES) if name.lower() in filter_names]
        print(f"✅ Filtering only: {', '.join([detector.CLASSES[i] for i in class_filter])}")
    
    # Wrapper function để apply filter
    original_predict = detector.predict
    if class_filter:
        def filtered_predict(frame):
            results = original_predict(frame)
            return [r for r in results if r['class_id'] in class_filter]
        detector.predict = filtered_predict
    
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