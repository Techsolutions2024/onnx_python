import argparse
import onnx
import onnx.shape_inference
import onnxruntime_extensions
from onnxruntime_extensions.tools.pre_post_processing import *
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time


def add_pre_post_processing(input_model_path: Path, output_model_path: Path):
    """
    Thêm pre và post processing vào model YOLOv8 Pose có sẵn
    """
    print(f"Đang tải model từ: {input_model_path}")
    model = onnx.load(str(input_model_path))
    
    # Lấy thông tin shape của model
    model_with_shape = onnx.shape_inference.infer_shapes(model)
    input_shape = model_with_shape.graph.input[0].type.tensor_type.shape
    
    # Kích thước input của model (thường là 640x640)
    w_in = input_shape.dim[-1].dim_value
    h_in = input_shape.dim[-2].dim_value
    print(f"Kích thước input của model: {h_in}x{w_in}")
    
    # Tạo pipeline pre/post processing
    onnx_opset = 18
    inputs = [create_named_value("image_bytes", onnx.TensorProto.UINT8, ["num_bytes"])]
    pipeline = PrePostProcessor(inputs, onnx_opset)
    
    # Pre-processing: chuyển đổi ảnh input
    pre_processing_steps = [
        ConvertImageToBGR(name="BGRImageHWC"),
        ChannelsLastToChannelsFirst(name="BGRImageCHW"),
        Resize((h_in, w_in), policy='not_larger', layout='CHW'),
        LetterBox(target_shape=(h_in, w_in), layout='CHW'),
        ImageBytesToFloat(),
        Unsqueeze([0]),
    ]
    pipeline.add_pre_processing(pre_processing_steps)
    
    # Post-processing: xử lý output và vẽ bounding boxes
    post_processing_steps = [
        Squeeze([0]),
        Transpose([1, 0]),
        Split(num_outputs=3, axis=1, splits=[4, 1, 51]),
        SelectBestBoundingBoxesByNMS(iou_threshold=0.7, score_threshold=0.25, has_mask_data=True),
        (ScaleNMSBoundingBoxesAndKeyPoints(num_key_points=17, layout='CHW'),
         [
             utils.IoMapEntry("BGRImageCHW", producer_idx=0, consumer_idx=1),
             utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
             utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
         ]),
    ]
    pipeline.add_post_processing(post_processing_steps)
    
    # Tạo model mới
    print("Đang thêm pre/post processing...")
    new_model = pipeline.run(model)
    
    # Validate và lưu
    _ = onnx.shape_inference.infer_shapes(new_model, strict_mode=True)
    onnx.save_model(new_model, str(output_model_path))
    print(f"Model đã được lưu tại: {output_model_path}")


class PoseDetector:
    """
    Class xử lý pose detection cho ảnh/video/camera
    """
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        
        print("Đang khởi tạo pose detector...")
        session_options = ort.SessionOptions()
        session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
        
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )
        self.input_name = self.session.get_inputs()[0].name
        
        # Skeleton connections (COCO format - 17 keypoints)
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
        
        print("Pose detector đã sẵn sàng!")
    
    def predict(self, image_bytes):
        """
        Chạy inference trên image bytes
        """
        outputs = self.session.run(None, {self.input_name: image_bytes})
        return outputs[0]
    
    def draw_results(self, frame, nms_output):
        """
        Vẽ bounding boxes và skeleton lên frame
        """
        for result in nms_output:
            box, score, _, keypoints = np.split(result, (4, 5, 6))
            keypoints = keypoints.reshape((17, 3))
            
            # Vẽ bounding box (format: center XYWH)
            half_w = box[2] / 2
            half_h = box[3] / 2
            x0, y0 = int(box[0] - half_w), int(box[1] - half_h)
            x1, y1 = int(box[0] + half_w), int(box[1] + half_h)
            
            # Vẽ box
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            
            # Hiển thị confidence score
            label = f"Person {score[0]:.2f}"
            cv2.putText(frame, label, (x0, y0 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Vẽ skeleton
            for i, bone in enumerate(self.skeleton):
                kp1 = keypoints[bone[0] - 1]
                kp2 = keypoints[bone[1] - 1]
                
                if kp1[2] > 0.5 and kp2[2] > 0.5:
                    pos1 = (int(kp1[0]), int(kp1[1]))
                    pos2 = (int(kp2[0]), int(kp2[1]))
                    
                    if (0 <= pos1[0] < frame.shape[1] and 0 <= pos1[1] < frame.shape[0] and
                        0 <= pos2[0] < frame.shape[1] and 0 <= pos2[1] < frame.shape[0]):
                        color = self.colors[i % len(self.colors)]
                        cv2.line(frame, pos1, pos2, color, 2)
            
            # Vẽ keypoints
            for kp in keypoints:
                if kp[2] > 0.5:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        
        return frame


def process_image(detector: PoseDetector, image_path: Path, output_path: Path = None):
    """
    Xử lý ảnh đơn lẻ
    """
    print(f"Đang xử lý ảnh: {image_path}")
    
    # Đọc ảnh
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    # Chuyển sang bytes để đưa vào model
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_bytes = np.frombuffer(img_encoded.tobytes(), dtype=np.uint8)
    
    # Predict
    nms_output = detector.predict(image_bytes)
    
    # Vẽ kết quả
    result_frame = detector.draw_results(frame.copy(), nms_output)
    
    # Lưu hoặc hiển thị
    if output_path:
        cv2.imwrite(str(output_path), result_frame)
        print(f"Đã lưu kết quả tại: {output_path}")
    
    cv2.imshow('Pose Detection - Nhấn phím bất kỳ để thoát', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector: PoseDetector, video_path: Path, output_path: Path = None):
    """
    Xử lý video
    """
    print(f"Đang xử lý video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo VideoWriter nếu cần lưu
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("Đang xử lý... Nhấn 'q' để dừng")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển sang bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        image_bytes = np.frombuffer(img_encoded.tobytes(), dtype=np.uint8)
        
        # Predict
        nms_output = detector.predict(image_bytes)
        
        # Vẽ kết quả
        result_frame = detector.draw_results(frame, nms_output)
        
        # Tính FPS
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Hiển thị FPS
        cv2.putText(result_frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị
        cv2.imshow('Pose Detection - Nhấn q để thoát', result_frame)
        
        # Lưu nếu cần
        if out:
            out.write(result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
        print(f"Đã lưu video tại: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"Đã xử lý {frame_count} frames với FPS trung bình: {frame_count / elapsed:.1f}")


def process_camera(detector: PoseDetector, camera_id: int = 0, output_path: Path = None):
    """
    Xử lý camera realtime
    """
    print(f"Đang mở camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera {camera_id}")
        return
    
    # Đặt resolution (tùy chọn)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Tạo VideoWriter nếu cần lưu
    out = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 20, (width, height))
    
    print("Camera đã sẵn sàng! Nhấn 'q' để thoát")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ camera")
            break
        
        # Chuyển sang bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        image_bytes = np.frombuffer(img_encoded.tobytes(), dtype=np.uint8)
        
        # Predict
        nms_output = detector.predict(image_bytes)
        
        # Vẽ kết quả
        result_frame = detector.draw_results(frame, nms_output)
        
        # Tính FPS
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Hiển thị thông tin
        cv2.putText(result_frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f'Persons: {len(nms_output)}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị
        cv2.imshow('Camera Pose Detection - Nhấn q để thoát', result_frame)
        
        # Lưu nếu cần
        if out:
            out.write(result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
        print(f"Đã lưu video tại: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"Đã xử lý {frame_count} frames với FPS trung bình: {frame_count / elapsed:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLOv8 Pose Detection - Hỗ trợ ảnh/video/camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Xử lý model
  python script.py --model yolov8n-pose.onnx --output model_processed.onnx
  
  # Xử lý ảnh
  python script.py --model model_processed.onnx --source image.jpg
  
  # Xử lý video
  python script.py --model model_processed.onnx --source video.mp4 --save output.mp4
  
  # Xử lý camera (mặc định camera 0)
  python script.py --model model_processed.onnx --source camera
  
  # Xử lý camera cụ thể
  python script.py --model model_processed.onnx --source camera --camera-id 1
        """
    )
    
    parser.add_argument('--model', type=Path, required=True,
                        help='Đường dẫn đến file model ONNX')
    parser.add_argument('--output', type=Path,
                        help='Đường dẫn lưu model đã xử lý (chỉ dùng khi chưa có processed model)')
    parser.add_argument('--source', type=str,
                        help='Nguồn input: đường dẫn ảnh/video hoặc "camera"')
    parser.add_argument('--save', type=Path,
                        help='Đường dẫn lưu kết quả (ảnh/video)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='ID của camera (mặc định: 0)')
    
    args = parser.parse_args()
    
    # Nếu chưa có processed model thì tạo
    if args.output:
        add_pre_post_processing(args.model, args.output)
        model_to_use = args.output
    else:
        model_to_use = args.model
    
    # Nếu có source thì xử lý
    if args.source:
        detector = PoseDetector(model_to_use)
        
        if args.source.lower() == 'camera':
            # Xử lý camera
            process_camera(detector, args.camera_id, args.save)
        elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Xử lý ảnh
            process_image(detector, Path(args.source), args.save)
        elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Xử lý video
            process_video(detector, Path(args.source), args.save)
        else:
            print(f"Lỗi: Định dạng file không được hỗ trợ hoặc file không tồn tại: {args.source}")
    else:
        print("Đã hoàn tất xử lý model. Sử dụng --source để chạy inference.")