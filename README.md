
# onnx_python 🚀

Repo này cung cấp ví dụ và công cụ chạy **YOLO inference** bằng **ONNX Runtime** trong Python.  
Mục tiêu là giúp bạn dễ dàng triển khai mô hình YOLO đã convert sang định dạng ONNX để thực hiện nhận diện vật thể nhanh chóng và hiệu quả.

---

## 📂 Cấu trúc thư mục

```
onnx_python/
│── models/              # Chứa file mô hình YOLO (.onnx)
│── box_utils.py         # Hàm hỗ trợ xử lý bounding box
│── detectv11onnx.py     # Script chạy YOLOv11 inference
│── posev11onnx.py       # Script chạy YOLOv11 pose inference
│── inf.py               # Module inference chung
│── main.py              # Entry point demo
│── onnxother.py         # Các tiện ích khác liên quan đến ONNX
│── 1.mp4                # Ví dụ video input
```

---

## ⚙️ Yêu cầu hệ thống

- Python >= 3.8  
- [onnxruntime](https://onnxruntime.ai/)  
- OpenCV (`cv2`)  
- NumPy  

Cài đặt nhanh:

```bash
pip install onnxruntime opencv-python numpy
```

---

## ▶️ Cách chạy demo

1. Clone repo:
   ```bash
   git clone https://github.com/Techsolutions2024/onnx_python.git
   cd onnx_python
   ```

2. Đặt mô hình YOLO ONNX vào thư mục `models/` (ví dụ: `yolov11.onnx`).

3. Chạy script inference:
   ```bash
   python detectv11onnx.py --source 1.mp4 --model models/yolov11.onnx
   ```

4. Kết quả sẽ hiển thị bounding boxes trên video hoặc ảnh đầu vào.

---

## 📌 Các tính năng chính

- Hỗ trợ **YOLOv11 ONNX inference** với onnxruntime.  
- Nhận diện vật thể từ ảnh hoặc video.  
- Hỗ trợ **pose estimation** (YOLO pose).  
- Tiện ích xử lý bounding box (NMS, scale, vẽ khung).  

---

## 🧩 Ví dụ sử dụng

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load model
session = ort.InferenceSession("models/yolov11.onnx")

# Đọc ảnh
img = cv2.imread("test.jpg")
input_blob = preprocess(img)  # Hàm tiền xử lý

# Chạy inference
outputs = session.run(None, {session.get_inputs()[0].name: input_blob})

# Hậu xử lý và hiển thị
draw_boxes(img, outputs)
cv2.imshow("Result", img)
cv2.waitKey(0)
```

---

## 📖 Hướng phát triển

- Thêm hỗ trợ nhiều phiên bản YOLO khác (YOLOv5, YOLOv8).  
- Tích hợp benchmark tốc độ giữa CPU/GPU.  
- Viết notebook demo để dễ thử nghiệm.  

---

## 📜 License

MIT License – bạn có thể sử dụng, chỉnh sửa và phát triển repo này cho mục đích cá nhân hoặc thương mại.
## email: thien.aiot@gmail.com
