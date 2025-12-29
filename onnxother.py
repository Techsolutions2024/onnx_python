import cv2
import numpy as np
import onnxruntime as ort

# --- CẤU HÌNH CHUẨN THEO NETRON ---
MODEL_PATH = "models\LPDNet_usa_pruned_tao5.onnx"
IN_W, IN_H = 720, 1168  # Rộng x Cao (Netron ghi 3, 1168, 720)
STRIDE = 16
GRID_H, GRID_W = 73, 45
CONF_THRESH = 0.5
NMS_THRESH = 0.4

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

def preprocess(frame):
    # 1. Resize đúng kích thước model yêu cầu
    img = cv2.resize(frame, (IN_W, IN_H))
    # 2. Chuyển BGR (OpenCV) sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. Normalize 0-1 (TF2ONNX thường dùng chuẩn này)
    img = img.astype(np.float32) / 255.0
    # 4. HWC -> CHW và thêm Batch Dimension
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, raw_w, raw_h):
    conf_map = outputs[0][0][0] # (73, 45)
    bbox_map = outputs[1][0]    # (4, 73, 45)
    
    rows, cols = np.where(conf_map > CONF_THRESH)
    
    boxes = []
    scores = []
    
    for r, c in zip(rows, cols):
        score = conf_map[r, c]
        
        # Lấy các giá trị offset từ model
        # Thông thường với TF License Plate: l, t, r, b
        l, t, r_dist, b = bbox_map[:, r, c]
        
        # Tính tâm của ô lưới (Grid Cell Center)
        center_x = c * STRIDE + (STRIDE // 2)
        center_y = r * STRIDE + (STRIDE // 2)
        
        # Tọa độ trên không gian 720x1168
        x1 = center_x - l
        y1 = center_y - t
        x2 = center_x + r_dist
        y2 = center_y + b
        
        # Scale lại về kích thước thật của Webcam
        x = int(x1 * raw_w / IN_W)
        y = int(y1 * raw_h / IN_H)
        w = int((x2 - x1) * raw_w / IN_W)
        h = int((y2 - y1) * raw_h / IN_H)
        
        boxes.append([x, y, w, h])
        scores.append(float(score))
        
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    return [(boxes[i], scores[i]) for i in (indices.flatten() if len(indices) > 0 else [])]

# --- CHẠY WEBCAM ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h_raw, w_raw = frame.shape[:2]
    
    blob = preprocess(frame)
    outputs = session.run(None, {input_name: blob})
    
    # Debug: In max confidence ra màn hình console để kiểm tra model có "thấy" gì không
    print(f"Max confidence: {np.max(outputs[0]):.4f}", end="\r")
    
    results = postprocess(outputs, w_raw, h_raw)
    
    for (box, score) in results:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"LP: {score:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
    cv2.imshow("Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()