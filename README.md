# Qwen3 Mode A Pipeline

Triển khai kiến trúc Pod ↔ Client theo pipeline Mode A:

## Pod (RunPod) – FastAPI + Qwen3-VL

1. Copy `Dockerfile` và `pod_server.py` vào workspace RunPod.
2. Tùy chọn đặt biến môi trường `MODEL_ID` trỏ đến checkpoint cục bộ (VD: `/workspace/models/qwen3-vl-30b-a3b`).
3. Build & chạy:
   ```bash
   docker build -t qwen3-vl-pod .
   docker run --gpus all -p 8000:8000 --env MODEL_ID=Qwen/Qwen3-VL-30B-A3B-Instruct qwen3-vl-pod
   ```
   Pod mở cổng `8000` cung cấp các endpoint:
   - `POST /chat`: hội thoại text thuần, đa lượt.
   - `POST /vision-chat`: hội thoại đa phương thức (ảnh/PDF + text).
   - `GET /health`: kiểm tra trạng thái.

## Client (Streamlit) – server khách

1. Cài phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
   > Ubuntu cần thêm `apt-get install -y poppler-utils` nếu muốn rasterize PDF phía client (tùy chọn).
2. Đặt biến môi trường `POD_URL="http://<POD_PUBLIC_IP>:8000"`.
3. Khởi chạy giao diện:
   ```bash
   streamlit run client_app.py --server.address 0.0.0.0 --server.port 8501
   ```
4. Người dùng mở trình duyệt tới `http://<client_server_ip>:8501`, nhập câu hỏi, tùy chọn upload ảnh/PDF:
   - Không file đính kèm → UI gọi `POST /chat`.
   - Có file đính kèm → UI gọi `POST /vision-chat`.

## Luồng hoạt động

1. Streamlit lưu `session_id` & lịch sử chat phía client để hiển thị.
2. Pod duy trì `session_id` tương ứng để nhớ hội thoại và gọi Qwen3-VL.
3. Kết quả được trả về và cập nhật trong chatbox trên giao diện client.
