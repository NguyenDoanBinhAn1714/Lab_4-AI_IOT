# LAB 4 - FORECASTING & PREDICTIVE ANALYTICS CHO DỮ LIỆU IoT

**Thông tin thực hiện:**
- Sinh viên: Nguyễn Doãn Bình An
- Khóa: Sinh viên năm 3 - Đại học Đại Nam
- Lĩnh vực: IT, AI, IoT, Cybersecurity & B.A.

---

## 1. Tổng Quan Dự Án

Lab 4 tập trung vào bài toán **Dự báo và Phân tích Dự đoán (Forecasting & Predictive Analytics)** cho hệ thống AIoT. Khác với Lab 3 (chỉ phát hiện sự cố hiện tại), Lab 4 trả lời câu hỏi: *Nếu xu hướng tiếp tục, giá trị sắp tới sẽ là bao nhiêu và hệ thống cần đưa ra khuyến nghị vận hành gì?*

Mô hình dự báo không được sử dụng trực tiếp để tự động điều khiển thiết bị (nhằm đảm bảo Safety Rule). Thay vào đó, kết quả dự báo được đưa qua một Decision Layer để tính toán **Mức độ rủi ro (Risk Level)** và **Khuyến nghị (Recommendation)**.

- **Dataset sử dụng:** [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction). Tập dữ liệu tiêu thụ năng lượng của thiết bị gia dụng (Appliances) đo mỗi 10 phút.
- **Forecast Horizon (Tầm nhìn dự báo):** 1 bước thời gian (10 phút tiếp theo).

---

## 2. Kiến Trúc Luồng Xử Lý (AIoT Forecasting Pipeline)

```text
Dữ liệu Telemetry thô (Past Data)
  ──> Tiền xử lý & Trích xuất đặc trưng chuỗi thời gian (Lag, Rolling Mean, Delta)
  ──> Tạo Target Tương lai (Shift -1)
  ──> Chia tập Train/Test theo trình tự thời gian (Time-based Split)
  ──> Huấn luyện Mô hình Machine Learning (Linear Regression / Random Forest)
  ──> Đánh giá bằng Metrics (MAE, RMSE, Bias)
  ──> Động cơ Quyết định (Decision Layer: Tính Risk Level & Recommendation)
  ──> Xuất Nhật ký Dự báo (forecast_log.csv)
  ──> Đóng gói Mô hình (.joblib)
  ──> Triển khai Inference Service (FastAPI: /forecast)
lab4_aiot_forecasting/
├── data/                               # Chứa dữ liệu tải về (nếu tải thủ công)
├── src/
│   ├── utils.py                        # Hàm hỗ trợ Feature Engineering và Tính Risk
│   ├── train_forecast.py               # Script huấn luyện, đánh giá mô hình và lưu output
│   ├── app.py                          # Ứng dụng FastAPI triển khai endpoint /forecast
│   └── test_api.py                     # Script kiểm thử gửi request payload tới API
├── models/
│   └── forecast_model_bundle_v1.joblib # Mô hình Machine Learning đã được đóng gói
├── outputs/
│   ├── forecast_metrics.json           # File lưu đánh giá độ đo (MAE, RMSE, Bias)
│   ├── forecast_test_predictions.csv   # Kết quả dự báo trên tập Test
│   ├── forecast_log.csv                # Nhật ký rủi ro (Risk & Recommendation)
│   └── api_test_result.json            # Kết quả JSON response trả về từ API
├── figures/                            # Biểu đồ trực quan hóa kết quả thực nghiệm
└── README.md
4. Hướng Dẫn Cài Đặt Môi Trường
Yêu cầu hệ thống cài đặt sẵn Python >= 3.8.

Trên Windows:

PowerShell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn requests pydantic joblib
Trên macOS/Linux:

Bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn requests pydantic joblib
5. Quy Trình Chạy Thực Nghiệm (Pipeline)
Bước 1: Huấn Luyện & Đánh Giá Mô Hình
Mở terminal (đã kích hoạt .venv), chạy lệnh sau để làm sạch dữ liệu, huấn luyện mô hình Linear Regression / Random Forest, sinh đặc trưng chuỗi thời gian, và tự động tạo ra các file logs ở thư mục outputs/:

Bash
python src/train_forecast.py
Bước 2: Triển Khai API Service (FastAPI)
Khởi chạy máy chủ API ở một cửa sổ terminal:

Bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
Giao diện tài liệu Swagger UI: http://127.0.0.1:8000/docs

Endpoint kiểm tra sức khỏe hệ thống: GET /health

Bước 3: Kiểm Thử (Test API)
Mở một cửa sổ terminal mới, chạy script kiểm thử để gửi dữ liệu môi trường (sensor data) lên Server:

Bash
python src/test_api.py
Nếu cấu hình đúng, API sẽ trả về dữ liệu định dạng JSON gồm hai khối chính:

"model_output": Kết quả dự báo bằng số (predicted_value).

"decision": Lớp ra quyết định chứa "risk_level", "recommendation" và "safety_note".

6. Phân Tích Độ Đo (Metrics Interpretation)
Kết quả huấn luyện được lưu tại outputs/forecast_metrics.json. Các tiêu chí đánh giá chính:

MAE (Mean Absolute Error): Trung bình trị tuyệt đối của sai số. Thể hiện trung bình mô hình dự báo lệch thực tế bao nhiêu Wh.

RMSE (Root Mean Squared Error): Phạt nặng các lỗi dự báo có sai số lớn. Nếu RMSE lớn hơn MAE đáng kể, chứng tỏ hệ thống đôi khi xuất hiện các đột biến (spike) tiêu thụ năng lượng mà mô hình chưa dự đoán sát được.

Bias: Thể hiện độ chệch. Nếu dương, mô hình đang có xu hướng dự đoán cao hơn thực tế; nếu âm, có xu hướng dự đoán thấp hơn thực tế.
