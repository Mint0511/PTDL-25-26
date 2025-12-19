================================================================================
DỰ ÁN: PHÂN TÍCH DỮ LIỆU DOANH SỐ WALMART (2010-2012)
================================================================================

Môn học: Phân tích dữ liệu (PTDL) - Giảng viên: TS. Đỗ Như Tài
Nhóm: 20
Sinh viên thực hiện: 
- Nguyễn Văn Minh (3122410242) - Trưởng nhóm
- Vũ Thị Thanh Ngân (3122410255)
- Nguyễn Trương Hiệp (3122410110)
- Trương Xuân Hưng (3122410161)

--------------------------------------------------------------------------------
1. TỔNG QUAN DỰ ÁN
--------------------------------------------------------------------------------
Dự án tập trung phân tích 8 câu hỏi về doanh số của 45 cửa hàng Walmart nhằm:
- Khám phá các yếu tố tác động: Ngày lễ, mùa vụ, quy mô, và chỉ số kinh tế.
- Ứng dụng Machine Learning: 
  + Phân cụm (K-Means): Chia 45 cửa hàng thành 3 nhóm chiến lược.
  + Dự đoán (Decision Tree): Phân loại tuần cao điểm với độ chính xác ~91%.

--------------------------------------------------------------------------------
2. CẤU TRÚC THƯ MỤC CODE
--------------------------------------------------------------------------------
📁 Code/
├── 📁 data/               : Chứa các file dữ liệu gốc (train, features, stores).
├── 📓 PhanTichWalmart_Nhom 20.ipynb : File phân tích chính (Jupyter Notebook).
├── 🐍 Nhom20_Walmart_App.py : Ứng dụng Dashboard tương tác (Streamlit).
└── 📄 requirements.txt    : Danh sách các thư viện cần cài đặt.

--------------------------------------------------------------------------------
3. KẾT QUẢ PHÂN TÍCH CHÍNH (INSIGHTS)
--------------------------------------------------------------------------------
- Doanh số trung bình đạt $1.047M/tuần; biến động mạnh vào Quý 4 (lễ hội).
- Ngày lễ (Black Friday, Christmas) thúc đẩy doanh số tăng từ 6-15%.
- Quy mô cửa hàng (Size) có tương quan thuận rất mạnh (r=0.81) với doanh số.
- Phân cụm thành công 3 nhóm cửa hàng: Quy mô Nhỏ - Vừa - Lớn.
- Mô hình Decision Tree đạt độ chính xác cao trong việc dự báo tuần doanh số cao.

--------------------------------------------------------------------------------
4. HƯỚNG DẪN CHẠY TRÊN VSCODE
--------------------------------------------------------------------------------
Yêu cầu: Đã cài đặt Python 3.8+.

Bước 1: Mở thư mục "Code" trong VSCode.
Bước 2: Cài đặt thư viện cần thiết.
        Mở Terminal (Ctrl+`) và chạy lệnh:
        pip install -r requirements.txt

Bước 3: Xem phân tích chi tiết.
        Mở file "PhanTichWalmart_Nhom 20.ipynb", chọn Kernel và nhấn "Run All".

Bước 4: Khởi chạy Dashboard tương tác.
        Trong Terminal, chạy lệnh:
        streamlit run Nhom20_Walmart_App.py
        (Sau đó truy cập địa chỉ http://localhost:8501 hiện ra trên màn hình).

--------------------------------------------------------------------------------
5. LƯU Ý VỀ DỮ LIỆU
--------------------------------------------------------------------------------
- Đã xử lý 1,285 dòng doanh số âm (do trả hàng) để tránh nhiễu mô hình.
- Các giá trị trống (NaN) ở cột khuyến mãi (MarkDown) được thay bằng 0.
- Các điểm dữ liệu đột biến (Outliers) được giữ lại vì mang giá trị phân tích 
  đặc thù của các dịp lễ lớn.

================================================================================
Ngày hoàn thành: 19/12/2025
Trường ĐH Sài Gòn (SGU)
================================================================================