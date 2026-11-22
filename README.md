# Phân tích dữ liệu CSV và xuất báo cáo khoa học PDF

Ứng dụng Python đọc dữ liệu từ `data/dataset.csv` (hoặc tệp CSV tải lên trực tiếp), làm sạch dữ liệu, tính thống kê mô tả, huấn luyện mô hình hồi quy/phân loại bằng `scikit-learn`, sinh nhận xét theo luật và xuất tệp `output/bao_cao_khoa_hoc.pdf` qua giao diện Streamlit.

## Cấu trúc dự án

```text
.
├── app.py
├── data/
│   └── dataset.csv
├── output/
│   └── bao_cao_khoa_hoc.pdf
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   └── pdf_report.py
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

Đặt tệp dữ liệu tại `data/dataset.csv` (hoặc tải tệp CSV trực tiếp trên giao diện), sau đó chạy:

```bash
streamlit run app.py
```

## Giấy phép

Phát hành theo giấy phép [MIT License](https://opensource.org/licenses/MIT).

