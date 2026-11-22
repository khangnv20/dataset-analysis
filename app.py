from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import (
    clean_dataset,
    compute_statistics,
    export_charts,
    generate_rule_based_insights,
    train_and_evaluate,
)
from src.pdf_report import generate_pdf_report


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
PDF_OUTPUT_PATH = OUTPUT_DIR / "bao_cao_khoa_hoc.pdf"


@st.cache_data(show_spinner=False)
def load_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_clean_and_stats(df: pd.DataFrame, target_col: str):
    cleaned, summary = clean_dataset(df, target_col=target_col)
    desc, corr = compute_statistics(cleaned)
    return cleaned, summary, desc, corr


@st.cache_data(show_spinner=False)
def cached_train_and_insights(
    cleaned_df: pd.DataFrame,
    target_col: str,
    feature_cols: tuple[str, ...],
    cleaning_summary: dict,
    corr_df: pd.DataFrame,
):
    model_res = train_and_evaluate(cleaned_df, target_col=target_col, feature_cols=list(feature_cols))
    rule_insights = generate_rule_based_insights(cleaning_summary, corr_df, target_col, model_res)
    return model_res, rule_insights


st.set_page_config(page_title="Phân tích dữ liệu & Báo cáo PDF", layout="wide")
st.title("Phân tích dữ liệu và xuất báo cáo khoa học PDF")

with st.sidebar:
    st.header("Nguồn dữ liệu")
    uploaded_file = st.file_uploader("Tải tệp CSV", type=["csv"])

    st.header("Thông tin báo cáo")
    project_title = st.text_input("Tên đề tài (tùy chọn)", value="")
    project_desc = st.text_area("Mục tiêu nghiên cứu (tùy chọn)", value="", height=90)

if uploaded_file is not None:
    raw_df = load_csv_from_bytes(uploaded_file.getvalue())
    csv_name = uploaded_file.name
elif DEFAULT_DATA_PATH.is_file():
    raw_df = load_csv_from_path(str(DEFAULT_DATA_PATH))
    csv_name = DEFAULT_DATA_PATH.name
else:
    st.warning("Chưa có tệp data/dataset.csv. Hãy tải tệp CSV từ thanh bên trái hoặc đặt sẵn tệp vào thư mục data/dataset.csv.")
    st.stop()

if raw_df.empty or raw_df.shape[1] < 2:
    st.error("Tập dữ liệu cần có ít nhất 2 cột và 1 dòng dữ liệu.")
    st.stop()

all_columns = raw_df.columns.tolist()

with st.sidebar:
    st.header("Thiết lập phân tích")
    target_col = st.selectbox("Biến mục tiêu", options=all_columns, index=len(all_columns) - 1)
    default_features = [c for c in all_columns if c != target_col and c.lower() != "id"]
    feature_cols = st.multiselect(
        "Biến độc lập",
        options=[c for c in all_columns if c != target_col],
        default=default_features or [c for c in all_columns if c != target_col],
    )

    st.header("Biểu đồ tương tác")
    chart_type = st.selectbox(
        "Loại biểu đồ",
        options=["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Heatmap tương quan"],
    )

cleaned_df, cleaning_summary, desc_df, corr_df = cached_clean_and_stats(raw_df, target_col)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Số dòng ban đầu", cleaning_summary["initial_rows"])
col2.metric("Số dòng sau làm sạch", cleaning_summary["final_rows"])
col3.metric("Giá trị thiếu đã xử lý", cleaning_summary["missing_handled"])
col4.metric("Dòng trùng đã loại bỏ", cleaning_summary["duplicates_removed"])

tab_data, tab_viz, tab_model, tab_report = st.tabs(
    ["Dữ liệu & Thống kê", "Biểu đồ tương tác", "Mô hình học máy", "Nhận xét & Báo cáo PDF"]
)

with tab_data:
    st.subheader("Dữ liệu sau làm sạch")
    st.dataframe(cleaned_df.head(50), width="stretch")

    st.subheader("Danh sách biến và kiểu dữ liệu")
    st.dataframe(pd.DataFrame(cleaning_summary["dtypes"]), width="stretch", hide_index=True)

    st.subheader("Bảng thống kê mô tả")
    if not desc_df.empty:
        st.dataframe(desc_df, width="stretch")
    else:
        st.info("Tập dữ liệu không có biến số để tính thống kê mô tả.")

with tab_viz:
    st.subheader(f"Biểu đồ {chart_type}")
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

    if chart_type == "Heatmap tương quan":
        if not corr_df.empty:
            fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Cần ít nhất 2 biến số để vẽ heatmap tương quan.")
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Trục X", options=numeric_cols, index=0)
            y_default = numeric_cols.index(target_col) if target_col in numeric_cols else min(1, len(numeric_cols) - 1)
            y_col = st.selectbox("Trục Y", options=numeric_cols, index=y_default)
            plot_df = cleaned_df.sample(min(len(cleaned_df), 5000), random_state=42)
            fig = px.scatter(plot_df, x=x_col, y=y_col, title=f"{x_col} và {y_col}")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Cần ít nhất 2 biến số để vẽ Scatter Plot.")
    elif chart_type == "Histogram":
        col_choice = st.selectbox("Biến hiển thị", options=all_columns, index=all_columns.index(target_col))
        fig = px.histogram(cleaned_df, x=col_choice, title=f"Phân phối {col_choice}")
        st.plotly_chart(fig, width="stretch")
    elif chart_type == "Box Plot":
        if numeric_cols:
            y_col = st.selectbox("Biến số", options=numeric_cols, index=0)
            fig = px.box(cleaned_df, y=y_col, title=f"Phân bố {y_col}")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Không có biến số để vẽ Box Plot.")
    elif chart_type == "Bar Chart":
        col_choice = st.selectbox("Biến phân nhóm", options=all_columns, index=all_columns.index(target_col))
        counts = cleaned_df[col_choice].astype(str).value_counts().reset_index()
        counts.columns = [col_choice, "Số lượng"]
        fig = px.bar(counts.head(20), x=col_choice, y="Số lượng", title=f"Tần suất theo {col_choice}")
        st.plotly_chart(fig, width="stretch")

model_result = None
insights = None
model_error = None

if feature_cols:
    try:
        with st.spinner("Đang huấn luyện và đánh giá mô hình..."):
            model_result, insights = cached_train_and_insights(
                cleaned_df,
                target_col,
                tuple(feature_cols),
                cleaning_summary,
                corr_df,
            )
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = "Hãy chọn ít nhất một biến độc lập ở thanh bên trái."

with tab_model:
    if model_error:
        st.warning(model_error)
    elif model_result is not None:
        task_name = "Hồi quy" if model_result["task_type"] == "regression" else "Phân loại"
        st.write(f"Loại bài toán nhận diện: **{task_name}** | Mô hình tốt nhất: **{model_result['best_model']}**")
        st.dataframe(model_result["metrics_df"], width="stretch", hide_index=True)

with tab_report:
    if model_error or model_result is None or insights is None:
        st.warning(model_error or "Chưa đủ thông tin để sinh nhận xét và báo cáo.")
    else:
        st.subheader("Nhận xét tự động")
        for item in insights["observations"]:
            st.write(f"- {item}")

        st.subheader("Đề xuất")
        for item in insights["recommendations"]:
            st.write(f"- {item}")

        st.divider()
        if st.button("Tạo báo cáo PDF", type="primary"):
            with st.spinner("Đang xuất biểu đồ và tạo báo cáo PDF..."):
                charts = export_charts(cleaned_df, target_col, corr_df, model_result, CHART_DIR)
                pdf_file = generate_pdf_report(
                    output_path=PDF_OUTPUT_PATH,
                    csv_name=csv_name,
                    target_col=target_col,
                    cleaning_summary=cleaning_summary,
                    desc_df=desc_df,
                    model_result=model_result,
                    chart_paths=charts,
                    insights=insights,
                    project_title=project_title,
                    project_description=project_desc,
                )
                st.session_state["pdf_bytes"] = pdf_file.read_bytes()
            st.success(f"Đã tạo báo cáo tại {pdf_file}")

        if "pdf_bytes" in st.session_state:
            st.download_button(
                label="Tải về bao_cao_khoa_hoc.pdf",
                data=st.session_state["pdf_bytes"],
                file_name="bao_cao_khoa_hoc.pdf",
                mime="application/pdf",
            )
