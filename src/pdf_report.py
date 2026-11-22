from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from fpdf import FPDF
from fpdf.enums import XPos, YPos


FONT_CANDIDATES = [
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
]


def resolve_fonts() -> tuple[Path, Path]:
    """Nạp font Arial tiếng Việt trên Windows, có phương án dự phòng cho macOS và Linux."""
    for regular, bold in FONT_CANDIDATES:
        if regular.is_file() and bold.is_file():
            return regular, bold
    raise FileNotFoundError(
        "Không tìm thấy font TrueType hỗ trợ tiếng Việt "
        "(C:/Windows/Fonts/arial.ttf và C:/Windows/Fonts/arialbd.ttf)."
    )


class ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("ArialVN", "", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, f"Trang {self.page_no()}", align="R")


def _add_section_title(pdf: FPDF, text: str) -> None:
    if pdf.get_y() > 260:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.set_font("ArialVN", "B", 12)
    pdf.set_text_color(27, 54, 93)
    pdf.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _draw_table(pdf: FPDF, headers: list[str], rows: list[list[str]], col_widths: list[float]) -> None:
    row_height = 7.5
    table_width = sum(col_widths)
    start_x = (pdf.w - table_width) / 2

    def _draw_header() -> None:
        pdf.set_x(start_x)
        pdf.set_font("ArialVN", "B", 9)
        pdf.set_fill_color(27, 54, 93)
        pdf.set_text_color(255, 255, 255)
        pdf.set_draw_color(148, 163, 184)
        for idx, (w, h) in enumerate(zip(col_widths, headers)):
            is_last = idx == len(headers) - 1
            pdf.cell(
                w,
                row_height,
                str(h)[:22],
                border=1,
                align="C",
                fill=True,
                new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                new_y=YPos.NEXT if is_last else YPos.TOP,
            )

    if pdf.get_y() + row_height * 3 > pdf.h - pdf.b_margin:
        pdf.add_page()
    _draw_header()

    pdf.set_font("ArialVN", "", 8.8)
    pdf.set_text_color(31, 41, 55)
    for row_idx, row in enumerate(rows):
        if pdf.get_y() + row_height > pdf.h - pdf.b_margin:
            pdf.add_page()
            _draw_header()
            pdf.set_font("ArialVN", "", 8.8)
            pdf.set_text_color(31, 41, 55)

        pdf.set_x(start_x)
        if row_idx % 2 == 1:
            pdf.set_fill_color(244, 247, 250)
        else:
            pdf.set_fill_color(255, 255, 255)

        for idx, (w, val) in enumerate(zip(col_widths, row)):
            is_last = idx == len(row) - 1
            pdf.cell(
                w,
                row_height,
                str(val)[:26],
                border=1,
                align="C",
                fill=True,
                new_x=XPos.LMARGIN if is_last else XPos.RIGHT,
                new_y=YPos.NEXT if is_last else YPos.TOP,
            )
    pdf.ln(3)


def generate_pdf_report(
    output_path: Path,
    csv_name: str,
    target_col: str,
    cleaning_summary: dict[str, Any],
    desc_df: pd.DataFrame,
    model_result: dict[str, Any],
    chart_paths: list[tuple[str, Path]],
    insights: dict[str, list[str]],
    project_title: str = "",
    project_description: str = "",
) -> Path:
    """Tạo tệp báo cáo khoa học PDF gồm 5 phần từ kết quả phân tích."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_regular, font_bold = resolve_fonts()

    title = project_title.strip() or f"Phân tích dữ liệu {csv_name} theo biến mục tiêu {target_col}"
    description = project_description.strip() or (
        f"Báo cáo trình bày quy trình làm sạch, thống kê mô tả, trực quan hóa và xây dựng mô hình "
        f"dự đoán biến mục tiêu {target_col} từ tập dữ liệu {csv_name}."
    )

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font("ArialVN", "", str(font_regular))
    pdf.add_font("ArialVN", "B", str(font_bold))
    pdf.add_page()

    # Tiêu đề báo cáo
    pdf.set_font("ArialVN", "B", 15.5)
    pdf.set_text_color(27, 54, 93)
    pdf.multi_cell(0, 8, "BÁO CÁO PHÂN TÍCH DỮ LIỆU KHOA HỌC", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("ArialVN", "", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 6.5, title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    y_rule = pdf.get_y()
    pdf.set_draw_color(203, 213, 225)
    pdf.line(pdf.l_margin, y_rule, pdf.w - pdf.r_margin, y_rule)
    pdf.ln(5)

    # 1. Giới thiệu đề tài
    _add_section_title(pdf, "1. Giới thiệu đề tài")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("ArialVN", "", 10)
    pdf.set_text_color(31, 41, 55)
    pdf.multi_cell(0, 5.8, f"Tên đề tài: {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5.8, f"Mục tiêu nghiên cứu: {description}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # 2. Mô tả dữ liệu
    _add_section_title(pdf, "2. Mô tả dữ liệu")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("ArialVN", "", 10)
    pdf.set_text_color(31, 41, 55)
    clean_text = (
        f"- Nguồn dữ liệu: {csv_name} | Biến mục tiêu: {target_col}\n"
        f"- Kích thước ban đầu: {cleaning_summary['initial_rows']} dòng, {cleaning_summary['initial_cols']} cột\n"
        f"- Kích thước sau làm sạch: {cleaning_summary['final_rows']} dòng, {cleaning_summary['final_cols']} cột\n"
        f"- Giá trị thiếu đã xử lý: {cleaning_summary['missing_handled']} ô ({cleaning_summary['missing_ratio']:.2f}%)\n"
        f"- Dòng trùng lặp đã loại bỏ: {cleaning_summary['duplicates_removed']} dòng ({cleaning_summary['duplicate_ratio']:.2f}%)"
    )
    pdf.multi_cell(0, 5.8, clean_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    dtype_rows = [[item["Biến"], item["Kiểu dữ liệu"]] for item in cleaning_summary["dtypes"]]
    _draw_table(pdf, ["Tên biến", "Kiểu dữ liệu"], dtype_rows, [95, 75])

    # 3. Kết quả phân tích
    _add_section_title(pdf, "3. Kết quả phân tích")
    if not desc_df.empty:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("ArialVN", "B", 10)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 6.5, "3.1. Bảng thống kê mô tả các biến số", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        stat_cols = list(desc_df.columns)
        chunk_size = 5
        for start in range(0, len(stat_cols), chunk_size):
            subset = stat_cols[start : start + chunk_size]
            headers = ["Chỉ số"] + [str(c) for c in subset]
            rows = []
            for idx_name, row_vals in desc_df[subset].iterrows():
                rows.append([str(idx_name)] + [f"{v:.2f}" if pd.notna(v) else "-" for v in row_vals])
            col_w = 150.0 / len(subset)
            _draw_table(pdf, headers, rows, [26.0] + [col_w] * len(subset))

    pdf.set_x(pdf.l_margin)
    pdf.set_font("ArialVN", "B", 10)
    pdf.set_text_color(31, 41, 55)
    task_label = "Hồi quy" if model_result["task_type"] == "regression" else "Phân loại"
    pdf.cell(
        0,
        6.5,
        f"3.2. Kết quả mô hình học máy ({task_label} | Train: {model_result['train_size']}, Test: {model_result['test_size']})",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    metrics_df = model_result["metrics_df"]
    m_headers = list(metrics_df.columns)
    m_rows = [[str(v) for v in row] for row in metrics_df.to_numpy().tolist()]
    m_widths = [70.0] + [50.0] * (len(m_headers) - 1)
    _draw_table(pdf, m_headers, m_rows, m_widths)

    # 4. Biểu đồ minh họa
    _add_section_title(pdf, "4. Biểu đồ minh họa")
    usable_w = 152.0
    for caption, img_path in chart_paths:
        if not img_path.is_file():
            continue
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        target_w = usable_w
        target_h = target_w * (img_h / img_w)
        if target_h > 105:
            target_h = 105.0
            target_w = target_h * (img_w / img_h)

        if pdf.get_y() + target_h + 14 > pdf.h - pdf.b_margin:
            pdf.add_page()

        img_x = (pdf.w - target_w) / 2
        y_top = pdf.get_y()
        pdf.image(str(img_path), x=img_x, y=y_top, w=target_w, h=target_h)
        pdf.set_y(y_top + target_h + 2)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("ArialVN", "", 9.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 5.5, caption, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # 5. Kết luận & đề xuất
    _add_section_title(pdf, "5. Kết luận & đề xuất")
    pdf.set_font("ArialVN", "", 10)
    pdf.set_text_color(31, 41, 55)

    lines: list[str] = ["Nhận xét chính:"]
    for obs in insights.get("observations", []):
        lines.append(f"- {obs}")
    if insights.get("recommendations"):
        lines.append("")
        lines.append("Đề xuất:")
        for rec in insights["recommendations"]:
            lines.append(f"- {rec}")

    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(240, 244, 248)
    pdf.set_draw_color(203, 213, 225)
    pdf.multi_cell(
        0,
        5.8,
        "\n".join(lines),
        border=1,
        fill=True,
        padding=3,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf.output(str(output_path))
    return output_path

