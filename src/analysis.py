from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataset(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Làm sạch dữ liệu: loại dòng trùng lặp và xử lý giá trị thiếu."""
    initial_rows, initial_cols = df.shape
    missing_before = int(df.isna().sum().sum())
    total_cells = max(initial_rows * initial_cols, 1)

    cleaned = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(cleaned)

    target_missing_dropped = 0
    if target_col and target_col in cleaned.columns:
        before_target_drop = len(cleaned)
        cleaned = cleaned.dropna(subset=[target_col]).copy()
        target_missing_dropped = before_target_drop - len(cleaned)

    missing_filled = 0
    for col in cleaned.columns:
        na_count = int(cleaned[col].isna().sum())
        if na_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            median_val = cleaned[col].median()
            fill_val = 0.0 if pd.isna(median_val) else median_val
            cleaned[col] = cleaned[col].fillna(fill_val)
        else:
            mode_series = cleaned[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Không xác định"
            cleaned[col] = cleaned[col].fillna(fill_val)
        missing_filled += na_count

    final_rows, final_cols = cleaned.shape
    summary = {
        "initial_rows": initial_rows,
        "initial_cols": initial_cols,
        "final_rows": final_rows,
        "final_cols": final_cols,
        "missing_before": missing_before,
        "missing_handled": missing_filled + target_missing_dropped,
        "missing_ratio": (missing_before / total_cells) * 100.0,
        "duplicates_removed": duplicates_removed,
        "duplicate_ratio": (duplicates_removed / max(initial_rows, 1)) * 100.0,
        "dtypes": [
            {"Biến": col, "Kiểu dữ liệu": str(cleaned[col].dtype)}
            for col in cleaned.columns
        ],
    }
    return cleaned, summary


def compute_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính bảng thống kê mô tả và ma trận tương quan cho các biến số."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    desc = numeric_df.describe().round(4)
    corr = numeric_df.corr().round(4) if numeric_df.shape[1] >= 2 else pd.DataFrame()
    return desc, corr


def detect_task_type(series: pd.Series) -> str:
    """Nhận diện bài toán hồi quy hay phân loại dựa trên biến mục tiêu."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return "classification"

    unique_count = series.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(series) and unique_count <= 10:
        return "classification"
    if unique_count <= 5:
        return "classification"
    return "regression"


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Huấn luyện và đánh giá mô hình hồi quy hoặc phân loại bằng scikit-learn."""
    valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
    if not valid_features:
        raise ValueError("Cần chọn ít nhất một biến độc lập khác biến mục tiêu.")
    if len(df) < 10:
        raise ValueError("Tập dữ liệu cần tối thiểu 10 dòng sau khi làm sạch để chia tập huấn luyện và kiểm thử.")

    X = df[valid_features].copy()
    y = df[target_col].copy()
    task_type = detect_task_type(y)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in valid_features if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    stratify = None
    if task_type == "classification":
        y = y.astype(str)
        class_counts = y.value_counts()
        if class_counts.min() >= 2 and len(class_counts) >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    metrics_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}

    if task_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=60, max_depth=16, n_jobs=-1, random_state=42
            ),
        }
        for name, estimator in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            r2 = float(r2_score(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics_rows.append({"Mô hình": name, "R2": round(r2, 4), "RMSE": round(rmse, 4)})
            predictions[name] = y_pred

        metrics_df = pd.DataFrame(metrics_rows).sort_values("R2", ascending=False).reset_index(drop=True)
        best_model = str(metrics_df.loc[0, "Mô hình"])
        primary_score = float(metrics_df.loc[0, "R2"])
        secondary_score = float(metrics_df.loc[0, "RMSE"])
    else:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=60, max_depth=16, n_jobs=-1, random_state=42
            ),
        }
        for name, estimator in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics_rows.append({"Mô hình": name, "Accuracy": round(acc, 4), "F1-score": round(f1, 4)})
            predictions[name] = y_pred

        metrics_df = pd.DataFrame(metrics_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        best_model = str(metrics_df.loc[0, "Mô hình"])
        primary_score = float(metrics_df.loc[0, "Accuracy"])
        secondary_score = float(metrics_df.loc[0, "F1-score"])

    return {
        "task_type": task_type,
        "metrics_df": metrics_df,
        "best_model": best_model,
        "primary_score": primary_score,
        "secondary_score": secondary_score,
        "y_test": y_test.to_numpy(),
        "y_pred": predictions[best_model],
        "labels": sorted(y.unique().tolist()) if task_type == "classification" else None,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def export_charts(
    df: pd.DataFrame,
    target_col: str,
    corr_df: pd.DataFrame,
    model_result: dict[str, Any],
    output_dir: Path,
) -> list[tuple[str, Path]]:
    """Xuất các biểu đồ PNG bằng matplotlib và seaborn."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    chart_paths: list[tuple[str, Path]] = []

    # 1. Biểu đồ phân phối biến mục tiêu
    dist_path = output_dir / "target_distribution.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    if model_result["task_type"] == "regression":
        sns.histplot(df[target_col], kde=True, color="#2563EB", ax=ax)
        ax.set_ylabel("Tần suất")
    else:
        counts = df[target_col].astype(str).value_counts()
        sns.barplot(x=counts.index, y=counts.values, color="#2563EB", ax=ax)
        ax.set_ylabel("Số lượng mẫu")
        if len(counts) > 6:
            ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Phân phối biến mục tiêu: {target_col}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(target_col)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(dist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append((f"Phân phối biến mục tiêu ({target_col})", dist_path))

    # 2. Heatmap ma trận tương quan
    if not corr_df.empty and corr_df.shape[0] >= 2:
        heatmap_path = output_dir / "correlation_heatmap.png"
        size = min(max(5.5, corr_df.shape[0] * 0.65), 10.0)
        fig, ax = plt.subplots(figsize=(size, size * 0.78), dpi=180)
        annot = corr_df.shape[0] <= 12
        sns.heatmap(
            corr_df,
            annot=annot,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.4,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Ma trận tương quan giữa các biến số", fontsize=12, fontweight="bold", pad=10)
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(("Ma trận tương quan giữa các biến số", heatmap_path))

    # 3. Biểu đồ kết quả mô hình
    model_chart_path = output_dir / "model_performance.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    best_name = model_result["best_model"]

    if model_result["task_type"] == "regression":
        y_test = model_result["y_test"]
        y_pred = model_result["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.7, color="#1D4ED8", edgecolors="white", linewidth=0.5)
        low = min(float(np.min(y_test)), float(np.min(y_pred)))
        high = max(float(np.max(y_test)), float(np.max(y_pred)))
        ax.plot([low, high], [low, high], linestyle="--", color="#DC2626", linewidth=1.2)
        ax.set_title(
            f"Giá trị thực tế và dự đoán ({best_name} | R2 = {model_result['primary_score']:.4f})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Giá trị thực tế")
        ax.set_ylabel("Giá trị dự đoán")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        labels = model_result["labels"]
        cm = confusion_matrix(model_result["y_test"], model_result["y_pred"], labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(
            f"Ma trận nhầm lẫn ({best_name} | Accuracy = {model_result['primary_score']:.4f})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")

    fig.tight_layout()
    fig.savefig(model_chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_paths.append((f"Kết quả mô hình tốt nhất ({best_name})", model_chart_path))

    return chart_paths


def generate_rule_based_insights(
    cleaning_summary: dict[str, Any],
    corr_df: pd.DataFrame,
    target_col: str,
    model_result: dict[str, Any],
) -> dict[str, list[str]]:
    """Sinh nhận xét và đề xuất hoàn toàn theo luật (if/else + f-string)."""
    observations: list[str] = []
    recommendations: list[str] = []

    # 1. Đánh giá chất lượng dữ liệu theo tỷ lệ thiếu và trùng lặp
    missing_ratio = cleaning_summary["missing_ratio"]
    dup_ratio = cleaning_summary["duplicate_ratio"]
    missing_handled = cleaning_summary["missing_handled"]
    duplicates_removed = cleaning_summary["duplicates_removed"]

    if missing_ratio == 0 and duplicates_removed == 0:
        observations.append(
            "Dữ liệu đầu vào sạch: không có giá trị thiếu và không có bản ghi trùng lặp."
        )
    else:
        if missing_ratio == 0:
            missing_level = "không có giá trị thiếu"
        elif missing_ratio < 5:
            missing_level = f"tỷ lệ giá trị thiếu ở mức thấp ({missing_ratio:.2f}%, đã xử lý {missing_handled} ô)"
        elif missing_ratio <= 20:
            missing_level = f"tỷ lệ giá trị thiếu ở mức trung bình ({missing_ratio:.2f}%, đã xử lý {missing_handled} ô)"
            recommendations.append(
                "Rà soát quy trình thu thập dữ liệu tại các cột thiếu nhiều và thử so sánh phương pháp điền khuyết KNNImputer."
            )
        else:
            missing_level = f"tỷ lệ giá trị thiếu cao ({missing_ratio:.2f}%, đã xử lý {missing_handled} ô)"
            recommendations.append(
                "Loại bỏ các biến có tỷ lệ thiếu vượt quá 30% hoặc bổ sung nguồn dữ liệu trước khi đưa mô hình vào sử dụng."
            )

        if duplicates_removed == 0:
            dup_level = "không phát hiện dòng trùng lặp"
        elif dup_ratio < 5:
            dup_level = f"đã loại bỏ {duplicates_removed} dòng trùng lặp ({dup_ratio:.2f}%)"
        else:
            dup_level = f"tỷ lệ trùng lặp đáng kể với {duplicates_removed} dòng ({dup_ratio:.2f}%)"
            recommendations.append(
                "Kiểm tra khóa định danh ở nguồn dữ liệu gốc để tránh ghi nhận lặp bản ghi."
            )

        observations.append(f"Chất lượng dữ liệu: {missing_level}; {dup_level}.")

    # 2. Biến có tương quan mạnh nhất với biến mục tiêu
    if not corr_df.empty and target_col in corr_df.columns and corr_df.shape[0] >= 2:
        target_corr = corr_df[target_col].drop(labels=[target_col]).dropna()
        if not target_corr.empty:
            top_var = str(target_corr.abs().idxmax())
            r_val = float(target_corr.loc[top_var])
            abs_r = abs(r_val)
            direction = "thuận" if r_val >= 0 else "nghịch"

            if abs_r >= 0.7:
                strength = "mạnh"
            elif abs_r >= 0.4:
                strength = "trung bình"
            else:
                strength = "yếu"

            observations.append(
                f"Biến có tương quan mạnh nhất với {target_col} là {top_var} "
                f"(r = {r_val:.4f}, tương quan {direction} mức {strength})."
            )
            if abs_r < 0.3:
                recommendations.append(
                    f"Tương quan tuyến tính với {target_col} đều thấp (|r| < 0.3); nên tạo thêm đặc trưng phi tuyến hoặc bổ sung biến giải thích."
                )
    elif not corr_df.empty and corr_df.shape[0] >= 2:
        upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
        stacked = upper.stack().dropna()
        if not stacked.empty:
            pair = stacked.abs().idxmax()
            r_val = float(stacked.loc[pair])
            observations.append(
                f"Biến mục tiêu {target_col} là biến phân loại; trong các biến số, cặp tương quan mạnh nhất là "
                f"{pair[0]} và {pair[1]} (r = {r_val:.4f})."
            )
            if abs(r_val) >= 0.85:
                recommendations.append(
                    f"Hai biến {pair[0]} và {pair[1]} có đa cộng tuyến cao (|r| >= 0.85); cân nhắc giữ lại một biến khi huấn luyện mô hình tuyến tính."
                )

    # 3. Đánh giá hiệu quả mô hình học máy theo ngưỡng R2 hoặc Accuracy
    best_model = model_result["best_model"]
    primary = model_result["primary_score"]
    secondary = model_result["secondary_score"]

    if model_result["task_type"] == "regression":
        if primary >= 0.85:
            tier = "cao"
            recommendations.append(
                f"Mô hình {best_model} đạt độ khớp cao; nên kiểm chứng thêm bằng k-fold cross-validation trước khi triển khai."
            )
        elif primary >= 0.65:
            tier = "khá"
            recommendations.append(
                f"Tinh chỉnh siêu tham số cho {best_model} hoặc thử thêm Gradient Boosting để cải thiện R2 và giảm RMSE."
            )
        elif primary >= 0.40:
            tier = "trung bình"
            recommendations.append(
                "Xem xét xử lý giá trị ngoại lai, biến đổi log cho biến lệch và mở rộng tập đặc trưng đầu vào."
            )
        else:
            tier = "thấp"
            recommendations.append(
                "Hiệu quả dự đoán hiện ở mức thấp; cần đánh giá lại biến giải thích, kiểm tra rò rỉ dữ liệu hoặc thu thập thêm mẫu."
            )

        observations.append(
            f"Bài toán hồi quy: mô hình {best_model} đạt kết quả tốt nhất với R2 = {primary:.4f} "
            f"và RMSE = {secondary:.4f} (mức hiệu quả {tier})."
        )
    else:
        if primary >= 0.90:
            tier = "cao"
            recommendations.append(
                f"Mô hình {best_model} phân loại chính xác cao; cần kiểm tra mất cân bằng lớp và đánh giá chéo trên tập mới."
            )
        elif primary >= 0.75:
            tier = "khá"
            recommendations.append(
                f"Điều chỉnh ngưỡng phân loại và siêu tham số cho {best_model} để nâng cao F1-score."
            )
        elif primary >= 0.60:
            tier = "trung bình"
            recommendations.append(
                "Cân nhắc cân bằng lại dữ liệu huấn luyện (class_weight='balanced') và bổ sung đặc trưng phân biệt giữa các lớp."
            )
        else:
            tier = "thấp"
            recommendations.append(
                "Độ chính xác phân loại còn thấp; cần kiểm tra lại nhãn dữ liệu, tỷ lệ lớp và thử các mô hình cây tăng cường."
            )

        observations.append(
            f"Bài toán phân loại: mô hình {best_model} đạt kết quả tốt nhất với Accuracy = {primary:.4f} "
            f"và F1-score = {secondary:.4f} (mức hiệu quả {tier})."
        )

    return {
        "observations": observations,
        "recommendations": recommendations,
    }

