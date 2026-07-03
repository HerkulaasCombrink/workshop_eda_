import io
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Brainy Data Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel data from a Streamlit uploaded file."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV, XLSX, or XLS file.")


def infer_column_groups(df: pd.DataFrame) -> Tuple[list, list, list]:
    """Infer numeric, categorical, and datetime-like columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols + datetime_cols]
    return numeric_cols, categorical_cols, datetime_cols


def infer_problem_type(y: pd.Series) -> str:
    """Infer whether the selected target looks like classification or regression."""
    y_non_missing = y.dropna()
    unique_count = y_non_missing.nunique()
    if unique_count <= 1:
        return "unsuitable"
    if pd.api.types.is_numeric_dtype(y_non_missing):
        unique_ratio = unique_count / max(len(y_non_missing), 1)
        if unique_count <= 10 or unique_ratio <= 0.05:
            return "classification"
        return "regression"
    return "classification"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a universal preprocessing pipeline for numeric and categorical predictors."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def safe_train_test_split(X: pd.DataFrame, y: pd.Series, problem_type: str, test_size: float):
    """Split data safely, using stratification only when valid."""
    stratify = None
    if problem_type == "classification":
        class_counts = y.value_counts(dropna=False)
        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify = y

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )


def show_missing_summary(df: pd.DataFrame) -> None:
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean().values * 100).round(2),
        }
    ).sort_values("missing_percent", ascending=False)
    st.dataframe(missing, use_container_width=True)


def show_categorical_summaries(df: pd.DataFrame, categorical_cols: list, max_categories: int) -> None:
    if not categorical_cols:
        st.info("No categorical columns detected.")
        return

    selected_col = st.selectbox("Select a categorical column", categorical_cols)
    freq = (
        df[selected_col]
        .astype("object")
        .fillna("Missing")
        .value_counts()
        .head(max_categories)
        .reset_index()
    )
    freq.columns = [selected_col, "count"]
    st.dataframe(freq, use_container_width=True)
    fig = px.bar(freq, x=selected_col, y="count", title=f"Top values in {selected_col}")
    st.plotly_chart(fig, use_container_width=True)


def show_numeric_visuals(df: pd.DataFrame, numeric_cols: list) -> None:
    if not numeric_cols:
        st.info("No numeric columns detected for visualisation.")
        return

    selected_num = st.selectbox("Select a numeric column", numeric_cols)
    fig_hist = px.histogram(df, x=selected_num, marginal="box", title=f"Distribution of {selected_num}")
    st.plotly_chart(fig_hist, use_container_width=True)

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Scatterplot X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Scatterplot Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)


def show_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> None:
    if len(numeric_cols) < 2:
        st.info("At least two numeric columns are required for a correlation heatmap.")
        return

    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)


def run_baseline_model(df: pd.DataFrame, target_col: str, problem_type: str, test_size: float, n_neighbors: int) -> None:
    model_df = df.dropna(subset=[target_col]).copy()
    if len(model_df) < 10:
        st.warning("Not enough complete target values for a meaningful train/test split.")
        return

    X = model_df.drop(columns=[target_col])
    y = model_df[target_col]

    usable_cols = [col for col in X.columns if X[col].nunique(dropna=True) > 1]
    X = X[usable_cols]

    if X.empty:
        st.warning("No usable predictor columns were found after removing constant columns.")
        return

    if problem_type == "classification" and y.nunique() < 2:
        st.warning("The selected target has fewer than two classes.")
        return

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, problem_type, test_size)
    preprocessor = build_preprocessor(X)

    if problem_type == "classification":
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except Exception as exc:
        st.error(f"The baseline model could not be trained: {exc}")
        return

    st.subheader("🧪 Baseline KNN Model")
    st.caption("This is a quick baseline only. It is not a final validated model.")

    if problem_type == "classification":
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
        col2.metric("Weighted F1", f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.3f}")

        st.write("Classification report")
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
        cm = confusion_matrix(pd.Series(y_test).astype(str), pd.Series(predictions).astype(str), labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mean_absolute_error(y_test, predictions):.3f}")
        col2.metric("RMSE", f"{rmse:.3f}")
        col3.metric("R²", f"{r2_score(y_test, predictions):.3f}")

        results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
        fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("🧠 Brainy Data Explorer")
    st.markdown(
        "Upload a CSV or Excel dataset and let the app inspect, summarise, visualise, "
        "and suggest a modelling pathway for your selected target variable. ⚡"
    )

    with st.sidebar:
        st.header("🧠 Control Centre")
        uploaded_file = st.file_uploader(
            "Upload your data",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, XLSX, XLS",
        )
        preview_rows = st.slider("Preview rows", 5, 100, 10)
        max_categories = st.slider("Maximum categories to display", 5, 50, 15)
        st.divider()
        st.caption("Built for flexible exploratory data analysis and quick baseline modelling.")

    if uploaded_file is None:
        st.info("👈 Upload a dataset to begin.")
        return

    try:
        df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    if df.empty:
        st.error("The uploaded file is empty.")
        return

    df = df.copy()
    numeric_cols, categorical_cols, datetime_cols = infer_column_groups(df)

    st.success("Dataset loaded successfully. 🧠")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Numeric columns", f"{len(numeric_cols):,}")
    col4.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    tabs = st.tabs(
        [
            "👀 Preview",
            "🧾 Structure",
            "🕳️ Missing Data",
            "📊 Visuals",
            "🎯 Target",
            "🧪 Baseline Model",
        ]
    )

    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(preview_rows), use_container_width=True)

    with tabs[1]:
        st.subheader("Column Structure")
        structure = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
                "missing_values": [df[col].isna().sum() for col in df.columns],
            }
        )
        st.dataframe(structure, use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        with st.expander("Detected column groups"):
            st.write("Numeric columns", numeric_cols if numeric_cols else "None detected")
            st.write("Categorical columns", categorical_cols if categorical_cols else "None detected")
            st.write("Datetime columns", datetime_cols if datetime_cols else "None detected")

    with tabs[2]:
        st.subheader("Missing-Data Summary")
        show_missing_summary(df)

        missing_total = int(df.isna().sum().sum())
        if missing_total == 0:
            st.success("No missing values detected. 🧠✨")
        else:
            st.warning(f"Total missing cells detected: {missing_total:,}")

    with tabs[3]:
        st.subheader("Automatic Visualisations")
        show_numeric_visuals(df, numeric_cols)
        st.subheader("Categorical Frequencies")
        show_categorical_summaries(df, categorical_cols, max_categories)
        st.subheader("Correlation")
        show_correlation_heatmap(df, numeric_cols)

    target_col: Optional[str] = None
    problem_type = "unsuitable"

    with tabs[4]:
        st.subheader("Target Variable Selection")
        target_col = st.selectbox("Choose the target variable", df.columns)
        y = df[target_col]
        problem_type = infer_problem_type(y)

        col1, col2, col3 = st.columns(3)
        col1.metric("Unique target values", f"{y.nunique(dropna=True):,}")
        col2.metric("Missing target values", f"{y.isna().sum():,}")
        col3.metric("Suggested task", problem_type.title())

        st.write("Target summary")
        if pd.api.types.is_numeric_dtype(y):
            st.dataframe(y.describe().to_frame(name=target_col), use_container_width=True)
            fig = px.histogram(df, x=target_col, marginal="box", title=f"Target distribution: {target_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            target_freq = y.astype("object").fillna("Missing").value_counts().head(max_categories).reset_index()
            target_freq.columns = [target_col, "count"]
            st.dataframe(target_freq, use_container_width=True)
            fig = px.bar(target_freq, x=target_col, y="count", title=f"Target frequency: {target_col}")
            st.plotly_chart(fig, use_container_width=True)

        if problem_type == "classification":
            st.info("The selected target appears suitable for a classification model.")
        elif problem_type == "regression":
            st.info("The selected target appears suitable for a regression model.")
        else:
            st.error("The selected target is not suitable because it has too little variation.")

    with tabs[5]:
        st.subheader("Optional Baseline Modelling")
        st.write(
            "This section uses a simple K-Nearest Neighbours baseline inspired by the original workshop script. "
            "It automatically preprocesses numeric and categorical predictors."
        )

        if target_col is None:
            st.info("Select a target variable first.")
            return

        if problem_type == "unsuitable":
            st.warning("The selected target is unsuitable for baseline modelling.")
            return

        test_size = st.slider("Test-set size", 0.1, 0.4, 0.2, 0.05)
        n_neighbors = st.slider("KNN neighbours", 1, 25, 5)
        run_model = st.button("Run baseline KNN model 🧪")

        if run_model:
            run_baseline_model(df, target_col, problem_type, test_size, n_neighbors)


if __name__ == "__main__":
    main()

