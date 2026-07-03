# app.py
# Streamlit Brainy Data Explorer

impsdfsdfsdfsfsdfsfdsdfsdfssfsdsdort streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

st.set_page_config(
    page_title="Brainy Data Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brainy Data Explorer")
st.caption("Upload CSV or Excel data, explore it, select a target variable, and get a simple modelling suggestion.")

uploaded_file = st.file_uploader(
    "📤 Upload your dataset",
    type=["csv", "xlsx", "xls"]
)


@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file type.")
            return None
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None


def infer_problem_type(series):
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.nunique(dropna=True)
        if unique_values <= 10:
            return "Classification"
        return "Regression"
    return "Classification"


if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        st.success("✅ Dataset uploaded successfully")

        st.subheader("👀 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("📌 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("🧬 Column Information")
        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(dropna=True).values
        })
        st.dataframe(column_info, use_container_width=True)

        st.subheader("🕳️ Missing Data Summary")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if missing.empty:
            st.info("No missing values detected.")
        else:
            st.dataframe(
                pd.DataFrame({
                    "Column": missing.index,
                    "Missing Values": missing.values,
                    "Missing Percentage": (missing.values / len(df) * 100).round(2)
                }),
                use_container_width=True
            )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            st.subheader("📈 Numeric Visualisations")
            selected_num_col = st.selectbox("Select a numeric variable", numeric_cols)

            fig, ax = plt.subplots()
            ax.hist(df[selected_num_col].dropna(), bins=30)
            ax.set_title(f"Distribution of {selected_num_col}")
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            if len(numeric_cols) >= 2:
                st.subheader("🔥 Correlation Heatmap")
                corr = df[numeric_cols].corr(numeric_only=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(corr)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=90)
                ax.set_yticklabels(corr.columns)
                fig.colorbar(im)
                st.pyplot(fig)

        if categorical_cols:
            st.subheader("🏷️ Categorical Frequency Summary")
            selected_cat_col = st.selectbox("Select a categorical variable", categorical_cols)

            freq = df[selected_cat_col].value_counts(dropna=False).head(20)
            st.dataframe(freq.rename("Count"), use_container_width=True)

            fig, ax = plt.subplots()
            freq.plot(kind="bar", ax=ax)
            ax.set_title(f"Top Categories in {selected_cat_col}")
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.subheader("🎯 Target Variable Selection")
        target_col = st.selectbox("Select the target variable", df.columns)

        if target_col:
            target_series = df[target_col]
            problem_type = infer_problem_type(target_series)

            st.info(f"Suggested problem type: **{problem_type}**")

            st.subheader("🧠 Target Variable Summary")
            st.write(f"Target variable: **{target_col}**")
            st.write(f"Data type: **{target_series.dtype}**")
            st.write(f"Unique values: **{target_series.nunique(dropna=True)}**")
            st.write(f"Missing values: **{target_series.isna().sum()}**")

            if problem_type == "Classification":
                st.dataframe(
                    target_series.value_counts(dropna=False).rename("Count"),
                    use_container_width=True
                )
            else:
                st.dataframe(
                    target_series.describe().rename("Summary"),
                    use_container_width=True
                )

            st.subheader("🤖 Optional Simple KNN Model")

            run_model = st.button("Run Simple KNN Model")

            if run_model:
                try:
                    model_df = df.dropna(subset=[target_col]).copy()

                    if model_df[target_col].nunique(dropna=True) < 2:
                        st.error("The selected target variable must contain at least two unique values.")
                    else:
                        X = model_df.drop(columns=[target_col])
                        y = model_df[target_col]

                        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
                        numeric_features = X.select_dtypes(include=np.number).columns.tolist()

                        if len(categorical_features) + len(numeric_features) == 0:
                            st.error("No usable feature columns were found.")
                        else:
                            numeric_pipeline = Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler())
                            ])

                            categorical_pipeline = Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore"))
                            ])

                            preprocessor = ColumnTransformer([
                                ("numeric", numeric_pipeline, numeric_features),
                                ("categorical", categorical_pipeline, categorical_features)
                            ])

                            if problem_type == "Classification":
                                model = KNeighborsClassifier(n_neighbors=5)

                                stratify = y if y.value_counts().min() >= 2 else None

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size=0.2,
                                    random_state=42,
                                    stratify=stratify
                                )

                                pipeline = Pipeline([
                                    ("preprocessor", preprocessor),
                                    ("model", model)
                                ])

                                pipeline.fit(X_train, y_train)
                                y_pred = pipeline.predict(X_test)

                                st.success("Classification model completed.")

                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
                                col2.metric("Precision", round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4))
                                col3.metric("Recall", round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4))
                                col4.metric("F1 Score", round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4))

                                st.text("Classification Report")
                                st.text(classification_report(y_test, y_pred, zero_division=0))

                                st.text("Confusion Matrix")
                                st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

                            else:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size=0.2,
                                    random_state=42
                                )

                                pipeline = Pipeline([
                                    ("preprocessor", preprocessor),
                                    ("model", KNeighborsRegressor(n_neighbors=5))
                                ])

                                pipeline.fit(X_train, y_train)
                                y_pred = pipeline.predict(X_test)

                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                                st.success("Regression model completed.")

                                col1, col2, col3 = st.columns(3)
                                col1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
                                col2.metric("RMSE", round(rmse, 4))
                                col3.metric("R²", round(r2_score(y_test, y_pred), 4))

                except Exception as e:
                    st.error(f"The model could not be completed: {e}")

    else:
        st.error("The uploaded file appears to be empty.")
else:
    st.info("Upload a CSV or Excel file to begin. 🧠")
