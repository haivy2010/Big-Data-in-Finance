import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.window import Window

# ML & clustering (sklearn cho SHAP/feature importance)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression as LRsk
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score
)

import warnings
warnings.filterwarnings("ignore", message="The default of observed=False is deprecated", category=FutureWarning)

# PySpark ML
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler as SparkStandardScaler, PCA as SparkPCA
from pyspark.ml.feature import Imputer ,VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.pipeline import PipelineModel


# SHAP (tuỳ chọn)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Streamlit
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    st = None
    HAS_STREAMLIT = False
    
@st.cache_resource(show_spinner=False)
def get_spark() -> SparkSession:
    """Khởi tạo và trả về một SparkSession đã được cấu hình."""
    return (
        SparkSession.builder
        .appName("CreditDashboard")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "512m") 
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") 
        .getOrCreate()
    )
spark = get_spark()


BASE = Path(__file__).parent
IN_CSV = BASE / "data.csv"
OUT_CSV = BASE / "data_with_years.csv"

# Features dùng cho ML (tự lọc chỉ lấy cột có trong dữ liệu)
RATIO_FEATURES = [
    "Debt-to-Assets Ratio", "Debt-to-Equity Ratio", "Debt-to-Capital Ratio",
    "Debt-to-EBITDA Ratio", "Asset-to-Equity Ratio",
    "Current Ratio", "Quick Ratio",
    "Interest Coverage Ratio", "EBITDA Interest Coverage",
    "Gross Profit Margin", "Net Profit Margin", "Operating Profit Margin", "ROA", "ROE",
    "Cash_Flow_to_Total_Debt", "Retained_Cash_Flow_to_Net_Debt"
]

# Expected metric columns
EXPECTED_METRICS = {
    "Gross Profit Margin", "Net Profit Margin", "General Profitability (Operating)",
    "EBITA_to_Assets", "Operating Profit Margin", "ROA", "ROE",
    "Cash_Flow_to_Total_Debt", "Retained_Cash_Flow_to_Net_Debt", "Score", "Rating"
}

# Rating order & helpers
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]

def normalize_rating_col(df, col_name):
    """Chuẩn hoá cột Rating trong PySpark DataFrame"""
    return df.withColumn(
        "_RATING_NORM",
        F.when(
            F.upper(F.trim(F.col(col_name))).isin(RATING_ORDER),
            F.upper(F.trim(F.col(col_name)))
        ).otherwise("NAN")
    )

# Formatting helpers
PCT_POINT = {
    "Gross Profit Margin", "Operating Profit Margin", "Net Profit Margin"
}
PCT_FRAC = {
    "ROA", "ROE", "EBITA_to_Assets",
    "Cash_Flow_to_Total_Debt", "Retained_Cash_Flow_to_Net_Debt"
}

def fmt_metric(name, val):
    import pandas as pd
    if pd.isna(val):
        return "N/A"
    if name in PCT_POINT:
        return f"{val:.2f}%"
    if name in PCT_FRAC:
        return f"{val*100:.2f}%"
    try:
        return f"{float(val):,.2f}"
    except Exception:
        return str(val)

def winsorize_col(df, col_name, p=0.99):
    """Cắt đuôi ngoại lệ cho một cột trong PySpark"""
    quantiles = df.approxQuantile(col_name, [1-p, p], 0.01)
    lo, hi = quantiles[0], quantiles[1]
    return df.withColumn(
        col_name,
        F.when(F.col(col_name) < lo, lo)
         .when(F.col(col_name) > hi, hi)
         .otherwise(F.col(col_name))
    )

def winsorize_df(df, cols, p=0.99):
    """Winsorize nhiều cột"""
    for c in cols:
        if c in df.columns:
            df = winsorize_col(df, c, p)
    return df

def read_and_expand(path=IN_CSV, start_year=2010, end_year=2024):
    """Đọc và expand dữ liệu với PySpark"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Input not found: {path}")
    
    s = get_spark()  # 👈 lấy SparkSession từ cache
    df = s.read.csv(str(path), header=True, inferSchema=True)
    
    # Detect company column
    cols = df.columns
    if "Company Code" in cols:
        comp_col = "Company Code"
    elif "Company" in cols:
        comp_col = "Company"
    else:
        first_col = cols[0]
        if first_col not in EXPECTED_METRICS:
            comp_col = first_col
        else:
            df = df.withColumn("Company", F.monotonically_increasing_id())
            comp_col = "Company"
    
    # If Year already exists, just normalize and return
    if "Year" in cols:
        df = df.withColumn("Year", F.col("Year").cast(IntegerType()))
        return df, comp_col
    
    # Keep one row per company and explode years
    df_unique = df.dropDuplicates([comp_col])
    
    # Create years array
    years = list(range(start_year, end_year + 1))
    
    # Explode years for each company
    df_expanded = df_unique.withColumn("Year", F.explode(F.array([F.lit(y) for y in years])))
    df_expanded = df_expanded.withColumn("Year", F.col("Year").cast(IntegerType()))
    
    return df_expanded, comp_col

# HELPER MACHINE LEARNING
@st.cache_resource(show_spinner=False)
def fit_isoforest(df_pd: pd.DataFrame, feats: list[str], contamination: float = 0.03):
    """Fit IsolationForest trên các feature đã chọn (chuẩn hoá + cache).
    Trả về sklearn Pipeline(StandardScaler -> IsolationForest) hoặc None nếu dữ liệu quá ít.
    """
    if not feats:
        return None

    X = df_pd[feats].apply(pd.to_numeric, errors="coerce").dropna()
    # Cần đủ mẫu để ổn định (ít nhất 50 hoặc 3x số feature)
    if len(X) < max(50, 3 * len(feats)):
        return None

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        IsolationForest(
            n_estimators=400,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        ),
    )
    pipe.fit(X.values)
    return pipe


def rank_anomalies(
    pipe, df_pd: pd.DataFrame, feats: list[str], meta_cols: list[str], top_n: int = 20
) -> pd.DataFrame:
    """Tính điểm bất thường, trả về bảng top_n có meta kèm score (cao = bất thường)."""
    if pipe is None or not feats:
        return pd.DataFrame()

    Xall = df_pd[feats].apply(pd.to_numeric, errors="coerce")
    mask = Xall.notna().all(axis=1)
    if mask.sum() == 0:
        return pd.DataFrame()

    # decision_function: cao = bình thường, thấp = bất thường → đảo dấu để "cao = bất thường"
    scores = -pipe.decision_function(Xall[mask].values)
    out = df_pd.loc[mask, meta_cols].copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False)
    return out.head(top_n)

def _label_from_rating(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    return s.isin(["AAA","AA","A","BBB"]).astype(int)  # IG = 1

def prepare_ml_pd(df_pd: pd.DataFrame, rating_col: str) -> tuple[pd.DataFrame, list[str]]:
    feats = [c for c in RATIO_FEATURES if c in df_pd.columns]
    if not feats:
        return pd.DataFrame(), []
    use = df_pd[[rating_col] + feats].copy()
    use["label"] = _label_from_rating(use[rating_col])
    for c in feats:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna()
    return use[["label"] + feats], feats

def build_ml_df(df_pd: pd.DataFrame, rating_col: str, chosen_feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Tạo dataframe ML theo danh sách feature người dùng chọn."""
    base_df, feats_all = prepare_ml_pd(df_pd, rating_col)
    feats = [c for c in chosen_feats if c in feats_all]
    if not feats:
        return pd.DataFrame(), []
    use = base_df[["label"] + feats].dropna()
    return use, feats

def cv_auc_lr_spark(ml_df_pd: pd.DataFrame, feats: list[str]) -> float:
    """3-fold CV AUC cho LR (Spark) với imputer + scaler (có grid nhỏ)."""
    if ml_df_pd.empty or not feats:
        return float("nan")
    s = get_spark()
    sdf = s.createDataFrame(ml_df_pd)
    imp_cols = [f"{c}__imp" for c in feats]
    imputer = Imputer(inputCols=feats, outputCols=imp_cols, strategy="median")
    assembler = VectorAssembler(inputCols=imp_cols, outputCol="features", handleInvalid="keep")
    scaler = SparkStandardScaler(inputCol="features", outputCol="z", withMean=True, withStd=True)
    lr = LogisticRegression(featuresCol="z", labelCol="label", maxIter=100)
    pipe = Pipeline(stages=[imputer, assembler, scaler, lr])

    grid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.0, 0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
            .build())
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    cv = CrossValidator(estimator=pipe, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3, seed=42)
    cvModel = cv.fit(sdf)
    return float(max(cvModel.avgMetrics))

def detect_anomalies_iso(df_pd: pd.DataFrame, feats: list[str], comp_col: str, contamination=0.05, topn=10):
    """IsolationForest cho outlier gợi ý cảnh báo sớm."""
    use_cols = [c for c in feats if c in df_pd.columns]
    if not use_cols or comp_col not in df_pd.columns or "Year" not in df_pd.columns:
        return pd.DataFrame()
    dft = df_pd[[comp_col, "Year"] + use_cols].copy()
    dft = dft.dropna()
    if dft.empty:
        return pd.DataFrame()
    X = dft[use_cols].to_numpy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    score = iso.fit_predict(X)
    dft["anomaly"] = (score == -1).astype(int)
    dft["score"] = iso.decision_function(X)  # thấp hơn = bất thường hơn
    out = dft[dft["anomaly"] == 1].sort_values("score").head(topn)
    return out[[comp_col, "Year", "score"] + use_cols]

@st.cache_resource(show_spinner=False)
def train_lr_spark(df_pd: pd.DataFrame, feats: list[str]) -> tuple[Pipeline, float, float]:
    """
    Huấn luyện LR (Spark) với Imputer(median) để xử lý thiếu dữ liệu.
    Trả về: (model_pipe, AUC, Accuracy)
    """
    if len(df_pd) < 200:
        return None, float("nan"), float("nan")

    # Ép kiểu số chắc chắn
    for c in feats:
        df_pd[c] = pd.to_numeric(df_pd[c], errors="coerce")
    sdf = get_spark().createDataFrame(df_pd)
    sdf = sdf.dropna(subset=["label"])

    # Split (Sau khi đã lọc)
    train, test = sdf.randomSplit([0.8, 0.2], seed=42)

    # Impute -> Assemble -> LR
    imp_cols = [f"{c}__imp" for c in feats]
    imputer = Imputer(inputCols=feats, outputCols=imp_cols, strategy="median")
    assembler = VectorAssembler(inputCols=imp_cols, outputCol="features", handleInvalid="keep")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
    model = Pipeline(stages=[imputer, assembler, lr]).fit(train)

    # Đánh giá
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    auc = float(evaluator.evaluate(model.transform(test)))
    pred = model.transform(test).select("label", "prediction").toPandas()
    acc = float((pred["label"] == pred["prediction"]).mean()) if not pred.empty else float("nan")
    return model, auc, acc


def predict_ig_proba_spark(model: Pipeline, row_pd: pd.Series, feats: list[str]) -> float | None:
    """
    Dự báo P(IG) cho 1 hàng pandas bằng pipeline có Imputer.
    """
    if model is None:
        return None

    # Lấy đúng các feature, ép số
    X = {f: pd.to_numeric(row_pd.get(f), errors="coerce") for f in feats}
    sdf = get_spark().createDataFrame(pd.DataFrame([X]))

    # Biến đổi qua pipeline (Imputer sẽ tự lấp median)
    out = model.transform(sdf).toPandas()
    if out.empty or "probability" not in out.columns:
        return None
    try:
        return float(out.loc[0, "probability"][1])  # P(label=1)=P(IG)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def train_kmeans_spark(df_pd: pd.DataFrame, feats: list[str], k: int = 5):
    s = get_spark()
    sdf = s.createDataFrame(df_pd[feats])
    assembler = VectorAssembler(inputCols=feats, outputCol="features")
    scaler = SparkStandardScaler(inputCol="features", outputCol="z", withMean=True, withStd=True)
    kmeans = SparkKMeans(featuresCol="z", k=k, seed=42)
    return Pipeline(stages=[assembler, scaler, kmeans]).fit(sdf)

def nearest_peers(df_pd_all: pd.DataFrame, comp: str, year: int, feats: list[str], n_peers=10):
    dft = df_pd_all.copy()
    dft = dft[dft["Company Code"].notna() & dft["Year"].notna()]
    cols = ["Company Code","Year"] + [c for c in feats if c in dft.columns]
    dft = dft[cols].dropna()
    if dft.empty or comp not in dft["Company Code"].values or year not in dft["Year"].values:
        return pd.DataFrame()
    X = dft[feats].to_numpy()
    z = StandardScaler().fit_transform(X)
    idx = (dft["Company Code"]==comp) & (dft["Year"]==year)
    z0 = z[idx.values][0]
    dist = np.sqrt(((z - z0)**2).sum(axis=1))
    out = dft.assign(distance=dist).sort_values("distance")
    return out.iloc[1:n_peers+1]

def _available_numeric_features(df_pd: pd.DataFrame) -> list[str]:
    # chỉ giữ các cột trong RATIO_FEATURES và là numeric (sau khi coerce)
    feats = []
    for c in RATIO_FEATURES:
        if c in df_pd.columns:
            s = pd.to_numeric(df_pd[c], errors="coerce")
            if s.notna().sum() > 0:
                feats.append(c)
    return feats

@st.cache_data(show_spinner=False)
def train_rf_sklearn(ml_df: pd.DataFrame, feats: list[str], seed: int = 42):
    X = ml_df[feats].to_numpy()
    y = ml_df["label"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced_subsample",
        min_samples_leaf=2, max_depth=None
    )
    rf.fit(Xtr, ytr)
    proba_te = rf.predict_proba(Xte)[:, 1]
    auc_roc = roc_auc_score(yte, proba_te)
    auc_pr  = average_precision_score(yte, proba_te)
    acc     = (rf.predict(Xte) == yte).mean()
    return rf, (Xtr, Xte, ytr, yte, proba_te), {"auc_roc": auc_roc, "auc_pr": auc_pr, "acc": acc}

def _altair_roc_pr(y_true: np.ndarray, y_score: np.ndarray):
    import altair as alt
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pr, rc, _  = precision_recall_curve(y_true, y_score)

    df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    df_pr  = pd.DataFrame({"Recall": rc, "Precision": pr})

    roc_ch = alt.Chart(df_roc).mark_line().encode(
        x=alt.X("FPR:Q", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("TPR:Q", scale=alt.Scale(domain=[0,1]))
    ).properties(height=260)

    pr_ch = alt.Chart(df_pr).mark_line().encode(
        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0,1]))
    ).properties(height=260)
    return roc_ch, pr_ch

def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return (tn, fp, fn, tp), prec, rec, f1

@st.cache_data(show_spinner=False)
def detect_anomaly_iso(ml_df: pd.DataFrame, feats: list[str], n_out: int = 10, seed: int = 42):
    X = ml_df[feats].to_numpy()
    mdl = IsolationForest(n_estimators=300, random_state=seed, contamination="auto")
    scores = mdl.score_samples(X)  # càng thấp càng bất thường
    out_idx = np.argsort(scores)[:n_out]
    out = ml_df.iloc[out_idx].copy()
    out.insert(0, "score", scores[out_idx])
    return out

def _company_row(df_all: pd.DataFrame, comp_col: str, comp: str, year: int) -> pd.Series | None:
    try:
        row = df_all[(df_all[comp_col].astype(str)==str(comp)) & (df_all["Year"].astype(int)==int(year))].iloc[0]
        return row
    except Exception:
        return None


# Main
def main_cli():
    try:
        df_exp, comp_col = read_and_expand()
    except Exception as e:
        print("Error:", e)
        return
    
    # Write to CSV
    df_exp.coalesce(1).write.csv(str(OUT_CSV), header=True, mode='overwrite')
    print(f"Wrote expanded data to {OUT_CSV} (rows={df_exp.count()})")
    
    # Show sample
    df_exp.show(5)

def main_streamlit():
    import pandas as pd
    
    # Cache function cho Streamlit
    @st.cache_data
    def _read_and_expand_cached(path):
        df_spark, comp_col = read_and_expand(path)
        df_pd = df_spark.toPandas()
        return df_pd, comp_col
    
    # Đọc dữ liệu
    try:
        df_exp, comp_col = _read_and_expand_cached(IN_CSV)
    except Exception as e:
        st.error(f"Error reading/expanding data: {e}")
        return
    
    # Chuẩn hoá Score
    if "Score" in df_exp.columns:
        df_exp["Score"] = pd.to_numeric(df_exp["Score"], errors="coerce")
    
    st.title("🏦 Hệ thống đánh giá tín dụng doanh nghiệp")
    
    # CSS
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # TAB CHÍNH
    tab1, tab2, tab3 = st.tabs(["📊 Toàn thị trường", "🏢 Phân tích doanh nghiệp", "🤖 Machine Learning & Phân tích"])
    
    # ========== TAB 1: TOÀN THỊ TRƯỜNG ==========
    with tab1:
        st.header("📈 Tổng quan xếp hạng toàn thị trường")
        
        # Lọc năm
        years_all = sorted(pd.to_numeric(df_exp["Year"], errors="coerce").dropna().astype(int).unique().tolist())
        default_years = [y for y in years_all if y >= max(years_all) - 9] or years_all
        year_sel_multi = st.multiselect(
            "Chọn năm (có thể chọn nhiều):",
            [str(y) for y in years_all],
            default=[str(y) for y in default_years],
            key="market_years"
        )
        
        vis_df = df_exp.copy()
        if year_sel_multi:
            vis_df = vis_df[vis_df["Year"].astype(str).isin([str(y) for y in year_sel_multi])].copy()
        
        # Tìm cột Rating
        rating_col = None
        for c in df_exp.columns:
            if str(c).strip().lower() == "rating":
                rating_col = c
                break
        if rating_col is None:
            for c in df_exp.columns:
                if "rating" in str(c).strip().lower():
                    rating_col = c
                    break
        
        if rating_col is None:
            st.info("Không tìm thấy cột 'Rating' trong dữ liệu.")
        else:
            # Metrics tổng quan
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số doanh nghiệp", len(vis_df[comp_col].unique()))
            with col2:
                st.metric("Tổng số bản ghi", len(vis_df))
            with col3:
                if "Score" in vis_df.columns:
                    st.metric("Điểm TB toàn thị trường", f"{vis_df['Score'].mean():.2f}")
            
            # Normalize rating
            def normalize_rating_col_pd(series):
                from pandas.api.types import CategoricalDtype
                s = series.astype(str).str.strip().str.upper()
                s = s.where(s.isin(RATING_ORDER), other="NAN")
                cat_type = CategoricalDtype(categories=RATING_ORDER + ["NAN"], ordered=True)
                return s.astype(cat_type)
            
            vis_df["_RATING_NORM"] = normalize_rating_col_pd(vis_df[rating_col])
            
            # Phân bổ xếp hạng
            total_counts = (
                vis_df["_RATING_NORM"]
                .value_counts(dropna=False)
                .reindex(RATING_ORDER + ["NAN"])
                .fillna(0).astype(int)
            )
            df_total = total_counts.rename_axis("Rating").reset_index(name="Count")
            df_total["Tỷ lệ (%)"] = (df_total["Count"] / df_total["Count"].sum() * 100).round(2)
            
            st.subheader("📊 Phân bổ xếp hạng tín dụng")
            st.table(df_total.set_index("Rating"))
            
            # Bar chart
            import altair as alt
            _dfbar = df_total.copy()
            _dfbar["Rating"] = pd.Categorical(_dfbar["Rating"], categories=RATING_ORDER + ["NAN"], ordered=True)
            _dfbar = _dfbar.sort_values("Rating")
            
            bars = alt.Chart(_dfbar).mark_bar().encode(
                x=alt.X("Rating:N", sort=RATING_ORDER + ["NAN"], title=""),
                y=alt.Y("Count:Q", title="Số lượng"),
                tooltip=["Rating:N", "Count:Q", alt.Tooltip("Tỷ lệ (%):Q", format=".2f")]
            ).properties(height=300)
            st.altair_chart(bars, use_container_width=True)
            
            # Pie chart
            try:
                st.write("📈 Biểu đồ tròn - Phân bổ theo Rating:")
                pie = alt.Chart(df_total).mark_arc().encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color(
                        "Rating:N",
                        scale=alt.Scale(domain=RATING_ORDER + ["NAN"], scheme="category20"),
                        sort=RATING_ORDER + ["NAN"]
                    ),
                    tooltip=["Rating:N", "Count:Q", "Tỷ lệ (%):Q"]
                ).properties(width=400, height=300)
                st.altair_chart(pie, use_container_width=False)
            except Exception:
                st.info("Không thể hiển thị biểu đồ tròn (Altair).")
            
            # Phân bổ Rating theo năm
            if len(year_sel_multi) > 1:
                st.subheader("📅 Phân bổ Rating theo năm")
                
                vis_df["Year"] = pd.to_numeric(vis_df["Year"], errors="coerce").astype("Int64")
                
                counts = (
                    vis_df
                    .groupby(["Year", "_RATING_NORM"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=RATING_ORDER + ["NAN"])
                    .sort_index()
                )
                st.dataframe(counts)
                
                counts_for_plot = counts.copy()
                counts_for_plot.index = counts_for_plot.index.astype(str)
                st.line_chart(counts_for_plot)
    
    # ========== TAB 2: PHÂN TÍCH DOANH NGHIỆP ==========
    with tab2:
        st.header("🏢 Phân tích chi tiết doanh nghiệp")
        
        # Chọn doanh nghiệp
        companies = sorted(df_exp[comp_col].dropna().astype(str).unique().tolist())
        sel = st.selectbox(f"Chọn doanh nghiệp ({comp_col}):", companies)
        
        df_company = df_exp[df_exp[comp_col].astype(str) == sel].copy()
        
        # Chọn năm
        years_company = sorted(df_company["Year"].astype(int).unique().tolist())
        sel_year = st.selectbox("Chọn năm phân tích:", years_company, index=len(years_company)-1)
        
        df_current = df_company[df_company["Year"] == sel_year]
        
        # Aggregate nếu có nhiều bản ghi
        if len(df_current) > 0:
            num_cols = df_current.select_dtypes(include="number").columns
            agg_num = df_current[num_cols].mean(numeric_only=True)
            
            non_num_cols = [c for c in df_current.columns if c not in num_cols]
            agg_cat = {}
            for c in non_num_cols:
                try:
                    agg_cat[c] = df_current[c].mode(dropna=True).iloc[0]
                except Exception:
                    agg_cat[c] = df_current[c].iloc[0] if len(df_current[c]) else None
            
            row = pd.concat([agg_num, pd.Series(agg_cat)])
            
            # Helper format number
            def format_number(value):
                if pd.isna(value):
                    return "N/A"
                abs_val = abs(value)
                if abs_val >= 1_000_000_000:
                    formatted = f"{value/1_000_000_000:.2f} tỷ"
                elif abs_val >= 1_000_000:
                    formatted = f"{value/1_000_000:.2f} tr"
                elif abs_val >= 1_000:
                    formatted = f"{value/1_000:.2f} nghìn"
                else:
                    formatted = f"{value:.2f}"
                return formatted
            
            # Metrics dòng 1
            metric_cols = st.columns(5)
            
            with metric_cols[0]:
                if rating_col and rating_col in row:
                    rating_val = str(row[rating_col]).upper()
                    st.metric("Xếp hạng tín dụng", rating_val)
                    if "Rating_Description" in row and pd.notna(row["Rating_Description"]):
                        st.caption(row["Rating_Description"])
            
            with metric_cols[1]:
                sc = row.get("Score", None)
                if sc is not None and pd.notna(sc):
                    sc = float(sc)
                    show = sc*100 if 0 <= sc <= 1 else sc
                    st.metric("Điểm tín dụng (0–100)", f"{show:.2f}")
            
            with metric_cols[2]:
                if "Total Assets" in row and pd.notna(row["Total Assets"]):
                    st.metric("Tổng tài sản", format_number(row["Total Assets"]))
            
            with metric_cols[3]:
                equity_col = "Owner's Equity"
                if equity_col in row and pd.notna(row[equity_col]):
                    st.metric("Vốn chủ sở hữu", format_number(row[equity_col]))
            
            with metric_cols[4]:
                revenue_col = "Sales and Service Revenue"
                if revenue_col in row and pd.notna(row[revenue_col]):
                    st.metric("Doanh thu", format_number(row[revenue_col]))
            
            # Metrics dòng 2
            metric_cols2 = st.columns(5)
            
            with metric_cols2[0]:
                if "Net Income" in row and pd.notna(row["Net Income"]):
                    st.metric("Lợi nhuận ròng", format_number(row["Net Income"]))
            
            with metric_cols2[1]:
                if "EBITDA" in row and pd.notna(row["EBITDA"]):
                    st.metric("EBITDA", format_number(row["EBITDA"]))
            
            with metric_cols2[2]:
                if "Total Debt" in row and pd.notna(row["Total Debt"]):
                    st.metric("Tổng nợ", format_number(row["Total Debt"]))
            
            with metric_cols2[3]:
                cashflow_col = "Operating Cash Flow"
                if cashflow_col in row and pd.notna(row[cashflow_col]):
                    st.metric("Dòng tiền HĐ", format_number(row[cashflow_col]))
            
            with metric_cols2[4]:
                cash_col = "Cash and Cash Equivalents"
                if cash_col in row and pd.notna(row[cash_col]):
                    st.metric("Tiền & tương đương", format_number(row[cash_col]))
        
        # CHỈ SỐ TÀI CHÍNH CHI TIẾT
        st.subheader("💰 Chỉ số tài chính chi tiết")
        
        def find_cols(df, aliases):
            cols = []
            lcols = {c.lower(): c for c in df.columns}
            for a in aliases:
                if a is None:
                    continue
                a_low = a.lower()
                if a_low in lcols:
                    cols.append(lcols[a_low])
                    continue
                found = None
                for c in df.columns:
                    cl = str(c).lower()
                    if a_low == cl or a_low in cl or cl in a_low:
                        found = c
                        break
                if found and found not in cols:
                    cols.append(found)
            return cols
        
        # Định nghĩa nhóm
        groups = {
            "Profitability": ["Gross Profit Margin", "Net Profit Margin", "General Profitability (Operating)", 
                             "EBITA_to_Assets", "Operating Profit Margin", "ROA", "ROE", 
                             "Gross Profit", "Operating Profit", "Net Income", "EBIT", "EBITDA"],
            "Liquidity": ["Current Assets", "Current Liabilities", "Quick Assets", "Current Ratio", "Quick Ratio",
                         "Cash and Cash Equivalents", "Cash", "Cash Equivalents"],
            "Leverage": ["Total Debt", "Total Equity", "Debt-to-Assets Ratio", "Debt-to-Equity Ratio", 
                        "Debt-to-Capital Ratio", "Debt-to-EBITDA Ratio", "Total Capital", "Asset-to-Equity Ratio"],
            "Coverage": ["Interest Coverage Ratio", "EBITDA Interest Coverage", "DSCR", "Cash Coverage Ratio",
                        "Interest Expenses", "Total Debt Service"],
            "Cash flow": ["Operating Cash Flow", "Retained Cash Flow", "Cash_Flow_to_Total_Debt", 
                         "Retained_Cash_Flow_to_Net_Debt", "Net Debt",
                         "Cash Flow from Operating Activities", "Cash Flow from Investing Activities",
                         "Cash Flow from Financing Activities"]
        }
        
        # Hiển thị các nhóm
        col1, col2 = st.columns(2)
        
        with col1:
            # Profitability
            st.write("**📊 Chỉ số sinh lời**")
            profit_cols = find_cols(df_current, groups["Profitability"])
            if profit_cols:
                profit_data = {}
                for col in profit_cols:
                    val = df_current[col].iloc[0]
                    if pd.notna(val):
                        profit_data[col] = fmt_metric(col, val)
                if profit_data:
                    st.table(pd.DataFrame.from_dict(profit_data, orient='index', columns=['Giá trị']))
            else:
                st.info("Không có dữ liệu")
            
            # Liquidity
            st.write("**💧 Chỉ số thanh khoản**")
            liquid_cols = find_cols(df_current, groups["Liquidity"])
            if liquid_cols:
                liquid_data = {}
                for col in liquid_cols:
                    val = df_current[col].iloc[0]
                    if pd.notna(val):
                        liquid_data[col] = fmt_metric(col, val)
                if liquid_data:
                    st.table(pd.DataFrame.from_dict(liquid_data, orient='index', columns=['Giá trị']))
            else:
                st.info("Không có dữ liệu")
            
            # Coverage
            st.write("**🛡️ Chỉ số bảo vệ**")
            coverage_cols = find_cols(df_current, groups["Coverage"])
            if coverage_cols:
                coverage_data = {}
                for col in coverage_cols:
                    val = df_current[col].iloc[0]
                    if pd.notna(val):
                        coverage_data[col] = fmt_metric(col, val)
                if coverage_data:
                    st.table(pd.DataFrame.from_dict(coverage_data, orient='index', columns=['Giá trị']))
            else:
                st.info("Không có dữ liệu")
        
        with col2:
            # Leverage
            st.write("**⚖️ Chỉ số đòn bẩy**")
            leverage_cols = find_cols(df_current, groups["Leverage"])
            if leverage_cols:
                leverage_data = {}
                for col in leverage_cols:
                    val = df_current[col].iloc[0]
                    if pd.notna(val):
                        leverage_data[col] = fmt_metric(col, val)
                if leverage_data:
                    st.table(pd.DataFrame.from_dict(leverage_data, orient='index', columns=['Giá trị']))
            else:
                st.info("Không có dữ liệu")
            
            # Cash flow
            st.write("**💵 Chỉ số dòng tiền**")
            cashflow_cols = find_cols(df_current, groups["Cash flow"])
            if cashflow_cols:
                cashflow_data = {}
                for col in cashflow_cols:
                    val = df_current[col].iloc[0]
                    if pd.notna(val):
                        cashflow_data[col] = fmt_metric(col, val)
                if cashflow_data:
                    st.table(pd.DataFrame.from_dict(cashflow_data, orient='index', columns=['Giá trị']))
            else:
                st.info("Không có dữ liệu")
        
        # XU HƯỚNG THEO THỜI GIAN
        if len(df_company) > 1:
            st.subheader("📈 Xu hướng theo thời gian")
            
            compare_years = st.multiselect(
                "Chọn các năm để so sánh:", 
                years_company, 
                default=years_company[-min(3, len(years_company)):],
                key="company_compare_years"
            )
            
            if compare_years:
                df_compare = df_company[df_company["Year"].isin(compare_years)]
                
                trend_tabs = st.tabs(["📊 Chỉ số chính", "💰 Doanh thu & Lợi nhuận", "💵 Dòng tiền", "⚖️ Cấu trúc tài chính"])
                
                # Tab 1: Chỉ số chính
                with trend_tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "Score" in df_compare.columns:
                            st.write("**Điểm tín dụng qua các năm**")
                            score_trend = df_compare.groupby("Year")["Score"].mean().reset_index()
                            st.line_chart(score_trend.set_index("Year"))
                        
                        if "ROA" in df_compare.columns and "ROE" in df_compare.columns:
                            st.write("**ROA & ROE**")
                            roe_roa = df_compare.groupby("Year")[["ROA", "ROE"]].mean()
                            st.line_chart(roe_roa)
                    
                    with col2:
                        if "Debt-to-Equity Ratio" in df_compare.columns:
                            st.write("**Tỷ lệ Nợ/Vốn chủ sở hữu**")
                            de_trend = df_compare.groupby("Year")["Debt-to-Equity Ratio"].mean().reset_index()
                            st.line_chart(de_trend.set_index("Year"))
                        
                        liquidity_cols = [c for c in ["Current Ratio", "Quick Ratio"] if c in df_compare.columns]
                        if liquidity_cols:
                            st.write("**Chỉ số thanh khoản**")
                            liq_trend = df_compare.groupby("Year")[liquidity_cols].mean()
                            st.line_chart(liq_trend)
                
                # Tab 2: Doanh thu & Lợi nhuận
                with trend_tabs[1]:
                    revenue_profit_cols = ["Sales and Service Revenue", "Gross Profit", "Operating Profit", 
                                          "EBIT", "EBITDA", "Net Income"]
                    available_rev_cols = [c for c in revenue_profit_cols if c in df_compare.columns]
                    
                    if available_rev_cols:
                        selected_rev = st.multiselect(
                            "Chọn chỉ số:",
                            available_rev_cols,
                            default=available_rev_cols[:3],
                            key="revenue_select"
                        )
                        
                        if selected_rev:
                            rev_data = df_compare.groupby("Year")[selected_rev].mean()
                            st.line_chart(rev_data)
                            
                            if len(compare_years) > 1:
                                st.write("**Tỷ lệ tăng trưởng (%)**")
                                growth = rev_data.pct_change() * 100
                                st.dataframe(growth.style.format("{:.2f}%"))
                
                # Tab 3: Dòng tiền
                with trend_tabs[2]:
                    cashflow_cols = ["Cash Flow from Operating Activities", 
                                    "Cash Flow from Investing Activities",
                                    "Cash Flow from Financing Activities",
                                    "Net Cash Flow during the Period"]
                    available_cf_cols = [c for c in cashflow_cols if c in df_compare.columns]
                    
                    if available_cf_cols:
                        cf_data = df_compare.groupby("Year")[available_cf_cols].mean()
                        st.bar_chart(cf_data)
                        st.dataframe(cf_data.style.format("{:,.0f}"))
                
                # Tab 4: Cấu trúc tài chính
                with trend_tabs[3]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "Total Assets" in df_compare.columns and "Total Debt" in df_compare.columns:
                            st.write("**Tổng tài sản vs Tổng nợ**")
                            asset_debt = df_compare.groupby("Year")[["Total Assets", "Total Debt"]].mean()
                            st.line_chart(asset_debt)
                
                # Bảng so sánh
                st.write("---")
                st.write("**📋 Bảng so sánh tổng hợp các chỉ số**")
                
                all_indicators = []
                for group_cols in groups.values():
                    all_indicators.extend(find_cols(df_compare, group_cols))
                
                all_indicators = list(dict.fromkeys(all_indicators))
                
                if all_indicators:
                    selected_indicators = st.multiselect(
                        "Chọn chỉ số để so sánh:",
                        all_indicators,
                        default=all_indicators[:5],
                        key="selected_indicators"
                    )
                    
                    if selected_indicators:
                        comparison_data = df_compare.pivot_table(
                            values=selected_indicators,
                            index="Year",
                            aggfunc='mean'
                        )
                        st.dataframe(comparison_data.style.format("{:.2f}"))
                        
                        try:
                            st.line_chart(comparison_data)
                        except Exception:
                            pass
        
        # SO SÁNH VỚI TRUNG BÌNH THỊ TRƯỜNG
        st.subheader("📊 So sánh với trung bình thị trường")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Điểm số & Xếp hạng**")
            if "Score" in df_exp.columns:
                company_score = None
                if "Score" in df_current.columns:
                    company_score = pd.to_numeric(df_current["Score"], errors="coerce").mean()
                
                market_avg = pd.to_numeric(
                    df_exp.loc[df_exp["Year"] == sel_year, "Score"], errors="coerce"
                ).mean()
                
                mult = 100.0 if (
                    pd.notna(company_score) and 0 <= company_score <= 1 and
                    pd.notna(market_avg) and 0 <= market_avg <= 1
                ) else 1.0
                
                compare_cols = st.columns(3)
                with compare_cols[0]:
                    st.metric("Điểm DN", f"{(company_score*mult) if pd.notna(company_score) else float('nan'):.2f}")
                with compare_cols[1]:
                    st.metric("TB thị trường", f"{(market_avg*mult) if pd.notna(market_avg) else float('nan'):.2f}")
                with compare_cols[2]:
                    diff = (company_score - market_avg) * mult if (pd.notna(company_score) and pd.notna(market_avg)) else float("nan")
                    st.metric("Chênh lệch (điểm)", f"{diff:+.2f}", delta=f"{diff:+.2f}")
            
            # So sánh rating với thị trường
            if rating_col:
                company_rating = str(df_current[rating_col].iloc[0]).upper()
                market_df = df_exp[df_exp["Year"] == sel_year].copy()
                
                def normalize_rating_col_pd(series):
                    from pandas.api.types import CategoricalDtype
                    s = series.astype(str).str.strip().str.upper()
                    s = s.where(s.isin(RATING_ORDER), other="NAN")
                    cat_type = CategoricalDtype(categories=RATING_ORDER + ["NAN"], ordered=True)
                    return s.astype(cat_type)
                
                market_df["_RATING_NORM"] = normalize_rating_col_pd(market_df[rating_col])
                
                market_counts = (
                    market_df["_RATING_NORM"]
                    .value_counts(dropna=False)
                    .reindex(RATING_ORDER + ["NAN"])
                    .fillna(0).astype(int)
                )
                
                st.write(f"**Xếp hạng doanh nghiệp: {company_rating}**")
                st.write("Phân bổ xếp hạng thị trường cùng năm:")
                
                import altair as alt
                _mdf = market_counts.rename_axis("Rating").reset_index(name="Count")
                _mdf["Rating"] = pd.Categorical(_mdf["Rating"], categories=RATING_ORDER + ["NAN"], ordered=True)
                _mdf = _mdf.sort_values("Rating")
                
                bars_year = alt.Chart(_mdf).mark_bar().encode(
                    x=alt.X("Rating:N", sort=RATING_ORDER + ["NAN"], title=""),
                    y=alt.Y("Count:Q", title="Số lượng")
                ).properties(height=260)
                st.altair_chart(bars_year, use_container_width=True)
        
        with col2:
            st.write("**So sánh chỉ số tài chính chính**")
            
            key_metrics = ["ROA", "ROE", "Debt-to-Equity Ratio", "Current Ratio", 
                          "Net Profit Margin", "Operating Profit Margin"]
            available_metrics = [m for m in key_metrics if m in df_exp.columns]
            
            if available_metrics:
                comparison_data = []
                for metric in available_metrics[:4]:
                    company_val = df_current[metric].iloc[0] if pd.notna(df_current[metric].iloc[0]) else 0
                    market_val = df_exp[df_exp["Year"] == sel_year][metric].mean()
                    
                    comparison_data.append({
                        "Chỉ số": metric,
                        "Doanh nghiệp": company_val,
                        "Thị trường": market_val,
                        "Chênh lệch": company_val - market_val
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.format({
                    "Doanh nghiệp": "{:.2f}",
                    "Thị trường": "{:.2f}",
                    "Chênh lệch": "{:.2f}"
                }))
                
                try:
                    import altair as alt
                    
                    plot_data = []
                    for _, row in comparison_df.iterrows():
                        plot_data.append({"Chỉ số": row["Chỉ số"], "Loại": "Doanh nghiệp", "Giá trị": row["Doanh nghiệp"]})
                        plot_data.append({"Chỉ số": row["Chỉ số"], "Loại": "Thị trường", "Giá trị": row["Thị trường"]})
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    chart = alt.Chart(plot_df).mark_bar().encode(
                        x=alt.X('Chỉ số:N', title=''),
                        y=alt.Y('Giá trị:Q', title='Giá trị'),
                        color=alt.Color('Loại:N', scale=alt.Scale(scheme='category10')),
                        xOffset='Loại:N'
                    ).properties(height=300)
                    
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass
        
        # BẢNG DỮ LIỆU CHI TIẾT
        st.subheader("📄 Dữ liệu chi tiết")
        show_all_years = st.checkbox("Hiển thị tất cả các năm", value=False)
        
        if show_all_years:
            st.dataframe(df_company.sort_values("Year", ascending=False))
        else:
            st.dataframe(df_current)
            
    # ========== TAB 3: Machine Learning & Phân tích (refactor) ==========
    with tab3:
        st.header("🤖 ML & Phân tích")

        # 0) Rating column
        if 'rating_col' not in locals() or rating_col is None:
            rating_col = next((c for c in df_exp.columns if "rating" in str(c).lower()), None)
        if rating_col is None:
            st.info("Không tìm thấy cột Rating → không thể gắn nhãn IG/Non-IG.")
            st.stop()

        # 1) Feature selector (chips)
        all_feats = _available_numeric_features(df_exp)
        if not all_feats:
            st.warning("Không có feature numeric phù hợp trong RATIO_FEATURES.")
            st.stop()

        st.caption("Chọn tập feature dùng cho mô hình:")
        # dùng all_feats (không dùng feats vì chưa có), và default an toàn
        default_pick = st.session_state.get("sel_feats", all_feats[:min(12, len(all_feats))])
        chosen_feats = st.multiselect(
            "Chọn tập feature dùng cho mô hình",
            options=all_feats,
            default=default_pick,
            key="sel_feats",
            label_visibility="collapsed"
        )

        # 2) Chuẩn bị dữ liệu ML
        ml_df_all, feats_in_data = prepare_ml_pd(df_exp, rating_col)   # tạo label + lọc numeric
        # giao giữa lựa chọn người dùng và những cột thực sự còn dữ liệu sau clean
        feats = [f for f in (chosen_feats or all_feats) if f in feats_in_data]
        if not feats:
            st.warning("Các feature đã chọn không có đủ dữ liệu. Hãy chọn thêm/khác.")
            st.stop()

        ml_df = ml_df_all[["label"] + feats].dropna()
        if ml_df.empty:
            st.warning("Sau khi làm sạch dữ liệu còn trống. Hãy chọn thêm feature hoặc nới lỏng tiêu chí.")
            st.stop()

        # 3) Quick actions row
        left, mid, right = st.columns([2,2,2])
        with left:
            use_lr = st.toggle("Dùng Logistic Regression (Spark)", value=True, help="Bỏ chọn để dùng RandomForest (sklearn)")
        with mid:
            cv3 = st.checkbox("Chạy Cross-Validation 3-fold (Spark LR) – có thể chậm", value=False)
        with right:
            st.write("")

        # ===== 3.1 LR (Spark) hoặc RF (sklearn) – Train & Evaluate =====
        if use_lr:
            lr_model, auc_lr, acc_lr = train_lr_spark(ml_df, feats)
            # lấy test scores để vẽ ROC/PR
            # (dùng lại pipeline đã train, split lại để có test approx)
            sdf = get_spark().createDataFrame(ml_df[["label"] + feats])
            train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)
            proba_pd = lr_model.transform(test_sdf).select("label", "probability").toPandas()
            if not proba_pd.empty:
                y_true = proba_pd["label"].to_numpy()
                y_score = proba_pd["probability"].apply(lambda v: float(v[1])).to_numpy()
                auc_pr  = average_precision_score(y_true, y_score)
            else:
                y_true = np.array([]); y_score = np.array([]); auc_pr = float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("AUC (holdout)", f"{auc_lr:.3f}" if np.isfinite(auc_lr) else "N/A")
            c2.metric("Accuracy (holdout)", f"{acc_lr:.3f}" if np.isfinite(acc_lr) else "N/A")

            # Dự báo cho DN đang chọn
            p_ig = None
            current = _company_row(df_exp, comp_col, sel, sel_year)
            if current is not None:
                p_ig = predict_ig_proba_spark(lr_model, current, feats)
            c3.metric("P(IG) – DN đang xem", f"{100*p_ig:.1f}%" if p_ig is not None else "N/A")

            # ROC & PR (Altair)
            if len(y_true) > 0:
                roc_ch, pr_ch = _altair_roc_pr(y_true, y_score)
                st.caption("Đường cong ROC & PR (Spark LR)")
                st.altair_chart(roc_ch, use_container_width=True)
                st.altair_chart(pr_ch, use_container_width=True)

                # Threshold slider + confusion matrix (dựa trên y_score)
                thr = st.slider("Ngưỡng phân loại (Spark LR)", 0.05, 0.95, 0.50, 0.01)
                (tn, fp, fn, tp), prec, rec, f1 = _metrics_at_threshold(y_true, y_score, thr)
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{prec:.3f}")
                m2.metric("Recall", f"{rec:.3f}")
                m3.metric("F1", f"{f1:.3f}")
                with st.expander("Confusion matrix [[TN, FP], [FN, TP]]"):
                    st.write([[int(tn), int(fp)], [int(fn), int(tp)]])
        else:
            # RandomForest (sklearn)
            rf, split, m = train_rf_sklearn(ml_df, feats, seed=42)
            Xtr, Xte, ytr, yte, proba_te = split
            c1, c2, c3 = st.columns(3)
            c1.metric("AUC (ROC)", f"{m['auc_roc']:.3f}")
            c2.metric("AUC (PR)",  f"{m['auc_pr']:.3f}")
            c3.metric("Accuracy",   f"{m['acc']:.3f}")

            # Prediction cho DN đang xem
            p_rf = None
            if 'sel' in locals() and 'sel_year' in locals():
                row = _company_row(df_exp, comp_col, sel, sel_year)
                if row is not None:
                    vec = np.array([[pd.to_numeric(row.get(f), errors="coerce") for f in feats]])
                    if np.isfinite(vec).all():
                        p_rf = float(rf.predict_proba(vec)[:,1][0])
            c3.metric("P(IG) – DN đang xem", f"{100*p_rf:.1f}%" if p_rf is not None else "N/A")

            # Importance (Permutation)
            pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=42)
            imp_df = pd.DataFrame({"feature": feats, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
            import altair as alt
            st.subheader("Feature Importance, SHAP & Model comparison")
            st.metric("RandomForest Accuracy (holdout)", f"{m['acc']:.3f}")
            ch_imp = alt.Chart(imp_df.head(15)).mark_bar().encode(
                x=alt.X("importance:Q", title="Importance"), y=alt.Y("feature:N", sort="-x", title="Feature")
            ).properties(height=340)
            st.altair_chart(ch_imp, use_container_width=True)

            # ROC / PR
            roc_ch, pr_ch = _altair_roc_pr(yte, proba_te)
            st.caption(f"AUC ROC: RF = {m['auc_roc']:.3f}  |  AUC PR: RF = {m['auc_pr']:.3f}")
            st.altair_chart(roc_ch, use_container_width=True)
            st.altair_chart(pr_ch, use_container_width=True)

            # Threshold slider + confusion matrix (RF)
            thr = st.slider("Ngưỡng phân loại (dựa trên proba RF)", 0.05, 0.95, 0.50, 0.01)
            (tn, fp, fn, tp), prec, rec, f1 = _metrics_at_threshold(yte, proba_te, thr)
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{prec:.3f}")
            m2.metric("Recall", f"{rec:.3f}")
            m3.metric("F1", f"{f1:.3f}")
            with st.expander("Confusion matrix [[TN, FP], [FN, TP]]"):
                st.write([[int(tn), int(fp)], [int(fn), int(tp)]])

            # SHAP (nếu có)
            if HAS_SHAP:
                try:
                    import shap, matplotlib.pyplot as plt
                    expl = shap.TreeExplainer(rf)
                    sample = np.random.RandomState(42).choice(len(Xtr), size=min(1000, len(Xtr)), replace=False)
                    sv = expl.shap_values(Xtr[sample])
                    fig = plt.figure()
                    shap.summary_plot(sv[1] if isinstance(sv, list) else sv, Xtr[sample],
                                    feature_names=feats, show=False, plot_type="dot", max_display=20)
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.info(f"Không vẽ được SHAP: {e}")

                # SHAP cho DN đang xem
                row = _company_row(df_exp, comp_col, sel, sel_year)
                if row is not None:
                    vec = np.array([[pd.to_numeric(row.get(f), errors="coerce") for f in feats]])
                    if np.isfinite(vec).all():
                        sv1 = expl.shap_values(vec)
                        vals = sv1[1][0] if isinstance(sv1, list) else sv1[0]
                        shap_df = pd.DataFrame({"feature": feats, "shap": vals, "abs_shap": np.abs(vals)})
                        with st.expander("🧭 Explain this company (SHAP)"):
                            st.dataframe(shap_df.sort_values("abs_shap", ascending=False).head(12))

            # Partial Dependence (tuỳ chọn)
            with st.expander("📈 Partial Dependence (top features)"):
                try:
                    from sklearn.inspection import PartialDependenceDisplay
                    import matplotlib.pyplot as plt
                    topk = imp_df.head(3)["feature"].tolist()
                    fig, ax = plt.subplots(1, len(topk), figsize=(4.2*len(topk), 3.2))
                    PartialDependenceDisplay.from_estimator(rf, Xtr, topk, ax=ax)
                    st.pyplot(fig, clear_figure=True)
                except Exception as e:
                    st.info(f"Không vẽ được PDP: {e}")

        st.caption("IG = {AAA, AA, A, BBB}. Kết quả phục vụ dashboard, không thay thế xếp hạng chính thức.")

        # ===== 4) Anomaly detection =====
        st.markdown("---")
        st.subheader("Phát hiện bất thường (IsolationForest)")

        col_iso1, col_iso2 = st.columns([1,1])
        with col_iso1:
            contam = st.slider(
                "Tỷ lệ bất thường kỳ vọng (contamination)",
                min_value=0.01, max_value=0.10, value=0.03, step=0.01,
                help="Ước lượng tỷ lệ điểm bất thường trong dữ liệu (1–10%)."
            )
        with col_iso2:
            top_n = st.slider(
                "Số dòng bất thường muốn xem (top-N)",
                min_value=5, max_value=50, value=10, step=5
            )

        # Bạn có thể reuse đúng bộ feature đang dùng cho ML/peer:
        feats_iso = feats  # hoặc danh sách feature người dùng chọn, nếu có biến đó

        iso_model = fit_isoforest(ml_df, feats_iso, contamination=contam)
        if iso_model is None:
            st.info("Không chạy được IsolationForest vì dữ liệu sau khi làm sạch quá ít. "
                    "Hãy chọn thêm feature hoặc nới lỏng điều kiện (giảm thiếu dữ liệu).")
        else:
            meta_cols = [c for c in ["Company Code", "Company", "Year"] if c in df_exp.columns]
            anom = rank_anomalies(iso_model, df_exp, feats_iso, meta_cols, top_n=top_n)
            if anom.empty:
                st.info("Không tạo được bảng bất thường (thiếu dữ liệu hợp lệ sau khi chuẩn hoá).")
            else:
                st.dataframe(anom.reset_index(drop=True))



if HAS_STREAMLIT:
    main_streamlit()
else:

    main_cli()





