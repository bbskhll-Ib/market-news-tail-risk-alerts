import os
import re
import json
import random
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
import statsmodels.api as sm


# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
START_DATE = "2008-01-01"
END_DATE = "2024-03-04"
TRAIN_END = "2018-12-31"
CALIB_START = "2017-01-01"
CAL_METHOD = "sigmoid"

NEWS_FILE = "sp500_headlines_2008_2024.csv"
OUT_DIR = "outputs_best_corrected"
os.makedirs(OUT_DIR, exist_ok=True)

H_MAIN = 5
Q_MAIN = 0.85
FPR_MAIN = 0.05

INDEX_SYMBOLS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
}

FINANCIAL_KEYWORDS = [
    "stock", "market", "fed", "inflation", "rate", "earnings",
    "bond", "economy", "recession", "volatility", "bank",
]

BOOT_BLOCK_LEN = 20
BOOT_B = 800

FEATURES_MKT = ["rolling_vol_21d", "vix_close", "negative_return_flag"]
FEATURES_CNT = FEATURES_MKT + ["has_news_lag1", "log_news_count_lag1"]
FEATURES_FULL_MEAN = FEATURES_CNT + ["neg_prob_mean_lag1"]
FEATURES_FULL_MAX = FEATURES_CNT + ["neg_prob_max_lag1"]
FEATURES_FULL_P90 = FEATURES_CNT + ["neg_prob_p90_lag1"]
HAR_X = ["rv_lag1", "rv_week", "rv_month"]

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# REPRODUCIBILITY METADATA
# ============================================================
def write_run_metadata():
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "train_inner_end": "2016-12-31",
        "calibration_start": CALIB_START,
        "calibration_end": TRAIN_END,
        "horizon": H_MAIN,
        "quantile": Q_MAIN,
        "fpr_target": FPR_MAIN,
        "bootstrap_method": "circular moving-block bootstrap",
        "bootstrap_block_length": BOOT_BLOCK_LEN,
        "bootstrap_replicates": BOOT_B,
        "calibration_method": CAL_METHOD,
        "finbert_model": "ProsusAI/finbert",
        "torch_version": torch.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    with open(os.path.join(OUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ============================================================
# DATA AND FEATURES
# ============================================================
def forward_rv(ret, h):
    squared_future_returns = pd.concat(
        [(ret.shift(-i) ** 2) for i in range(1, h + 1)], axis=1
    )
    return np.sqrt(squared_future_returns.sum(axis=1))


def build_market(symbol, start=START_DATE, end=END_DATE):
    end_exclusive = (
        pd.to_datetime(end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    prices = yf.download(symbol, start=start, end=end_exclusive, progress=False)
    vix = yf.download("^VIX", start=start, end=end_exclusive, progress=False)

    df = pd.concat([prices["Close"], vix["Close"]], axis=1)
    df.columns = ["close", "vix_close"]
    df["vix_close"] = df["vix_close"].ffill()

    df["ret"] = df["close"].pct_change()
    df["rolling_vol_21d"] = df["ret"].rolling(21).std()
    df["negative_return_flag"] = (df["ret"] < 0).astype(int)
    df["rv_daily"] = np.abs(df["ret"])

    df = df.dropna().reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def load_news(news_file=NEWS_FILE):
    if not os.path.exists(news_file):
        raise FileNotFoundError(f"News file not found: {news_file}")

    news = pd.read_csv(news_file)
    required_columns = {"Date", "Title"}
    if not required_columns.issubset(news.columns):
        raise ValueError(
            f"CSV must contain {required_columns}. Found: {list(news.columns)}"
        )

    news["Date"] = pd.to_datetime(news["Date"], errors="coerce").dt.normalize()
    news["Title"] = news["Title"].fillna("").astype(str)
    news = news.dropna(subset=["Date"])
    news = news.drop_duplicates(subset=["Date", "Title"])

    return news.sort_values("Date").reset_index(drop=True)


def filter_financial_news(news_df):
    pattern = "|".join(FINANCIAL_KEYWORDS)
    mask = news_df["Title"].str.lower().str.contains(pattern, regex=True, na=False)
    return news_df.loc[mask].copy().reset_index(drop=True)


def compute_finbert_negprob(news_df, cache_path=None, batch_size=64):
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
        cached["Title"] = cached["Title"].astype(str)
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "ProsusAI/finbert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    label_map = {str(k): str(v) for k, v in model.config.id2label.items()}
    print("FinBERT id2label:", label_map)

    negative_index = None
    for index, label in model.config.id2label.items():
        if str(label).lower() == "negative":
            negative_index = int(index)
            break

    if negative_index is None:
        raise ValueError(
            "Could not identify the negative FinBERT class from model.config.id2label. "
            f"Observed mapping: {label_map}"
        )

    texts = news_df["Title"].tolist()
    neg_probs = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().cpu().numpy()
            probabilities = softmax(logits, axis=1)
            neg_probs.extend(probabilities[:, negative_index])

    scored = news_df[["Date", "Title"]].copy()
    scored["neg_prob"] = neg_probs

    if cache_path:
        scored.to_parquet(cache_path, index=False)

    return scored


def map_news_to_trading_days(news_scored, trading_days):
    """Assign a dated headline to the first trading session on or after its date.

    Headlines after the final available trading session are excluded rather than being
    incorrectly assigned to the final in-sample trading day.
    """
    calendar_dates = pd.to_datetime(
        news_scored["Date"]
    ).values.astype("datetime64[D]")
    market_dates = pd.to_datetime(
        trading_days
    ).values.astype("datetime64[D]")

    indices = np.searchsorted(market_dates, calendar_dates, side="left")
    valid = indices < len(market_dates)

    mapped = news_scored.loc[valid].copy()
    mapped["TradeDate"] = pd.to_datetime(market_dates[indices[valid]])
    return mapped


def daily_news_agg(mapped_news):
    def q90(values):
        return values.quantile(0.9)

    daily = (
        mapped_news.groupby("TradeDate")
        .agg(
            news_count=("neg_prob", "size"),
            neg_prob_mean=("neg_prob", "mean"),
            neg_prob_max=("neg_prob", "max"),
            neg_prob_p90=("neg_prob", q90),
        )
        .reset_index()
        .rename(columns={"TradeDate": "Date"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    return daily


def add_har_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["rv_lag1"] = df["rv_daily"].shift(1)
    df["rv_week"] = df["rv_daily"].rolling(5).mean().shift(1)
    df["rv_month"] = df["rv_daily"].rolling(22).mean().shift(1)
    return df


def build_dataset_for_index(symbol, h, news_scored):
    market = build_market(symbol)
    mapped_news = map_news_to_trading_days(news_scored, market["Date"].values)
    daily_news = daily_news_agg(mapped_news)

    df = market.merge(daily_news, on="Date", how="left").fillna(0)

    df["has_news_lag1"] = (df["news_count"].shift(1) > 0).astype(int)
    df["log_news_count_lag1"] = np.log1p(df["news_count"].shift(1))
    df["neg_prob_mean_lag1"] = df["neg_prob_mean"].shift(1).fillna(0)
    df["neg_prob_max_lag1"] = df["neg_prob_max"].shift(1).fillna(0)
    df["neg_prob_p90_lag1"] = df["neg_prob_p90"].shift(1).fillna(0)

    df = add_har_features(df)
    df["future_rv"] = forward_rv(df["ret"], h)

    return df.dropna().reset_index(drop=True)


def time_split_train_calib_test(df, train_end=TRAIN_END, calib_start=CALIB_START):
    df = df.sort_values("Date").reset_index(drop=True)
    train_end = pd.to_datetime(train_end)
    calib_start = pd.to_datetime(calib_start)

    train_inner = df[df["Date"] < calib_start].copy()
    calibration = df[
        (df["Date"] >= calib_start) & (df["Date"] <= train_end)
    ].copy()
    test = df[df["Date"] > train_end].copy()
    return train_inner, calibration, test


# ============================================================
# MODELS AND CALIBRATION
# ============================================================
def make_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ])


def make_lgbm():
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_base_and_predict(estimator, x_train, y_train, x_calib, x_test):
    estimator.fit(x_train, y_train)
    p_calib = estimator.predict_proba(x_calib)[:, 1]
    p_test = estimator.predict_proba(x_test)[:, 1]
    return estimator, p_calib, p_test


def fit_prefit_calibrator(estimator_fitted, x_calib, y_calib, method=CAL_METHOD):
    calibrator = CalibratedClassifierCV(
        estimator_fitted,
        method=method,
        cv="prefit",
    )
    calibrator.fit(x_calib, y_calib)
    return calibrator


def fit_predict_har_ols(train_df, test_df):
    x_train = sm.add_constant(train_df[HAR_X])
    y_train = train_df["future_rv"].values
    model = sm.OLS(y_train, x_train).fit()

    x_test = sm.add_constant(test_df[HAR_X])
    return model.predict(x_test), model


# ============================================================
# METRICS
# ============================================================
def compute_ece(y_true, probability, n_bins=10):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probability, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        empirical_rate = y_true[mask].mean()
        mean_probability = probability[mask].mean()
        ece += mask.mean() * abs(empirical_rate - mean_probability)
    return float(ece)


def threshold_at_fpr(y_true, probability, target_fpr=FPR_MAIN):
    fpr, _, thresholds = roc_curve(y_true, probability)
    admissible = np.where(fpr <= target_fpr)[0]
    if len(admissible) == 0:
        return 1.0
    return thresholds[admissible[-1]]


def recall_at_fpr_from_reference(
    y_reference,
    p_reference,
    y_evaluation,
    p_evaluation,
    target_fpr=FPR_MAIN,
):
    threshold = threshold_at_fpr(y_reference, p_reference, target_fpr)
    alerts = (p_evaluation >= threshold).astype(int)
    true_positive = ((y_evaluation == 1) & (alerts == 1)).sum()
    false_negative = ((y_evaluation == 1) & (alerts == 0)).sum()
    denominator = true_positive + false_negative
    return float(true_positive / denominator) if denominator > 0 else np.nan


def eval_prob_metrics(y_reference, p_reference, y_evaluation, p_evaluation):
    if len(np.unique(y_evaluation)) < 2:
        return {
            "AUC": np.nan,
            "PR_AUC": np.nan,
            "Recall@FPR": np.nan,
            "Brier": np.nan,
            "ECE": np.nan,
        }

    return {
        "AUC": roc_auc_score(y_evaluation, p_evaluation),
        "PR_AUC": average_precision_score(y_evaluation, p_evaluation),
        "Recall@FPR": recall_at_fpr_from_reference(
            y_reference,
            p_reference,
            y_evaluation,
            p_evaluation,
            target_fpr=FPR_MAIN,
        ),
        "Brier": brier_score_loss(y_evaluation, p_evaluation),
        "ECE": compute_ece(y_evaluation, p_evaluation),
    }


def metric_auc(y, probability):
    return roc_auc_score(y, probability) if len(np.unique(y)) > 1 else np.nan


def metric_prauc(y, probability):
    return average_precision_score(y, probability) if len(np.unique(y)) > 1 else np.nan


# ============================================================
# CIRCULAR MOVING-BLOCK BOOTSTRAP
# ============================================================
def moving_block_bootstrap_indices(n, block_len, rng):
    indices = []
    while len(indices) < n:
        start = rng.integers(0, n)
        block = (start + np.arange(block_len)) % n
        indices.extend(block.tolist())
    return np.asarray(indices[:n])


def bootstrap_delta_ci(
    y,
    p_news,
    p_mkt,
    metric_func,
    block_len=BOOT_BLOCK_LEN,
    b_replicates=BOOT_B,
    seed=SEED,
):
    y = np.asarray(y)
    p_news = np.asarray(p_news)
    p_mkt = np.asarray(p_mkt)

    point_estimate = metric_func(y, p_news) - metric_func(y, p_mkt)

    rng = np.random.default_rng(seed)
    bootstrap_deltas = []
    n = len(y)

    for _ in range(b_replicates):
        resample_index = moving_block_bootstrap_indices(n, block_len, rng)
        delta = (
            metric_func(y[resample_index], p_news[resample_index])
            - metric_func(y[resample_index], p_mkt[resample_index])
        )
        if np.isfinite(delta):
            bootstrap_deltas.append(delta)

    bootstrap_deltas = np.asarray(bootstrap_deltas)

    if len(bootstrap_deltas) < 50:
        return float(point_estimate), (np.nan, np.nan)

    ci_low, ci_high = np.percentile(bootstrap_deltas, [2.5, 97.5])
    return float(point_estimate), (float(ci_low), float(ci_high))


# ============================================================
# EXPERIMENT PIPELINE
# ============================================================
def run_protocol_for_index(index_name, symbol, h, q, news_scored):
    news_scored = news_scored.drop_duplicates(subset=["Date", "Title"]).copy()

    df = build_dataset_for_index(symbol, h, news_scored)
    train_inner, calibration, test = time_split_train_calib_test(df)

    threshold = train_inner["future_rv"].quantile(q)
    y_train = (train_inner["future_rv"] > threshold).astype(int).values
    y_calib = (calibration["future_rv"] > threshold).astype(int).values
    y_test = (test["future_rv"] > threshold).astype(int).values

    print(
        f"{index_name}: train_inner={len(train_inner)}, "
        f"calib={len(calibration)}, test={len(test)}, "
        f"test_dates={test['Date'].min().date()} to {test['Date'].max().date()}, "
        f"event_rate_test={y_test.mean():.3f}"
    )

    variants = {
        "MKT": FEATURES_MKT,
        "CNT": FEATURES_CNT,
        "MEAN": FEATURES_FULL_MEAN,
        "MAX": FEATURES_FULL_MAX,
        "P90": FEATURES_FULL_P90,
    }
    model_factories = {
        "Logistic": make_logistic,
        "LightGBM": make_lgbm,
    }

    absolute_rows = []
    delta_rows = []

    for model_name, model_factory in model_factories.items():
        stored_predictions = {"uncal": {}, "cal": {}}

        for variant_name, feature_names in variants.items():
            estimator = model_factory()
            fitted_estimator, p_calib, p_test = fit_base_and_predict(
                estimator,
                train_inner[feature_names],
                y_train,
                calibration[feature_names],
                test[feature_names],
            )

            stored_predictions["uncal"][variant_name] = (p_calib, p_test)
            metrics = eval_prob_metrics(y_calib, p_calib, y_test, p_test)
            absolute_rows.append({
                "Index": index_name,
                "Model": model_name,
                "Calib": "uncal",
                "Variant": variant_name,
                "H": h,
                "Q": q,
                **metrics,
                "EventRate_Test": float(y_test.mean()),
                "N_Test": len(y_test),
            })

            calibrator = fit_prefit_calibrator(
                fitted_estimator,
                calibration[feature_names],
                y_calib,
                method=CAL_METHOD,
            )
            p_calib_cal = calibrator.predict_proba(calibration[feature_names])[:, 1]
            p_test_cal = calibrator.predict_proba(test[feature_names])[:, 1]

            stored_predictions["cal"][variant_name] = (p_calib_cal, p_test_cal)
            calibrated_metrics = eval_prob_metrics(
                y_calib,
                p_calib_cal,
                y_test,
                p_test_cal,
            )
            absolute_rows.append({
                "Index": index_name,
                "Model": model_name,
                "Calib": f"cal_{CAL_METHOD}",
                "Variant": variant_name,
                "H": h,
                "Q": q,
                **calibrated_metrics,
                "EventRate_Test": float(y_test.mean()),
                "N_Test": len(y_test),
            })

        for tag, tag_label in [("uncal", "uncal"), ("cal", f"cal_{CAL_METHOD}")]:
            p_market_test = stored_predictions[tag]["MKT"][1]

            for comparator in ["CNT", "MEAN", "MAX", "P90"]:
                p_news_test = stored_predictions[tag][comparator][1]

                delta_auc, ci_auc = bootstrap_delta_ci(
                    y_test,
                    p_news_test,
                    p_market_test,
                    metric_auc,
                )
                delta_prauc, ci_prauc = bootstrap_delta_ci(
                    y_test,
                    p_news_test,
                    p_market_test,
                    metric_prauc,
                )

                delta_rows.append({
                    "Index": index_name,
                    "Model": model_name,
                    "Calib": tag_label,
                    "Compare": f"{comparator}-MKT",
                    "DELTA_AUC": delta_auc,
                    "CI_AUC_LO": ci_auc[0],
                    "CI_AUC_HI": ci_auc[1],
                    "DELTA_PRAUC": delta_prauc,
                    "CI_PRAUC_LO": ci_prauc[0],
                    "CI_PRAUC_HI": ci_prauc[1],
                    "H": h,
                    "Q": q,
                    "N_Test": len(y_test),
                })

    har_score, _ = fit_predict_har_ols(train_inner, test)
    absolute_rows.append({
        "Index": index_name,
        "Model": "HAR(OLS_score)",
        "Calib": "NA",
        "Variant": "HAR",
        "H": h,
        "Q": q,
        "AUC": metric_auc(y_test, har_score),
        "PR_AUC": metric_prauc(y_test, har_score),
        "Recall@FPR": np.nan,
        "Brier": np.nan,
        "ECE": np.nan,
        "EventRate_Test": float(y_test.mean()),
        "N_Test": len(y_test),
    })

    return pd.DataFrame(absolute_rows), pd.DataFrame(delta_rows)


def run_all_experiments():
    write_run_metadata()

    news_all = load_news(NEWS_FILE)
    news_filtered = filter_financial_news(news_all)

    print("Unique cleaned unfiltered headlines:", len(news_all))
    print("Unique cleaned keyword-filtered headlines:", len(news_filtered))

    cache_path = os.path.join(OUT_DIR, "news_finbert_ALL_unique.parquet")
    news_scored_all = compute_finbert_negprob(
        news_all,
        cache_path=cache_path,
        batch_size=64,
    )
    news_scored_all["Date"] = pd.to_datetime(news_scored_all["Date"]).dt.normalize()
    news_scored_all["Title"] = news_scored_all["Title"].astype(str)
    news_scored_all = news_scored_all.drop_duplicates(subset=["Date", "Title"])

    filtered_keys = news_filtered[["Date", "Title"]].drop_duplicates()
    news_scored_filtered = (
        news_scored_all.merge(filtered_keys, on=["Date", "Title"], how="inner")
        .drop_duplicates(subset=["Date", "Title"])
        .reset_index(drop=True)
    )

    print("Scored FILTER_OFF headlines:", len(news_scored_all))
    print("Scored FILTER_ON headlines:", len(news_scored_filtered))

    runs = {
        "FILTER_OFF": news_scored_all,
        "FILTER_ON": news_scored_filtered,
    }

    for run_name, scored_news in runs.items():
        print(f"\n========== Running {run_name} ==========")

        all_absolute = []
        all_delta = []

        for index_name, symbol in INDEX_SYMBOLS.items():
            absolute_df, delta_df = run_protocol_for_index(
                index_name=index_name,
                symbol=symbol,
                h=H_MAIN,
                q=Q_MAIN,
                news_scored=scored_news,
            )
            absolute_df["RUN"] = run_name
            delta_df["RUN"] = run_name
            all_absolute.append(absolute_df)
            all_delta.append(delta_df)

        absolute_results = pd.concat(all_absolute, ignore_index=True)
        delta_results = pd.concat(all_delta, ignore_index=True)

        absolute_path = os.path.join(
            OUT_DIR,
            f"best_protocol_absolute_{run_name}.csv",
        )
        delta_path = os.path.join(
            OUT_DIR,
            f"best_protocol_deltas_{run_name}.csv",
        )

        absolute_results.to_csv(absolute_path, index=False)
        delta_results.to_csv(delta_path, index=False)

        print(f"Saved: {absolute_path}")
        print(f"Saved: {delta_path}")

        print("\nLightGBM uncalibrated absolute results:")
        print(
            absolute_results.loc[
                (absolute_results["Model"] == "LightGBM")
                & (absolute_results["Calib"] == "uncal"),
                [
                    "Index", "Variant", "AUC", "PR_AUC", "Recall@FPR",
                    "Brier", "ECE", "EventRate_Test", "N_Test",
                ],
            ].to_string(index=False)
        )

        print("\nLightGBM uncalibrated incremental results:")
        print(
            delta_results.loc[
                (delta_results["Model"] == "LightGBM")
                & (delta_results["Calib"] == "uncal"),
                [
                    "Index", "Compare", "DELTA_AUC", "CI_AUC_LO", "CI_AUC_HI",
                    "DELTA_PRAUC", "CI_PRAUC_LO", "CI_PRAUC_HI",
                ],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    run_all_experiments()
