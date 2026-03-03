"""
Enhanced XAI Stock Prediction Dashboard
--------------------------------------

This Streamlit application builds on the original thesis dashboard by
adding several improvements:

* Self-documenting plots: The predicted vs actual price plot now
  uses real dates on the x-axis and highlights train/validation/test
  regions using coloured spans.
* Robust SHAP visualisation: The SHAP summary is generated
  using the figure returned by `shap.summary_plot` to avoid empty
  figures, and a separate global bar chart shows the mean absolute
  importance of each lagged input feature.
* Explicit evaluation metrics: RMSE and MAE are reported for the
  train, validation and test sets in the sidebar for quick
  comparison.

To run this app on your own machine you will need to install the
dependencies: `streamlit`, `yfinance`, `tensorflow`, `shap` and
`scikit-learn`. Launch with `streamlit run RA_Thesis_full_improved.py`.

Multi-source reliability upgrade:
- Try Yahoo via yfinance
- If blocked: yfinance history() fallback
- If still blocked: Stooq (via pandas_datareader) for daily bars
- If everything fails: local CSV cache fallback
"""

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr  # pip install pandas_datareader

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt

import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap


# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="XAI Stock Dashboard")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        h1 { color: #1f77b4; }
        h2, h3 { color: #333; }
        .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Data loading and preprocessing (multi-source)
# -------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe = f"{ticker}_{period}_{interval}".replace("/", "-")
    return CACHE_DIR / f"{safe}.csv"


def _load_cached_single(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """Load a single-ticker cached CSV and return normal columns (Open/High/Low/Close/Volume)."""
    p = _cache_path(ticker, period, interval)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _save_cached_single(ticker: str, period: str, interval: str, df_single: pd.DataFrame) -> None:
    """Save a single-ticker DF (normal columns) to cache."""
    p = _cache_path(ticker, period, interval)
    try:
        df_single.to_csv(p)
    except Exception:
        pass


def _normalize_to_multiindex(ticker: str, df_single: pd.DataFrame) -> pd.DataFrame:
    """Convert single-ticker columns to MultiIndex columns (ticker, field)."""
    df_single = df_single.copy()
    df_single.columns = pd.MultiIndex.from_product([[ticker], df_single.columns])
    return df_single


def _fetch_yahoo_download(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """Try Yahoo via yf.download."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            group_by="column",
            auto_adjust=True,
            progress=False,
            threads=False,  # important
        )
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _fetch_yahoo_history(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """Fallback: Yahoo via yf.Ticker().history(). Often works when download() fails."""
    try:
        df = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _fetch_stooq_daily(ticker: str) -> pd.DataFrame | None:
    """
    Fallback: Stooq daily OHLCV via pandas_datareader.
    Tries ticker as-is and ticker+'.US' for US equities.
    Works best for interval='1d'. No API key required.
    """
    candidates = [ticker, f"{ticker}.US"]

    for sym in candidates:
        try:
            df = pdr.DataReader(sym, "stooq")
            if df is None or df.empty:
                continue

            df = df.sort_index()

            # Normalize columns to match yfinance style
            rename = {c: c.title() for c in df.columns}
            df = df.rename(columns=rename)

            needed = {"Open", "High", "Low", "Close"}
            if not needed.issubset(set(df.columns)):
                continue

            if "Volume" not in df.columns:
                df["Volume"] = np.nan

            return df[["Open", "High", "Low", "Close", "Volume"]].copy()

        except Exception:
            continue

    return None


@st.cache_data(show_spinner=False)
def download_data(
    tickers: list[str] | str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame | None:
    """
    Download historical data for one or more tickers using multiple sources.

    Priority:
    1) Yahoo (yf.download) with retries
    2) Yahoo fallback (Ticker().history) with retries
    3) Stooq (daily only; interval must be '1d')
    4) Local CSV cache (if previously saved)

    Returns:
      - DataFrame with MultiIndex columns: (TICKER, field)
      - None if everything fails for all tickers
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    tickers_clean = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers_clean:
        return None

    frames = []

    for ticker in tickers_clean:
        df_single = None

        # ---- Attempt 1: Yahoo download (retry) ----
        for attempt in range(3):
            df_single = _fetch_yahoo_download(ticker, period, interval)
            if df_single is not None and not df_single.empty:
                break
            time.sleep(1.2 * (attempt + 1))

        # ---- Attempt 2: Yahoo history fallback (retry) ----
        if df_single is None or df_single.empty:
            for attempt in range(3):
                df_single = _fetch_yahoo_history(ticker, period, interval)
                if df_single is not None and not df_single.empty:
                    break
                time.sleep(1.2 * (attempt + 1))

        # ---- Attempt 3: Stooq fallback (daily only) ----
        if (df_single is None or df_single.empty) and interval == "1d":
            df_try = _fetch_stooq_daily(ticker)
            if df_try is not None and not df_try.empty:
                df_single = df_try

        # ---- Attempt 4: Local cache ----
        if df_single is None or df_single.empty:
            cached = _load_cached_single(ticker, period, interval)
            if cached is not None and not cached.empty:
                df_single = cached

        # If still none, skip ticker
        if df_single is None or df_single.empty:
            continue

        # Clean + persist
        df_single = df_single.sort_index()
        df_single.ffill(inplace=True)
        df_single.dropna(inplace=True)

        _save_cached_single(ticker, period, interval, df_single)

        frames.append(_normalize_to_multiindex(ticker, df_single))

    if not frames:
        return None

    data = pd.concat(frames, axis=1)
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    return data


def get_close_col(data: pd.DataFrame, ticker: str) -> str:
    """Return the correct closing price column name for a given ticker."""
    for col in ["Adj Close", "Close"]:
        if (ticker, col) in data.columns:
            return col
    raise KeyError(
        f"No close price column found for {ticker}. Available: {data.columns.tolist()}"
    )


# -------------------------------------------------------------------
# Technical indicators
# -------------------------------------------------------------------
def calculate_indicators(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Add SMA, RSI and MACD technical indicators for each ticker."""
    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            continue

        close_col = get_close_col(data, ticker)
        close = data[(ticker, close_col)]

        # Simple moving averages
        data[(ticker, "SMA_30")] = close.rolling(window=30, min_periods=1).mean()
        data[(ticker, "SMA_60")] = close.rolling(window=60, min_periods=1).mean()

        # Relative Strength Index (RSI)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        data[(ticker, "RSI")] = 100 - (100 / (1 + rs))

        # Moving Average Convergence/Divergence (MACD)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        data[(ticker, "MACD")] = macd
        data[(ticker, "MACD_Signal")] = macd.ewm(span=9, adjust=False).mean()

    return data

# -------------------------------------------------------------------
# SMA crossover strategy for buy/sell signals
# -------------------------------------------------------------------
def sma_strategy(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a DataFrame with buy/sell signals based on SMA_30 vs SMA_60."""
    close_col = get_close_col(data, ticker)
    signals = pd.DataFrame(index=data.index)
    signals["signal"] = 0.0
    signals.loc[data[(ticker, "SMA_30")] > data[(ticker, "SMA_60")], "signal"] = 1.0
    signals["positions"] = signals["signal"].diff()
    return signals


# -------------------------------------------------------------------
# Helper to create supervised learning dataset from a 1D array
# -------------------------------------------------------------------
def create_dataset(dataset: np.ndarray, look_back: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Construct input/output sequences for LSTM from a univariate time series."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i : (i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# -------------------------------------------------------------------
# LSTM model training with train/val/test split
# -------------------------------------------------------------------
def train_lstm(
    df_scaled: np.ndarray,
    look_back: int = 10,
    epochs: int = 50,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    batch_size: int = 16,
    validation_split: tuple[float, float] = (0.70, 0.15),
) -> tuple:
    """Train an LSTM on the scaled series and return the model and splits."""
    X, Y = create_dataset(df_scaled, look_back)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    n = len(X_lstm)
    train_end = int(n * validation_split[0])
    val_end = int(n * sum(validation_split))
    X_train, y_train = X_lstm[:train_end], Y[:train_end]
    X_val, y_val = X_lstm[train_end:val_end], Y[train_end:val_end]
    X_test, y_test = X_lstm[val_end:], Y[val_end:]

    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    return model, X_train, y_train, X_val, y_val, X_test, y_test, X_lstm, Y, history


# -------------------------------------------------------------------
# SHAP calculation helper using DeepExplainer
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# SHAP calculation helper (DeepExplainer with safe fallback)
# -------------------------------------------------------------------
def calculate_shap_values(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Compute SHAP values for LSTM predictions.

    Strategy:
    1) Try SHAP DeepExplainer (fast, but can break on some TF/Keras versions)
    2) If DeepExplainer fails, fallback to KernelExplainer (slower, but robust)

    Returns:
      shap_values: (n_samples, look_back)
      X_flat:      (n_samples, look_back) flattened inputs used for SHAP
      method:      "deep" or "kernel"
    """
    # Small samples to keep runtime reasonable
    bg_n = min(50, len(X_train))
    n_samples = min(20, len(X_test))

    # Ensure float32 (important for TF + SHAP)
    background = X_train[:bg_n].astype(np.float32)
    sample = X_test[:n_samples].astype(np.float32)

    # -----------------------------
    # Attempt 1: DeepExplainer
    # -----------------------------
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(sample)

        sv = np.array(shap_values).squeeze()

        # Normalize shape -> (n_samples, look_back)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        if sv.shape[0] != n_samples and sv.shape[-1] == n_samples:
            sv = sv.T

        if sv.shape[-1] != look_back:
            sv = sv.reshape(n_samples, look_back)

        X_flat = sample.reshape(n_samples, look_back)
        return sv, X_flat, "deep"

    except Exception:
        # -----------------------------
        # Attempt 2: KernelExplainer fallback
        # -----------------------------
        background_flat = background.reshape(bg_n, look_back)
        sample_flat = sample.reshape(n_samples, look_back)

        def predict_from_flat(x_flat: np.ndarray) -> np.ndarray:
            x = x_flat.reshape((-1, look_back, 1)).astype(np.float32)
            preds = model.predict(x, verbose=0).reshape(-1)
            return preds

        explainer = shap.KernelExplainer(predict_from_flat, background_flat)

        # nsamples controls speed/quality tradeoff
        sv = explainer.shap_values(sample_flat, nsamples=100)

        # KernelExplainer may return a list
        if isinstance(sv, list):
            sv = sv[0]

        sv = np.array(sv)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        return sv, sample_flat, "kernel"

# -------------------------------------------------------------------
# Plotting routines
# -------------------------------------------------------------------
def plot_stock_signals(data: pd.DataFrame, ticker: str) -> None:
    """Display price with SMA crossover signals, RSI and MACD indicators."""
    close_col = get_close_col(data, ticker)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    # Price & SMAs with signals
    ax = axes[0]
    ax.plot(data.index, data[(ticker, close_col)], label=f"{ticker} Close", color="#1f77b4", linewidth=1.5)
    ax.plot(data.index, data[(ticker, "SMA_30")], label="SMA 30", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.plot(data.index, data[(ticker, "SMA_60")], label="SMA 60", color="#2ca02c", linestyle="--", linewidth=1.5)
    signals = sma_strategy(data, ticker)
    buy_signals = signals.positions == 1.0
    sell_signals = signals.positions == -1.0
    ax.plot(data.index[buy_signals], data[(ticker, close_col)][buy_signals], "^", color="green", label="Buy", markersize=10)
    ax.plot(data.index[sell_signals], data[(ticker, close_col)][sell_signals], "v", color="red", label="Sell", markersize=10)
    ax.set_title(f"{ticker} — Price & SMA Strategy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4)
    # RSI
    ax2 = axes[1]
    ax2.plot(data.index, data[(ticker, "RSI")], color="purple", linewidth=1.2)
    ax2.axhline(70, color="red", linestyle="--", linewidth=0.8, label="Overbought (70)")
    ax2.axhline(30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", linewidth=0.4)
    # MACD
    ax3 = axes[2]
    ax3.plot(data.index, data[(ticker, "MACD")], color="blue", linewidth=1.2, label="MACD")
    ax3.plot(data.index, data[(ticker, "MACD_Signal")], color="orange", linewidth=1.2, label="Signal")
    ax3.axhline(0, color="black", linewidth=0.6)
    ax3.set_ylabel("MACD")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_predicted_vs_actual(
    Y_all: np.ndarray,
    predictions_all: np.ndarray,
    scaler: MinMaxScaler,
    split_indices: tuple[int, int],
    dates: pd.Index,
) -> None:
    """
    Show predicted vs actual prices on a single plot with train/val/test zones highlighted.

    The dates parameter is used to label the x‑axis. `split_indices` should be
    a tuple (train_end, val_end) specifying the boundaries in index units.
    """
    train_end, val_end = split_indices
    actual_prices = scaler.inverse_transform(Y_all.reshape(-1, 1)).flatten()
    pred_prices = scaler.inverse_transform(predictions_all.reshape(-1, 1)).flatten()
    date_index = dates[-len(actual_prices) :]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axvspan(date_index[0], date_index[train_end], alpha=0.07, color="blue", label="Train (70%)")
    ax.axvspan(date_index[train_end], date_index[val_end], alpha=0.07, color="orange", label="Validation (15%)")
    ax.axvspan(date_index[val_end], date_index[-1], alpha=0.07, color="green", label="Test (15%)")
    ax.plot(date_index, actual_prices, label="Actual Price", color="#1f77b4", linewidth=1.5)
    ax.plot(date_index, pred_prices, label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.set_title("Predicted vs Actual Price — Train / Validation / Test Split", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_training_loss(history) -> None:
    """Plot training and validation loss curves on one chart."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Train Loss", color="blue")
    ax.plot(history.history["val_loss"], label="Val Loss", color="orange")
    ax.set_title("Model Training Loss (with Early Stopping)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_shap_summary(shap_values: np.ndarray, X_sample: np.ndarray, feature_names: list[str]) -> None:
    """
    Display the SHAP summary plot and a bar chart of mean absolute SHAP values.
    """
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False, plot_type="dot")
    fig_dot = plt.gcf()
    fig_dot.suptitle("SHAP Summary — Feature Importance Across Test Predictions", fontsize=13, fontweight="bold")
    st.pyplot(fig_dot)
    plt.close(fig_dot)
    shap_sum = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(shap_sum)[::-1]
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    ax_bar.bar(range(len(shap_sum)), shap_sum[indices], color="steelblue")
    ax_bar.set_xticks(range(len(shap_sum)))
    ax_bar.set_xticklabels([feature_names[i] for i in indices], rotation=45)
    ax_bar.set_ylabel("Mean |SHAP value|")
    ax_bar.set_title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)


def plot_shap_instance(shap_values: np.ndarray, X_sample: np.ndarray, feature_names: list[str], instance_idx: int) -> None:
    """Plot a horizontal bar chart explaining a single prediction instance."""
    sv = shap_values[instance_idx]
    vals = X_sample[instance_idx]
    colours = ["#2ca02c" if v >= 0 else "#d62728" for v in sv]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feature_names, sv, color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on model output)")
    ax.set_title(
        f"Per‑Prediction SHAP Explanation — Test Sample {instance_idx + 1}\n"
        f"Green = pushes price UP  |  Red = pushes price DOWN",
        fontsize=12,
        fontweight="bold",
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2, f"  input={val:.3f}", va="center", fontsize=8, color="black")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------------
# Streamlit app layout
# -------------------------------------------------------------------
st.title("📈 XAI Stock Prediction Dashboard")
st.caption("LSTM forecasting with explainability via SHAP DeepExplainer — Enhanced Version")

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_input("Stock Tickers (comma‑separated)", value="AAPL")
    period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Data Interval", ["1d", "1wk"], index=0)
    st.markdown("---")
    st.subheader("🔧 Model Hyperparameters")
    look_back = st.slider("Look‑back Window (days)", 5, 30, 10)
    epochs = st.slider("Max Epochs", 10, 100, 50)
    lstm_units = st.select_slider("LSTM Units", options=[32, 64, 128], value=64)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
    batch_size = st.slider("Batch Size", 8, 64, 16, step=8)
    st.markdown("---")
    st.info(
        "ℹ️ **Train / Val / Test split:** 70 / 15 / 15 %\n\nEarly stopping monitors validation loss."
    )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Please enter at least one valid stock ticker.")
        st.stop()

    data = download_data(tickers, period, interval)
    # Handle cases where the download fails or returns no rows
    if data is None or data.empty:
        st.error(
            "No data was returned for the selected tickers. This may be due to an invalid symbol "
            "or a network issue. Please verify your internet connection or try a different ticker."
        )
        st.stop()
    data = calculate_indicators(data, tickers)

selected_ticker = st.selectbox("Select Stock for Detailed Analysis", tickers)

tab1, tab2, tab3 = st.tabs(["📊 Price & Signals", "🤖 LSTM Prediction", "🔍 XAI / SHAP"])

with tab1:
    st.subheader(f"{selected_ticker} — Technical Analysis")
    plot_stock_signals(data, selected_ticker)
    st.caption("SMA crossover signals + RSI (overbought/oversold) + MACD momentum indicator.")

close_col = get_close_col(data, selected_ticker)
df_close = data[(selected_ticker, close_col)].values.reshape(-1, 1)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_close)

if len(df_scaled) <= look_back + 20:
    with tab2:
        st.warning("Not enough data points. Try a longer period or shorter look‑back.")
    with tab3:
        st.warning("Not enough data for SHAP analysis.")
    st.stop()

with st.spinner("Training LSTM model… this may take a moment ☕"):
    (
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_all,
        Y_all,
        history,
    ) = train_lstm(
        df_scaled,
        look_back=look_back,
        epochs=epochs,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
    )

n = len(X_all)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

pred_all = model.predict(X_all, verbose=0)
pred_train = model.predict(X_train, verbose=0)
pred_val = model.predict(X_val, verbose=0)
pred_test = model.predict(X_test, verbose=0)

rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
mae_train = mean_absolute_error(y_train, pred_train)
rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
mae_val = mean_absolute_error(y_val, pred_val)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test = mean_absolute_error(y_test, pred_test)

feature_names = [f"Day -{look_back - i}" for i in range(look_back)]

with tab2:
    st.subheader("LSTM — Predicted vs Actual Price")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Train RMSE", f"{rmse_train:.4f}")
    col2.metric("Train MAE", f"{mae_train:.4f}")
    col3.metric("Val RMSE", f"{rmse_val:.4f}")
    col4.metric("Val MAE", f"{mae_val:.4f}")
    col5.metric("Test RMSE", f"{rmse_test:.4f}")
    col6.metric("Test MAE", f"{mae_test:.4f}")
    st.caption("⚠️ Metrics reported separately for train, validation and test sets. Lower = better.")
    plot_predicted_vs_actual(Y_all, pred_all.flatten(), scaler, (train_end, val_end), dates=data.index)
    st.markdown("---")
    st.subheader("Training Loss Curve")
    plot_training_loss(history)
    st.caption("Validation loss guides early stopping. Divergence between curves indicates overfitting.")

with tab3:
    st.subheader("XAI — SHAP Explainability (DeepExplainer)")
    st.markdown(
        """
        **What are SHAP values?**  
        SHAP (SHapley Additive exPlanations) measures how much each past day's price
        contributed to the model's prediction. A **positive SHAP value** pushes the
        predicted price **up**; a **negative value** pushes it **down**.  

        > ⚠️ SHAP values reveal **correlation**, not causation. These are patterns the model
        learned from historical data — not guarantees of future performance.
        """
    )
    if len(X_test) < 2:
        st.warning("Not enough test samples for SHAP. Try a longer time period.")
        st.stop()
    with st.spinner("Computing SHAP values…"):
        try:
            shap_values, X_sample, method = calculate_shap_values(model, X_train, X_test, look_back)
            if method == "kernel":
                st.info("DeepExplainer failed on this setup. Using KernelExplainer fallback (slower but compatible).")
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            st.stop()
    st.subheader("Global Feature Importance")
    plot_shap_summary(shap_values, X_sample, feature_names)
    st.markdown("---")
    st.subheader("Per‑Prediction Explanation (Single Instance)")
    max_idx = len(shap_values) - 1
    instance_idx = st.slider("Select test prediction to explain", 0, max_idx, 0)
    plot_shap_instance(shap_values, X_sample, feature_names, instance_idx)
    st.caption(
        "Each bar shows how much that day's price pushed the prediction up (green) or down (red).\n"
        "Input value shown to the right of each bar."
    )
    st.markdown("---")
    st.subheader("SHAP Heatmap — All Test Predictions")
    fig_heat, ax_heat = plt.subplots(figsize=(12, max(4, len(shap_values) // 2)))
    im = ax_heat.imshow(shap_values, aspect="auto", cmap="RdYlGn")
    ax_heat.set_xticks(range(look_back))
    ax_heat.set_xticklabels(feature_names, rotation=45, ha="right")
    ax_heat.set_yticks(range(len(shap_values)))
    ax_heat.set_yticklabels([f"Sample {i + 1}" for i in range(len(shap_values))])
    ax_heat.set_title("SHAP Value Heatmap — Test Set", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, label="SHAP Value")
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)

st.markdown("---")
st.caption("📌 This dashboard is for **educational and research purposes only**. Not financial advice.")