from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    # Streamlit Secrets -> normal env vars for old model code
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[str(k)] = str(v)
    except Exception:
        pass

    load_dotenv(ROOT / ".env", override=False)


def money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def pct(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        val = 0.0

    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def clean_chart(df: pd.DataFrame, trades: pd.DataFrame | None = None) -> go.Figure:
    """
    Main chart helper.

    Supports two chart types:

    1. Portfolio equity chart:
       df columns: time, equity
       trade markers use trades["equity"]

    2. Individual ticker price-path chart:
       df columns: time, close, open, high, low, volume
       trade markers use trades["price"]

    This fixes the old problem where the graph only showed BUY/SELL points.
    Now the chart can show the actual stock movement while the trade was open.
    """

    fig = go.Figure()

    if df is None or df.empty:
        fig.add_trace(go.Scatter(x=[], y=[]))
        _apply_chart_layout(fig)
        return fig

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    is_price_path = "close" in df.columns
    is_equity_path = "equity" in df.columns

    if is_price_path:
        y_col = "close"
        chart_name = "Stock Price"
        hovertemplate = "$%{y:,.2f}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[y_col],
                mode="lines",
                line={
                    "width": 4,
                    "color": "#00e676",
                    "shape": "spline",
                    "smoothing": 1.2,
                },
                fill="tozeroy",
                fillcolor="rgba(0,230,118,0.10)",
                hovertemplate=hovertemplate,
                name=chart_name,
            )
        )

        # Optional high/low range shading, if available
        if {"high", "low"}.issubset(df.columns):
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["high"],
                    mode="lines",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    name="High",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["low"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(255,255,255,0.045)",
                    hoverinfo="skip",
                    showlegend=False,
                    name="Low",
                )
            )

    elif is_equity_path:
        y_col = "equity"
        chart_name = "Net Worth"
        hovertemplate = "$%{y:,.2f}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[y_col],
                mode="lines",
                line={
                    "width": 4,
                    "color": "#00e676",
                    "shape": "spline",
                    "smoothing": 1.2,
                },
                fill="tozeroy",
                fillcolor="rgba(0,230,118,0.12)",
                hovertemplate=hovertemplate,
                name=chart_name,
            )
        )

    else:
        fig.add_trace(go.Scatter(x=[], y=[]))
        _apply_chart_layout(fig)
        return fig

    if trades is not None and not trades.empty:
        marker_df = trades.copy()
        marker_df["time"] = pd.to_datetime(marker_df["time"], errors="coerce")
        marker_df = marker_df.dropna(subset=["time"])

        if "side" in marker_df.columns:
            marker_df["side"] = marker_df["side"].astype(str).str.lower()
        else:
            marker_df["side"] = ""

        if is_price_path:
            marker_y_col = "price"
        else:
            marker_y_col = "equity"

        if marker_y_col in marker_df.columns:
            buys = marker_df[marker_df["side"] == "buy"].copy()
            sells = marker_df[marker_df["side"] == "sell"].copy()

            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys["time"],
                        y=buys[marker_y_col],
                        mode="markers",
                        marker={
                            "size": 13,
                            "symbol": "triangle-up",
                            "color": "#00e676",
                            "line": {"width": 1, "color": "#06110a"},
                        },
                        customdata=_marker_customdata(buys),
                        hovertemplate=(
                            "BUY %{customdata[0]}<br>"
                            "Qty: %{customdata[1]}<br>"
                            "Price: $%{customdata[2]:,.2f}<extra></extra>"
                        ),
                        name="Buy",
                    )
                )

            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells["time"],
                        y=sells[marker_y_col],
                        mode="markers",
                        marker={
                            "size": 13,
                            "symbol": "triangle-down",
                            "color": "#ff5c5c",
                            "line": {"width": 1, "color": "#150707"},
                        },
                        customdata=_marker_customdata(sells),
                        hovertemplate=(
                            "SELL %{customdata[0]}<br>"
                            "Qty: %{customdata[1]}<br>"
                            "Price: $%{customdata[2]:,.2f}<extra></extra>"
                        ),
                        name="Sell",
                    )
                )

        # Optional special event markers:
        # event_type examples: history_peak, stop_loss, model_flip, friday_exit
        if is_price_path and {"event_type", "price"}.issubset(marker_df.columns):
            events = marker_df[
                ~marker_df["event_type"].fillna("").astype(str).eq("")
            ].copy()

            if not events.empty:
                fig.add_trace(
                    go.Scatter(
                        x=events["time"],
                        y=events["price"],
                        mode="markers",
                        marker={
                            "size": 11,
                            "symbol": "diamond",
                            "color": "#ffd166",
                            "line": {"width": 1, "color": "#171100"},
                        },
                        customdata=_event_customdata(events),
                        hovertemplate=(
                            "%{customdata[0]}<br>"
                            "%{customdata[1]}<br>"
                            "Price: $%{y:,.2f}<extra></extra>"
                        ),
                        name="Trade Event",
                    )
                )

    _apply_chart_layout(fig)
    return fig


def _marker_customdata(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["symbol"] = df["symbol"] if "symbol" in df.columns else ""
    out["qty"] = df["qty"] if "qty" in df.columns else 0
    out["price"] = df["price"] if "price" in df.columns else 0

    return out[["symbol", "qty", "price"]]


def _event_customdata(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["event_type"] = df["event_type"] if "event_type" in df.columns else ""
    out["reason"] = df["reason"] if "reason" in df.columns else ""

    return out[["event_type", "reason"]]


def _apply_chart_layout(fig: go.Figure) -> None:
    fig.update_layout(
        height=390,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
        xaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "fixedrange": True,
        },
        yaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "fixedrange": True,
        },
    )


def apply_style() -> None:
    st.set_page_config(
        page_title="XANAX Paper Trading Dashboard",
        layout="wide",
        page_icon="📈",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #070a0d;
            color: #f6f7f9;
        }

        [data-testid="stSidebar"] {
            background: #0d1117;
            border-right: 1px solid #1f2937;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }

        .hero {
            background: linear-gradient(180deg, #0d1117 0%, #0a0e13 100%);
            border: 1px solid #1f2937;
            border-radius: 28px;
            padding: 28px;
            box-shadow: 0 18px 60px rgba(0,0,0,.30);
        }

        .small-muted {
            color: #8b949e;
            font-size: 0.92rem;
        }

        .gain {
            color: #00e676;
            font-weight: 700;
        }

        .loss {
            color: #ff5c5c;
            font-weight: 700;
        }

        .big-money {
            font-size: 3.4rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -.05em;
            margin: .2rem 0 .35rem 0;
        }

        div.stButton > button {
            border-radius: 14px;
            height: 3rem;
            font-weight: 700;
            border: 1px solid #243040;
            background: #111827;
            color: white;
        }

        div.stButton > button[kind="primary"] {
            background: #00e676;
            color: #06110a;
            border-color: #00e676;
        }

        div[data-testid="stRadio"] label {
            color: #c9d1d9;
            font-weight: 700;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.45rem;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            border-color: #00e676;
        }

        .pill {
            display: inline-block;
            padding: .28rem .7rem;
            border-radius: 999px;
            background: #111827;
            border: 1px solid #263244;
            color: #c9d1d9;
            margin-right: .35rem;
            font-size: .85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def scan_dataframe(rows: list[dict]) -> pd.DataFrame:
    out = []

    for i, r in enumerate(rows[:25], start=1):
        out.append(
            {
                "Rank": i,
                "Ticker": str(r.get("ticker", "")).upper(),
                "Signal": str(r.get("final_action", "WATCH")),
                "Direction": str(r.get("forecast_direction", "")),
                "Conviction": int(float(r.get("conviction_score") or 0)),
                "Edge": str(r.get("estimated_edge", "")),
                "Setup": str(r.get("setup_type", "")),
                "History %": r.get("history_true_during_week_rate", ""),
            }
        )

    return pd.DataFrame(out)
