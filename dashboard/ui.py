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
    fig = go.Figure()

    if df.empty:
        fig.add_trace(go.Scatter(x=[], y=[]))
    else:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["equity"],
                mode="lines",
                line={
                    "width": 4,
                    "color": "#00e676",
                    "shape": "spline",
                    "smoothing": 1.2,
                },
                fill="tozeroy",
                fillcolor="rgba(0,230,118,0.12)",
                hovertemplate="$%{y:,.2f}<extra></extra>",
                name="Net Worth",
            )
        )

    if trades is not None and not trades.empty:
        buys = trades[trades["side"].astype(str).str.lower() == "buy"]
        sells = trades[trades["side"].astype(str).str.lower() == "sell"]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["time"],
                    y=buys["equity"],
                    mode="markers",
                    marker={
                        "size": 12,
                        "symbol": "triangle-up",
                        "color": "#00e676",
                        "line": {"width": 1, "color": "#06110a"},
                    },
                    customdata=buys[["symbol", "qty", "price"]],
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
                    y=sells["equity"],
                    mode="markers",
                    marker={
                        "size": 12,
                        "symbol": "triangle-down",
                        "color": "#ff5c5c",
                        "line": {"width": 1, "color": "#150707"},
                    },
                    customdata=sells[["symbol", "qty", "price"]],
                    hovertemplate=(
                        "SELL %{customdata[0]}<br>"
                        "Qty: %{customdata[1]}<br>"
                        "Price: $%{customdata[2]:,.2f}<extra></extra>"
                    ),
                    name="Sell",
                )
            )

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

    return fig


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
