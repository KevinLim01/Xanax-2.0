from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass

from dashboard.alpaca_client import (
    build_trade_chart_data,
    filter_trades_for_chart,
    get_account,
    get_portfolio_history,
    get_trade_activities,
    infer_trade_windows,
    positions_dataframe,
)
from dashboard.command_runner import (
    execute_top_trades,
    latest_logs,
    load_scan_results,
    monitor_positions,
    run_scan_chunks,
)
from dashboard.ui import apply_style, clean_chart, load_env, money, pct, scan_dataframe


load_env()
apply_style()

st.sidebar.markdown("# XANAX")
st.sidebar.markdown("**PAPER TRADING DASHBOARD**")
st.sidebar.caption("Manual dashboard. No GitHub Actions needed.")
st.sidebar.divider()
st.sidebar.markdown("### Trading controls")

universe = st.sidebar.selectbox("Universe", ["custom", "top50"], index=0)

chunk_count = st.sidebar.number_input(
    "Scan chunks",
    min_value=1,
    max_value=30,
    value=int(os.getenv("DASHBOARD_SCAN_CHUNKS", "15")),
    step=1,
)

max_slots = st.sidebar.number_input(
    "Max trade slots",
    min_value=1,
    max_value=25,
    value=int(os.getenv("AUTO_TRADE_MAX_TRADES_PER_SCAN", "5")),
    step=1,
)

second_chance = st.sidebar.toggle(
    "Second-chance filter",
    value=os.getenv("DASHBOARD_SECOND_CHANCE", "true").lower() == "true",
)

dry_run = st.sidebar.toggle("Dry run", value=False)

run_scan = st.sidebar.button("Run Scan", use_container_width=True)
execute = st.sidebar.button("Execute Top Trades", use_container_width=True)
monitor = st.sidebar.button("Run Monitor Check", type="primary", use_container_width=True)

st.sidebar.success("Run Monitor Check is always available.")

st.title("XANAX Paper Trading Dashboard")
st.caption("Dashboard build: individual stock price-path chart enabled.")

account_error = None

try:
    account = get_account()
except Exception as exc:
    account = {}
    account_error = str(exc)

equity = float(account.get("equity") or account.get("portfolio_value") or 0)
last_equity = float(account.get("last_equity") or equity or 0)

change = equity - last_equity
change_pct = (change / last_equity * 100) if last_equity else 0

range_choice = st.radio(
    "Chart range",
    ["1D", "1W", "1M", "1Y", "Total"],
    horizontal=True,
    label_visibility="collapsed",
)

range_map = {
    "1D": {
        "period": "1D",
        "timeframe": "5Min",
        "days_back": 2,
        "bar_timeframe": "5Min",
    },
    "1W": {
        "period": "1W",
        "timeframe": "15Min",
        "days_back": 10,
        "bar_timeframe": "15Min",
    },
    "1M": {
        "period": "1M",
        "timeframe": "1H",
        "days_back": 40,
        "bar_timeframe": "1H",
    },
    "1Y": {
        "period": "1A",
        "timeframe": "1D",
        "days_back": 370,
        "bar_timeframe": "1D",
    },
    "Total": {
        "period": "all",
        "timeframe": "1D",
        "days_back": 1500,
        "bar_timeframe": "1D",
    },
}

chart_settings = range_map[range_choice]

with st.container(border=False):
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    st.markdown("<div class='small-muted'>Net Worth</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='big-money'>{money(equity)}</div>", unsafe_allow_html=True)

    klass = "gain" if change >= 0 else "loss"
    sign = "+" if change >= 0 else ""

    st.markdown(
        f"<div class='{klass}'>{sign}{money(change)} ({pct(change_pct)}) Today</div>",
        unsafe_allow_html=True,
    )

    try:
        hist = get_portfolio_history(
            period=chart_settings["period"],
            timeframe=chart_settings["timeframe"],
        )
    except Exception:
        hist = pd.DataFrame(columns=["time", "equity"])

    try:
        raw_trades = get_trade_activities(days_back=int(chart_settings["days_back"]))
    except Exception:
        raw_trades = pd.DataFrame(columns=["time", "symbol", "side", "qty", "price"])

    chart_mode = st.radio(
        "Chart type",
        ["Portfolio", "Individual stock trade"],
        horizontal=True,
        key="chart_mode",
    )

    if chart_mode == "Portfolio":
        if hist.empty:
            st.info(
                "No usable Alpaca portfolio history yet. Once Alpaca returns real equity or P/L history, the graph will move."
            )

        try:
            trade_points = filter_trades_for_chart(raw_trades, hist)
        except Exception:
            trade_points = pd.DataFrame(
                columns=["time", "symbol", "side", "qty", "price", "equity"]
            )

        st.plotly_chart(
            clean_chart(hist, trade_points),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.caption("Portfolio chart: green triangles = buys. Red triangles = sells.")

    else:
        windows = infer_trade_windows(raw_trades)

        if windows.empty:
            st.info("No completed or open buy/sell trade windows found yet.")
        else:
            windows = windows.sort_values("open_time", ascending=False).reset_index(drop=True)

            def window_label(row: pd.Series) -> str:
                symbol = str(row.get("symbol", "")).upper()
                open_time = row.get("open_time")
                close_time = row.get("close_time")
                is_open = bool(row.get("is_open"))

                try:
                    open_label = pd.to_datetime(open_time).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    open_label = str(open_time)

                if is_open or close_time is None or pd.isna(close_time):
                    close_label = "OPEN"
                else:
                    try:
                        close_label = pd.to_datetime(close_time).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        close_label = str(close_time)

                return f"{symbol} | {open_label} → {close_label}"

            labels = [window_label(row) for _, row in windows.iterrows()]

            selected_label = st.selectbox(
                "Select stock trade",
                labels,
                index=0,
                key="selected_trade_window",
            )

            selected_idx = labels.index(selected_label)
            selected = windows.iloc[selected_idx]

            symbol = str(selected.get("symbol", "")).upper()
            open_time = selected.get("open_time")
            close_time = selected.get("close_time")

            if close_time is None or pd.isna(close_time):
                close_time = datetime.now(timezone.utc)

            price_path, markers = build_trade_chart_data(
                symbol=symbol,
                trades=raw_trades,
                open_time=open_time,
                close_time=close_time,
                timeframe=str(chart_settings["bar_timeframe"]),
            )

            if price_path.empty:
                st.warning(
                    f"No Alpaca price bars returned for {symbol}. Try 1D or 1W first, or check Alpaca market-data access."
                )

            st.plotly_chart(
                clean_chart(price_path, markers),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            entry_price = selected.get("entry_price")
            exit_price = selected.get("exit_price")
            is_open = bool(selected.get("is_open"))

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Ticker", symbol)

            with c2:
                st.metric("Entry", money(entry_price))

            with c3:
                if is_open or exit_price is None or pd.isna(exit_price):
                    latest_price = (
                        float(price_path["close"].iloc[-1])
                        if not price_path.empty and "close" in price_path.columns
                        else 0
                    )
                    st.metric("Current", money(latest_price))
                else:
                    st.metric("Exit", money(exit_price))

            with c4:
                try:
                    if is_open or exit_price is None or pd.isna(exit_price):
                        latest_price = float(price_path["close"].iloc[-1])
                        move_pct = ((latest_price - float(entry_price)) / float(entry_price)) * 100
                    else:
                        move_pct = ((float(exit_price) - float(entry_price)) / float(entry_price)) * 100

                    st.metric("Move", pct(move_pct))
                except Exception:
                    st.metric("Move", pct(0))

            st.caption(
                "Individual stock chart: line = actual stock price movement while trade was open. Triangles = buy/sell fills."
            )

    st.markdown("</div>", unsafe_allow_html=True)

if account_error:
    st.warning(f"Alpaca data not loaded: {account_error}")

if run_scan:
    st.info("Running scan chunks. This is manual and can take a while depending on your ticker list.")

    progress = st.progress(0)
    log_box = st.empty()
    results = []

    for idx, (code, path, out) in enumerate(
        run_scan_chunks(
            universe=universe,
            chunk_count=int(chunk_count),
            second_chance=second_chance,
        ),
        start=1,
    ):
        results.append((code, path, out))
        progress.progress(idx / int(chunk_count))

        tail = "\n".join(out.splitlines()[-12:])

        log_box.code(
            f"Chunk {idx}/{chunk_count} finished with code {code}\n"
            f"Saved: {path}\n\n"
            f"{tail}"
        )

    if all(code == 0 for code, _, _ in results):
        st.success("Scan finished. Review results before executing trades.")
    else:
        st.warning("Scan finished, but one or more chunks returned errors. Check logs below.")

if execute:
    st.warning("Executing top trades sends orders to Alpaca unless Dry run is on.")

    code, out = execute_top_trades(
        max_slots=int(max_slots),
        dry_run=dry_run,
        second_chance=second_chance,
    )

    st.code(out[-12000:])

    if code == 0:
        st.success("Execute command finished.")
    else:
        st.error(f"Execute command failed with code {code}.")

if monitor:
    code, out = monitor_positions(dry_run=dry_run)

    st.code(out[-12000:])

    if code == 0:
        st.success("Monitor check finished.")
    else:
        st.error(f"Monitor check failed with code {code}.")

st.divider()

left, middle, right = st.columns([1.2, 1.2, 1])

with left:
    st.subheader("Latest Scan Results")

    scan_rows = load_scan_results()
    df = scan_dataframe(scan_rows)

    if df.empty:
        st.caption("No scan results yet. Click Run Scan first.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

with middle:
    st.subheader("Open Positions")

    try:
        pos = positions_dataframe()
    except Exception as exc:
        pos = pd.DataFrame()
        st.caption(f"Could not load positions: {exc}")

    if pos.empty:
        st.caption("No open positions, or Alpaca keys are missing.")
    else:
        st.dataframe(pos, use_container_width=True, hide_index=True)

with right:
    st.subheader("Recent Activity / Logs")

    logs = latest_logs(8)

    if not logs:
        st.caption("No logs yet.")

    for p in logs:
        with st.expander(p.name):
            try:
                st.code(p.read_text()[-5000:])
            except Exception:
                st.caption("Could not read log.")
