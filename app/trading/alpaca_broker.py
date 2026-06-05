from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any

import yfinance as yf

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, ContractType, QueryOrderStatus
    from alpaca.trading.requests import MarketOrderRequest, GetOptionContractsRequest, GetOrdersRequest
except ImportError:  # pragma: no cover
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    AssetStatus = None
    ContractType = None
    QueryOrderStatus = None
    MarketOrderRequest = None
    GetOptionContractsRequest = None
    GetOrdersRequest = None


@dataclass(frozen=True)
class PositionSnapshot:
    ticker: str
    qty: float
    side: str
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    asset_class: str = ""
    underlying_ticker: str | None = None
    option_type: str | None = None
    expiration_date: str | None = None


@dataclass(frozen=True)
class OptionContractChoice:
    underlying_ticker: str
    option_symbol: str
    option_type: str
    strike_price: float
    expiration_date: str
    days_to_expiration: int
    underlying_price: float
    estimated_contract_price: float | None = None


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        if TradingClient is None:
            raise RuntimeError("alpaca-py is not installed. Run: pip install alpaca-py")
        if not api_key or not secret_key:
            raise RuntimeError("Missing Alpaca keys. Add ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def get_account(self) -> Any:
        return self.client.get_account()

    def is_tradable(self, ticker: str) -> bool:
        asset = self.client.get_asset(ticker)
        return bool(asset.tradable and asset.status == AssetStatus.ACTIVE)

    def is_shortable(self, ticker: str) -> bool:
        asset = self.client.get_asset(ticker)
        return bool(asset.tradable and asset.shortable and asset.status == AssetStatus.ACTIVE)

    def is_options_enabled(self, ticker: str) -> bool:
        asset = self.client.get_asset(ticker)
        return bool(getattr(asset, "options_enabled", False))

    def get_latest_price(self, ticker: str) -> float:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
        if hist.empty:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if hist.empty:
            raise RuntimeError(f"Could not fetch latest price for {ticker}.")
        return float(hist["Close"].dropna().iloc[-1])

    def list_positions(self) -> list[PositionSnapshot]:
        out: list[PositionSnapshot] = []
        for p in self.client.get_all_positions():
            symbol = str(p.symbol).upper()
            out.append(
                PositionSnapshot(
                    ticker=symbol,
                    qty=float(p.qty),
                    side=str(p.side).lower(),
                    market_value=float(p.market_value),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    unrealized_pl=float(p.unrealized_pl),
                    asset_class=str(getattr(p, "asset_class", "") or ""),
                    underlying_ticker=_infer_option_underlying(symbol),
                    option_type=_infer_option_type(symbol),
                    expiration_date=_infer_option_expiration(symbol),
                )
            )
        return out

    def get_position(self, ticker: str) -> PositionSnapshot | None:
        try:
            p = self.client.get_open_position(ticker)
            symbol = str(p.symbol).upper()
            return PositionSnapshot(
                ticker=symbol,
                qty=float(p.qty),
                side=str(p.side).lower(),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                asset_class=str(getattr(p, "asset_class", "") or ""),
                underlying_ticker=_infer_option_underlying(symbol),
                option_type=_infer_option_type(symbol),
                expiration_date=_infer_option_expiration(symbol),
            )
        except Exception:
            return None

    def get_latest_filled_order_time(self, symbol: str) -> datetime | None:
        """
        Used for same-day exit protection.
        If the most recent filled order for this symbol happened today, the monitor
        treats the position as same-day and will not take profit yet.
        """
        if GetOrdersRequest is None or QueryOrderStatus is None:
            return None

        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                limit=50,
                nested=False,
            )
            orders = self.client.get_orders(filter=req)
        except Exception:
            return None

        filled_times: list[datetime] = []

        for order in orders:
            filled_at = getattr(order, "filled_at", None)
            if filled_at is None:
                continue

            if isinstance(filled_at, str):
                try:
                    filled_at = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
                except ValueError:
                    continue

            if isinstance(filled_at, datetime):
                if filled_at.tzinfo is None:
                    filled_at = filled_at.replace(tzinfo=timezone.utc)
                filled_times.append(filled_at)

        if not filled_times:
            return None

        return max(filled_times)

    def has_position_for_underlying(self, underlying: str) -> bool:
        underlying = underlying.upper()
        for p in self.list_positions():
            if p.ticker == underlying:
                return True
            if p.underlying_ticker == underlying:
                return True
        return False

    def submit_market_order_by_qty(self, ticker: str, side: str, qty: float):
        if qty <= 0:
            raise ValueError("Order quantity must be positive.")
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=_round_qty(qty),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(req)

    def submit_option_market_order(self, option_symbol: str, side: str, contracts: int):
        if contracts <= 0:
            raise ValueError("Option contract quantity must be positive.")
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=option_symbol,
            qty=int(contracts),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(req)

    def close_position(self, ticker: str):
        return self.client.close_position(ticker)

    def choose_simple_option_contract(
        self,
        underlying: str,
        option_type: str,
        min_dte: int,
        max_dte: int,
        strike_offset_pct: float,
        max_contract_price: float | None = None,
    ) -> OptionContractChoice:
        if GetOptionContractsRequest is None:
            raise RuntimeError("This alpaca-py version does not expose GetOptionContractsRequest.")

        underlying = underlying.upper()
        option_type = option_type.lower()
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")

        if not self.is_options_enabled(underlying):
            raise RuntimeError(f"{underlying} is not options-enabled on Alpaca.")

        underlying_price = self.get_latest_price(underlying)
        today = date.today()
        exp_gte = today + timedelta(days=min_dte)
        exp_lte = today + timedelta(days=max_dte)

        contract_type = _contract_type_value(option_type)
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            type=contract_type,
            limit=1000,
        )

        response = self.client.get_option_contracts(req)
        contracts = getattr(response, "option_contracts", response)

        choices = []
        for c in contracts:
            strike = float(c.strike_price)
            exp = c.expiration_date
            if isinstance(exp, str):
                exp_date = date.fromisoformat(exp)
                exp_str = exp
            else:
                exp_date = exp
                exp_str = exp.isoformat()

            dte = (exp_date - today).days

            if option_type == "call":
                target_strike = underlying_price * (1 + strike_offset_pct / 100.0)
            else:
                target_strike = underlying_price * (1 - strike_offset_pct / 100.0)

            choices.append(
                (
                    abs(strike - target_strike),
                    dte,
                    OptionContractChoice(
                        underlying_ticker=underlying,
                        option_symbol=str(c.symbol),
                        option_type=option_type,
                        strike_price=strike,
                        expiration_date=exp_str,
                        days_to_expiration=dte,
                        underlying_price=underlying_price,
                        estimated_contract_price=None,
                    ),
                )
            )

        if not choices:
            raise RuntimeError(f"No suitable {option_type} contracts found for {underlying}.")

        choices.sort(key=lambda x: (x[0], x[1]))
        return choices[0][2]


def _round_qty(qty: float) -> float:
    return float(Decimal(str(qty)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def _contract_type_value(option_type: str):
    if ContractType is None:
        return option_type
    return ContractType.CALL if option_type == "call" else ContractType.PUT


def _infer_option_underlying(symbol: str) -> str | None:
    # OCC option symbols usually look like AAPL260116C00150000
    symbol = symbol.upper()
    for i, ch in enumerate(symbol):
        if ch.isdigit():
            return symbol[:i] or None
    return None


def _infer_option_type(symbol: str) -> str | None:
    symbol = symbol.upper()
    for marker in ("C", "P"):
        idx = symbol.find(marker)
        if idx > 0 and any(ch.isdigit() for ch in symbol[:idx]):
            return "call" if marker == "C" else "put"
    return None


def _infer_option_expiration(symbol: str) -> str | None:
    symbol = symbol.upper()
    digits = ""
    for ch in symbol:
        if ch.isdigit():
            digits += ch
            if len(digits) >= 6:
                break
    if len(digits) < 6:
        return None
    try:
        yy = int(digits[0:2])
        mm = int(digits[2:4])
        dd = int(digits[4:6])
        year = 2000 + yy
        return date(year, mm, dd).isoformat()
    except Exception:
        return None
