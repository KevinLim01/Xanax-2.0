from pathlib import Path

root = Path.cwd()
pipeline = root / 'app' / 'pipeline.py'
main = root / 'main.py'

if not pipeline.exists() or not main.exists():
    raise SystemExit('Run this from the project root. Expected app/pipeline.py and main.py.')

p = pipeline.read_text()
m = main.read_text()

old = '''    if _is_semi_or_ai_name(ticker):
        penalty *= 0.82
'''
new = '''    # Semis/AI can trend longer than ordinary names, but a huge prior run should
    # still lower the odds of another clean continuation week. The old 0.82
    # multiplier was too forgiving for NVDA/AMD-type names after a very large run.
    if _is_semi_or_ai_name(ticker):
        if prev_5d >= 0.18 or prev_20d >= 0.32 or rsi >= 90:
            penalty *= 1.05
        else:
            penalty *= 0.86
'''
if old in p:
    p = p.replace(old, new, 1)
elif 'prev_5d >= 0.18 or prev_20d >= 0.32 or rsi >= 90' not in p:
    raise SystemExit('Could not find semi/AI overextension block in app/pipeline.py')

insert_after = '''    return {
        "score_penalty": penalty,
        "move_multiplier": max(0.55, 1.0 - 0.55 * penalty / 0.45) if penalty > 0 else 1.0,
        "conviction_penalty": round(22 * penalty / 0.45) if penalty > 0 else 0,
        "is_overextended": penalty >= 0.12,
    }
'''
fresh_func = r'''


def _fresh_setup_profile(ticker: str, row: dict[str, Any]) -> dict[str, Any]:
    """Reward fresh setups and cool off crowded continuation names.

    This is not a replacement for the model. It is a ranking/discipline layer.
    It tries to answer: is this name setting up for next week, or did it already
    spend most of the move last week?
    """
    rsi = float(row["rsi_14"])
    prev_1d = float(row["prev_1d_return"])
    prev_5d = float(row["prev_5d_return"])
    prev_20d = float(row["prev_20d_return"])
    z20 = float(row["zscore_20"])
    ma_gap = float(row["ma_gap_10_50"])
    macd = float(row["macd_hist"])
    vol = float(row["realized_vol_20"])

    score = 0.0
    tags: list[str] = []

    if ma_gap > 0 and macd > 0 and -0.055 <= prev_5d <= 0.035 and 42 <= rsi <= 68:
        score += 0.24
        tags.append("fresh setup: constructive pullback inside an uptrend")

    if -0.12 <= prev_20d <= -0.015 and prev_1d > -0.025 and 28 <= rsi <= 48 and z20 < -0.45:
        score += 0.18
        tags.append("fresh setup: possible early reversal after weakness")

    if abs(prev_5d) <= 0.035 and abs(z20) <= 0.90 and vol <= 0.45:
        score += 0.12
        tags.append("fresh setup: quiet consolidation")

    if ma_gap > 0 and macd > 0 and 0.025 < prev_5d <= 0.085 and rsi <= 74 and z20 <= 1.45:
        score += 0.10
        tags.append("fresh setup: controlled breakout")

    if prev_5d >= 0.10:
        score -= 0.16
        tags.append("continuation caution: prior 5-day move is already large")
    if prev_5d >= 0.18:
        score -= 0.16
        tags.append("continuation caution: prior week was extreme")
    if prev_20d >= 0.30:
        score -= 0.14
        tags.append("continuation caution: 20-day run is stretched")
    if rsi >= 86:
        score -= 0.13
        tags.append("continuation caution: RSI is stretched")
    if z20 >= 2.00:
        score -= 0.10
        tags.append("continuation caution: price is far above recent range")

    if _is_ev_or_story_stock(ticker) and prev_5d >= 0.08:
        score -= 0.08
        tags.append("continuation caution: story-stock rallies fade faster")

    score = max(-0.40, min(0.35, score))
    return {
        "score": round(score, 4),
        "score_delta": round(score * 0.55, 4),
        "move_delta": round(score * 0.012, 4),
        "conviction_delta": int(round(score * 18)),
        "tags": tags[:4],
    }
'''
if '_fresh_setup_profile' not in p:
    if insert_after not in p:
        raise SystemExit('Could not find insertion point after _overextension_penalty return block.')
    p = p.replace(insert_after, insert_after + fresh_func, 1)

old = '    overext = _overextension_penalty(ticker, row)\n'
new = '''    overext = _overextension_penalty(ticker, row)
    fresh_setup = _fresh_setup_profile(ticker, row)

    # Fresh setups get a small forward-looking lift. Stretched continuation names
    # get cooled off without being automatically flipped bearish.
    expected_move += fresh_setup["move_delta"]
'''
if old in p and 'fresh_setup = _fresh_setup_profile(ticker, row)' not in p:
    p = p.replace(old, new, 1)

old = '    raw_score = _signal_score(ticker, adjusted_probs, expected_move, row, model_validity, news, sector_rt)\n'
new = '''    raw_score = _signal_score(ticker, adjusted_probs, expected_move, row, model_validity, news, sector_rt)
    raw_score = float(max(-1.0, min(1.0, raw_score + fresh_setup["score_delta"])))
'''
if old in p and 'raw_score + fresh_setup["score_delta"]' not in p:
    p = p.replace(old, new, 1)

old = '    conviction = _conviction_from_score(ticker, raw_score, model_validity, news, sector_rt, adjusted_probs, expected_move, row)\n'
new = '''    conviction = _conviction_from_score(ticker, raw_score, model_validity, news, sector_rt, adjusted_probs, expected_move, row)
    conviction = int(max(5, min(95, conviction + fresh_setup["conviction_delta"])))
'''
if old in p and 'conviction + fresh_setup["conviction_delta"]' not in p:
    p = p.replace(old, new, 1)

old = '''    if overext["is_overextended"]:
        drivers.append("risk: stock is extended after a large recent run")
'''
new = '''    if overext["is_overextended"]:
        drivers.append("risk: stock is extended after a large recent run")
    for tag in fresh_setup["tags"]:
        if tag.startswith("fresh setup"):
            drivers.append(tag)
'''
if old in p and 'tag.startswith("fresh setup")' not in p:
    p = p.replace(old, new, 1)

old = '''    if overext["is_overextended"]:
        conflicts.append("momentum: prior move may already be overextended")
    conflicts = conflicts[:3]
'''
new = '''    if overext["is_overextended"]:
        conflicts.append("momentum: prior move may already be overextended")
    for tag in fresh_setup["tags"]:
        if tag.startswith("continuation caution"):
            conflicts.append(tag)
    conflicts = conflicts[:3]
'''
if old in p and 'tag.startswith("continuation caution")' not in p:
    p = p.replace(old, new, 1)

old = '            f"Overextension penalties now reduce conviction after extreme prior-week moves."\n'
new = '            f"Fresh-setup calibration now favors cleaner next-week setups over crowded continuation names."\n'
if old in p:
    p = p.replace(old, new, 1)

old = '''        "expected_move": expected_move,
        "raw_signal_score": raw_score,
        "overextension": overext,
        "override": override,
'''
new = '''        "expected_move": expected_move,
        "raw_signal_score": raw_score,
        "overextension": overext,
        "fresh_setup": fresh_setup,
        "override": override,
'''
if old in p and '"fresh_setup": fresh_setup' not in p:
    p = p.replace(old, new, 1)

old = '''        "final_output": result.to_dict(),
    }

    run_id = save_run(ticker, result.run_timestamp, result.forecast_window, result.to_dict(), debug_payload)
    logger.info("Saved run %s", run_id)
    return {"run_id": run_id, "ticker": ticker, "final_output": result.to_dict(), "debug": debug_payload}
'''
new = '''        "final_output": result_dict,
    }

    run_id = save_run(ticker, result.run_timestamp, result.forecast_window, result_dict, debug_payload)
    logger.info("Saved run %s", run_id)
    return {"run_id": run_id, "ticker": ticker, "final_output": result_dict, "debug": debug_payload}
'''
if 'result_dict = result.to_dict()' not in p:
    marker = '''        monday_close_reference=row["monday_close"],
    )
'''
    replacement = '''        monday_close_reference=row["monday_close"],
    )

    # Keep this extra field out of the pydantic/dataclass constructor in case the
    # model schema is strict. It will still appear in saved/debug output.
    result_dict = result.to_dict()
    result_dict["fresh_setup_score"] = fresh_setup["score"]
'''
    if marker in p:
        p = p.replace(marker, replacement, 1)
    else:
        raise SystemExit('Could not find Result constructor ending marker.')
if old in p:
    p = p.replace(old, new, 1)
elif 'return {"run_id": run_id, "ticker": ticker, "final_output": result_dict' not in p:
    raise SystemExit('Could not update save_run/final_output block.')

old = '''def _rank_key(item: Dict[str, Any]) -> tuple:
    action_rank = {'BUY': 2, 'WATCH': 1, 'SELL': 0}.get(item.get('final_action', 'WATCH'), 1)
    direction_rank = {'UP': 2, 'NEUTRAL': 1, 'DOWN': 0}.get(item.get('forecast_direction', 'NEUTRAL'), 1)
    return (action_rank, direction_rank, int(item.get('conviction_score', 0)))
'''
new = '''def _rank_key(item: Dict[str, Any]) -> tuple:
    action_rank = {'BUY': 2, 'SELL': 2, 'WATCH': 1}.get(item.get('final_action', 'WATCH'), 1)
    direction_rank = {'UP': 2, 'DOWN': 2, 'NEUTRAL': 1}.get(item.get('forecast_direction', 'NEUTRAL'), 1)
    fresh_setup = float(item.get('fresh_setup_score', 0.0) or 0.0)
    raw_score = abs(float(item.get('raw_signal_score', 0.0) or 0.0))
    expected_move = abs(float(item.get('expected_move_pct', 0.0) or 0.0))
    conviction = int(item.get('conviction_score', 0) or 0)

    # Sort by quality of next-week setup first, then conviction. This keeps the
    # screen from constantly promoting last week’s winners just because they have
    # high momentum and noisy news.
    return (action_rank, direction_rank, round(fresh_setup, 4), conviction, raw_score, expected_move)
'''
if old in m:
    m = m.replace(old, new, 1)
elif 'fresh_setup_score' not in m:
    raise SystemExit('Could not find _rank_key block in main.py')

pipeline.write_text(p)
main.write_text(m)
print('Applied fresh setup focus fix to app/pipeline.py and main.py')
