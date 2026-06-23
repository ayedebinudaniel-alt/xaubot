"""
╔══════════════════════════════════════════════════════════════╗
║       XAUUSD TELEGRAM SIGNAL BOT — Linux/Cloud Version       ║
║  Data: Yahoo Finance (free, no API key needed)               ║
║  Alerts: Telegram  |  Runs on Railway / Render / any Linux   ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
1. pip install -r requirements.txt
2. Set environment variables (see below)
3. python bot.py

ENVIRONMENT VARIABLES:
    TELEGRAM_TOKEN   = your bot token from @BotFather
    TELEGRAM_CHAT_ID = your chat ID from @userinfobot
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────
TG_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT   = os.environ.get("TELEGRAM_CHAT_ID", "")
SYMBOL    = "GC=F"          # Yahoo Finance symbol for Gold/XAU
INTERVAL  = "1m"            # 1-minute candles
MIN_SCORE = 6               # minimum confluence score out of 9
CHECK_EVERY = 60            # seconds between scans

# Sessions (UTC hours) — London + New York only
SESSIONS = [(7, 10), (12, 16)]

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
#  TELEGRAM
# ────────────────────────────────────────────────────────────
def send(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram not configured.")
        print(msg)  # fallback: print to console
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": TG_CHAT,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ────────────────────────────────────────────────────────────
#  DATA — Yahoo Finance (no API key needed)
# ────────────────────────────────────────────────────────────
def fetch_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval={interval}&range={period}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        chart = data["chart"]["result"][0]
        timestamps = chart["timestamp"]
        ohlcv = chart["indicators"]["quote"][0]
        df = pd.DataFrame({
            "time":   pd.to_datetime(timestamps, unit="s", utc=True),
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": ohlcv["volume"],
        }).dropna()
        return df.reset_index(drop=True)
    except Exception as e:
        log.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────
#  INDICATORS
# ────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_engulfing(df: pd.DataFrame) -> str | None:
    if len(df) < 2:
        return None
    p, c = df.iloc[-2], df.iloc[-1]
    pb = p["close"] - p["open"]
    cb = c["close"] - c["open"]
    if pb < 0 and cb > 0 and c["open"] <= p["close"] and c["close"] >= p["open"]:
        return "bull"
    if pb > 0 and cb < 0 and c["open"] >= p["close"] and c["close"] <= p["open"]:
        return "bear"
    return None

def detect_pin_bar(df: pd.DataFrame) -> str | None:
    if len(df) < 1:
        return None
    c = df.iloc[-1]
    body   = abs(c["close"] - c["open"])
    candle = c["high"] - c["low"]
    if candle == 0:
        return None
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    if lower > 2 * body and lower > upper:
        return "bull"
    if upper > 2 * body and upper > lower:
        return "bear"
    return None

# ────────────────────────────────────────────────────────────
#  SESSION FILTER
# ────────────────────────────────────────────────────────────
def in_session() -> bool:
    h = datetime.now(timezone.utc).hour
    return any(s <= h < e for s, e in SESSIONS)

def session_name() -> str:
    h = datetime.now(timezone.utc).hour
    if 7 <= h < 10:  return "🇬🇧 London Open"
    if 12 <= h < 16: return "🇺🇸 New York"
    return "Outside Session"

# ────────────────────────────────────────────────────────────
#  CONFLUENCE SCORE ENGINE
# ────────────────────────────────────────────────────────────
def analyse() -> dict | None:
    # Fetch M1 (1 minute, last 2 days for enough data)
    m1 = fetch_candles(SYMBOL, "1m", "2d")
    if m1.empty or len(m1) < 60:
        log.warning("Not enough M1 data")
        return None

    # Simulate M5 and M15 by resampling M1
    m1_indexed = m1.set_index("time")

    def resample(df, rule):
        r = df.resample(rule).agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum"
        }).dropna().reset_index()
        return r

    m5  = resample(m1_indexed, "5min")
    m15 = resample(m1_indexed, "15min")

    if len(m5) < 55 or len(m15) < 55:
        log.warning("Not enough resampled data")
        return None

    # M15 trend
    m15_f = ema(m15["close"], 8).iloc[-1]
    m15_s = ema(m15["close"], 50).iloc[-1]

    # M5 trend
    m5_f = ema(m5["close"], 8).iloc[-1]
    m5_s = ema(m5["close"], 50).iloc[-1]

    # M1 indicators
    close  = m1["close"]
    e_fast = ema(close, 8)
    e_mid  = ema(close, 21)
    e_slow = ema(close, 50)
    m1_rsi = rsi(close, 14)
    m1_atr = atr(m1, 14)

    f   = e_fast.iloc[-1]
    m   = e_mid.iloc[-1]
    s   = e_slow.iloc[-1]
    f2  = e_fast.iloc[-2]
    m2  = e_mid.iloc[-2]
    r   = m1_rsi.iloc[-1]
    a   = m1_atr.iloc[-1]
    a_avg = m1_atr.rolling(10).mean().iloc[-1]
    price = close.iloc[-1]

    pat_e = detect_engulfing(m1)
    pat_p = detect_pin_bar(m1)

    # ── Score BUY ──
    bull = 0
    if m15_f > m15_s:           bull += 1   # M15 uptrend
    if m5_f  > m5_s:            bull += 1   # M5 uptrend
    if f > m > s:               bull += 1   # M1 EMA stack bullish
    if f2 < m2 and f > m:       bull += 1   # fresh bullish crossover
    if 40 < r < 65:             bull += 1   # RSI healthy zone
    if a > a_avg:               bull += 1   # good volatility
    if pat_e == "bull":         bull += 1   # bullish engulfing
    if pat_p == "bull":         bull += 1   # bullish pin bar
    if price > f > m > s:       bull += 1   # price above all EMAs

    # ── Score SELL ──
    bear = 0
    if m15_f < m15_s:           bear += 1
    if m5_f  < m5_s:            bear += 1
    if f < m < s:               bear += 1
    if f2 > m2 and f < m:       bear += 1
    if 35 < r < 60:             bear += 1
    if a > a_avg:               bear += 1
    if pat_e == "bear":         bear += 1
    if pat_p == "bear":         bear += 1
    if price < f < m < s:       bear += 1

    sl_mult = 1.5
    tp_mult = 3.0

    if bull >= MIN_SCORE and bull > bear:
        direction = "BUY"
        score = bull
        sl = round(price - a * sl_mult, 2)
        tp = round(price + a * tp_mult, 2)
    elif bear >= MIN_SCORE and bear > bull:
        direction = "SELL"
        score = bear
        sl = round(price + a * sl_mult, 2)
        tp = round(price - a * tp_mult, 2)
    else:
        return None

    rr = tp_mult / sl_mult
    return {
        "direction": direction,
        "score":     score,
        "price":     round(price, 2),
        "sl":        sl,
        "tp":        tp,
        "rsi":       round(r, 1),
        "atr":       round(a, 4),
        "pattern":   pat_e or pat_p or "none",
        "rr":        rr,
        "session":   session_name(),
        "time":      datetime.now(timezone.utc).strftime("%H:%M UTC"),
    }

# ────────────────────────────────────────────────────────────
#  FORMAT TELEGRAM MESSAGE
# ────────────────────────────────────────────────────────────
def format_signal(sig: dict) -> str:
    arrow  = "🟢 BUY  ▲" if sig["direction"] == "BUY" else "🔴 SELL ▼"
    stars  = "⭐" * sig["score"]
    return (
        f"{'━'*28}\n"
        f"{arrow}  <b>XAU/USD SIGNAL</b>\n"
        f"{'━'*28}\n"
        f"📍 Entry  : <b>{sig['price']}</b>\n"
        f"🛑 SL     : {sig['sl']}\n"
        f"🎯 TP     : {sig['tp']}\n"
        f"{'━'*28}\n"
        f"📊 Score  : {sig['score']}/9  {stars}\n"
        f"📈 RSI    : {sig['rsi']}\n"
        f"🕯 Pattern: {sig['pattern']}\n"
        f"⚖️ R:R    : 1:{sig['rr']:.1f}\n"
        f"🕐 Session: {sig['session']}\n"
        f"🕒 Time   : {sig['time']}\n"
        f"{'━'*28}\n"
        f"⚠️ <i>Execute manually on MT5 mobile</i>\n"
        f"<i>Lot: 0.01 | Risk: 1% only</i>"
    )

# ────────────────────────────────────────────────────────────
#  MAIN LOOP
# ────────────────────────────────────────────────────────────
def main():
    log.info("═" * 50)
    log.info("  XAUUSD TELEGRAM SIGNAL BOT — STARTED")
    log.info("═" * 50)

    send(
        "🤖 <b>XAU/USD Signal Bot Started</b>\n"
        "Scanning M1 for high-confluence setups.\n"
        "Sessions: London 07–10 UTC | NY 12–16 UTC\n"
        "Min score: 6/9 to fire a signal."
    )

    last_signal_minute = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            current_minute = now.replace(second=0, microsecond=0)

            if not in_session():
                log.info(f"[{now.strftime('%H:%M')}] Outside session. Waiting...")
                time.sleep(CHECK_EVERY)
                continue

            if last_signal_minute == current_minute:
                time.sleep(15)
                continue

            log.info(f"[{now.strftime('%H:%M')}] Scanning market...")
            signal = analyse()

            if signal:
                msg = format_signal(signal)
                send(msg)
                log.info(f"✅ Signal sent: {signal['direction']} | Score {signal['score']}/9")
                last_signal_minute = current_minute
            else:
                log.info("No qualifying signal this candle.")

            time.sleep(CHECK_EVERY)

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            send("⛔ <b>Bot Stopped</b>")
            break
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
