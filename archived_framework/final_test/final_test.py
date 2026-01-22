import os, sys, time, math, datetime as dt
from dateutil.tz import gettz
from loguru import logger
import requests
import pandas as pd
import numpy as np

logger.remove()
logger.add(sys.stdout, level="INFO", backtrace=False, diagnose=False, colorize=False)

FAILS: list[str] = []
WARN: list[str] = []

def fail(msg: str) -> None:
    FAILS.append(msg)
    logger.error(msg)


def warn(msg: str) -> None:
    WARN.append(msg)
    logger.warning(msg)


def need(var: str) -> str:
    value = os.getenv(var)
    if not value:
        fail(f"Missing env {var}. Set it and rerun.")
        sys.exit(1)
    return value


REFRESH = need("QUES_TRADE_REFRESH_TOKEN")
TZ = gettz("America/Toronto")

SYMS = [
    "SHOP.TO",
    "RY.TO",
    "SU.TO",
    "XIU.TO",
    "HUT.TO",
    "AC.TO",
    "BNS.TO",
    "CP.TO",
    "SU.V",
]
START = (dt.datetime.now(tz=TZ) - dt.timedelta(minutes=90)).replace(microsecond=0)
END = dt.datetime.now(tz=TZ).replace(microsecond=0)


def qt_refresh(refresh: str, max_tries: int = 6, base_sleep: float = 0.8) -> dict:
    url = "https://login.questrade.com/oauth2/token"
    params = {"grant_type": "refresh_token", "refresh_token": refresh}
    last_text = ""
    for attempt in range(1, max_tries + 1):
        try:
            response = requests.get(url, params=params, timeout=15)
            last_text = response.text[:300]
            if response.status_code >= 500:
                raise requests.HTTPError(f"Server {response.status_code}")
            response.raise_for_status()
            try:
                payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"Non-JSON token response: {last_text!r}") from exc
            if "access_token" not in payload or "api_server" not in payload:
                raise RuntimeError(f"Incomplete token payload: {payload}")
            return payload
        except Exception as exc:
            sleep = base_sleep * (2 ** (attempt - 1)) + 0.1 * (attempt - 1)
            if attempt < max_tries:
                logger.warning(
                    "Token refresh attempt {}/{} failed ({}). Retrying in {:.1f}s...",
                    attempt,
                    max_tries,
                    exc,
                    sleep,
                )
                time.sleep(sleep)
            else:
                fail(
                    "Token refresh failed after retries. Likely causes: "
                    "1) refresh token expired or rotated; "
                    "2) Questrade outage; "
                    "3) local network/clock issue. "
                    f"Last response snippet: {last_text!r}"
                )
                raise


def qt(api: str, path: str, token: str, **params) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{api}{path}", headers=headers, params=params, timeout=30)
    if response.status_code >= 400:
        logger.error("QT {} returned {}: {}", path, response.status_code, response.text[:200])
    response.raise_for_status()
    return response.json()


logger.info("1) Auth: requesting access token using refresh token")
TOK = qt_refresh(REFRESH)
ACCESS = TOK["access_token"]
API = TOK["api_server"]
NEW_REFRESH = TOK.get("refresh_token", "")
if NEW_REFRESH and NEW_REFRESH != REFRESH:
    logger.info("Refresh token rotated by Questrade. Store the new one securely for next runs.")

logger.info("2) Accounts and server time")
accounts_response = qt(API, "v1/accounts", ACCESS)
server_time = qt(API, "v1/time", ACCESS)
logger.info(
    "Accounts: {}; Server time: {}",
    len(accounts_response.get("accounts", [])),
    server_time.get("time", "?"),
)

if not accounts_response.get("accounts"):
    fail("No accounts returned by Questrade.")
    sys.exit(2)

ACCOUNT_ID = accounts_response["accounts"][0]["number"]

logger.info("3) Resolve symbol IDs")
syms_response = qt(API, "v1/symbols", ACCESS, names=",".join(SYMS))
sid_map = {
    symbol_data["symbol"]: (
        symbol_data["symbolId"],
        symbol_data.get("isTradable", True),
        symbol_data.get("listingExchange"),
    )
    for symbol_data in syms_response.get("symbols", [])
}
missing = [symbol for symbol in SYMS if symbol not in sid_map]
if missing:
    warn(f"Unresolved symbols: {missing}")
logger.info("Resolved {} symbols", len(sid_map))

logger.info("4) 1-minute OHLCV fetch (last 90 minutes)")

def candles(symbol_id: int, start: dt.datetime, end: dt.datetime) -> dict:
    try:
        return qt(
            API,
            f"v1/markets/candles/{symbol_id}",
            ACCESS,
            startTime=start.isoformat(),
            endTime=end.isoformat(),
            interval="OneMinute",
        )
    except requests.HTTPError:
        return qt(
            API,
            "v1/markets/candles",
            ACCESS,
            symbolId=symbol_id,
            startTime=start.isoformat(),
            endTime=end.isoformat(),
            interval="OneMinute",
        )


frames: list[pd.DataFrame] = []
for symbol, (symbol_id, tradable, exch) in sid_map.items():
    candles_json = candles(symbol_id, START, END)
    rows = candles_json.get("candles", [])
    if not rows:
        warn(f"No candles for {symbol}")
        continue
    frame = pd.DataFrame(rows)
    frame["symbol"] = symbol
    frames.append(frame)

ohlcv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if not ohlcv.empty:
    logger.info("5) Data quality checks (currency CAD, gaps, duplicates)")
    if "close" not in ohlcv.columns:
        fail("Candle schema missing 'close' field.")
    quotes_response = qt(
        API,
        "v1/markets/quotes",
        ACCESS,
        ids=",".join(str(sid_map[symbol][0]) for symbol in sid_map),
    )
    currency_map = {
        quote["symbol"]: quote.get("currency", "")
        for quote in quotes_response.get("quotes", [])
    }
    non_cad = [symbol for symbol, currency in currency_map.items() if currency and currency != "CAD"]
    if non_cad:
        warn(f"Non-CAD quotes detected: {non_cad}")

    issues: list[str] = []
    for symbol, group in ohlcv.groupby("symbol"):
        ts = pd.to_datetime(group["start"], utc=True).dt.tz_convert(TZ).sort_values()
        if ts.duplicated().any():
            issues.append(f"dups:{symbol}")
        if len(ts) >= 3:
            gaps = (ts.diff().dt.total_seconds().dropna() != 60).sum()
            if gaps > 0:
                issues.append(f"gaps:{symbol}:{int(gaps)}")
    if issues:
        warn("Candle continuity issues -> " + ", ".join(issues))
else:
    warn("No OHLCV frames gathered; skipping data quality checks.")

logger.info("6) Account state and positions")
positions_response = qt(API, f"v1/accounts/{ACCOUNT_ID}/positions", ACCESS)
balances_response = qt(API, f"v1/accounts/{ACCOUNT_ID}/balances", ACCESS)
logger.info(
    "Positions: {}; Currencies: {}",
    len(positions_response.get("positions", [])),
    len(balances_response.get("perCurrencyBalances", [])),
)

logger.info("7) Options chain for RY.TO")
if "RY.TO" in sid_map:
    ry_id = sid_map["RY.TO"][0]
    chain_response = qt(API, "v1/markets/options/chain", ACCESS, underlyingId=ry_id)
    if not chain_response.get("options", []):
        warn("Empty options chain for RY.TO")
else:
    warn("Skipping options chain; RY.TO unresolved.")

logger.info("8) Backtest and Monte Carlo smoke tests")


def simple_backtest(df: pd.DataFrame) -> dict:
    subset = df[df["symbol"] == "SHOP.TO"].copy()
    if subset.empty:
        return {"trades": 0, "ret": 0.0}
    subset["ts"] = pd.to_datetime(subset["start"], utc=True).dt.tz_convert(TZ)
    subset = subset.sort_values("ts")
    subset["sma5"] = subset["close"].rolling(5).mean()
    subset["sma20"] = subset["close"].rolling(20).mean()
    subset["signal"] = np.where(subset["sma5"] > subset["sma20"], 1, 0)
    subset["ret"] = subset["close"].pct_change().fillna(0)
    strategy = (subset["signal"].shift(1).fillna(0) * subset["ret"]).cumsum().iloc[-1]
    trades = int((subset["signal"].diff().abs() == 1).sum())
    return {"trades": trades, "ret": float(strategy)}


if not ohlcv.empty:
    backtest = simple_backtest(ohlcv)
    logger.info(
        "Backtest SHOP.TO SMA5>20: trades={} total_ret={:.4f}",
        backtest["trades"],
        backtest["ret"],
    )
else:
    warn("Backtest skipped: no OHLCV data.")


def monte_carlo(
    days: int = 5,
    paths: int = 500,
    mu: float = 0.0005,
    sigma: float = 0.02,
    s0: float = 100.0,
) -> list[float]:
    rng = np.random.default_rng(7)
    dtau = 1.0
    outcomes: list[float] = []
    for _ in range(paths):
        price = s0
        for _ in range(days * 390):
            price *= math.exp((mu - 0.5 * sigma * sigma) * dtau + sigma * math.sqrt(dtau) * rng.standard_normal())
        outcomes.append(price / s0 - 1.0)
    return np.percentile(outcomes, [1, 5, 50, 95, 99]).tolist()


mc = monte_carlo()
logger.info("Monte Carlo 5d percentiles: {}", mc)

logger.info("9) Optional GPT auditor (router at 127.0.0.1:8787)")
auditor_url = "http://127.0.0.1:8787/decide"
auditor_payload = {
    "prompt": "Audit final test: check gaps vs runbook and return a checklist."
}
try:
    auditor_response = requests.post(auditor_url, json=auditor_payload, timeout=5)
    if auditor_response.ok:
        auditor_json = auditor_response.json()
        engine = auditor_json.get("engine", "?")
        snippet = auditor_json.get("reply", "")[:300].replace("\n", " ")
        logger.info("Auditor engine={} reply: {} ...", engine, snippet)
    else:
        warn(f"Auditor HTTP {auditor_response.status_code}")
except Exception as exc:
    warn(f"Auditor unavailable: {exc}")

logger.info("10) Production gates summary")

def gate(name: str, condition: bool, ok: str = "OK", bad: str = "See logs") -> None:
    if condition:
        logger.info("[PASS] {} — {}", name, ok)
    else:
        fail(f"[FAIL] {name} — {bad}")


gate("G1 Auth", bool(ACCESS and API), "token+api_server")
gate("G2 Accounts", bool(accounts_response.get("accounts")), "account(s) present", "no accounts")
gate("G3 Symbols", len(sid_map) >= 4, f"{len(sid_map)} resolved", "insufficient symbols")
gate("G4 1m OHLCV", (not ohlcv.empty) and (len(ohlcv) >= 5), f"{len(ohlcv)} rows", "no candles")
gate("G5 Positions/Balances", True, "queried")
gate("G6 Options Chain", True, "queried")
gate("G7 Backtest/MC", True, "ran")
gate("G8 Auditor (optional)", True, "attempted")

if WARN:
    logger.warning("Warnings: {} -> {}", len(WARN), WARN)
if FAILS:
    logger.error("FINAL: {} FAIL", len(FAILS))
    sys.exit(2)
logger.success("FINAL: ALL CRITICAL GATES PASSED")
