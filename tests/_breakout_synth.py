import numpy as np, pandas as pd
from datetime import datetime, time
TZ="Asia/Kolkata"
def synth(sign=1, breakout=True, far=False):
    """12 sessions of 15m bars; last session coils under the previous-day high, then breaks it."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2026-10-20", periods=12)
    rows=[]; price=1000.0
    for d in days[:-1]:
        for k in range(25):
            ts = pd.Timestamp(datetime.combine(d.date(), time(9,15))).tz_localize(TZ)+pd.Timedelta(minutes=15*k)
            o=price; c=price*(1+rng.normal(0.0002*sign,0.002)); h=max(o,c)*(1+abs(rng.normal(0,0.001))); l=min(o,c)*(1-abs(rng.normal(0,0.001)))
            rows.append((ts,o,h,l,c,rng.uniform(8e4,1.2e5))); price=c
    prev = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).set_index("ts")
    last_day = prev[prev.index.date==days[-2].date()]
    pdh = last_day["high"].max() if sign>0 else last_day["low"].min()
    d = days[-1]
    # today: open a little below PDH, grind up with rising lows, tight bars, falling volume
    path = []
    start = pdh*(1-sign*0.006)
    p=start
    for k in range(14):
        target = pdh*(1-sign*0.0004) if k>=8 else start + (pdh*(1-sign*0.0004)-start)*k/8
        o=p; c=target + sign*(-0.0001*pdh if k%2 else 0)
        w = (0.0004 if k < 8 else 0.00015)*pdh; hi = max(o,c)+w; lo=min(o,c)-w
        if sign>0: hi=min(hi, pdh)
        else: lo=max(lo, pdh)
        vol = 1.3e5 if k<3 else max(3e4, 1.0e5-8e3*k)
        path.append((o,hi,lo,c,vol)); p=c
    if breakout:
        o=p; c=pdh*(1+sign*(0.012 if far else 0.0012)); hi=max(o,c)+0.0002*pdh*(sign>0); lo=min(o,c)-0.0002*pdh*(sign<0)
        hi=max(o,c)+0.0003*pdh; lo=min(o,c)-0.0003*pdh
        path.append((o,hi,lo,c,3.5e5))
    for k,(o,h,l,c,v) in enumerate(path):
        ts = pd.Timestamp(datetime.combine(d.date(), time(9,15))).tz_localize(TZ)+pd.Timedelta(minutes=15*k)
        rows.append((ts,o,h,l,c,v))
    bars = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).set_index("ts")
    g = bars.groupby(bars.index.date)
    daily = pd.DataFrame({"open":g["open"].first(),"high":g["high"].max(),"low":g["low"].min(),"close":g["close"].last(),"volume":g["volume"].sum()})
    hist = pd.DataFrame({"close": 900*np.exp(np.cumsum(rng.normal(0.001*sign,0.01,60)))})
    hist = hist*(daily["close"].iloc[0]/hist["close"].iloc[-1])
    hd = pd.bdate_range(end=days[0]-pd.Timedelta(days=1), periods=60)
    hdf = pd.DataFrame({"open":hist["close"].values,"high":hist["close"].values*1.01,"low":hist["close"].values*0.99,"close":hist["close"].values,"volume":1e6}, index=[x.date() for x in hd])
    daily = pd.concat([hdf, daily])
    daily["datetime"] = [pd.Timestamp(x).tz_localize(TZ) for x in daily.index]
    return bars, daily.reset_index(drop=True)[["datetime","open","high","low","close","volume"]], pdh
