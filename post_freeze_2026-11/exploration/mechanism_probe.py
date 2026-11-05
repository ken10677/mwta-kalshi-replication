import json, re, sys
import numpy as np, pandas as pd
from datetime import datetime
LOGS = sys.argv[1]
rng = np.random.default_rng(20260824)
rows = []
for line in open(f"{LOGS}/retro_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("r") not in ("yes", "no"): continue
    b, a = int(r.get("b") or 0), int(r.get("a") or 100)
    rows.append({"t": r["t"], "event": r.get("event") or r["t"], "series": r.get("series") or "", "b": b, "a": a, "y": 1 if r["r"]=="yes" else 0})
df = pd.DataFrame(rows); df = df[(df.b>0)&(df.a<100)&(df.a>df.b)].copy()
df["mid"]=(df.a+df.b)/2; df["spread"]=df.a-df.b; df["gap"]=df.y*100-df.mid
df["mbucket"]=np.minimum((df.mid//10).astype(int),9); df["ev_n"]=df.groupby("event")["t"].transform("count")
df["suffix"]=df.t.str.split("-").str[-1]
df["ladder"]=df.suffix.str.match(r"^T[\d.]+$")
df["multi"]=df.ev_n>=3
def boot(sub, col="gap", draws=2000):
    g=sub.groupby("event")[col].agg(["sum","count"]); S,N=g["sum"].to_numpy(),g["count"].to_numpy(); k=len(S)
    if k==0: return (np.nan,)*3+(0,0)
    out=np.empty(draws)
    for i in range(0,draws,400):
        m=min(400,draws-i); idx=rng.integers(0,k,size=(m,k)); out[i:i+m]=S[idx].sum(1)/N[idx].sum(1)
    lo,hi=np.percentile(out,[2.5,97.5]); return (S.sum()/N.sum(),lo,hi,len(sub),k)
f=lambda t:f"{t[0]:+6.2f} [{t[1]:+6.2f}, {t[2]:+6.2f}] n={t[3]:>6} clus={t[4]:>5}"
print("== gap by spread bin, two-market events vs 3+ market events ==")
for lo,hi in [(1,4),(5,20),(21,50),(51,70),(71,99)]:
    s=df[(df.spread>=lo)&(df.spread<=hi)]
    print(f"  {lo:>2}-{hi:<2}c  2-mkt {f(boot(s[s.ev_n==2]))}   3+ {f(boot(s[s.ev_n>=3]))}")
print("\n== share of rows by event size within spread groups ==")
for lab,m in (("<=20",df.spread<=20),(">=51",df.spread>=51)):
    s=df[m]; print(f"  spread {lab}: 1-mkt {100*(s.ev_n==1).mean():.1f}%  2-mkt {100*(s.ev_n==2).mean():.1f}%  3+ {100*(s.ev_n>=3).mean():.1f}%  n={len(s)}")
print("\n== top series among 3+-market events with spread>=51 ==")
s=df[(df.ev_n>=3)&(df.spread>=51)]
print(s.series.value_counts().head(10).to_string())
print(f"  threshold-ladder share: {100*s.ladder.mean():.1f}%")
print("\n== within series x bucket FE (wide>=51 vs <=20), by event size ==")
def fe(d):
    d=d[(d.spread<=20)|(d.spread>=51)].copy(); d["w"]=(d.spread>=51).astype(float); d["cell"]=d.series+"|"+d.mbucket.astype(str)
    gm=d.groupby("cell").gap.transform("mean"); wm=d.groupby("cell").w.transform("mean")
    dw=d.w-wm; den=(dw*dw).sum(); return float((dw*(d.gap-gm)).sum()/den) if den>0 else np.nan, d
for lab,m in (("2-market",df.ev_n==2),("3+ market",df.ev_n>=3),("3+ non-ladder",(df.ev_n>=3)&~df.ladder)):
    pt,d=fe(df[m]); ev=d.event.unique(); idx=d.groupby("event").indices; bs=[]
    for _ in range(300):
        pick=rng.choice(ev,len(ev)); ix=np.concatenate([idx[e] for e in pick]); bs.append(fe(d.iloc[ix])[0])
    lo,hi=np.nanpercentile(bs,[2.5,97.5]); print(f"  {lab:<14} wide-minus-narrow {pt:+.2f} [{lo:+.2f}, {hi:+.2f}] (300 draws)")
print("\n== non-ladder multi-outcome events (3+ markets): sum of midpoints vs YES count ==")
nl=df[(df.ev_n>=3)&~df.ladder]
g=nl.groupby("event").agg(k=("t","count"),summid=("mid","sum"),yes=("y","sum"),allwide=("spread",lambda x:(x>=51).all()),allnarrow=("spread",lambda x:(x<=20).all()))
for lab,m in (("all books narrow (<=20c)",g.allnarrow),("all books near-empty (>=51c)",g.allwide)):
    gg=g[m]; print(f"  {lab:<30} events={len(gg):>5} mean outcomes listed {gg.k.mean():.1f}  mean sum of midpoints {gg.summid.mean():6.1f}  mean YES {gg.yes.mean():.2f}")
