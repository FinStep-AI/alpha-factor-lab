#!/usr/bin/env python3
"""
ROE Stability Quality Factor v1 — roe_stability_v1
"""
import argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

LAG, EPS, MAD_K = 45, 1.0, 5.5

def build(fund_path, kline_path):
    fund  = pd.read_csv(fund_path)
    kline = pd.read_csv(kline_path)
    fund["rdate"]  = pd.to_datetime(fund["report_date"])
    kline["date"]  = pd.to_datetime(kline["date"])
    fund["stock_code"]  = fund["stock_code"].astype(int)
    kline["stock_code"] = kline["stock_code"].astype(int)
    kline.sort_values(["stock_code","date"], inplace=True)

    # 8q rolling std(ROE), shift(1) before rolling = 纯粹历史已发布季报
    fund.sort_values(["stock_code","rdate"], inplace=True)
    fund["avail"] = fund["rdate"] + pd.Timedelta(days=LAG)
    fund["roe_std8"] = fund.groupby("stock_code")["roe"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=6).std())

    # 因子 = 1/(std+eps); 越稳越高
    fund["fr"] = EPS / (fund["roe_std8"] + EPS)

    # 只保留 std 有效且 lag 观察不浅的行
    fund = fund.dropna(subset=["fr","avail"])

    # kline 20日均额 作为 size proxy
    kline["ln_a20"] = np.log(
        kline.groupby("stock_code")["amount"]
        .transform(lambda x: x.rolling(20, min_periods=10).mean()) + 1)

    # fund→trade_date: daily forward-fill per stock
    recs=[]
    for dt, dk in kline[["date","stock_code","ln_a20"]].groupby("date", sort=False):
        sub = fund[fund["avail"] <= dt].sort_values(
            ["stock_code","avail"], ascending=[True,False]).drop_duplicates("stock_code")
        m   = dk.merge(sub[["stock_code","fr"]], on="stock_code", how="left")
        recs.append(m.dropna(subset=["fr"]))
    df = pd.concat(recs, ignore_index=True)

    # cross-section OLS neutralize, MAD, z-score
    out=[]
    for dt, g in df.groupby("date", sort=False):
        x = g["ln_a20"].values; y = g["fr"].values
        if len(g) < 30: continue
        b = float(np.linalg.lstsq(np.vstack([x,np.ones_like(x)]).T, y, rcond=None)[0][0])
        r = y - b*x
        med=np.median(r); mad=np.median(np.abs(r-med))+1e-9
        lo,hi = med-MAD_K*mad, med+MAD_K*mad
        r=np.clip(r,lo,hi); mu,sd=r.mean(),r.std()
        g2 = g[["date","stock_code"]].copy()
        g2["f"]=(r-mu)/(sd+1e-9); out.append(g2)
    return pd.concat(out, ignore_index=True)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--fund",default="data/csi1000_fundamental_cache.csv")
    pa.add_argument("--kline",default="data/csi1000_kline_raw.csv")
    pa.add_argument("--output",required=True)
    a=pa.parse_args()
    base=Path(__file__).resolve().parents[2]
    df=build(str(base/a.fund),str(base/a.kline))
    df["stock_code"]=df["stock_code"].astype(str).str.zfill(6)
    df["date"]=df["date"].dt.strftime("%Y-%m-%d")
    df.rename(columns={"f":"factor_value"}).to_csv(a.output,index=False)
    print(f"wrote {len(df)} rows  {df['date'].min()} ~ {df['date'].max()}")

if __name__=="__main__": main()
