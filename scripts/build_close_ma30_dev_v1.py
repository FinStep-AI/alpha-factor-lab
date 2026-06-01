#!/usr/bin/env python3
"""
因子: 30日均线偏离度 v1 (MA30 Price Deviation v1)
factor_id: close_ma30_dev_v1

逻辑:
  因子 = close / MA30 - 1  （window=20日滚动计算, log_amount中性化）
  做多"价格持续站上MA30"的股票（正向 momentum / trend）。
  MA20的bias因子已经有了(close_bias_20/turnover_bias 系列)，MA30偏度这个单独长度是干净的。

  方向验证: 先正向做（均衡）；若IC显示方向反转再掉头。
"""
import numpy as np, pandas as pd
from pathlib import Path

def main():
    D = Path("data")
    print("loading...")
    df = pd.read_csv(D/"csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code','date']).reset_index(drop=True)

    # MA30
    df['ma30'] = df.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(30, min_periods=20).mean()
    )
    raw = df['close'] / df['ma30'] - 1.0              # price above(+)/below(-) MA30
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))
    df['raw_factor'] = raw

    res = df[['date','stock_code','raw_factor','log_amount']].dropna()
    print(f"raw rows={len(res):,}  dates={res['date'].min().date()}~{res['date'].max().date()}")

    # 5% winsorize
    parts=[]
    for dt,g in res.groupby('date',sort=False):
        v=g['raw_factor'].values
        lo,hi=np.nanquantile(v,.025),np.nanquantile(v,.975)
        g2=g.copy(); g2['raw_factor']=np.clip(v,lo,hi); parts.append(g2)
    res=pd.concat(parts,ignore_index=True)

    # cross-sectional z-score
    res['z'] = res.groupby('date')['raw_factor'].transform(
        lambda x:(x-x.mean())/x.std() if x.std()>0 else 0)

    # amount OLS neutralization
    parts2=[]
    for dt,g in res.groupby('date',sort=False):
        g=g.dropna(subset=['z','log_amount'])
        if len(g)<20:
            g['n']=np.nan; parts2.append(g[['n']]); continue
        x=g['log_amount'].values; y=g['z'].values
        xm=np.nanmean(x); ym=np.nanmean(y)
        b=np.nansum((x-xm)*(y-ym))/(np.nansum((x-xm)**2)+1e-10)
        a=ym-b*xm; g=g.copy(); g['n']=y-(a+b*x); parts2.append(g[['n']])
    neu=pd.concat(parts2,ignore_index=False)
    res['z2']=neu['n']
    res=res.dropna(subset=['z2'])

    res['factor_value']=res.groupby('date')['z2'].transform(
        lambda x:(x-x.mean())/x.std() if x.std()>0 else 0).clip(-3,3)

    out=res[['date','stock_code','factor_value']].dropna()
    out['date']=out['date'].dt.strftime('%Y-%m-%d')
    out_path=D/"factor_close_ma30_dev_v1.csv"
    out.to_csv(out_path,index=False)
    print(f"saved {out_path}  ({len(out):,} rows, {out['date'].min()}~{out['date'].max()})")
    print(out['factor_value'].describe())

if __name__=="__main__": main()
