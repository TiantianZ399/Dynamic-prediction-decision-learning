#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from dynport.data import load_returns
from dynport.metrics import add_nav, performance_metrics
from scripts.run_full_oos_fair_eval import equal_weight_baselines, active_metrics, _date_str


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--chunks_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--encoder', default='cnn_lstm')
    p.add_argument('--switch_thresholds', type=float, nargs='+', default=[0,0.1,0.2])
    p.add_argument('--start_date', default='2019-01-02')
    p.add_argument('--end_date', default='2024-12-31')
    p.add_argument('--test_len', type=int, default=20)
    p.add_argument('--eval_cost', type=float, default=0.002)
    a=p.parse_args()
    chunks=Path(a.chunks_dir); out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dates, assets, returns=load_returns(a.data_path,start_date='2011-01-01',end_date=a.end_date)
    ds=pd.Series(pd.to_datetime(dates)); start_idx=int(np.where(ds>=pd.Timestamp(a.start_date))[0][0]); end_idx=int(np.where(ds<=pd.Timestamp(a.end_date))[0][-1])-1
    all_rolls=list(range(start_idx, end_idx-a.test_len+1, a.test_len))
    oos_end=min(all_rolls[-1]+a.test_len, len(returns)-1)
    baseline_curves=equal_weight_baselines(dates,returns,start_idx,oos_end,cost=a.eval_cost)
    curves={}
    for name,df in baseline_curves.items():
        df=add_nav(df); df.to_csv(out/f'{name}_daily_curve.csv',index=False); curves[name]=df
    for thr in a.switch_thresholds:
        m=f'{a.encoder}_cql_thr{thr:g}'
        parts=[]
        for f in sorted(chunks.glob(f'{m}_daily_curve_chunk_*.csv'), key=lambda x:int(x.stem.split('_')[-1])):
            parts.append(pd.read_csv(f))
        if parts:
            df=pd.concat(parts,ignore_index=True)
            df=add_nav(df)
            df.to_csv(out/f'{m}_daily_curve.csv',index=False)
            curves[m]=df
    agg=[]
    for m,df in curves.items():
        met=performance_metrics(df.net_return,df.gross_return,df.turnover); met.update({'method':m}); agg.append(met)
    agg_df=pd.DataFrame(agg).sort_values('net_sharpe',ascending=False).reset_index(drop=True)
    agg_df.to_csv(out/'aggregate_results_full_oos.csv',index=False)
    active=[]
    for m,df in curves.items():
        if m.startswith('ew_'): continue
        for b,bdf in baseline_curves.items():
            row=active_metrics(df,bdf,b); row.update({'method':m}); active.append(row)
    active_df=pd.DataFrame(active).sort_values(['baseline','active_ir'],ascending=[True,False]).reset_index(drop=True)
    active_df.to_csv(out/'active_metrics_vs_equal_weight.csv',index=False)
    # action counts merge
    ac=[]
    for f in sorted(chunks.glob('action_counts_chunk_*.csv'), key=lambda x:int(x.stem.split('_')[-1])):
        ac.append(pd.read_csv(f))
    if ac:
        tmp=pd.concat(ac,ignore_index=True).fillna(0)
        cols=[c for c in tmp.columns if c!='method']
        counts=tmp.groupby('method',as_index=False)[cols].sum()
        counts.to_csv(out/'action_counts.csv',index=False)
    # roll levels
    rl=[]
    for f in sorted(chunks.glob('roll_level_results_chunk_*.csv'), key=lambda x:int(x.stem.split('_')[-1])):
        rl.append(pd.read_csv(f))
    if rl:
        pd.concat(rl,ignore_index=True).to_csv(out/'roll_level_results.csv',index=False)
    meta={'assets':assets,'n_rolls':len(all_rolls),'covered_oos_start':_date_str(dates[start_idx+1]),'covered_oos_end':_date_str(dates[oos_end]),'start_idx':start_idx,'oos_end_idx':oos_end,'note':'Uses the aligned seven-ETF return panel available in this runtime. The code accepts public Yahoo/Stooq replacement CSVs in the same wide-return format.'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('Aggregate results')
    print(agg_df[['method','n_days','net_nav','net_sharpe','gross_sharpe','max_drawdown','avg_turnover']].to_string(index=False))
    print('\nActive metrics')
    print(active_df[['method','baseline','active_nav','active_ir','active_hit_rate','active_max_drawdown']].to_string(index=False))
    print('Saved to',out)
if __name__=='__main__': main()
