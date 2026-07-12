#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from dynport.data import load_returns
from dynport.actions import ACTION_NAMES, ActionConfig, make_candidate_actions
from dynport.metrics import add_nav, performance_metrics
from dynport.deep_temporal_cql_fqi import DeepTemporalCQLConfig, _standardize_returns, build_transition_data, train_model
from scripts.run_full_oos_fair_eval import eval_model_continuous, _date_str


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--encoder', default='cnn_lstm')
    p.add_argument('--start_date', default='2019-01-02')
    p.add_argument('--end_date', default='2024-12-31')
    p.add_argument('--train_len', type=int, default=250)
    p.add_argument('--test_len', type=int, default=20)
    p.add_argument('--seq_len', type=int, default=120)
    p.add_argument('--warmup', type=int, default=120)
    p.add_argument('--vol_window', type=int, default=20)
    p.add_argument('--eval_cost', type=float, default=0.002)
    p.add_argument('--reward_cost', type=float, default=0.002)
    p.add_argument('--risk_coef', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--fqi_iters', type=int, default=1)
    p.add_argument('--epochs_per_iter', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--q_scale', type=float, default=100.0)
    p.add_argument('--pred_loss_weight', type=float, default=0.05)
    p.add_argument('--cql_alpha', type=float, default=0.05)
    p.add_argument('--dropout', type=float, default=0.10)
    p.add_argument('--kernel_size', type=int, default=5)
    p.add_argument('--lstm_layers', type=int, default=1)
    p.add_argument('--switch_thresholds', type=float, nargs='+', default=[0,0.1,0.2])
    p.add_argument('--roll_offset', type=int, default=0)
    p.add_argument('--roll_count', type=int, default=25)
    p.add_argument('--init_prev_json', default='')
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--torch_num_threads', type=int, default=1)
    a=p.parse_args()
    torch.set_num_threads(max(1,a.torch_num_threads)); torch.manual_seed(a.seed); np.random.seed(a.seed)
    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg=DeepTemporalCQLConfig(start_date=a.start_date,end_date=a.end_date,train_len=a.train_len,test_len=a.test_len,warmup=a.warmup,seq_len=a.seq_len,vol_window=a.vol_window,eval_cost=a.eval_cost,reward_cost=a.reward_cost,risk_coef=a.risk_coef,gamma=a.gamma,fqi_iters=a.fqi_iters,epochs_per_iter=a.epochs_per_iter,batch_size=a.batch_size,hidden=a.hidden,lr=a.lr,weight_decay=a.weight_decay,q_scale=a.q_scale,pred_loss_weight=a.pred_loss_weight,cql_alpha=a.cql_alpha,dropout=a.dropout,encoder=a.encoder,kernel_size=a.kernel_size,lstm_layers=a.lstm_layers,max_rolls=None,switch_thresholds=tuple(a.switch_thresholds),seed=a.seed,torch_num_threads=a.torch_num_threads)
    dates, assets, returns = load_returns(a.data_path, start_date='2011-01-01', end_date=a.end_date)
    ds=pd.Series(pd.to_datetime(dates)); start_idx=int(np.where(ds>=pd.Timestamp(a.start_date))[0][0]); end_idx=int(np.where(ds<=pd.Timestamp(a.end_date))[0][-1])-1
    all_rolls=list(range(start_idx, end_idx-cfg.test_len+1, cfg.test_len))
    roll_starts=all_rolls[a.roll_offset:a.roll_offset+a.roll_count]
    base_actions,covs=make_candidate_actions(returns, assets, ActionConfig(cfg.warmup,cfg.vol_window))
    methods=[f'{cfg.encoder}_cql_thr{thr:g}' for thr in cfg.switch_thresholds]
    if a.init_prev_json and Path(a.init_prev_json).exists():
        prev_data=json.loads(Path(a.init_prev_json).read_text())
        method_prev={m: np.array(prev_data.get(m, np.ones(returns.shape[1])/returns.shape[1]), dtype=float) for m in methods}
    else:
        method_prev={m: np.ones(returns.shape[1])/returns.shape[1] for m in methods}
    segs={m:[] for m in methods}; action_counts={m:{k:0 for k in ACTION_NAMES} for m in methods}; roll_rows=[]
    t0=time.time()
    for j,start in enumerate(roll_starts):
        train_idx=np.arange(start-cfg.train_len-1,start-1); train_idx=train_idx[(train_idx>=cfg.warmup)&(train_idx<len(returns)-1)]
        Rs=_standardize_returns(returns,train_idx)
        data=build_transition_data(returns,Rs,base_actions,covs,train_idx,cfg)
        model=train_model(data,cfg,returns.shape[1])
        te0,te1=start,min(start+cfg.test_len,len(returns)-1)
        for thr in cfg.switch_thresholds:
            m=f'{cfg.encoder}_cql_thr{thr:g}'
            df, counts, prev=eval_model_continuous(model,m,dates,returns,Rs,base_actions,te0,te1,cfg,thr,method_prev[m])
            method_prev[m]=prev; segs[m].append(df)
            for k,v in counts.items(): action_counts[m][k]+=v
            met=performance_metrics(df.net_return,df.gross_return,df.turnover); met.update({'global_roll':a.roll_offset+j,'method':m}); roll_rows.append(met)
        if (j+1)%5==0 or j==len(roll_starts)-1:
            print(f'chunk {a.roll_offset}: roll {j+1}/{len(roll_starts)} elapsed={time.time()-t0:.1f}s', flush=True)
    for m,lst in segs.items():
        if lst: add_nav(pd.concat(lst,ignore_index=True)).to_csv(out/f'{m}_daily_curve_chunk_{a.roll_offset}.csv',index=False)
    pd.DataFrame(roll_rows).to_csv(out/f'roll_level_results_chunk_{a.roll_offset}.csv',index=False)
    pd.DataFrame([{'method':m,**c} for m,c in action_counts.items()]).to_csv(out/f'action_counts_chunk_{a.roll_offset}.csv',index=False)
    Path(out/f'final_prev_chunk_{a.roll_offset}.json').write_text(json.dumps({m: method_prev[m].tolist() for m in methods}, indent=2))
    meta={'assets':assets,'roll_offset':a.roll_offset,'roll_count':len(roll_starts),'first_roll':a.roll_offset,'last_roll':a.roll_offset+len(roll_starts)-1,'covered_start':_date_str(dates[roll_starts[0]+1]) if roll_starts else None,'covered_end':_date_str(dates[min(roll_starts[-1]+cfg.test_len,len(returns)-1)]) if roll_starts else None,'elapsed_sec':time.time()-t0}
    Path(out/f'metadata_chunk_{a.roll_offset}.json').write_text(json.dumps(meta,indent=2))
    print('Saved chunk', a.roll_offset, 'to', out)

if __name__=='__main__': main()
