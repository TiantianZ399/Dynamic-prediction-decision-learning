# Full OOS fair equal-weight and active-return evaluation

This addendum runs the fair-equal-weight and active-return experiments requested for the workshop draft.

## Setup

- Universe: AGG.P, DBC.P, GLD.P, IWM.P, LQD.P, MUB.P, VTI.P
- OOS covered: 2019-01-03 to 2024-12-17
- Rolling blocks: 75
- OOS decisions: 1500
- Train lookback: 250 trading days
- Test block: 20 trading days
- State window: 120 days
- Transaction cost: 20 bps times L1 turnover
- Learned policy: CNN-LSTM Conservative FQI proxy with thresholds 0, 0.1, 0.2

Note: the runtime could not fetch Yahoo/Stooq data because outbound DNS is unavailable. The experiment was run on the aligned seven-ETF return panel available in the working directory. The included scripts accept a public Yahoo/Stooq replacement CSV in the same wide-return format.

## Aggregate results

| method                |   n_days |   net_nav |   ann_return |   ann_vol |   net_sharpe |   gross_sharpe |   max_drawdown |   avg_turnover |   total_turnover |
|:----------------------|---------:|----------:|-------------:|----------:|-------------:|---------------:|---------------:|---------------:|-----------------:|
| ew_zero_cost_target   |     1500 |    1.5167 |       0.0778 |    0.0996 |       0.7529 |         0.7529 |        -0.2070 |         0.0000 |           0.0000 |
| ew_monthly_rebalanced |     1500 |    1.4948 |       0.0751 |    0.0987 |       0.7337 |         0.7397 |        -0.2074 |         0.0012 |           1.7506 |
| ew_buy_hold           |     1500 |    1.5239 |       0.0792 |    0.1041 |       0.7324 |         0.7324 |        -0.2075 |         0.0000 |           0.0000 |
| ew_daily_rebalanced   |     1500 |    1.4915 |       0.0748 |    0.0996 |       0.7246 |         0.7529 |        -0.2076 |         0.0056 |           8.3901 |
| cnn_lstm_cql_thr0.2   |     1500 |    1.6747 |       0.1016 |    0.1420 |       0.6814 |         1.0105 |        -0.2033 |         0.0917 |         137.4752 |
| cnn_lstm_cql_thr0.1   |     1500 |    0.6863 |      -0.0449 |    0.1852 |      -0.2480 |         0.2951 |        -0.4146 |         0.1992 |         298.7522 |
| cnn_lstm_cql_thr0     |     1500 |    0.5886 |      -0.0715 |    0.1715 |      -0.4324 |         0.3266 |        -0.4684 |         0.2575 |         386.1871 |

## Active metrics for best learned policy

Best learned policy by net Sharpe among learned methods: `cnn_lstm_cql_thr0.2`.

| method              | baseline              |   active_nav |   active_ir |   active_hit_rate |   active_max_drawdown |
|:--------------------|:----------------------|-------------:|------------:|------------------:|----------------------:|
| cnn_lstm_cql_thr0.2 | ew_buy_hold           |       1.0973 |      0.2065 |            0.4887 |               -0.2331 |
| cnn_lstm_cql_thr0.2 | ew_daily_rebalanced   |       1.1232 |      0.2439 |            0.5160 |               -0.2180 |
| cnn_lstm_cql_thr0.2 | ew_monthly_rebalanced |       1.1218 |      0.2430 |            0.4933 |               -0.2197 |
| cnn_lstm_cql_thr0.2 | ew_zero_cost_target   |       1.1045 |      0.2160 |            0.4907 |               -0.2219 |

## Interpretation

The stronger, fairer evaluation changes the story slightly. The best learned policy has higher terminal NAV than all EW variants, but it still has lower annualized Sharpe because it takes more active risk and pays turnover costs. It has a small positive active information ratio versus the fair EW variants, but the active advantage is not yet strong enough to claim robust superiority.

The workshop-safe conclusion is: the uncertainty/no-trade gate helps the learned policy avoid the severe overtrading seen in lower-threshold policies, but equal-weight remains a difficult benchmark. The framework should be presented as an active overlay whose goal is positive risk-controlled active return, not as a full replacement for equal weight.

## Files

- `aggregate_results_full_oos.csv`: absolute performance metrics.
- `active_metrics_vs_equal_weight.csv`: active NAV / information ratio versus each EW variant.
- `*_daily_curve.csv`: daily strategy and baseline curves.
- `figures/`: NAV, active NAV, Sharpe, and turnover plots.
