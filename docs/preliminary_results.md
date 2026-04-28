# Preliminary seven-ETF diagnostic results

These results are included for transparency and should not be treated as final empirical evidence. The real ETF data used for these diagnostics are not included in the public repository.

## Setting

- Universe: seven broad ETFs: AGG, DBC, GLD, IWM, LQD, MUB, VTI.
- Out-of-sample start: 2019-01-02.
- Rolling lookback: 250 days.
- Test block: 20 trading days.
- State window: 120 days for Deep CQL-FQI.
- Transaction cost: 20 bps times L1 turnover.
- Diagnostic length: 12 rolls / 240 daily decisions.

## Main comparison

| Method | Net NAV | Net Sharpe | Gross Sharpe | Max DD | Avg Turnover |
|---|---:|---:|---:|---:|---:|
| Equal weight | 1.1439 | 2.7689 | 2.7689 | -0.0210 | 0.0000 |
| Deep CQL-FQI, threshold 0.20 | 1.1078 | 1.6897 | 1.8478 | -0.0504 | 0.0213 |
| Deep CQL-FQI, threshold 0.10 | 1.0974 | 1.3974 | 2.0006 | -0.0504 | 0.0892 |
| Deep CQL-FQI, threshold 0.00 | 1.0988 | 1.2908 | 2.7152 | -0.0603 | 0.2121 |
| Two-stage, H=10 | 1.0740 | 1.4769 | -- | -- | 0.1226 |
| Integrated, H=10 | 1.0572 | 1.1455 | -- | -- | 0.1549 |
| Plain FQI, gamma=0 | 1.0821 | 0.6968 | 1.8673 | -0.0690 | 0.2970 |
| Plain FQI, gamma=0.95 | 0.9949 | 0.0299 | 0.9978 | -0.0765 | 0.2589 |

## Interpretation

Equal weight is a strong benchmark in this universe because the ETF set is diversified and the diagnostic period was favorable to broad passive exposure. The learned methods must create enough active value to compensate for estimation error and transaction costs.

The most important observation is not that the current Deep CQL-FQI beats equal weight; it does not. The observation is that conservative dynamic decision learning materially reduces the overtrading problem observed in plain FQI and performs better than the tested two-stage/integrated configurations in this short diagnostic.

## Next steps

1. Run a full 2019--2024 walk-forward evaluation.
2. Add fairer equal-weight variants: buy-and-hold equal weight, daily rebalanced equal weight, and monthly rebalanced equal weight.
3. Report active excess return relative to equal weight.
4. Tune conservative penalty, switch threshold, and reward risk coefficient on validation windows.
5. Add continuous-action TD3 or TD3+BC only after finite-action Conservative FQI shows stable value.
