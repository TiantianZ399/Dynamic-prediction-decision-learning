# Deep Temporal CQL-FQI Diagnostic

This note records a minimal 12-roll diagnostic after adding temporal deep encoders to the conservative finite-action FQI decision layer.

## Setup

- Universe: 7 aligned ETFs.
- OOS start: 2019-01-02.
- Rolling windows: 12.
- OOS days: 240.
- Training lookback: 250 days.
- Test block: 20 days.
- State history: past 120 daily returns.
- Transaction cost: 20 bps times L1 turnover.
- Reward: realized return minus transaction cost minus covariance risk penalty.
- RL method: finite-action conservative CQL-FQI.
- Encoders tested: MLP, TCN, CNN-LSTM.
- Fast diagnostic training: hidden=16, fqi_iters=1, epochs_per_iter=1.

## Results

```text
          run              method  n_days  net_nav  net_sharpe  gross_sharpe  max_drawdown  avg_turnover
     mlp_fast        equal_weight     240 1.143944    2.768878      2.768878     -0.020990      0.000000
     tcn_fast        equal_weight     240 1.143944    2.768878      2.768878     -0.020990      0.000000
cnn_lstm_fast        equal_weight     240 1.143944    2.768878      2.768878     -0.020990      0.000000
cnn_lstm_fast cnn_lstm_cql_thr0.2     240 1.072693    1.104407      1.797179     -0.052170      0.093021
cnn_lstm_fast   cnn_lstm_cql_thr0     240 1.025334    0.327042      1.114759     -0.081902      0.146317
     tcn_fast      tcn_cql_thr0.2     240 1.016136    0.289890      2.046654     -0.051200      0.216571
cnn_lstm_fast cnn_lstm_cql_thr0.1     240 1.012005    0.183952      0.937808     -0.082720      0.134380
     mlp_fast      mlp_cql_thr0.2     240 1.002852    0.079256      1.607572     -0.065366      0.180203
     mlp_fast      mlp_cql_thr0.1     240 0.918901   -1.200727      1.315302     -0.133545      0.348464
     tcn_fast      tcn_cql_thr0.1     240 0.929725   -1.284964      1.535652     -0.090248      0.313090
     tcn_fast        tcn_cql_thr0     240 0.870504   -1.689976      0.795564     -0.140428      0.411127
     mlp_fast        mlp_cql_thr0     240 0.894151   -1.784940      1.425526     -0.127613      0.409592
```

## Interpretation

The CNN-LSTM encoder is the best temporal encoder in this fast diagnostic. It beats the fast MLP and TCN variants, mainly because it trades less aggressively than TCN and has better net Sharpe after transaction costs.

However, equal weight remains the strongest benchmark in this short OOS slice. The result should be interpreted as an implementation and diagnostic result, not as a deployable strategy. The main takeaway is that adding temporal deep learning does not automatically create alpha; the decision layer still needs stronger no-trade discipline and more robust validation.

## Recommended current default

For this repository, use `--encoder cnn_lstm` as the default temporal deep encoder for the next diagnostic experiments, while keeping TCN as an ablation.
