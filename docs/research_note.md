# Research note: prediction layer + dynamic decision layer

## Motivation

Classical portfolio optimizers are often static. They solve a one-period or receding-horizon allocation problem from current return and risk estimates, but the previous portfolio weight is not always treated as a state variable whose value affects future costs and future rebalancing opportunities.

The project studies a modular alternative:

\[
X_{t-L:t}, w_{t-1} \rightarrow z_t \rightarrow \{\hat r_{t+1:t+H}, Q(s_t,a)\} \rightarrow w_t.
\]

The prediction layer uses deep learning to capture historical market states and future return/risk information. The decision layer uses dynamic programming / reinforcement learning to model the continuation value of actions under transaction costs.

## Compared approaches

| Approach | Training objective | Main advantage | Main weakness |
|---|---|---|---|
| Two-stage | Forecast loss, then optimize | Simple, fast, stable | Prediction-decision mismatch |
| Integrated/IPMO | Portfolio loss through optimizer | Direct decision alignment | Slow; optimizer-in-loop; noisy portfolio loss |
| Bellman planning DP | Backward planning over candidate paths | Cheap path-dependent decision layer | Does not learn alpha source |
| FQI | Offline Bellman value learning | Fits fixed historical data | Can overtrade due to Q overestimation |
| Deep CQL-FQI | Deep state + conservative Bellman learning | Learns dynamic value with no-trade discipline | Still approximate finite-action method |
| TD3/PPO | Deep continuous-action RL | More expressive | Data-hungry and harder to stabilize in small financial datasets |

## Reward design

The decision reward is built to discourage over-aggressive active trading:

\[
R_t(w_t,w_{t-1}) = w_t^\top r_{t+1}
- c \|w_t-w_{t-1}\|_1
- \eta w_t^\top \Sigma_t w_t.
\]

This turns portfolio choice into a state-dependent problem where previous holdings matter. The conservative Deep CQL-FQI version adds a no-trade regularizer and a switching threshold.

## Current conclusion

The preliminary tests do not support using integrated learning as-is: two-stage was faster and stronger in the small rolling tests. Plain FQI also overtraded. The most promising current direction is a constrained deep prediction + conservative Bellman decision layer.
