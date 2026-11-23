# RL-Based Adaptive Queue Flushing for Real-Time Video

This project investigates whether tabular Q-learning can improve real-time video delivery over a congested wireless link. The agent observes network conditions and decides whether to retain the queue or flush eligible outdated frames.

The project builds on the queue-flushing approach presented in *Real-Time Video Gets a Fast Lane via Smart Queue Flushing at the Wireless Edge* (IEEE INFOCOM 2026).

## Implementation

The Python simulator models a video sender, a finite router queue, and a receiver. Frames arrive at 30 FPS, and available bandwidth follows synthetic traces representing stable conditions, fluctuations, and a sudden capacity drop.

| Setting | Value |
|---|---|
| Video rate | 30 FPS |
| Frame size | 140,000–240,000 bits |
| Queue capacity | 12 frames |
| Playback deadline | 150 ms |
| Important-frame frequency | One every 15 frames |
| Training | 600 episodes per agent per seed |
| Training seeds | 5 |
| Evaluation | 20 held-out traces per scenario |

The simulator tracks partially transmitted frames and drains the remaining queue after frame generation ends.

## Policies

| Policy | Behavior |
|---|---|
| FIFO | Retains queued frames without proactive flushing |
| Fixed threshold | Triggers selective flushing when predicted delay exceeds the playback deadline |
| Q-learning with importance | Uses 54 states based on occupancy, bandwidth, frame delay, and frame importance |
| Q-learning without importance | Uses 27 states with the importance input removed |
| Gecko A/B selective | Uses adaptive keep/flush decisions with the same selective flush rule as Q-learning |
| Gecko A/B atomic | Uses adaptive decisions and clears the flow queue, with a veto when important frames are at risk |

Selective flushing removes normal frames predicted to miss their deadlines and preserves important frames. Both Q-learning variants use this same rule, allowing the experiment to isolate the effect of the importance input.

The adaptive A/B baselines include initial keep/flush trials, time-decayed outcome averages, forced exploration after five identical actual responses, and adaptive alert thresholds.

## Q-Learning

The agent selects one of two actions.

- **Keep** retains the queue.
- **Flush** removes eligible outdated normal frames.

The reward penalizes normalized delivery delay, useful-frame loss, and missed deadlines.

\[
R = -0.5D - 0.5L - 2.0M
\]

Training uses a learning rate of 0.10, a discount factor of 0.95, and epsilon-greedy exploration that decays from 1.0 to a minimum of 0.05.

All policies use the same held-out video traffic and bandwidth traces for evaluation. The RL reward, A/B bad-frame counting, and evaluation share a 150 ms delay target.

## Results

The table shows on-time delivery as a percentage of all generated frames. Q-learning values are averaged across five training seeds.

| Policy | Stable | Fluctuating | Sudden drop |
|---|---:|---:|---:|
| FIFO | 100.00% | 52.60% | 64.33% |
| Fixed threshold | 100.00% | **86.82%** | **77.70%** |
| Q-learning with importance | 100.00% | 81.00% | 72.01% |
| Q-learning without importance | 100.00% | 81.48% | 72.82% |
| Gecko A/B selective | 100.00% | 75.55% | 66.68% |
| Gecko A/B atomic | 100.00% | 69.83% | 65.82% |

Q-learning improved on-time delivery over FIFO and both adaptive A/B baselines. The importance-aware agent reduced P95 delay compared with selective A/B, with increased frame loss.

| Scenario | Policy | P95 delay | Frame loss |
|---|---|---:|---:|
| Fluctuating | Q-learning with importance | 162 ms | 13.14% |
| Fluctuating | Gecko A/B selective | 273 ms | 10.62% |
| Sudden drop | Q-learning with importance | 163 ms | 23.03% |
| Sudden drop | Gecko A/B selective | 498 ms | 20.02% |

The fixed threshold policy achieved the best overall performance under the tested conditions. Adding frame importance to the Q-learning state produced no consistent improvement.

## Run in Google Colab

1. Upload `RLv2.ipynb` to Google Colab.
2. Select a CPU runtime.
3. Select **Runtime → Run all**.
4. Review the metrics and comparison plot.
5. Run the final cell to download the results ZIP.

The notebook installs NumPy, Matplotlib, and pandas.

## Output Files

| File | Contents |
|---|---|
| `per_seed_metrics.csv` | Metrics for each policy, scenario, and training seed |
| `summary.csv` | Mean metrics and standard deviations |
| `gecko_comparison.png` | Policy comparison plots |
| `alert_audit.csv` | Requested actions, actual actions, and adaptive thresholds |
| `q_*.npy` | Trained Q-tables |

Metrics include mean delay, P95 delay, overall loss, overall on-time delivery, and separate on-time delivery and loss rates for important and normal frames.

Delay metrics describe delivered frames. On-time delivery and loss rates use all generated frames as their denominator. RL standard deviations measure variation across training seeds on a fixed evaluation set.

## Limitations

- Bandwidth traces and video frame sizes are synthetic.
- The simulator uses a fixed video workload and does not implement sender congestion control.
- The Gecko baselines are simulation adaptations of the paper’s decision logic.
- A/B observation timing, averaging rate, and threshold bounds are assumed parameters.
- Signaling is immediate, with no RTP/RTCP communication or feedback delay.
- Frame importance is represented by a binary label, without codec dependencies or decoded video quality.
- The current evaluation uses one queue capacity, bitrate range, and playback deadline.

## Remaining Work

- Vary video bitrate, queue capacity, and playback deadline.
- Evaluate sensitivity to the A/B observation window and averaging rate.
- Compare alternative reward weights.
- Test longer sessions and real wireless traces.
- Complete the final IEEE report and presentation.

## Reference

Y. Zhang et al., “Real-Time Video Gets a Fast Lane via Smart Queue Flushing at the Wireless Edge,” IEEE INFOCOM, 2026. DOI 10.1109/INFOCOM59046.2026.11571420.

### Bitrate robustness experiments

The [bitrate experiments](experiments/bitrate/README.md) reproduce the high-load pilot, evaluate the archived agents on fresh traffic, and compare single-rate and mixed-rate training with independent training seeds.
