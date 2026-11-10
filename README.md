# Bipedal Payload Variation

**How well do deep reinforcement learning walking policies hold up when a bipedal robot carries extra weight?**

This project trains continuous-control agents on the MuJoCo **Walker2d-v5** biped and builds toward a study of *payload robustness*: what happens to a learned gait when the robot's torso gets heavier, as if it were wearing a heavy backpack. All code is written from scratch in PyTorch, with no RL libraries, so that every part of each algorithm is visible and easy to modify.

| Stage | Status |
|---|---|
| Environment and payload wrapper | Complete |
| Scripted (no-RL) baseline | Complete |
| DDPG agent | Trained, ~850k steps |
| TD3 agent | Trained, 600k steps |
| Payload-variation experiments | In progress |

---

## Table of Contents

- [Results So Far](#results-so-far)
- [Repository Structure](#repository-structure)
- [The Environment](#the-environment)
- [Simulating a Payload](#simulating-a-payload)
- [Algorithms](#algorithms)
- [Training and Evaluation Protocol](#training-and-evaluation-protocol)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [References](#references)

---

## Results So Far

All results below use the **normal torso mass** (`torso_mass_scale = 1.0`). They establish the reference performance that later payload experiments will be compared against.

### TD3

![TD3 convergence](td3_convergence.png)

- Learning starts quickly once the 25k-step random warmup ends. Evaluation return passes **~3,000 by roughly 300k steps** and settles around **4,000–5,000** from about 350k steps onward, peaking above 5,000.
- The critic loss stays bounded, and the actor loss (−Q) falls steadily, with no sign of the value estimates blowing up.
- Some individual evaluations still dip sharply (for example, around 500k steps), so the best checkpoint is kept rather than the final one.

### DDPG

![DDPG convergence](ddpg_convergence.png)

- Learning is slower and much noisier. After the first 600k steps, the best evaluation score was **2,295**, and the most recent one was **941 ± 743**.
- Training was then **resumed from the saved state** (same networks, optimizers and replay buffer) with a health-based stopping rule. It continued to **~850k total steps** and finished with an evaluation return of **~3,800**.
- The evaluation curve swings far more than TD3's, and the critic loss is roughly an order of magnitude more volatile. This is the overestimation instability that TD3 was designed to fix.

### Head-to-head (normal mass)

| | DDPG | TD3 |
|---|---|---|
| Steps to reach ~2,000 eval return | ~550k | ~200k |
| Typical eval return at 600k steps | ~1,000–2,500 | ~4,000–5,000 |
| Stability of evaluation curve | High variance, frequent collapses | Occasional dips, recovers quickly |

> These figures are read from single-seed training curves, so treat them as indicative rather than statistically conclusive. The two runs also differ in random warmup length (10k steps for DDPG, 25k for TD3). See the [Roadmap](#roadmap).

### Media

| File | Description |
|---|---|
| [`walker_no_rl_baseline.mp4`](walker_no_rl_baseline.mp4) | Scripted PD stepping controller with no learning. It falls within ~1.2 s every episode, which confirms the task isn't trivially solvable. |
| [`ddpg_walker.mp4`](ddpg_walker.mp4) | DDPG policy rollout, with a camera that follows the walker. |

---

## Repository Structure

```
.
├── bipedal_ddpg.ipynb          # Environment, scripted baseline, and DDPG agent
├── bipedal_td3.ipynb           # Same environment and building blocks, plus the TD3 agent
├── ddpg_convergence.png        # DDPG learning curve and losses
├── td3_convergence.png         # TD3 learning curve and losses
├── ddpg_walker.mp4             # Trained DDPG rollout
├── walker_no_rl_baseline.mp4   # Scripted (no-RL) baseline rollout
├── LICENSE
└── README.md
```

Each notebook is self-contained and runs top to bottom. Both share identical code for the environment factory, `Actor`, `Critic` and `ReplayBuffer`, so differences in results come from the algorithm itself.

---

## The Environment

[**Walker2d-v5**](https://gymnasium.farama.org/environments/mujoco/walker2d/) (Gymnasium ≥ 1.2, MuJoCo ≥ 3.3) is a planar biped with a torso and two legs, each with a thigh, a shin and a foot.

| Property | Value |
|---|---|
| Observation | 17-D: torso height and angle, 6 joint angles, torso velocities, 6 joint velocities |
| Action | 6-D continuous joint torques in [−1, 1] |
| Control timestep | 0.008 s (125 Hz) |
| Episode length | Up to 1,000 steps (8 s) |
| Reward | Forward velocity + 1.0 per step for staying healthy − 0.001 × ‖action‖² |
| Termination ("unhealthy") | Torso height leaves [0.8, 2.0] m, or torso angle leaves [−1, 1] rad |
| Default masses | Torso 3.665 kg; whole robot 23.677 kg |

The reward and physics are **left at the Gymnasium defaults**, so results are comparable with published benchmarks.

---

## Simulating a Payload

Payload is modelled by the `FixedTorsoMass` wrapper, created through a single factory function:

```python
env = make_env()                        # normal torso
env = make_env(torso_mass_scale=1.5)    # torso 50% heavier
env = make_env(torso_mass_scale=2.0)    # torso mass doubled
```

The wrapper multiplies the torso's **mass and rotational inertia** by the same factor, then calls `mujoco.mj_setConst` so that MuJoCo recomputes the quantities that depend on mass and inertia. This represents a *denser torso of unchanged shape*:

- It does **not** shift the centre of mass or attach an offset backpack.
- Doubling the torso mass does **not** double the whole robot's mass (3.7 kg of 23.7 kg).

---

## Algorithms

Both agents share the same network architecture:

| Network | Architecture |
|---|---|
| Actor | 17 → 256 → 256 → 6, ReLU hidden layers, `tanh` output |
| Critic | (17 + 6) → 256 → 256 → 1, ReLU hidden layers |

### DDPG: Deep Deterministic Policy Gradient

One actor and one critic, each with a slowly updated target copy. The critic learns from the Bellman target $r + \gamma(1-d)\,Q'(s', \mu'(s'))$, and the actor follows the critic's gradient. Because a single critic's errors are maximised by the actor, DDPG is prone to **Q-value overestimation**.

### TD3: Twin Delayed DDPG

TD3 adds three fixes to DDPG:

1. **Clipped double-Q learning.** Two independent critics; the target uses the *minimum* of their two estimates, which counteracts overestimation.
2. **Target policy smoothing.** Clipped Gaussian noise is added to the target action, so the critic can't exploit sharp, narrow peaks in its own value estimate.
3. **Delayed policy updates.** The actor and all target networks update once for every 2 critic updates, so the policy follows a steadier critic.

### Hyperparameters

| Parameter | DDPG | TD3 |
|---|---|---|
| Discount γ | 0.99 | 0.99 |
| Target update rate τ | 0.005 | 0.005 |
| Optimizer / learning rate | Adam, 3e-4 (actor and critic) | Adam, 3e-4 (actor and critics) |
| Batch size | 256 | 256 |
| Replay buffer | 1M transitions | 1M transitions |
| Random warmup | 10k steps | 25k steps |
| Exploration noise (std) | 0.1 | 0.1 |
| Target policy noise / clip | — | 0.2 / 0.5 |
| Policy delay | 1 (every step) | 2 |
| Training steps | 600k, then continued to ~850k | 600k |

---

## Training and Evaluation Protocol

- **Off-policy loop:** one gradient update per environment step after warmup. Transitions store `terminated` (the robot fell), not `truncated` (the time limit was reached), so a time limit is never treated as a true ending when computing targets.
- **Periodic evaluation:** every 10k steps, the policy runs **without exploration noise** for 5 episodes on a separate environment. The mean ± standard deviation is plotted live.
- **Best-checkpoint selection:** the actor with the highest evaluation return is saved and restored at the end of training, because both algorithms can dip after peaking.
- **Health-based continuation (DDPG):** a resumable training cell continues from the current state and stops once all 5 fixed evaluation seeds (100–104) survive the full 1,000-step episode.
- **Videos:** a side-view camera follows the torso's forward position each frame, so the walker stays centred for the whole episode.
- **Reproducibility:** fixed seeds for PyTorch, NumPy, Python's `random` module and the environment.

---

## Getting Started

### Requirements

- Python 3.10+ (developed on 3.11)
- No GPU required. Development used an Apple-silicon Mac; DDPG trains at about 380–420 environment steps per second.

### Install

```bash
pip install "gymnasium[mujoco]>=1.2,<2" "mujoco>=3.3,<4" torch numpy matplotlib imageio imageio-ffmpeg pillow
```

The first cell of each notebook also installs these automatically.

### Rendering backend

The environment cell selects the renderer automatically:

| Platform | `MUJOCO_GL` | Notes |
|---|---|---|
| macOS | `glfw` | Works out of the box |
| Linux / Google Colab | `osmesa` | CPU software rendering; installs `libosmesa6` via `apt-get` if it's missing |

### Run

1. Open `bipedal_ddpg.ipynb` or `bipedal_td3.ipynb` in Jupyter, VS Code or Cursor.
2. **Restart the kernel and run all cells.**
3. Watch the live convergence plot. When training finishes, the evaluation and video cells use the restored best policy.

> **Tip:** use `torch.set_num_threads(1)`. With networks this small, extra CPU threads add overhead and slowed training by about 25% in testing.

---

## Roadmap

- [ ] **Payload sweep:** evaluate the trained DDPG and TD3 policies, without retraining, at `torso_mass_scale ∈ {1.0, 1.25, 1.5, 1.75, 2.0, …}`. Measure return, distance walked and survival time.
- [ ] **Payload-aware training:** train with a randomised torso mass (domain randomisation) and compare robustness against agents trained only on the nominal mass.
- [ ] **Offset backpack:** add a payload that shifts the centre of mass backward, closer to a real backpack.
- [ ] **Statistical rigour:** run multiple seeds per configuration and report the mean with confidence intervals.
- [ ] **Matched comparison:** equalise warmup length (and other settings) between DDPG and TD3, so differences can be attributed to the algorithm alone.
- [ ] **SAC:** add Soft Actor-Critic as a third algorithm.
- [ ] Save trained weights (`.pt`) so policies can be evaluated without retraining.

---

## References

1. Lillicrap, T. P., et al. *Continuous control with deep reinforcement learning* (DDPG). ICLR 2016. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
2. Fujimoto, S., van Hoof, H., & Meger, D. *Addressing Function Approximation Error in Actor-Critic Methods* (TD3). ICML 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
3. Towers, M., et al. *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* 2024. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)
4. Todorov, E., Erez, T., & Tassa, Y. *MuJoCo: A physics engine for model-based control.* IROS 2012.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
