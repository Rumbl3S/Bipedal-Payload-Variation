import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from walker_env import RandomTorsoMass, make_env

torch.set_num_threads(1)

GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
BUFFER_CAPACITY = 1_000_000
EXPLORATION_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
LOG_STD_MIN, LOG_STD_MAX = -20, 2
EVAL_SEED_BASE = 10_000
REFERENCE_SEEDS = range(100, 105)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, state):
        return torch.tanh(self.mean_head(self.body(state)))

    def sample(self, state):
        features = self.body(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = float(terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[idx])
            for array in (self.states, self.actions, self.rewards, self.next_states, self.terminated)
        )

    def __len__(self):
        return self.size


def soft_update(target, source):
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1 - TAU)
            target_parameter.add_(TAU * parameter)


def frozen_copy(network):
    copy = deepcopy(network)
    copy.requires_grad_(False)
    return copy


class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = frozen_copy(self.actor)
        self.target_critic = frozen_copy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def act(self, obs):
        with torch.no_grad():
            action = self.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        action = action + np.random.normal(0.0, EXPLORATION_NOISE, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch):
        states, actions, rewards, next_states, terminated = batch
        with torch.no_grad():
            next_values = self.target_critic(next_states, self.target_actor(next_states))
            targets = rewards + GAMMA * (1 - terminated) * next_values

        critic_loss = ((self.critic(states, actions) - targets) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        soft_update(self.target_actor, self.actor)
        soft_update(self.target_critic, self.critic)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.target_actor = frozen_copy(self.actor)
        self.target_critic_1 = frozen_copy(self.critic_1)
        self.target_critic_2 = frozen_copy(self.critic_2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=LEARNING_RATE
        )
        self.update_count = 0
        self.last_actor_loss = float("nan")

    def act(self, obs):
        with torch.no_grad():
            action = self.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        action = action + np.random.normal(0.0, EXPLORATION_NOISE, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch):
        states, actions, rewards, next_states, terminated = batch
        self.update_count += 1
        with torch.no_grad():
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = (self.target_actor(next_states) + noise).clamp(-1.0, 1.0)
            next_values = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            )
            targets = rewards + GAMMA * (1 - terminated) * next_values

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = ((q1 - targets) ** 2).mean() + ((q2 - targets) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.update_count % POLICY_DELAY == 0:
            self.critic_1.requires_grad_(False)
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic_1.requires_grad_(True)

            soft_update(self.target_actor, self.actor)
            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            self.last_actor_loss = actor_loss.item()

        return {"critic_loss": critic_loss.item(), "actor_loss": self.last_actor_loss}


class SAC:
    def __init__(self, state_dim, action_dim):
        self.actor = GaussianActor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.target_critic_1 = frozen_copy(self.critic_1)
        self.target_critic_2 = frozen_copy(self.critic_2)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=LEARNING_RATE
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LEARNING_RATE)

    def act(self, obs):
        with torch.no_grad():
            action, _ = self.actor.sample(torch.as_tensor(obs, dtype=torch.float32))
        return action.numpy().astype(np.float32)

    def update(self, batch):
        states, actions, rewards, next_states, terminated = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_values = torch.min(
                self.target_critic_1(next_states, next_actions),
                self.target_critic_2(next_states, next_actions),
            ) - alpha * next_log_probs
            targets = rewards + GAMMA * (1 - terminated) * next_values

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = ((q1 - targets) ** 2).mean() + ((q2 - targets) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        new_actions, log_prob = self.actor.sample(states)
        q_new = torch.min(self.critic_1(states, new_actions), self.critic_2(states, new_actions))
        actor_loss = (alpha * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1)
        soft_update(self.target_critic_2, self.critic_2)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "entropy": -log_prob.mean().item(),
        }


AGENTS = {"ddpg": DDPG, "td3": TD3, "sac": SAC}


def run_policy(actor, env, seed):
    obs, _ = env.reset(seed=seed)
    start_x = float(env.unwrapped.data.qpos[0])
    total, done, terminated = 0.0, False, False
    while not done:
        with torch.no_grad():
            action = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    return total, float(env.unwrapped.data.qpos[0]) - start_x, bool(terminated)


def evaluate(actor, eval_envs, episodes):
    per_mass = {}
    for scale, env in eval_envs.items():
        returns = [run_policy(actor, env, EVAL_SEED_BASE + i)[0] for i in range(episodes)]
        per_mass[scale] = float(np.mean(returns))
    all_means = list(per_mass.values())
    return float(np.mean(all_means)), per_mass


def reference_distances(actor):
    env = make_env(torso_mass_scale=1.0)
    try:
        return {seed: run_policy(actor, env, seed)[1] for seed in REFERENCE_SEEDS}
    finally:
        env.close()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, tag, warmup, episodes, evals, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    steps = [row["step"] for row in episodes]
    returns = np.array([row["return"] for row in episodes])
    ax1.plot(steps, returns, color="tab:blue", alpha=0.25, label="Training episode")
    if len(returns) >= 20:
        ax1.plot(steps[19:], np.convolve(returns, np.ones(20) / 20, mode="valid"), color="tab:blue", label="Training (20-episode avg)")
    ax1.plot([row["step"] for row in evals], [row["eval_return"] for row in evals], color="tab:orange", marker="o", markersize=3, label="Evaluation (no noise)")
    ax1.axvline(warmup, color="gray", linestyle=":", label="End of random warmup")
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Episode return")
    ax1.set_title(f"{tag}: return")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    loss_steps = [row["step"] for row in losses]
    ax2.plot(loss_steps, [row["critic_loss"] for row in losses], color="tab:red")
    ax2.set_yscale("log")
    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Critic loss (log)", color="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(loss_steps, [row["actor_loss"] for row in losses], color="tab:green")
    ax2b.set_ylabel("Actor loss", color="tab:green")
    ax2.set_title("Losses")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPG, TD3 or SAC on Walker2d-v5.")
    parser.add_argument("--algo", choices=sorted(AGENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=600_000)
    parser.add_argument("--warmup", type=int, default=25_000)
    parser.add_argument("--dr", action="store_true", help="Randomise torso mass every episode.")
    parser.add_argument("--mass-low", type=float, default=1.0)
    parser.add_argument("--mass-high", type=float, default=2.0)
    parser.add_argument("--eval-masses", type=float, nargs="+", default=None)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    tag = f"{args.algo}_dr_seed{args.seed}" if args.dr else f"{args.algo}_seed{args.seed}"
    eval_masses = args.eval_masses or ([1.0, 1.5, 2.0] if args.dr else [1.0])

    models_dir = os.path.join(args.root, "models")
    logs_dir = os.path.join(args.root, "results", "training")
    figures_dir = os.path.join(args.root, "figures", "training")
    for folder in (models_dir, logs_dir, figures_dir):
        os.makedirs(folder, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    if args.dr:
        env = RandomTorsoMass(env, args.mass_low, args.mass_high, seed=args.seed)
    eval_envs = {scale: make_env(torso_mass_scale=scale) for scale in eval_masses}
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.action_space.seed(args.seed)

    agent = AGENTS[args.algo](state_dim, action_dim)
    buffer = ReplayBuffer(BUFFER_CAPACITY, state_dim, action_dim)

    episodes, evals, losses = [], [], []
    best_eval, best_state, best_step = -np.inf, None, 0
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    start = time.time()
    print(f"{tag}: {args.steps:,} steps, warmup {args.warmup:,}, "
          f"{'torso mass ~ U[%.2f, %.2f]' % (args.mass_low, args.mass_high) if args.dr else 'normal mass'}, "
          f"checkpoint chosen on masses {eval_masses}", flush=True)

    for step in range(1, args.steps + 1):
        action = env.action_space.sample() if step <= args.warmup else agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        episode_return += float(reward)
        obs = next_obs

        if step > args.warmup and len(buffer) >= BATCH_SIZE:
            stats = agent.update(buffer.sample(BATCH_SIZE))
            if step % 1_000 == 0:
                losses.append({"step": step, **stats})

        if terminated or truncated:
            episodes.append({
                "step": step,
                "return": episode_return,
                "fell": bool(terminated),
                "torso_mass_scale": getattr(env, "scale", 1.0),
            })
            obs, _ = env.reset()
            episode_return = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            mean_return, per_mass = evaluate(agent.actor, eval_envs, args.eval_episodes)
            evals.append({"step": step, "eval_return": mean_return, **{f"eval_x{k:g}": v for k, v in per_mass.items()}})
            if mean_return > best_eval:
                best_eval, best_step = mean_return, step
                best_state = deepcopy(agent.actor.state_dict())
            elapsed = time.time() - start
            print(f"{tag} | step {step:,}/{args.steps:,} | {elapsed / 60:.1f} min | {step / elapsed:.0f} steps/s | "
                  f"eval {mean_return:.0f} | best {best_eval:.0f} @ {best_step:,}", flush=True)

    for eval_env in eval_envs.values():
        eval_env.close()
    env.close()

    agent.actor.load_state_dict(best_state)
    agent.actor.eval()
    torch.save(agent.actor.state_dict(), os.path.join(models_dir, f"{tag}.pt"))
    with open(os.path.join(models_dir, f"{tag}_reference.json"), "w") as f:
        json.dump(reference_distances(agent.actor), f, indent=2)
    with open(os.path.join(models_dir, f"{tag}_config.json"), "w") as f:
        json.dump({
            **vars(args),
            "tag": tag,
            "eval_masses": eval_masses,
            "best_eval_return": best_eval,
            "best_step": best_step,
            "minutes": (time.time() - start) / 60,
            "gamma": GAMMA, "tau": TAU, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "optimizer": "Adam", "buffer_capacity": BUFFER_CAPACITY,
        }, f, indent=2)

    write_csv(os.path.join(logs_dir, f"{tag}_episodes.csv"), episodes)
    write_csv(os.path.join(logs_dir, f"{tag}_evals.csv"), evals)
    write_csv(os.path.join(logs_dir, f"{tag}_losses.csv"), losses)
    save_plot(os.path.join(figures_dir, f"{tag}.png"), tag, args.warmup, episodes, evals, losses)
    print(f"{tag}: saved best actor (eval {best_eval:.0f} at step {best_step:,}) to models/{tag}.pt", flush=True)


if __name__ == "__main__":
    main()
