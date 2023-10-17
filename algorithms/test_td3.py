import argparse
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from torch.distributions.categorical import Categorical
from utils import *


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="JoinGym",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--disable-cartesian-product", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, cartesian product will be disabled")
    parser.add_argument("--enable-bushy", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, allow bushy join trees")
    parser.add_argument("--eval-frequency", type=float, default=20000, help="frequency of evaluation ")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
                        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--gradient-clip", type=float, default=5.0,
        help="max l2 norm of gradient")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
                        help="the replay memory buffer size")  # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--per", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use prioritized experience replay")
    parser.add_argument("--per-alpha", type=float, default=0.5,
        help="")
    parser.add_argument("--per-start-beta", type=float, default=0.4,
        help="")
    parser.add_argument("--per-end-beta", type=float, default=1.0,
        help="")


    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 1)")  # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
                        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--update-frequency", type=int, default=4,
                        help="the frequency of training updates")
    parser.add_argument("--policy-frequency", type=int, default=2,
                        help="the frequency of updates for the target networks")
    args = parser.parse_args()
    # fmt: on
    return args


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = get_critic(env.observation_space, env.action_space.n)

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = get_actor(env.observation_space, env.action_space.n)

    def forward(self, x):
        return self.network(x)

    def get_action(self, x, mask):
        logits = self(x)
        logits[~mask] = -float("inf")
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        return action, action_probs


if __name__ == "__main__":
    args = parse_args()
    if args.run_name:
        run_name = args.run_name
    else:
        cli = get_cli_overrides()
        override_str = "".join([f"[{arg}]" for arg in cli])
        run_name = f"[td3]{override_str}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    split = get_query_ids_split()
    train_env = make_env(args.seed, split["train"], args.disable_cartesian_product, args.enable_bushy)()
    eval_envs_map = {
        key: make_env(seed=args.seed+i, query_ids=split[key],
                    disable_cartesian_product=args.disable_cartesian_product,
                    enable_bushy=args.enable_bushy)()
            for i, key in enumerate(["train", "val", "test"])
    }

    actor = Actor(train_env).to(device)
    qf1 = SoftQNetwork(train_env).to(device)
    qf2 = SoftQNetwork(train_env).to(device)
    qf1_target = SoftQNetwork(train_env).to(device)
    qf2_target = SoftQNetwork(train_env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    env_dict = {
        "obs": {"shape": train_env.observation_space.shape, "dtype": np.float32},
        "action": {"shape": (1,), "dtype": np.int64},
        "reward": {"dtype": np.float32},
        "next_obs": {"shape": train_env.observation_space.shape, "dtype": np.float32},
        "done": {"dtype": np.bool_},
        "poss_actions_mask": {"shape": (train_env.action_space.n,), "dtype": np.bool_},
        "next_poss_actions_mask": {
            "shape": (train_env.action_space.n,),
            "dtype": np.bool_,
        },
    }

    if args.per:
        rb = PrioritizedReplayBuffer(
            args.buffer_size, env_dict=env_dict, alpha=args.per_alpha
        )
    else:
        rb = ReplayBuffer(args.buffer_size, env_dict=env_dict)

    start_time = time.time()
    best_agent_stats = {"val_multiple": float("inf")}

    # TRY NOT TO MODIFY: start the game
    obs, info = train_env.reset()
    poss_actions_mask = info["action_mask"]

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        wandb.run.log_code(".")

    for global_step in range(args.total_timesteps):
        # evaluate the agent
        if args.track and global_step % args.eval_frequency == 0:

            def agent(obs, poss_actions_mask):
                _, action_probs = actor.get_action(obs, poss_actions_mask)
                return torch.argmax(action_probs).cpu().numpy()

            eval_results, eval_log_dict = evaluate_agent(eval_envs_map, agent, device)
            val_results = eval_results["val"]
            if best_agent_stats["val_multiple"] > val_results["avg_multiple"]:
                best_agent_stats.update({
                    "val_return": val_results["avg_return"],
                    "val_multiple": val_results["avg_multiple"],
                    "test_return": eval_results["test"]["avg_return"],
                    "test_multiple": eval_results["test"]["avg_multiple"],
                    "train_return": eval_results["train"]["avg_return"],
                    "train_multiple": eval_results["train"]["avg_multiple"],
                })
                eval_log_dict.update({f"best/{k}": v for k, v in best_agent_stats.items()})
            wandb.log(eval_log_dict, commit=False)


        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.random.choice(poss_actions_mask.nonzero()[0])
        else:
            actions, _ = actor.get_action(
                torch.Tensor(obs).to(device),
                torch.Tensor(poss_actions_mask).bool().to(device),
            )
            actions = actions.detach().cpu().numpy()

        assert poss_actions_mask[
            actions
        ].all(), f"actions={actions}, poss_actions_mask={poss_actions_mask}"

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, term, trunc, info = train_env.step(actions)
        done = term or trunc
        next_poss_actions_mask = info["action_mask"]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if done and random.random() < 0.01:
            assert "episode" in info
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            if args.track:
                wandb.log({
                    "charts/episodic_return": info["episode"]["r"],
                    "charts/episodic_length": info["episode"]["l"],
                }, commit=False)

        rb.add(
            obs=obs,
            action=actions,
            reward=reward,
            next_obs=next_obs,
            done=done,
            poss_actions_mask=poss_actions_mask,
            next_poss_actions_mask=next_poss_actions_mask,
        )

        # TRY NOT TO MODIFY: reset environment
        if done:
            rb.on_episode_end()
            next_obs, info = train_env.reset()
            next_poss_actions_mask = info["action_mask"]

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        poss_actions_mask = next_poss_actions_mask


        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                if args.per:
                    beta = linear_schedule(
                        args.per_start_beta,
                        args.per_end_beta,
                        args.total_timesteps,
                        global_step,
                    )
                    sample = rb.sample(args.batch_size, beta=beta)
                    indexes = sample.pop("indexes")
                else:
                    sample = rb.sample(args.batch_size)
                sample = {k: torch.tensor(v).to(device) for k, v in sample.items()}

                # CRITIC training
                with torch.no_grad():
                    observations_mask = sample["done"] == 1
                    sample_next_poss_actions_mask = (
                        sample["next_poss_actions_mask"].bool() | observations_mask
                    )
                    _, next_state_action_probs = actor.get_action(
                        sample["next_obs"], sample_next_poss_actions_mask
                    )
                    qf1_next_target = qf1_target(sample["next_obs"])
                    qf2_next_target = qf2_target(sample["next_obs"])
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    assert next_state_action_probs.shape == qf1_next_target.shape
                    min_qf_next_target = next_state_action_probs * torch.min(
                        qf1_next_target, qf2_next_target
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = sample["reward"].flatten() + (
                        ~sample["done"].flatten()
                    ) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(sample["obs"])
                qf2_values = qf2(sample["obs"])
                qf1_a_values = qf1_values.gather(1, sample["action"].long()).view(-1)
                qf2_a_values = qf2_values.gather(1, sample["action"].long()).view(-1)
                if args.per:
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value, reduction="none")
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value, reduction="none")
                    elementwise_qf_loss = qf1_loss + qf2_loss
                    weights = sample["weights"].float()
                    qf_loss = (weights * elementwise_qf_loss).mean()
                    with torch.no_grad():
                        priority1 = F.smooth_l1_loss(qf1_a_values, next_q_value, reduction="none")
                        priority2 = F.smooth_l1_loss(qf2_a_values, next_q_value, reduction="none")
                        priority = priority1 + priority2
                        rb.update_priorities(indexes, priority.cpu().numpy())
                else:
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                qf1_grad_norm = nn.utils.clip_grad_norm_(
                    qf1.parameters(), args.gradient_clip
                )
                qf2_grad_norm = nn.utils.clip_grad_norm_(
                    qf2.parameters(), args.gradient_clip
                )
                q_optimizer.step()

                # ACTOR training
                actual_policy_frequency = args.policy_frequency * args.update_frequency
                if global_step % actual_policy_frequency == 0:
                    sample_poss_actions_mask = sample["poss_actions_mask"].bool()
                    _, action_probs = actor.get_action(
                        sample["obs"], sample_poss_actions_mask
                    )
                    with torch.no_grad():
                        min_qf_values = qf1(sample["obs"])
                        # qf1_values = qf1(sample['obs'])
                        # qf2_values = qf2(sample['obs'])
                        # min_qf_values = torch.min(qf1_values, qf2_values)

                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    masked_action_probs = action_probs[sample_poss_actions_mask]
                    masked_min_qf_values = min_qf_values[sample_poss_actions_mask]
                    actor_loss = (
                        -(masked_action_probs * masked_min_qf_values).sum()
                        / action_probs.shape[0]
                    )

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        actor.parameters(), args.gradient_clip
                    )
                    actor_optimizer.step()

                    # update the target networks
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                if args.track:
                    log_dict = {
                        "charts/SPS": sps,
                        "charts/global_step": global_step,
                        "losses/qf1_values": qf1_a_values.mean().item(),
                        "losses/qf2_values": qf2_a_values.mean().item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/actor_loss": actor_loss.item(),
                        "charts/qf1_grad_norm": qf1_grad_norm.item(),
                        "charts/qf2_grad_norm": qf2_grad_norm.item(),
                        "charts/actor_grad_norm": actor_grad_norm.item(),
                    }
                    if args.per:
                        log_dict.update({
                            "charts/per_beta": beta,
                            "losses/per_weight": weights.mean()
                        })
                    wandb.log(log_dict)

