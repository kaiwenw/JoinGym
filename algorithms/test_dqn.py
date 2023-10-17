import argparse
import random
import time
import logging
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from utils import *


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    parser.add_argument("--learning-rate", type=float, default=0.0003,
        help="the learning rate of the optimizer")
    parser.add_argument("--gradient-clip", type=float, default=5.0,
        help="max l2 norm of gradient")

    parser.add_argument("--double-dqn", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use double DQN")

    parser.add_argument("--buffer-size", type=int, default=250000,
        help="the replay memory buffer size")
    parser.add_argument("--per", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use prioritized experience replay")
    parser.add_argument("--per-alpha", type=float, default=0.5,
        help="")
    parser.add_argument("--per-start-beta", type=float, default=0.4,
        help="")
    parser.add_argument("--per-end-beta", type=float, default=1.0,
        help="")

    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = get_critic(env.observation_space, env.action_space.n)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    args = parse_args()
    if args.run_name:
        run_name = args.run_name
    else:
        cli = get_cli_overrides()
        override_str = "".join([f"[{arg}]" for arg in cli])
        run_name = f"[dqn]{override_str}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    split = get_query_ids_split()
    train_env = make_env(seed=args.seed, query_ids=split["train"],
                    disable_cartesian_product=args.disable_cartesian_product,
                    enable_bushy=args.enable_bushy)()
    eval_envs_map = {
        key: make_env(seed=args.seed+i, query_ids=split[key],
                    disable_cartesian_product=args.disable_cartesian_product,
                    enable_bushy=args.enable_bushy)()
            for i, key in enumerate(["train", "val", "test"])
    }

    q_network = QNetwork(train_env).to(device)
    optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)
    target_network = QNetwork(train_env).to(device)
    target_network.load_state_dict(q_network.state_dict())

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
                q_values = q_network(obs)
                q_values[~poss_actions_mask] = -float("inf")
                return torch.argmax(q_values).cpu().numpy()

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
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        random_number = random.random()
        if random_number < epsilon:
            actions = np.random.choice(poss_actions_mask.nonzero()[0])
        else:
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                q_values[~torch.Tensor(poss_actions_mask).bool().to(device)] = -float(
                    "inf"
                )
                actions = torch.argmax(q_values, dim=0).cpu().numpy()

        assert poss_actions_mask[
            actions
        ].all(), f"actions={actions}, poss_actions_mask={poss_actions_mask}"

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, term, trunc, info = train_env.step(actions)
        done = term or trunc
        next_poss_actions_mask = info["action_mask"]

        # TRY NOT TO MODIFY: record reward for plotting purposes
        if done:
            assert "episode" in info
            if random.random() < 0.01:
                log.info(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                if args.track:
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                        "charts/epsilon": epsilon,
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
            # Sample from Replay Buffer
            if global_step % args.train_frequency == 0:
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

                # Calculate Loss
                with torch.no_grad():
                    target_q_vals = target_network(sample["next_obs"])
                    target_q_vals[~sample["next_poss_actions_mask"].bool()] = -float(
                        "inf"
                    )
                    if args.double_dqn:
                        next_q_vals = q_network(sample["next_obs"])
                        next_q_vals[~sample["next_poss_actions_mask"].bool()] = -float(
                            "inf"
                        )
                        target_max = target_q_vals.gather(
                            1, next_q_vals.argmax(dim=1, keepdim=True)
                        ).squeeze()
                    else:
                        target_max, _ = target_q_vals.max(dim=1)
                    # When episode is done, we need to set it to zero.
                    target_max[sample["done"].squeeze().bool()] = 0.0
                    td_target = sample["reward"].squeeze() + args.gamma * target_max
                old_val = q_network(sample["obs"]).gather(1, sample["action"]).squeeze()

                if args.per:
                    elementwise_loss = F.mse_loss(td_target, old_val, reduction="none")
                    weights = sample["weights"].float()
                    loss = (weights * elementwise_loss).mean()
                    with torch.no_grad():
                        priority = F.smooth_l1_loss(td_target, old_val, reduction="none")
                        rb.update_priorities(indexes, priority.cpu().numpy())
                else:
                    loss = F.mse_loss(td_target, old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    q_network.parameters(), args.gradient_clip
                )
                optimizer.step()

                if global_step % 100 == 0:
                    sps = int(global_step / (time.time() - start_time))
                    if args.track:
                        log_dict = {
                            "charts/global_step": global_step,
                            "charts/SPS": sps,
                            "losses/td_loss": loss,
                            "losses/q_values": old_val.mean().item(),
                            "losses/grad_norm": grad_norm.item(),
                        }

                        if args.per:
                            log_dict.update({
                                "charts/per_beta": beta,
                                "losses/per_weight": weights.mean(),
                            })
                        wandb.log(log_dict)


            # update target network
            actual_frequency = int(args.target_network_frequency * args.tau)
            if global_step % actual_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )
