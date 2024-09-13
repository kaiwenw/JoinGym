# JoinGym

JoinGym is an efficient and lightweight query optimization environment for reinforcement learning (RL). JoinGym also comes with a new dataset of intermediate result cardinalities for $3300$ queries, located in `imdb/joingym`.

## Quick Start
First, install [Gymnasium](https://gymnasium.farama.org/) in Python 3.9.
Then, to install JoinGym, run
```
cd join-optimization
pip install -e .
```

You can verify that JoinGym is installed correctly by running `test_env.py` provided in the main directory.
As shown in that test file, creating a JoinGym environment is as simple as
```
import gymnasium as gym
import join_optimization  # register JoinGym

env = gym.make(
    "join_optimization_left-v0",
    db_schema=db_schema,
    join_contents=join_contents,
)
```
where `db_schema` is the database schema (e.g. `imdb/schema.txt`) and `join_contents` is a map from `query_id` to the IR cardinalities of that query (e.g. `imdb/joingym/q1_0.json`).

JoinGym adheres to the standard Gymnasium API, with two key methods.
<ol>
  <li>
  state, info = env.reset(options={query_id=x})
  </li>
  <li>
  next_state, reward, done, _, info = env.step(action)
  </li>
</ol>
There is one key distinction from standard Gym environments: info['action_mask'] contains a multi-hot encoding of the possible actions at the current step. The RL algorithm should make use of this information to learn and act only from valid actions, which is more efficient than learning from all actions.
Example usage of the action mask can be found in our RL implementations in the `algorithms` directory.


## RL Algorithms

We provide implementations of DQN, PPO, SAC and TD3 in the `algorithms` folder. These implementations were modified from [CleanRL](https://github.com/vwxyzjn/cleanrl) to handle action masks and prioritized replay.

To get started, install
[PyTorch](https://pytorch.org/), [cpprb](https://ymd_h.gitlab.io/cpprb/api/), and [wandb](https://docs.wandb.ai/quickstart).
Our scripts accept two flags.
First, `--enable-bushy` can be used to enable bushy plans; otherwise, only left-deep plans are allowed by default. Second, `--disable-cartesian-product` can be used to disable Cartesian product (CP)actions; otherwise, CPs will be allowed by default.
For example, to run PPO on the left-deep environment and without CPs,
```
python test_ppo.py --disable-cartesian-product
```
To run SAC on the bushy environment and with CPs,
```
python test_sac.py --enable-bushy
```
By default, these scripts will run JoinGym with our whole dataset of $3300$ queries in `imdb/joingym`. You can modify the environment initialization to use a subset of queries, or to use queries from the Join Order Benchmark (JOB). Data for the JOB is stored in `imdb/job`.


### Citation
If you find this project inspiring or use our repository, kindly please cite

```
@article{wang2024joingym,
    title={{JoinGym}: {A}n Efficient Join Order Selection Environment},
    author={Wang, Junxiong and Wang, Kaiwen and Li, Yueying and Kallus, Nathan and Trummer, Immanuel and Sun, Wen},
    journal={Reinforcement Learning Journal},
    volume={1},
    issue={1},
    year={2024}
}
```
