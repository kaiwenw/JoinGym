import join_optimization
import json, re
import numpy as np
import torch
import gymnasium as gym
from collections import defaultdict
import torch.nn as nn
import os


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    if end_e < start_e:
        return max(slope * t + start_e, end_e)
    else:
        return min(slope * t + start_e, end_e)


def get_cli_overrides():
    import sys

    ignore_keys = [
        "track",
    ]
    value_only_keys = [
        "query-id",
    ]
    out = []
    # first one is the script name
    for arg in sorted(sys.argv[1:]):
        arg = arg[2:]  # remove --
        splitted = arg.split("=")
        # assert len(splitted) == 2
        key = splitted[0]
        if key in ignore_keys:
            continue
        elif key in value_only_keys:
            assert len(splitted) == 2, f"{splitted} should have length 2"
            out.append(splitted[1])
        else:
            out.append(arg)
    return out


def make_env(seed, query_ids, disable_cartesian_product, enable_bushy):
    # read db schema and convert to dict
    schema_regex = r"(.*)\((.*)\)"
    db_schema = {}

    random_query_id = next(iter(query_ids))
    if isinstance(random_query_id, int):
        job_base_path = "imdb/job"
    else:
        job_base_path = "imdb/joingym"

    with open("imdb/schema.txt", "r") as f:
        schemas = f.readlines()
        for schema in schemas:
            match = re.match(schema_regex, schema.strip())
            table_name = match.group(1).strip()
            columns = match.group(2).strip().split(",")
            columns = [column.strip() for column in columns]
            db_schema[table_name] = columns

    join_contents = {}
    for id in query_ids:
        path = os.path.join(job_base_path, f"q{id}.json")
        with open(path, "r") as file:
            join_contents[id] = json.load(file)

    def thunk():
        env_name = "join_optimization_bushy-v0" if enable_bushy else "join_optimization_left-v0"
        env = gym.make(
            env_name, db_schema=db_schema, join_contents=join_contents,
            disable_cartesian_product=disable_cartesian_product,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@torch.no_grad()
def evaluate_agent(
    envs_map, agent, device, agent_name="agent",
):
    f"""
    envs_map: mapping from env_type (train, val, test) to env

    Returns:
        mapping from env_type to a dict(avg_return, avg_multiple)
    """

    out = {}
    log_dict = {}
    for env_type, env in envs_map.items():
        returns_per_query = defaultdict(list)
        total_returns = []
        optimal_multiplier_per_query = defaultdict(list)
        total_optimal_multipliers = []
        for cur_query_id in env.query_ids:
            eval_obs, infos = env.reset(options=dict(query_id=cur_query_id))
            min_plan_cost = infos["min_plan_cost"]
            eval_poss_actions_mask = infos["action_mask"]
            eval_done = False
            while not eval_done:
                action = agent(
                    torch.Tensor(eval_obs).to(device),
                    torch.Tensor(eval_poss_actions_mask).bool().to(device),
                )
                eval_obs, eval_rewards, eval_term, eval_trunc, infos = env.step(
                    action
                )
                eval_done = eval_term or eval_trunc
                eval_poss_actions_mask = infos["action_mask"]


            optimal_multiplier = env.cumulative_cardinality / min_plan_cost
            optimal_multiplier_per_query[cur_query_id].append(optimal_multiplier)
            total_optimal_multipliers.append(optimal_multiplier)

            returns_per_query[cur_query_id].append(infos["episode"]["r"])
            total_returns.append(infos["episode"]["r"])

        log_dict[f"{env_type}/{agent_name}/avg_query"] = np.mean(total_returns)
        for query_id, query_returns in returns_per_query.items():
            log_dict[f"{env_type}/{agent_name}/query_{query_id}"] = np.mean(query_returns)

        log_dict[f"{env_type}/cost_multiple/{agent_name}/avg_query"] = np.mean(total_optimal_multipliers)
        for query_id, query_returns in optimal_multiplier_per_query.items():
            log_dict[f"{env_type}/cost_multiple/{agent_name}/query_{query_id}"] = np.mean(query_returns)

        out[env_type] = {
            "avg_return": np.mean(total_returns),
            "avg_multiple": np.mean(total_optimal_multipliers),
        }

    return out, log_dict

def get_query_ids_split(num_total: int = 100):
    query_templates = {
        1: range(num_total),
        2: range(num_total),
        3: range(num_total),
        4: range(num_total),
        5: range(num_total),
        6: range(num_total),
        7: range(num_total),
        8: range(num_total),
        9: range(num_total),
        10: range(num_total),
        11: range(num_total),
        12: range(num_total),
        13: range(num_total),
        14: range(num_total),
        15: range(num_total),
        16: range(num_total),
        17: range(num_total),
        18: range(num_total),
        19: range(num_total),
        20: range(num_total),
        21: range(num_total),
        22: range(num_total),
        23: range(num_total),
        24: range(num_total),
        25: range(num_total),
        26: range(num_total),
        27: range(num_total),
        28: range(num_total),
        29: range(num_total),
        30: range(num_total),
        31: range(num_total),
        32: range(num_total),
        33: range(num_total),
    }
    query_ids = {f"{k}_{v}" for k, vs in query_templates.items() for v in vs}
    for query_id in query_ids:
        path = f"imdb/joingym/q{query_id}.json"
        assert os.path.exists(path), f"{path} doesn't exist"

    val_query_ids = set()
    test_query_ids = set()
    for k, vs in query_templates.items():
        for i in range(60, 80):
            val_query_ids.add(f"{k}_{vs[i]}")
        for i in range(80, 100):
            test_query_ids.add(f"{k}_{vs[i]}")
    train_query_ids = query_ids - test_query_ids - val_query_ids
    print(f"train: {len(train_query_ids)}, val: {len(val_query_ids)}, test: {len(test_query_ids)}")
    return {
        "train": train_query_ids,
        "test": test_query_ids,
        "val": val_query_ids,
    }



def job_get_query_ids_split():
    # query_templates = {
    #     1: [1, 2, 3, 4],
    #     2: [5, 6, 7, 8],
    #     3: [9, 10, 11],
    #     4: [12, 13, 14],
    #     5: [15, 16, 17],
    #     6: [18, 19, 20, 21, 22, 23],
    #     7: [24, 25, 26],
    #     8: [27, 28, 29, 30],
    #     9: [31, 32, 33, 34],
    #     10: [35, 36, 37],
    #     11: [38, 39, 40, 41],
    #     15: [52, 53, 54, 55],
    #     16: [56, 57, 58, 59],
    #     17: [60, 61, 62, 63, 64, 65],
    #     19: [69, 70, 71, 72],
    #     21: [76, 77, 78],
    #     23: [83, 84, 85],
    #     24: [86, 87],
    # }

    query_templates = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11], 4: [12, 13, 14], 5: [15, 16, 17],
                       6: [18, 19, 20, 21, 22, 23], 7: [24, 25, 26], 8: [27, 28, 29, 30], 9: [31, 32, 33, 34],
                       10: [35, 36, 37], 11: [38, 39, 40, 41], 12: [42, 43, 44], 13: [45, 46, 47, 48], 14: [49, 50, 51],
                       15: [52, 53, 54, 55], 16: [56, 57, 58, 59], 17: [60, 61, 62, 63, 64, 65], 18: [66, 67, 68],
                       19: [69, 70, 71, 72], 20: [73, 74, 75], 21: [76, 77, 78], 22: [79, 80, 81, 82], 23: [83, 84, 85],
                       24: [86, 87], 25: [88, 89, 90], 26: [91, 92, 93], 27: [94, 95, 96], 28: [97, 98, 99],
                       29: [100, 101, 102], 30: [103, 104, 105], 31: [106, 107, 108], 32: [109, 110],
                       33: [111, 112, 113]}

    query_ids = {value for values in query_templates.values() for value in values}
    test_query_ids = {
        per_query_template[0] for per_query_template in query_templates.values()
    }
    val_query_ids = {
        per_query_template[1] for per_query_template in query_templates.values()
    }
    # test_query_ids = {random.choice(per_query_template) for per_query_template in query_templates.values()}
    train_query_ids = query_ids - test_query_ids - val_query_ids
    return {
        "train": train_query_ids,
        "test": test_query_ids,
        "val": val_query_ids,
    }


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_critic(observation_space, output_dim=1):
    return nn.Sequential(
        layer_init(nn.Linear(np.array(observation_space.shape).prod(), 256)),
        nn.ELU(),
        layer_init(nn.Linear(256, 128)),
        nn.ELU(),
        layer_init(nn.Linear(128, output_dim), std=1.0),
    )


def get_actor(observation_space, output_dim):
    return nn.Sequential(
        layer_init(nn.Linear(np.array(observation_space.shape).prod(), 128)),
        nn.ELU(),
        layer_init(nn.Linear(128, 64)),
        nn.ELU(),
        layer_init(nn.Linear(64, output_dim), std=1.0),
    )
