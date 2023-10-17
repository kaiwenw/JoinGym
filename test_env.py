
import gymnasium as gym
import numpy as np
import join_optimization  # register JoinGym
import json
import re
import os


def get_schema(schema_path):
    """Get DB schema."""
    schema_regex = r"(.*)\((.*)\)"
    db_schema = {}
    with open(schema_path, "r") as f:
        schemas = f.readlines()
        for schema in schemas:
            match = re.match(schema_regex, schema.strip())
            table_name = match.group(1).strip()
            columns = match.group(2).strip().split(",")
            columns = [column.strip() for column in columns]
            db_schema[table_name] = columns
    return db_schema

def get_join_contents(dataset_dir, query_ids):
    """Get IR cardinality data for each query."""
    join_contents = {}
    for query_id in query_ids:
        path = os.path.join(dataset_dir, f"{query_id}.json")
        with open(path, "r") as file:
            join_contents[query_id] = json.load(file)
    return join_contents

## First, test our left deep environment.
enable_bushy = True
disable_cp = True
env_name = "join_optimization_left-v0"
db_schema = get_schema("imdb/schema.txt")
query_ids = [
    f"q{template_id}_{i}" for template_id, i in zip(range(1, 11), range(10))
]
join_contents = get_join_contents("imdb/joingym", query_ids)
env = gym.make(
    env_name, db_schema=db_schema, join_contents=join_contents,
    disable_cartesian_product=disable_cp,
)

done = False
step = 0
obs, info = env.reset()
query_id = info['query_id']
min_plan_cost = info['min_plan_cost']
print(f'Running query: {query_id}')
while not done:
    # sample a random valid action
    poss_actions_mask = info['action_mask']
    action = np.random.choice(poss_actions_mask.nonzero()[0])
    next_obs, reward, done, _, info = env.step(action)

    print(f'step {step}')
    # print(f'obs {obs}')
    print(f'action {action}')
    print(f'reward {reward}')
    print('')

    obs = next_obs
    step += 1


