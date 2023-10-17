# -----------------------------------------------------------------------
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import logging
import random
from .join_opt_env_base import JoinOptEnvBase, TreeNode


log = logging.getLogger(__name__)



class JoinOptEnvBushy(JoinOptEnvBase):
    """the environment of DBMS optimizer"""

    def map_2d_to_1d_action(self, i, j, n):
        # create a mapping from 2d strictly upper triangular matrix index to 1d
        return int(((n - 1) + (n - i)) * (i) / 2 + j - (i + 1))

    # db_schema is a dict from table -> a column list
    def __init__(self, db_schema, join_contents, disable_cartesian_product=True):
        super().__init__(db_schema, join_contents, disable_cartesian_product)

        # observation is concatenation of query encoding and current join columns and selectivity encoding
        obs_size = self.nr_columns + self.nr_columns + self.nr_base_table
        self.observation_space = gym.spaces.Box(low=-1, high=3, shape=(obs_size,), dtype=np.float32)
        self.nr_action = int((self.nr_base_table * (self.nr_base_table - 1)) / 2)
        self.map_1d_to_2d = {self.map_2d_to_1d_action(left_table_idx, right_table_idx, self.nr_base_table): (left_table_idx, right_table_idx) for
                    left_table_idx in range(self.nr_base_table - 1) for right_table_idx in range(left_table_idx + 1, self.nr_base_table)}
        self.map_2d_to_1d = {(left_table_idx, right_table_idx) : self.map_2d_to_1d_action(left_table_idx, right_table_idx, self.nr_base_table) for
                    left_table_idx in range(self.nr_base_table - 1) for right_table_idx in range(left_table_idx + 1, self.nr_base_table)}
        self.action_space = gym.spaces.Discrete(self.nr_action)


    def get_partial_plan_encoding(self, query_plans):
        encoding = np.zeros(self.nr_columns)
        bushy_tree_label = 1
        for query_plan_idx in range(len(query_plans)):
            query_plan = query_plans[query_plan_idx]
            if query_plan is not None:
                # return the encoding of a given state
                # the encoding of a give state should contain three information
                # 1) query graph encoding
                # 2) current join columns
                # 3) its left child and right child
                current_joined_columns = set(query_plan.value)
                previous_joined_columns = set(query_plan._get_descendant_values())
                # set for columns are already joined
                joined_columns = current_joined_columns.union(previous_joined_columns)
                current_joined_tables = set()
                for joined_column in joined_columns:
                    encoding[self.db_schema_linear.index(joined_column)] = bushy_tree_label
                    origin_join_table, _ = joined_column.split(".")
                    current_joined_tables.add(origin_join_table)
                # for stage columns
                for join_column in self.join_columns:
                    join_table_name, join_column_name = join_column.split(".")
                    origin_join_column = f"{self.alias_dict[join_table_name]}.{join_column_name}"
                    # if this table is already added into current joined tables, but its columns has not joined yet, set it to 2
                    if self.alias_dict[join_table_name] in current_joined_tables and origin_join_column not in joined_columns:
                        encoding[self.db_schema_linear.index(origin_join_column)] = -1
                bushy_tree_label += 1

        return encoding

    # give left tables and right tables, retrieve all joins in this node
    def retrieve_join_columns(self, left_tables, right_tables):
        columns_to_join = []
        for join_expression in self.join_expressions:
            left_table, left_column = join_expression["left"].split(".")
            right_table, right_column = join_expression["right"].split(".")
            if (left_table in left_tables and right_table in right_tables) \
                    or (right_table in left_tables and left_table in right_tables):
                columns_to_join.append(f'{self.alias_dict[left_table]}.{left_column}')
                columns_to_join.append(f'{self.alias_dict[right_table]}.{right_column}')
        return columns_to_join

    def step(self, action):
        action = action.item()
        # action is a tuple, representing two components to merge
        (left_base_table_idx, right_base_table_idx) = self.map_1d_to_2d[action]
        left_table_idx = self.base_idx_to_join_idx[left_base_table_idx]
        right_table_idx = self.base_idx_to_join_idx[right_base_table_idx]
        # retrieve two trees
        # action is a table index
        left_table_name = self.join_tables[left_table_idx]
        right_table_name = self.join_tables[right_table_idx]
        # retrieve a tree of left table and right table
        left_tree_component_idx = self.tree_components[left_table_name]
        right_tree_component_idx = self.tree_components[right_table_name]
        # left query plan and right query plan
        left_tree_plan = self.current_query_plan[left_tree_component_idx]
        right_tree_plan = self.current_query_plan[right_tree_component_idx]
        left_tables = self.current_merged_tables[left_tree_component_idx]
        right_tables = self.current_merged_tables[right_tree_component_idx]
        # get all join columns of left tables and right tables
        current_join_columns = self.retrieve_join_columns(left_tables, right_tables)
        # create a tree node and merge left tree and right tree
        query_plan = TreeNode(current_join_columns)
        query_plan.left = left_tree_plan
        query_plan.right = right_tree_plan
        # get cardinality info
        log_left_cardinality = self.log_cardinalities[left_tree_component_idx]
        log_right_cardinality = self.log_cardinalities[right_tree_component_idx]
        log_query_cardinality = self.compute_log_query_cardinality(left_tables, right_tables, log_left_cardinality, log_right_cardinality)
        # receive a reward
        reward = self.log_cardinality_to_reward(log_query_cardinality)
        self.cumulative_cardinality += round(np.exp(log_query_cardinality))

        # update the tree components
        if right_tree_component_idx < left_tree_component_idx:
            left_tree_component_idx, right_tree_component_idx = right_tree_component_idx, left_tree_component_idx
        # merge right component to left
        self.current_merged_tables[left_tree_component_idx].update(self.current_merged_tables[right_tree_component_idx])
        self.log_cardinalities[left_tree_component_idx] = log_query_cardinality
        self.current_query_plan[left_tree_component_idx] = query_plan
        # delete right component in the tree
        self.current_merged_tables.pop(right_tree_component_idx)
        self.log_cardinalities.pop(right_tree_component_idx)
        self.current_query_plan.pop(right_tree_component_idx)
        # change tree component map
        for tree_idx in range(len(self.current_merged_tables)):
            for table_name in self.current_merged_tables[tree_idx]:
                self.tree_components[table_name] = tree_idx
        # return observation state
        self.partial_plan_encoding = self.get_partial_plan_encoding(self.current_query_plan)
        observation = self.get_observation(self.partial_plan_encoding)
        done = (len(self.current_merged_tables) == 1)  # episode terminates when we have only one tree

        if self.disable_cartesian_product:
            self._valid_actions_mask = self.valid_action_mask_with_heuristic()
        else:
            self._valid_actions_mask = self.valid_action_mask()

        info = {
            'action_mask': self._valid_actions_mask,
            "log_cardinality": log_query_cardinality,
        }

        return observation, reward, done, done, info

    def reset(self, *, seed=None, options=None):

        # Get the query_id
        if options is not None and "query_id" in options:
            query_id = options["query_id"]
            assert query_id in self.query_ids, f"Invalid query id {query_id}, should be one of {self.query_ids}"
        else:
            query_id = random.choice(self.query_ids)

        self.initialize_join_and_set_query_encoding(query_id)

        # each table is a tree at beginning
        self.current_merged_tables = [set([table]) for table in self.join_tables]
        # map from a table name to its tree component id
        self.tree_components = {self.join_tables[tid]: tid for tid in range(self.nr_join_tables)}
        # construct trees, each query tree is an empty node
        self.current_query_plan = [None for _ in range(self.nr_join_tables)]
        # cardinalities information
        self.log_cardinalities = [np.log(self.cardinality_after_filter[table]) for table in self.join_tables]
        self.cumulative_cardinality = 0

        # set actions mask
        if self.disable_cartesian_product:
            self._valid_actions_mask = self.valid_action_mask_with_heuristic()
        else:
            self._valid_actions_mask = self.valid_action_mask()

        info = {
            'action_mask': self._valid_actions_mask,
            "query_id": self.cur_query_id,
            "min_plan_cost": self.min_plan_cost,
        }

        return self.get_observation(np.zeros(self.nr_columns)), info

    def compute_log_query_cardinality(self, left_tables, right_tables, log_left_cardinality, log_right_cardinality):
        # get the set of tables if we join left tables with right tables
        joined_tables_frozen = frozenset(left_tables.union(right_tables))
        if joined_tables_frozen in self.join_cardinality:
            # compute the reward of join left tables with right tables
            return np.log(self.join_cardinality[joined_tables_frozen])
        else:
            # intermediate result is the cartesian product
            return log_left_cardinality + log_right_cardinality

    def valid_action_mask_with_heuristic(self):
        valid_actions_mask = np.zeros(self.nr_action)
        for left_table_idx in range(0, self.nr_join_tables - 1):
            for right_table_idx in range(left_table_idx + 1, self.nr_join_tables):
                left_table_name = self.join_tables[left_table_idx]
                right_table_name = self.join_tables[right_table_idx]
                # table1 and table2 are not in a same component
                # and we avoid cartesian product
                if (self.tree_components[left_table_name] != self.tree_components[right_table_name]) \
                        and right_table_name in self.joins[left_table_name]:
                    left_base_table_idx = self.join_idx_to_base_idx[left_table_idx]
                    right_base_table_idx = self.join_idx_to_base_idx[right_table_idx]
                    if left_base_table_idx > right_base_table_idx:
                        left_base_table_idx, right_base_table_idx = right_base_table_idx, left_base_table_idx
                    action_idx = self.map_2d_to_1d[(left_base_table_idx, right_base_table_idx)]
                    valid_actions_mask[action_idx] = 1.0
        return valid_actions_mask

    def valid_action_mask(self):
        valid_actions_mask = np.zeros(self.nr_action)
        for left_table_idx in range(0, self.nr_join_tables - 1):
            for right_table_idx in range(left_table_idx + 1, self.nr_join_tables):
                left_table_name = self.join_tables[left_table_idx]
                right_table_name = self.join_tables[right_table_idx]
                # table1 and table2 are not in a same component
                if self.tree_components[left_table_name] != self.tree_components[right_table_name]:
                    left_base_table_idx = self.join_idx_to_base_idx[left_table_idx]
                    right_base_table_idx = self.join_idx_to_base_idx[right_table_idx]
                    if left_base_table_idx > right_base_table_idx:
                        left_base_table_idx, right_base_table_idx = right_base_table_idx, left_base_table_idx
                    action_idx = self.map_2d_to_1d[(left_base_table_idx, right_base_table_idx)]
                    valid_actions_mask[action_idx] = 1.0
        return valid_actions_mask
