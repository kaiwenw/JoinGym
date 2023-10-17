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

import json

import gymnasium as gym
import numpy as np
import logging
import random
from .join_opt_env_base import JoinOptEnvBase, TreeNode

log = logging.getLogger(__name__)


class JoinOptEnvLeft(JoinOptEnvBase):
    def __init__(self, db_schema, join_contents, disable_cartesian_product=True):
        super().__init__(db_schema, join_contents, disable_cartesian_product)

        # observation is concatenation of query_encoding and current join columns
        obs_size = self.nr_columns + self.nr_columns + self.nr_base_table
        self.observation_space = gym.spaces.Box(low=-1, high=2, shape=(obs_size,), dtype=np.float32)
        # action space is the table index.
        self.action_space = gym.spaces.Discrete(self.nr_base_table)


    def get_partial_plan_encoding(self, query_plan, current_joined_tables):
        if query_plan is None:
            return np.zeros(self.nr_columns)
        else:
            # return the encoding of a given state
            # the encoding of a give state should contain three information
            # 1) query graph encoding
            # 2) current join columns
            # 3) its left child and right child
            current_join_columns = set(query_plan.value)
            previous_join_columns = set()
            if query_plan.left is not None:
                previous_join_columns = set(query_plan.left._get_descendant_values())

            # for previous join information
            encoding = np.zeros(self.nr_columns)
            for previous_join_column in previous_join_columns:
                encoding[self.db_schema_linear.index(previous_join_column)] = 1
            # for current join information
            for current_join_column in current_join_columns:
                encoding[self.db_schema_linear.index(current_join_column)] = 1

            # set for columns are already joined
            join_columns = current_join_columns.union(previous_join_columns)
            for join_column in self.join_columns:
                join_table_name, join_column_name = join_column.split(".")
                origin_join_column = f"{self.alias_dict[join_table_name]}.{join_column_name}"
                # if this table is already added into current joined tables, but its columns has not joined yet, set it to 2
                if join_table_name in current_joined_tables and origin_join_column not in join_columns:
                    encoding[self.db_schema_linear.index(origin_join_column)] = -1

            # combine 3 encoding together
            return encoding

    # give a new table to join and list of tables, retrieve all joins in this node
    def retrieve_join_columns(self, table_to_join, current_joined_tables):
        columns_to_join = []
        for join_expression in self.join_expressions:
            left_table, left_column = join_expression["left"].split(".")
            right_table, right_column = join_expression["right"].split(".")
            if (left_table == table_to_join and right_table in current_joined_tables) \
                    or (right_table == table_to_join and left_table in current_joined_tables):
                columns_to_join.append(f'{self.alias_dict[left_table]}.{left_column}')
                columns_to_join.append(f'{self.alias_dict[right_table]}.{right_column}')
        return columns_to_join

    def step(self, action):
        assert self._valid_actions_mask[
                   action] > 0, f"Invalid action {action}, should be one of {self._valid_actions_mask}"

        action = action.item()
        action = self.base_idx_to_join_idx[action]
        # action is a table index
        select_table = self.join_tables[action]

        # retrieve columns to join at this step
        join_columns = self.retrieve_join_columns(select_table, self.current_joined_tables)

        # update the join plan
        if len(self.current_joined_tables) <= 1:
            # here we only have one table
            self.query_plan = TreeNode(join_columns)
        else:
            # here extend the tree
            parent_node = TreeNode(join_columns)
            parent_node.left = self.query_plan
            parent_node.right = None
            self.query_plan = parent_node

        ## Compute reward
        # no table joined yet, so first table is simply selected
        is_first_step = len(self.current_joined_tables) == 0
        if is_first_step:
            self.log_current_tuple_cardinality = np.log(self.cardinality_after_filter[select_table])
            reward = 0.0
        else:
            self.update_log_cardinality(self.current_joined_tables, select_table)
            reward = self.log_cardinality_to_reward(self.log_current_tuple_cardinality)
            self.cumulative_cardinality += round(np.exp(self.log_current_tuple_cardinality))

        # add the current table into the current joined table
        self.current_joined_tables.add(select_table)

        # return observation state
        self.partial_plan_encoding = self.get_partial_plan_encoding(self.query_plan, self.current_joined_tables)
        observation = self.get_observation(self.partial_plan_encoding)
        done = (len(self.current_joined_tables) == self.nr_join_tables)  # episode terminates when all tables are joined

        if self.disable_cartesian_product:
            self._valid_actions_mask = self.valid_action_mask_with_heuristic()
        else:
            self._valid_actions_mask = self.valid_action_mask()
        info = {
            'action_mask': self._valid_actions_mask,
            "log_cardinality": self.log_current_tuple_cardinality if not is_first_step else float("-inf"),
        }
        return observation, reward, done, done, info

    def reset(self, *, seed=None, options=None):
        # we start from init state
        self.current_joined_tables = set()
        self.query_plan = None
        self.log_current_tuple_cardinality = float("-inf")
        self.cumulative_cardinality = 0

        # Get the query_id
        if options is not None and "query_id" in options:
            query_id = options["query_id"]
            assert query_id in self.query_ids, f"Invalid query id {query_id}, should be one of {self.query_ids}"
        else:
            query_id = random.choice(self.query_ids)
        self.initialize_join_and_set_query_encoding(query_id)

        # set actions mask
        self._valid_actions_mask = np.zeros(self.nr_base_table)
        for i in range(self.nr_join_tables):
            self._valid_actions_mask[self.join_idx_to_base_idx[i]] = 1
        info = {
            'action_mask': self._valid_actions_mask,
            'query_id': self.cur_query_id,
            'min_plan_cost': self.min_plan_cost,
        }
        return self.get_observation(np.zeros(self.nr_columns)), info

    def update_log_cardinality(self, current_joined_tables, select_table):
        assert len(current_joined_tables) > 0
        # get the set of tables if we join current_joined_tables with select_table
        next_joined_tables_frozen = frozenset(current_joined_tables.union({select_table}))
        if next_joined_tables_frozen in self.join_cardinality:
            # compute the reward of join previous tables with the new table
            self.log_current_tuple_cardinality = np.log(self.join_cardinality[next_joined_tables_frozen])
        else:
            # intermediate result is the cartesian product
            self.log_current_tuple_cardinality += np.log(self.cardinality_after_filter[select_table])


    def valid_action_mask_with_heuristic(self):
        # this is the heuristic with removing cartesian product
        valid_actions_mask = np.zeros(self.nr_base_table)
        for i in range(self.nr_join_tables):
            table_name = self.join_tables[i]
            if table_name not in self.current_joined_tables:
                # test whether it has a connected join
                if self.joins[table_name].intersection(self.current_joined_tables):
                    valid_actions_mask[self.join_idx_to_base_idx[i]] = 1.0
        return valid_actions_mask

    def valid_action_mask(self):
        # get validate actions of the current state
        # valida action is the table index which corresponding table are not in the current_joined_tables set
        valid_actions_mask = np.zeros(self.nr_base_table)
        for i in range(self.nr_join_tables):
            if self.join_tables[i] not in self.current_joined_tables:
                valid_actions_mask[self.join_idx_to_base_idx[i]] = 1.0
        return valid_actions_mask