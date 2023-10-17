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
import math


log = logging.getLogger(__name__)


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __str__(self):
        return json.dumps(self._generate_tree_dict(self), indent=2)

    def _generate_tree_dict(self, node):
        # Base case: if the node is None, return None
        if node is None:
            return None

        # Generate the dictionary representation of the node and its children
        tree_dict = {}
        tree_dict['value'] = node.value  # Add the value of the node
        tree_dict['left'] = self._generate_tree_dict(node.left)  # Recursively generate left subtree
        tree_dict['right'] = self._generate_tree_dict(node.right)  # Recursively generate right subtree
        return tree_dict

    def _get_descendant_values(self):
        """
        Returns a list of all descendant values in the given tree.

        Args:
            tree (dict): Tree represented as a nested dictionary with 'value' and 'children' keys.

        Returns:
            list: List of all descendant values in the tree.
        """
        if self == None:
            return []
        else:
            return self.value + (self.left._get_descendant_values() if self.left is not None else []) + (
                self.right._get_descendant_values() if self.right is not None else [])


def build_inverse_map_one_to_one(d):
    out = {value: [key for key, val in d.items() if val == value] for value in set(d.values())}
    assert all(len(v) == 1 for v in out.values()), f"Input map is not one-to-one, resulting in inverse: {out}"
    out = {k: v[0] for k, v in out.items()}
    return out


class JoinOptEnvBase(gym.Env):
    # db_schema is a dict from table -> a column list
    def __init__(self, db_schema, join_contents, disable_cartesian_product):
        """
        Args:
            db_schema: Mapping from table to a list of columns in that table.
            join_contents: Mapping from query_id to the json containing all join information.
            disable_cartesian_product: whether to disable joins that lead to cartesian product.
                Typically, DBMS would disable cartesian product to limit search space, potentially
                leaving the optimal join plan on the table, for the benefit of smaller action space
                and less variance.
        """
        super().__init__()
        self.join_contents = join_contents
        self.query_ids = list(join_contents.keys())
        self.disable_cartesian_product = disable_cartesian_product
        self.n_queries = len(self.query_ids)

        # number of tables in database
        # TODO: rename base_table to alias_table
        self.db_base_tables = list(db_schema.keys())
        self.nr_base_table = len(self.db_base_tables)
        self.base_table_name_to_idx = {table: idx for idx, table in enumerate(self.db_base_tables)}

        # get linearized db schema
        self.db_schema_linear = [f"{table}.{column}" for table, columns in db_schema.items() for column in columns]
        self.nr_columns = len(self.db_schema_linear)

        self.joins_total_queries = {}
        self.join_tables_total_queries = {}
        self.nr_join_tables_total_queries = {}
        self.join_cardinality_total_queries = {}
        self.join_expressions_total_queries = {}
        self.min_plan_cost_total_queries = {}
        self.min_plan_cost_per_step_total_queries = {}
        self.cap_on_cost_total_queries = {}
        self.log_cap_on_cost_total_queries = {}
        self.selectivity_total_queries = {}
        self.join_base_tables_total_queries = {}
        self.join_idx_to_base_idx_total_queries = {}
        self.base_idx_to_join_idx_total_queries = {}
        self.join_columns_total_queries = {}
        self.alias_dict_total_queries = {}
        self.cardinality_after_filter_total_queries = {}
        self.query_encoding_total_queries = {}
        self.selectivity_encoding_total_queries = {}

        # initialize informations for all queries
        for query_id in self.query_ids:
            join_content = self.join_contents[query_id]
            # retrieve the join table field in query plan represented by json
            join_table_field = join_content["relations"]
            # retrieve the join size field in query plan represented by json
            join_cost_field = join_content["sizes"]
            # retrieve all joins
            joins = {relation: set(other_rel
                                   for join in join_content['joins'] for other_rel in join['relations'] if
                                   relation in join['relations'] and other_rel != relation)
                     for join in join_content['joins'] for relation in join['relations']}
            join_tables = list(map(lambda obj: obj['name'], join_table_field))
            nr_join_tables = len(join_tables)

            join_cardinality = {frozenset(obj['relations']): obj['cardinality'] for obj in
                                map(lambda obj: {'relations': obj['relations'], 'cardinality': obj['cardinality']},
                                    join_cost_field)}

            join_expressions = join_content["join expressions"]

            min_plan_cost = max(float(join_content["left deep tree min cost"]), 1)
            min_plan_cost_per_step = min_plan_cost / (nr_join_tables - 1)

            cap_on_cost = min_plan_cost * 100.0
            log_cap_on_cost = np.log(cap_on_cost)
            selectivity = {
                relations['name']: float(relations['cardinality']) / float(relations['unfilteredCardinality']) for
                relations in join_table_field}

            # map from table id in database to table id in the query
            join_base_tables = list(map(lambda obj: obj['aliastable'], join_table_field))
            join_idx_to_base_idx = {index: self.db_base_tables.index(elem) for index, elem in
                                    enumerate(join_base_tables)}
            base_idx_to_join_idx = build_inverse_map_one_to_one(join_idx_to_base_idx)

            join_columns = join_content["join columns"]
            # init dict from query alias to original table name
            alias_dict = {relation["name"]: relation["aliastable"] for relation in join_content["relations"]}
            cardinality_after_filter = {relation["name"]: relation["cardinality"] for relation in
                                        join_content["relations"]}

            # init query encoding
            query_encoding = np.zeros(self.nr_columns)
            for join_column in join_columns:
                # set join column in the given query to 1
                join_table_name, join_column_name = join_column.split(".")
                origin_join_column = f"{alias_dict[join_table_name]}.{join_column_name}"
                query_encoding[self.db_schema_linear.index(origin_join_column)] = 1

            selectivity_encoding = np.zeros(self.nr_base_table)
            for join_table in join_tables:
                base_pos = self.db_base_tables.index(alias_dict[join_table])
                selectivity_encoding[base_pos] = selectivity[join_table]

            # retrieve all joins
            self.joins_total_queries[query_id] = joins
            self.join_tables_total_queries[query_id] = join_tables
            self.nr_join_tables_total_queries[query_id] = nr_join_tables
            self.join_cardinality_total_queries[query_id] = join_cardinality
            self.join_expressions_total_queries[query_id] = join_expressions
            self.min_plan_cost_total_queries[query_id] = min_plan_cost
            self.min_plan_cost_per_step_total_queries[query_id] = min_plan_cost_per_step
            self.cap_on_cost_total_queries[query_id] = cap_on_cost
            self.log_cap_on_cost_total_queries[query_id] = log_cap_on_cost
            self.selectivity_total_queries[query_id] = selectivity
            self.join_base_tables_total_queries[query_id] = join_base_tables
            self.join_idx_to_base_idx_total_queries[query_id] = join_idx_to_base_idx
            self.base_idx_to_join_idx_total_queries[query_id] = base_idx_to_join_idx
            self.join_columns_total_queries[query_id] = join_columns
            self.alias_dict_total_queries[query_id] = alias_dict
            self.cardinality_after_filter_total_queries[query_id] = cardinality_after_filter
            self.query_encoding_total_queries[query_id] = query_encoding
            self.selectivity_encoding_total_queries[query_id] = selectivity_encoding

    def initialize_join_and_set_query_encoding(self, query_id):
        """This function is called at the beginning of each episode to initialize the join tree and query encoding.

        Args:
            query_id: the id of the query to be optimized.
        """
        assert query_id in self.query_ids, f"Invalid query id {query_id}, should be one of {self.query_ids}"
        self.cur_query_id = query_id

        self.joins = self.joins_total_queries[query_id]
        self.join_tables = self.join_tables_total_queries[query_id]
        self.nr_join_tables = self.nr_join_tables_total_queries[query_id]
        self.join_cardinality = self.join_cardinality_total_queries[query_id]
        self.join_expressions = self.join_expressions_total_queries[query_id]
        self.min_plan_cost = self.min_plan_cost_total_queries[query_id]
        self.min_plan_cost_per_step = self.min_plan_cost_per_step_total_queries[query_id]
        self.cap_on_cost = self.cap_on_cost_total_queries[query_id]
        self.log_cap_on_cost = self.log_cap_on_cost_total_queries[query_id]
        self.selectivity = self.selectivity_total_queries[query_id]
        self.join_base_tables = self.join_base_tables_total_queries[query_id]
        self.join_idx_to_base_idx = self.join_idx_to_base_idx_total_queries[query_id]
        self.base_idx_to_join_idx = self.base_idx_to_join_idx_total_queries[query_id]
        self.join_columns = self.join_columns_total_queries[query_id]
        self.alias_dict = self.alias_dict_total_queries[query_id]
        self.cardinality_after_filter = self.cardinality_after_filter_total_queries[query_id]
        self.query_encoding = self.query_encoding_total_queries[query_id]
        self.selectivity_encoding = self.selectivity_encoding_total_queries[query_id]

    def get_action_idx_from_query_alias(self, query_alias):
        """
        Given the base table, return the index of the action.
        """
        base_table = self.alias_dict[query_alias]
        return self.base_table_name_to_idx[base_table]

    def get_observation(self, partial_plan_encoding):
        return np.concatenate((self.query_encoding, partial_plan_encoding, self.selectivity_encoding)).astype(np.float32)


    def step(self, action):
        """
        Should also set the mask for possible actions in info['action_mask'].
        """
        raise NotImplementedError()

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError()

    def compute_current_cardinality(self):
        """
        Compute the current cardinality of the query plan.
        """
        raise NotImplementedError()

    def log_cardinality_to_reward(self, log_ir_cardinality):
        """
        Given the cardinality of the intermediate result (cost), compute reward as follows:
        Let c_h denote the current cost at time h.
        First, we threshold it by c_h = min(c_h, c_max/H) to avoid numerical issues.

        Then, we compute the step-wise regret reg_h = c_h - c^*/H,
        where c^* is the cost of the optimal plan. Note reg_h can be negative, but is at least -c^*/H.
        Thus, the total cumulative regret is sum_h reg_h = (sum_h c_h) - c^*.

        Since maximum possible cost is c_max, we can upper bound the total cumulative regret by c_max.
        So we normalize by c_max to get a cumulative returns in [0, 1].

        Thus, we have each reg_h \in [-c^*/H / (c_max), 1/H]
        """
        cost = np.exp(min(log_ir_cardinality, self.log_cap_on_cost))
        regret = cost - self.min_plan_cost_per_step
        reward = -regret / self.cap_on_cost
        assert not math.isnan(reward)
        return reward


