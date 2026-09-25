# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import sqlite3

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from graph_utils import get_dpeth, get_subgraph_by_time, plot_graph_like_tree


class prop_graph:

    def __init__(self, source_post_content, db_path="", viz=False):
        # Source tweet content for propagation
        self.source_post_content = source_post_content
        self.db_path = db_path  # Path to the db file obtained after simulation
        self.viz = viz  # Whether to visualize the result
        # Determine if the simulation ran successfully, False if the db
        # is empty
        self.post_exist = False

    def build_graph(self):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_path)

        # Execute SQL query and load the results into a DataFrame
        query = "SELECT * FROM post"
        df = pd.read_sql(query, conn)
        # import pdb; pdb.set_trace()
        # Close the database connection
        conn.close()

        source_df = df[df['content'].str.contains(self.source_post_content[0:10], na=False, regex=False)]
        if source_df.empty:
            print("Source post not in DB")
            self.post_exist = False
            return

        self.post_exist = True
        source_post_id = int(source_df.iloc[0]['post_id'])
        self.root_id = str(int(source_df.iloc[0]['user_id']))
        start_time = source_df.iloc[0]['created_at']

        #all_reposts_and_time = []
        #Codul asta este foarte funky, ce avem noi nu functioneaza nici pe departe asa
        # Collect repost data
        # for i in range(len(df)):
        #     content = df.loc[i]["content"]
        #     # There are some encoding issues with the data, use [0:10] to
        #     # avoid them
        #     if self.post_exist is False and self.source_post_content[
        #             0:10] in content:
        #         self.post_exist = True
        #         self.root_id = df.loc[i]["user_id"]
        #     if "repost from" in content and (self.source_post_content[0:10]
        #                                      in content):
        #         repost_history = content.split(". original_post: ")[:-1]
        #         repost_time = df.loc[i]["created_at"]
        #         all_reposts_and_time.append((repost_history, repost_time))


        self.G = nx.DiGraph()
        self.G.add_node(self.root_id, timestamp=0)
        post_user_map = {int(row['post_id']): str(int(row['user_id'])) for _, row in df.iterrows()}
        reposts_df = df[df['original_post_id'].notna()].sort_values('created_at')

        for _, row in reposts_df.iterrows():
            orig_post_id = int(row['original_post_id'])

            if orig_post_id not in post_user_map:
                continue

            orig_user = post_user_map[orig_post_id]
            repost_user = str(int(row['user_id']))
            time_diff = int(row['created_at'] - start_time)

            if orig_user == repost_user:
                continue

            # Enforce single incoming edge: this node already has a parent, skip
            if repost_user in self.G and self.G.in_degree(repost_user) > 0:
                continue

            if repost_user not in self.G:
                self.G.add_node(repost_user, timestamp=time_diff)

            try:
                has_cycle = nx.has_path(self.G, repost_user, orig_user)
            except nx.NetworkXNoPath:
                has_cycle = False
            if not has_cycle:
                self.G.add_edge(orig_user, repost_user)
            

        # Get the start and end timestamps of propagation
        self.start_timestamp = 0
        timestamps = nx.get_node_attributes(self.G, "timestamp")
        if timestamps:
            self.end_timestamp = max(timestamps.values()) + 3
        else:
            self.end_timestamp = 3

        # Calculate propagation graph depth, scale, maximum width
        # (max_breadth), and total structural virality
        self.total_depth = get_dpeth(self.G, source=self.root_id)
        self.total_scale = self.G.number_of_nodes()
        self.total_max_breadth = 0
        last_breadth_list = [1]

        for depth in range(self.total_depth):
            breadth = len(
                list(
                    nx.bfs_tree(
                        self.G, source=self.root_id, depth_limit=depth +
                        1).nodes())) - sum(last_breadth_list)
            last_breadth_list.append(breadth)
            if breadth > self.total_max_breadth:
                self.total_max_breadth = breadth

        if self.root_id in self.G:
            component = nx.node_connected_component(self.G.to_undirected(), self.root_id)
            self.G = self.G.subgraph(component).copy()

    def viz_graph(self, time_threshold=10000):
        # Visualize the graph, can choose to only view the propagation graph
        # within the first time_threshold seconds
        subG = get_subgraph_by_time(self.G, time_threshold)
        plot_graph_like_tree(subG, self.root_id)

    def plot_depth_time(self, separate_ratio: float = 1):
        """
        Entire propagation process
        Detailed depiction of the data for the process before separate_ratio
        Rough depiction of the data afterwards
        Default to 1
        Use this parameter when the propagation time is very long, can be set
        to 0.01
        """
        # Calculate depth-time information
        depth_list = []
        # Normal interval is 1 for the time list, depth-time information needs
        # to be detailed enough
        self.d_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp), 1))
        depth = 0
        for t in self.d_t_list:
            if depth < self.total_depth:
                try:
                    sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                    depth = get_dpeth(sub_g, source=self.root_id)
                except Exception:
                    import pdb

                    pdb.set_trace()
            depth_list.append(depth)
        self.depth_list = depth_list

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.d_t_list, self.depth_list)

            # Add titles and labels
            plt.title("Propagation depth-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Depth")

            # Display the figure
            plt.show()
        else:
            return self.d_t_list, self.depth_list

    def plot_scale_time(self, separate_ratio: float = 1.0):
        """
        Detailed depiction of the data between the start and separate_ratio*T
        of the entire propagation process
        Rough depiction of the data afterwards
        Default to 1
        Use this parameter when the propagation time is very long, can be set
        to 0.1
        """
        self.node_nums = []
        # Detailed depiction of the data from start_time to separate point,
        # rough depiction from separate point to end_time
        separate_point = int(
            int(self.start_timestamp) + separate_ratio *
            (int(self.end_timestamp) - int(self.start_timestamp)))

        self.s_t_list = list(
            range(
                int(self.start_timestamp), separate_point,
                1))  # + list(range(separate_point, int(self.end_time), 1000))
        for t in self.s_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                node_num = sub_g.number_of_nodes()
            except Exception:
                import pdb

                pdb.set_trace()

            self.node_nums.append(node_num)

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.s_t_list, self.node_nums)
            # Set the x-axis to log scale
            # ax.set_xscale('log')

            # Set the x-axis tick positions
            # ax.set_xticks([1, 10, 100, 1000, 10000])

            # Set the x-axis tick labels
            # ax.set_xticklabels(['1', '10', '100', '1k', '10k'])

            # Add titles and labels
            plt.title("Propagation scale-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Scale")

            # Display the figure
            plt.show()
        else:
            return self.s_t_list, self.node_nums

    def plot_max_breadth_time(self, interval=1):
        self.max_breadth_list = []

        self.b_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  interval))
        for t in self.b_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb

                pdb.set_trace()
            max_depth = self.depth_list[t - self.b_t_list[0]]
            max_breadth = 0
            last_breadth_list = [1]
            for depth in range(max_depth):
                breadth = len(
                    list(
                        nx.bfs_tree(
                            sub_g, source=self.root_id, depth_limit=depth +
                            1).nodes())) - sum(last_breadth_list)
                last_breadth_list.append(breadth)
                if breadth > max_breadth:
                    max_breadth = breadth
            self.max_breadth_list.append(max_breadth)

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.b_t_list, self.max_breadth_list)

            # Add titles and labels
            plt.title("Propagation max breadth-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Max breadth")

            # Display the figure
            plt.show()
        else:
            return self.b_t_list, self.max_breadth_list

    def plot_structural_virality_time(self, interval=1):
        self.sv_list = []
        self.sv_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  interval))

        for t in self.sv_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb

                pdb.set_trace()
            sub_g = sub_g.to_undirected()
            sv = nx.average_shortest_path_length(sub_g)
            self.sv_list.append(sv)

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.sv_t_list, self.sv_list)

            # Add titles and labels
            plt.title("Propagation structural virality-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Structural virality")

            # Display the figure
            plt.show()
        else:
            return self.sv_t_list, self.sv_list
