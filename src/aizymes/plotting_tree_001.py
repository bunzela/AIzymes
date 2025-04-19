def plot_tree(self, 
              max_generation=0, 
              radius_offset=0, 
              max_angle=360, 
              color="combined_score",
              generation_labels=None, 
              start_angle_label=0, 
              filename=None):

    import os
    import sys
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    import math

    from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm, PowerNorm
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    if max_generation == 0:
        max_generation = int(self.plot_scores_df['generation'].max() + 1)

    if "redox" not in self.SELECTED_SCORES:
        self.TARGET_REDOX=0
        
    # Create a directed graph
    G = nx.DiGraph()
    G.add_node(0, 
               sequence="X",
               generation=0,
               hamming_distance=np.nan,
               BioDC_redox=np.nan,
               **{f"{score}_score": np.nan for score in self.SELECTED_SCORES}, 
               final_score=np.nan, 
               combined_score=np.nan)
    
    # Build a mapping from the original DataFrame index to a new sequential index (starting at 1)
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.plot_scores_df.index, start=1)}
    
    # Iterate over the rows using iterrows() to get both the original index and the row data
    for old_idx, row in self.plot_scores_df.iterrows():
        new_idx = index_mapping[old_idx]  # new sequential row number
    
        if pd.isna(row['parent_index']):
            continue
        if row['generation'] + 1 > max_generation and max_generation != 0:
            continue
    
        # Adjust the parent's index using the mapping.
        if row['parent_index'] != "Parent":
            parent_old = int(float(row['parent_index']))
            parent_idx = index_mapping.get(parent_old, 0)
        else:
            parent_idx = 0
    
        distance = row['parent_mutations']
        if not isinstance(distance, (int, float)) or np.isnan(distance):
            print(f"Illegal hamming_distance value detected: {distance} for variant {new_idx}")
            sys.exit()

        if "redox" in self.SELECTED_SCORES:
            BioDC_redox = row['BioDC_redox']
        else:
            BioDC_redox = np.nan            

        G.add_node(new_idx,
                   sequence            = row['sequence'],
                   generation          = row['generation'] + 1,
                   hamming_distance    = distance,
                   BioDC_redox         = abs(BioDC_redox-self.TARGET_REDOX),
                   **{f"{score}_score" : self.scores[f"{score}_score"][new_idx - 1] for score in self.SELECTED_SCORES},
                   final_score         = self.scores[f"final_score"][new_idx - 1],
                   combined_score      = self.scores['combined_score'][new_idx - 1])
    
        G.add_edge(parent_idx,
                   new_idx,
                   hamming_distance    = distance,
                   BioDC_redox         = abs(BioDC_redox-self.TARGET_REDOX),
                   **{f"{score}_score" : self.scores[f"{score}_score"][new_idx - 1] for score in self.SELECTED_SCORES},
                   final_score         = self.scores[f"final_score"][new_idx - 1],
                   combined_score      = self.scores['combined_score'][new_idx - 1])

    # Calculate subtree sizes
    def calculate_subtree_sizes(graph, node, subtree_sizes):
        size = 1
        for child in graph.successors(node):
            size += calculate_subtree_sizes(graph, child, subtree_sizes)
        subtree_sizes[node] = size
        return size

    subtree_sizes = {}
    calculate_subtree_sizes(G, 0, subtree_sizes)

    # Assign polar coordinates with grouping
    pos_polar = {}

    def assign_polar_positions(node, radius=0, angle_start=0, angle_range=max_angle):
        pos_polar[node] = (radius, math.radians(angle_start))
        children = list(G.successors(node))
        if children:
            children.sort(key=lambda x: subtree_sizes[x])
            num_children = len(children)
            child_angle_range = angle_range / num_children
            for i, child in enumerate(children):
                child_angle_start = angle_start + i * child_angle_range
                assign_polar_positions(child, radius + 1, child_angle_start, child_angle_range)

    assign_polar_positions(0, radius=0, angle_start=0, angle_range=max_angle)

    def adjust_uniform_angles(pos_polar):
        unique_angles = sorted({math.degrees(angle) for _, angle in pos_polar.values()})
        num_angles = len(unique_angles)
        uniform_angles = [max_angle / num_angles * i for i in range(num_angles)]
        angle_mapping = {unique_angles[i]: uniform_angles[i] for i in range(num_angles)}
        for node, (radius, angle) in pos_polar.items():
            pos_polar[node] = (radius, math.radians(angle_mapping[math.degrees(angle)]))
    adjust_uniform_angles(pos_polar)

    for node, (radius, angle) in pos_polar.items():
        if radius != 0:
            pos_polar[node] = (radius + radius_offset, angle)

    def polar_to_cartesian(radius, angle):
        return radius * math.cos(angle), radius * math.sin(angle)

    # Set up the colormap and normalization
    if color == "hamming_distance":
        min_color = 0
        max_color = int(max([G.edges[edge]['hamming_distance'] for edge in G.edges]))
        base_cmap = LinearSegmentedColormap.from_list("WhiteToBlue", [(1, 1, 1), (0, 0, 1)])
        bounds = np.linspace(min_color, max_color + 1, max_color + 1)
        norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1, clip=True)
        custom_cmap = ['salmon'] + [base_cmap(i / max_color) for i in range(1, max_color)]
        custom_cmap = ListedColormap(custom_cmap)
        norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1, clip=True)
        color_values = [G.edges[edge]['hamming_distance'] for edge in G.edges]
    else:
        values = [G.edges[edge][color] for edge in G.edges]

        if color ==  "BioDC_redox":

            """
            vmin=-1
            vmax=1
            vmid=self.HIGHSCORE["BioDC_redox"]
            base_cmap = LinearSegmentedColormap.from_list(
                "RedYellowBlue",
                [
                    (0.0, (1, 0, 0)),
                    (0.3, (1, 0.5, 0)),
                    (0.5, (1, 1, 0)),
                    (0.7, (0.5, 0.5, 1)),
                    (1.0, (0, 0, 1))
                ]
            )
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)   
            """

            #vmax = abs(np.amax(self.scores['BioDC_redox'] - self.TARGET_REDOX))
            #norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
            #base_cmap = LinearSegmentedColormap.from_list("BlueToWhie", [(0, 0, 1), (1, 1, 1)])
            
            # Grab your edge values
            color_values = [G.edges[edge][color] for edge in G.edges]
            
            # Pride flag colors
            pride_colors = [
                "#750787",  # violet
                "#004dff",  # blue
                "#008026",  # green
                "#ffed00",  # yellow
                "#ff8c00",  # orange
                "#e40303"   # red
            ]
            n_colors = len(pride_colors)
            
            # Compute max value for bounds
            vmax = max(color_values)
            boundaries = np.linspace(0, vmax, n_colors + 1)

            # clip at 0.3
            vmax = 0.3
            boundaries = np.linspace(0, vmax, n_colors + 1)
            color_values = [min(val, vmax) for val in color_values]

            # Set colormap and normalization
            base_cmap = ListedColormap(pride_colors)
            norm = BoundaryNorm(boundaries, base_cmap.N)
            
            # Apply to edges
            edge_colors = [base_cmap(norm(val)) for val in color_values]

        else:
            
            norm = Normalize(vmin=0, vmax=max(values))
            base_cmap = LinearSegmentedColormap.from_list("WhiteToBlue", [(1, 1, 1), (0, 0, 1)])
            
        color_values = [G.edges[edge][color] for edge in G.edges]

    edge_colors = [base_cmap(norm(val)) for val in color_values]

    # Create a figure and main axis; add a colorbar to the same figure.
    fig, ax = plt.subplots(figsize=(6,6))

    # Plot arcs, edges, and nodes
    for node in G.nodes:
        children = list(G.successors(node))
        if len(children) > 1:
            parent_angle = pos_polar[node][1]
            child_angles = [(child, abs(pos_polar[child][1] - parent_angle)) for child in children]
            furthest_child = max(child_angles, key=lambda x: x[1])[0]
            r0, theta0 = pos_polar[node]
            r1, theta1 = pos_polar[furthest_child]
            arc_radius = r1 - 1.0
            arc_theta = np.linspace(theta0, theta1, 10)
            arc_x = arc_radius * np.cos(arc_theta)
            arc_y = arc_radius * np.sin(arc_theta)
            ax.plot(arc_x, arc_y, color="black", linewidth=1.5, zorder=3)

    for edge_idx, edge in enumerate(G.edges):
        start, end = edge
        r0, theta0 = pos_polar[start]
        r1, theta1 = pos_polar[end]
        x_mid2, y_mid2 = polar_to_cartesian(r1 - 1.0, theta1)
        x1, y1 = polar_to_cartesian(r1, theta1)
        ax.plot([x_mid2, x1], [y_mid2, y1], color='black', linewidth=1.5, zorder=4)
        ax.scatter(x1, y1, s=20, c="k", zorder=5)
        ax.scatter(x1, y1, s=5, c=[edge_colors[edge_idx]], zorder=6)

    ax.scatter(0, 0, s=25, c="k", zorder=10, alpha=0.5)

    for radius in range(1, max_generation + radius_offset + 1):
        circle = plt.Circle((0, 0), radius, color='grey', linewidth=1.0, fill=False, 
                            linestyle=(0, (1, 5)), zorder=-3)
        ax.add_artist(circle)
        if generation_labels and radius in generation_labels:
            label_text = f"generation {radius}" if radius == generation_labels[-1] else f"{radius}"
            char_spacing = 0.3
            start_angle = start_angle_label
            label_length = (len(label_text) - 1) * char_spacing
            start_arc_length = (math.pi * radius * (start_angle / 360)) - (label_length / 2)
            for i, char in enumerate(label_text):
                arc_length = start_arc_length + i * char_spacing
                angle = 180 - (arc_length / (math.pi * radius) * 360)
                x = radius * math.cos(math.radians(angle))
                y = radius * math.sin(math.radians(angle))
                ax.text(x, y, " ", ha="center", va="center", rotation=angle - 90,
                        rotation_mode="anchor", fontsize=8, zorder=-2,
                        bbox=dict(boxstyle="round,pad=0.5", edgecolor="none", facecolor="white"))
                ax.text(x, y, char, ha="center", va="center", rotation=angle - 90,
                        rotation_mode="anchor", fontsize=8, color="grey", zorder=-1)

    ax.set_xlim(-(max_generation + radius_offset + 0.5), max_generation + radius_offset + 0.5)
    ax.set_ylim(-(max_generation + radius_offset + 0.5), max_generation + radius_offset + 0.5)
    ax.axis("off")

    # Create a ScalarMappable for the colorbar using the same normalization and cmap
    if color == "hamming_distance":
        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=plt.Normalize(vmin=0, vmax=max_color))
    else:
        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    if color == "combined_score":
        cbar.set_label('combined score')
    if color == "final_score":
        cbar.set_label('final score')
    elif color == "BioDC_redox":
        cbar.set_label('Difference to target potential')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
        """
        tick_vals = [
            vmin,
            vmin + 0.5 * (vmid - vmin),
            vmid,
            vmid + 0.5 * (vmax - vmid),
            vmax
        ]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{tv:.2f}" for tv in tick_vals])
        """
        
    elif color == "hamming_distance":
        cbar.set_label('Hamming Distance')
    else:
        cbar.set_label(color)
        
    # Always save to self.FOLDER_HOME with the naming scheme
    save_path = os.path.join(self.FOLDER_HOME, "plots", f"plot_tree_{color}.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=300)