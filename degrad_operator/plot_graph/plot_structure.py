import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from networkx import topological_sort
from omegaconf import OmegaConf
from .draw_edge import DrawEdge
from degrad_operator.graph_sampler.components.structure_sampler import Structure
from degrad_operator.grafx import Grafx
import os

opj = os.path.join

class PlotStructure:
    def __init__(self,
                 afx_config,
                 plot_config,
                 color_config,
                 unit_config,
                 synonyms,
                 mini_synonyms,
                 ):
        self.afx_config = afx_config
        self.plot_config = plot_config
        self.color_config = color_config
        self.unit_config = unit_config
        self.synonyms = synonyms
        self.mini_synonyms = mini_synonyms
#         self.draw_edge = DrawEdge()
        
        self.set_plot_mode()
    
    def set_plot_mode(self, plot_mode="default"):
        self.plot_mode = plot_mode
        assert plot_mode in ["mini", "default", "big"]
            
    def plot(
        self,
        S: Structure,
        plot_mode: str,
        apply_node_info: bool = True,
        with_text=False,
        name="structure_plot.png",
        path="plot",
    ):
        self.set_plot_mode(plot_mode)

        self.setup_structure(S, apply_node_info)
        fig, ax = plt.subplots()
        self.prepare_canvas(ax)
        self.draw_structure(ax, S, with_text=with_text)
        self.save_canvas(ax, opj(path, name))

    def setup_structure(self, S: Structure, apply_node_info: bool = True):
        """
        Structure has
        - width, height
        - padding_x, padding_y
        - wrapper_height
        - rel_xpos, rel_ypos
        - xpos, ypos : final
        """
        # Topological sort to get column order of structure nodes
        topological_order = list(topological_sort(S))

        # Compute depths of each nodes (for node x pos)
        node_depths = {node: -1 for node in S.nodes()}  # Initialize
        for node in topological_order:
            if S.in_degree(node) == 0:  # If no incoming edges, input node
                node_depths[node] = 0
            else:
                # Maximum depth of parents + 1
                node_depths[node] = (
                    max([node_depths[parent] for parent, _ in S.in_edges(node)]) + 1
                )

        def get_depth(node):
            return node_depths.get(node, None)

        """ Xpos"""
        # Assign X position of structure nodes
        if apply_node_info is False:
            def get_xpos(depth):
                return (self.plot_config['structure']['width'][self.plot_mode] 
                        + self.plot_config['structure']['margin_x'][self.plot_mode]) * depth

            for node in list(S.nodes):
                S.nodes[node]["width"] = self.plot_config['structure']['width'][self.plot_mode]
                S.nodes[node]["xpos"] = get_xpos(get_depth(node))

        else:  # Apply node info is True
            # Count row node num and column node num of Structure
            def row_count(node_list):
                filtered = [
                    s
                    for s in node_list
                    if not (
                        s.startswith("rir")
                        or s.startswith("micir")
                        or (s.startswith("noise") and not s.startswith("noisegate"))
                        or s.startswith("lfo")
                        or s.startswith("stereo_lfo")
                    )
                ]
                return len(filtered)

            def column_count(node_list):
                # for node in node_list:
                #     if node.startswith("lfo") or node.startswith("stereo_lfo"):
                #         return 2
                return 1

            for node in S.nodes():
                G_nodes = S.nodes[node]["node_names"]
                S.nodes[node]["row"] = row_count(G_nodes)
                S.nodes[node]["column"] = column_count(G_nodes)
                S.nodes[node]["height"] = (
                    S.nodes[node]["column"] * self.plot_config['structure']['height'][self.plot_mode]
                )

            max_node_width_per_depth = {
                depth: -1 for depth in node_depths.values()  # .keys()
            }  # Initialize

            for node in S.nodes():
                depth = get_depth(node)
                node_width = (
                    S.nodes[node]["row"] * (self.plot_config['node']['width'][self.plot_mode] + self.plot_config['node']['margin'][self.plot_mode])
                    + self.plot_config['node']['margin'][self.plot_mode]
                )
                S.nodes[node]["width"] = node_width
                if node_width > max_node_width_per_depth[depth]:  #
                    max_node_width_per_depth[depth] = node_width

            def acc_max_node_width_per_depth(depth):
                acc = 0
                for i in range(depth):
                    acc += max_node_width_per_depth[i]
                return acc

            node_xpos_per_depth = {0: 0}  # Initialize
            for node in list(S.nodes):
                depth = get_depth(node)
                if depth in node_xpos_per_depth:
                    xpos = node_xpos_per_depth[depth]
                else:
                    xpos = (
                        acc_max_node_width_per_depth(depth)
                        + (self.plot_config['structure']['margin_x'][self.plot_mode] * 2) * depth
                    )
                    node_xpos_per_depth[depth] = xpos
                S.nodes[node]["xpos"] = xpos

        """ Ypos """
        # Count nodes in each depths (for node y pos)
        node_depth_counts = {}
        for depth in node_depths.values():
            if depth in node_depth_counts:
                node_depth_counts[depth] += 1
            else:
                node_depth_counts[depth] = 1

        # Assign wrapper_height (applying leaf node heights) of structure nodes
        def set_wrapper_height(S: Structure, node):
            leaf_num = len(list(S.successors(node)))
            S.nodes[node]["wrapper_height"] = self.plot_config['structure']['wrapper_height_multiplier'][self.plot_mode] * leaf_num

            for leaf in list(S.successors(node)):
                set_wrapper_height(S, leaf)

        input_node = 0
        set_wrapper_height(S, input_node)
        
        # Assign ypos from parent node
        def set_ypos_from_in(S: Structure, node, from_parent_y):
            if "ypos" in S.nodes[node]:
                S.nodes[node]["ypos"] = None  # Set merged nodes later
            else:
                S.nodes[node]["ypos"] = from_parent_y

            leaves = list(S.successors(node))
            leaf_wrapper_height_sum = sum(
                S.nodes[leaf]["wrapper_height"] for leaf in leaves
            )

            leaf_y = from_parent_y + (
                S.nodes[node]["wrapper_height"] - leaf_wrapper_height_sum
            ) / 2
            
            for leaf in list(S.successors(node)):
                set_ypos_from_in(S, leaf, leaf_y)
                leaf_y += S.nodes[leaf]["wrapper_height"]

        set_ypos_from_in(S, node=input_node, from_parent_y=0)
        

        # For merged nodes
        for node in S.nodes():
            if S.nodes[node]["ypos"] is None:
                preds = list(S.predecessors(node))
                S.nodes[node]["ypos"] = sum(
                    [S.nodes[pred]["ypos"] for pred in preds]
                ) / len(preds)

        # Reassign ypos from children nodes
        for node in S.nodes():
            if any(True for _ in S.successors(node)) and len(list(S.successors(node))) > 1:
                succs = list(S.successors(node))
                S.nodes[node]['ypos'] = sum([S.nodes[succ]['ypos'] for succ in succs]) / len(succs)
        
        if 0 in S.nodes() and 'out' in S.nodes():
            S.nodes[0]['ypos'] = S.nodes['out']['ypos']


    def draw_structure(self, ax: Axes, S: Structure, plot_mode, with_text=False):
        # Draw structure nodes
        for node in S.nodes():
            patch = patches.FancyBboxPatch(
                (S.nodes[node]["xpos"], S.nodes[node]["ypos"]),
                S.nodes[node]["width"],
                S.nodes[node]["height"],
                alpha=0.05,
                facecolor="black",
                edgecolor="black",
                linewidth=self.plot_config['edge']['line_thickness'][self.plot_mode],
                fill=True,
                zorder=3,
                boxstyle="round,pad=0.1,rounding_size=20",
            )
            patch.set_clip_on(False)
            ax.add_patch(patch)

            if with_text:
                ax.text(
                    x=S.nodes[node]["xpos"] + S.nodes[node]["width"] / 2,
                    y=S.nodes[node]["ypos"] + self.plot_config['structure']['height'][self.plot_mode] / 2,
                    s=node,
                    ha="center",
                    fontsize=self.plot_config['ui']['font_size'][self.plot_mode],
                    zorder=4,
                )
                ax.text(
                    x=S.nodes[node]["xpos"] + S.nodes[node]["width"] / 2,
                    y=S.nodes[node]["ypos"] + self.plot_config['structure']['height'][self.plot_mode] / 2 - 50,
                    s=S.nodes[node]["node_names"],
                    ha="center",
                    fontsize=self.plot_config['ui']['small_font_size'][self.plot_mode],
                    zorder=4,
                )

        # Draw structure edges
        def draw_structure_connection(ax: Axes, S: Structure, output_connection, node_from):
            node_to = output_connection[1]
            p_from = (
                S.nodes[node_from]["xpos"] + S.nodes[node]["width"],
                S.nodes[node_from]["ypos"] + self.plot_config['structure']['height'][self.plot_mode] / 2,
            )
            p_to = (
                S.nodes[node_to]["xpos"],
                S.nodes[node_to]["ypos"] + self.plot_config['structure']['height'][self.plot_mode] / 2,
            )
            self.draw_edge.add_connection_curves(ax, [p_from, p_to])

        for node in S:
            output_connections = S.out_edges(nbunch=[node], data=True)
            for output_connection in output_connections:
                draw_structure_connection(ax, S, output_connection, node)

    def prepare_canvas(self, ax: Axes):
        plt.style.use("default")
        ax.axis("off")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 400)

    def save_canvas(self, ax, name="plot.png", path="plot"):
        plt.savefig(opj(path, name), bbox_inches="tight", dpi=300)
        print("Saved Plot.")
