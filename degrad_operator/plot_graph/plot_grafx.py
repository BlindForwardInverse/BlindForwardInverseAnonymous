import copy
import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from networkx import topological_sort
from omegaconf import OmegaConf
from .draw_edge import DrawEdge
from .draw_node import DrawNode
from .plot_structure import PlotStructure

from degrad_operator.grafx import Grafx

opj = os.path.join


class PlotGrafx:
    def __init__(self, verbose=False):
        self.afx_config = OmegaConf.load("configs/degrad_operator/afx_module_configs.yaml")
        self.plot_config = OmegaConf.load("configs/degrad_operator/plot_configs/plot.yaml")
        self.color_config = OmegaConf.load("configs/degrad_operator/plot_configs/colors.yaml")
        self.unit_config = OmegaConf.load("configs/degrad_operator/plot_configs/units.yaml")
        self.synonyms = OmegaConf.load("configs/degrad_operator/plot_configs/synonyms.yaml")
        self.mini_synonyms = OmegaConf.load("configs/degrad_operator/plot_configs/mini_synonyms.yaml")
        self.input_types = ["vocal", "drums", "guitar", "bass", "piano", "speech"]
        self.verbose = verbose

        self.lfo_nodes = ["lfo", "stereo_lfo"]
        self.ir_nodes = ["rir", "micir", "noise"]
        self.half_nodes = self.lfo_nodes + self.ir_nodes
        self.compressor_nodes = ["compressor", "inverted_compressor"]
        
        self.plot_structure = PlotStructure(self.afx_config, self.plot_config, self.color_config,
                                            self.unit_config, self.synonyms, self.mini_synonyms)
        self.draw_node = DrawNode(self.afx_config, self.plot_config, self.color_config,
                                  self.unit_config, self.synonyms, self.mini_synonyms)
        self.draw_edge = DrawEdge(self.afx_config, self.plot_config, self.color_config, self.draw_node)


    def set_plot_mode(self, plot_mode="default"):
        assert plot_mode in ["mini", "default", "big"]
        self.plot_mode = plot_mode
        self.plot_structure.set_plot_mode(plot_mode)
        self.draw_node.set_plot_mode(plot_mode)
        self.draw_edge.set_plot_mode(plot_mode)


    def plot(
        self, G: Grafx, plot_mode="default", name="grafx_plot.png"
    ):
        self.set_plot_mode(plot_mode)
        G = copy.deepcopy(G)
        S = G.structure
        if self.verbose:
            print(G)

#         self.plot_structure.setup_structure(S, apply_node_info=True)
        self.setup_grafx_chain(G)

        fig, ax = plt.subplots()
        self.prepare_canvas(ax)
#         self.plot_structure.draw_structure(ax, S, self.plot_mode)
        self.draw_grafx_chain(ax, G)
        ax.invert_yaxis()
        self.save_canvas(name=name)
        plt.close()

    def setup_grafx_chain(self, G: Grafx):
        """
        Grafx node has
        - rel_xpos, rel_ypos
        - inlets, outlets
        """

        """ Get first, last node of Grafx """
        topo_order = list(topological_sort(G))
        for item in topo_order:
            if not (G.nodes[item]['afx_type'] in self.half_nodes):
                first_node = item
                break
        last_node = topo_order[-1]
        
        """ Get depth for every node"""
        def get_depth(node, depth=0):
            node_config = G.nodes[node]
            if ("depth" not in node_config or
                node_config["depth"] < depth):
                node_config["depth"] = depth
                for suc in G.successors(node):
                    get_depth(suc, depth=depth+1)

        get_depth(first_node, 0)

        for node in topo_order:
            node_config = G.nodes[node]
            if "depth" not in node_config:
                suc = [suc for suc in G.successors(node)][0]
                node_config["depth"] = G.nodes[suc]["depth"] - 1

        """ Get level for every node"""
        for node in topo_order:
            node_config = G.nodes[node]
            structure_node = node_config["structure_node"]
            node_config["level"] = G.structure.nodes[structure_node]["level"]


        """ Update relative position x, y from first node """
        half_node_pos_history = []
        for node in topo_order:
            node_config = G.nodes[node]
            node_config["rel_xpos"] = (
                self.plot_config["node"]['width'][self.plot_mode] 
                + self.plot_config["node"]['margin'][self.plot_mode]
            ) * node_config["depth"] + self.plot_config["node"]['margin'][self.plot_mode]

            node_config["rel_ypos"] = (
                self.plot_config["structure"]['wrapper_height_multiplier'][self.plot_mode]
#                 + self.plot_config["node"]['height'][self.plot_mode]
#                 + self.plot_config["node"]['margin'][self.plot_mode]
                ) * node_config["level"] + self.plot_config["structure"]['padding_y'][self.plot_mode]

            # Adjust positions for special nodes
            # (inv)compressor node between crossover and mix
            if (
                G.nodes[node]["afx_type"] in self.compressor_nodes # Compressor
                and any(True for _ in G.successors(node)) # Not last Node
                and G.nodes[list(G.successors(node))[0]]["afx_type"] == "mix" # Mix at the successor
            ):
                G.nodes[node]["rel_ypos"] += self.plot_config["node"]['crossover_offset_y'][self.plot_mode]

            # ir, noise, lfo
            ''' If there is a node in front of (ir, noise, lfo)-attached node,
                locate the (ir, noise, lfo) to just right below the front node.
                Otherwise, locate it to next to the node'''

            if node_config["afx_type"] in self.half_nodes:
                attached_node = [suc for suc in G.successors(node)][0]
                half_node_position = (node_config['depth'], node_config['level'])

                if all(half_node_position != (G.nodes[pred]['depth'], G.nodes[pred]['level'])
                        for pred in G.predecessors(attached_node) if pred != node):
                    node_config["rel_ypos"] += (self.plot_config["node"]['lfo_offset_y'][self.plot_mode])
                else:
                    node_config["rel_ypos"] += (self.plot_config["node"]['height'][self.plot_mode])

                if (node_config["rel_xpos"], node_config["rel_ypos"]) in half_node_pos_history:
                    node_config["rel_ypos"] += self.plot_config["node"]['lfo_offset_y'][self.plot_mode] * 2
                half_node_pos_history.append((node_config["rel_xpos"], node_config["rel_ypos"]))

        """ Save inlets, outlets """
        for node, data in G.nodes(data=True):
            if "afx_type" in G.nodes[node]:  # else it is node from other structure
                afx_type = G.nodes[node]["afx_type"]
                G.nodes[node]["inlets"] = self.afx_config[afx_type]["inlets"]
                G.nodes[node]["outlets"] = self.afx_config[afx_type]["outlets"]

    def draw_grafx_chain(self, ax: Axes, G: Grafx):
        """ Draw nodes """
        for node in G.nodes():
            if "afx_type" not in G.nodes[node]:
                continue
            if G.nodes[node]["afx_type"] == "in":  # input
                edge_color = self.color_config["inout_module"]["input_color"]
            elif G.nodes[node]["afx_type"] == "out":  # output
                edge_color = self.color_config["inout_module"]["output_color"]
            else:
                edge_color = self.color_config['colors']['header_color']
                # node_class = self.afx_config[node]['class']

            name = G.nodes[node]["afx_type"]
            module_class = self.afx_config[G.nodes[node]["afx_type"]]["class"]
            if name in self.half_nodes: module_class = 'controller' # Match micir, rir, noise nodes with controller.
            face_color = self.color_config["colors"]["header_colors"][module_class]
            self.draw_node.draw_node(
                ax=ax,
                x= G.nodes[node]["rel_xpos"],
                y= G.nodes[node]["rel_ypos"],
                w=self.plot_config["node"]['width'][self.plot_mode], # self.node_width,
                h=self.plot_config["node"]['height'][self.plot_mode], # self.node_height,
                name=name,
                inlets=G.nodes[node]["inlets"],
                outlets=G.nodes[node]["outlets"],
                font_size=self.plot_config['ui']['font_size'][self.plot_mode],# self.font_size,
                face_color=face_color if face_color is not None else "w",
                edge_color=edge_color,
                desc=G.nodes[node]["parameters"] if self.plot_mode == "big" else None,
            )

        """ Draw edges between nodes in chain """
        for node in G.nodes:
            module = G.nodes[node]
            output_connections = G.out_edges(nbunch=[node], data=True)
            filtered_connections = [
                conn
                for conn in output_connections
            ]
            for output_connection in filtered_connections:
                self.draw_edge.draw_node_connection(ax, G, output_connection, node)

    def prepare_canvas(self, ax: Axes):
        plt.style.use("default")
        ax.axis("off")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 400)

    def save_canvas(self, name="plot.png", path="plot"):
        plt.savefig(opj(path, name), bbox_inches="tight", dpi=300)
        if self.verbose:
            print("Saved Plot.")
        
