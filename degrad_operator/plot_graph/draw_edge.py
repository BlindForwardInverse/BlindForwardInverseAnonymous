import copy
import os
from itertools import accumulate
from pprint import pprint

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from omegaconf import OmegaConf
# from sampler.components.structure_sampler import Structure

from .bezier import Bezier

from .draw_node import DrawNode
from degrad_operator.grafx import Grafx


class DrawEdge:
    def __init__(self,
                 afx_config,
                 plot_config,
                 color_config,
                 draw_node):
        self.afx_config = afx_config
        self.plot_config = plot_config
        self.color_config = color_config

        self.plot_mode = "default"
        self.set_plot_mode()
#         self.draw_node = DrawNode()
        self.draw_node = draw_node

    def set_plot_mode(self, plot_mode="default"):
        self.plot_mode = plot_mode
        assert plot_mode in ["mini", "default", "big"]

    def add_connection_curves(self, ax: Axes, point_list):
        for i in range(len(point_list) - 1):
            self.add_connection_curve(ax, point_list[i], point_list[i + 1])

    def add_connection_curve(self, ax: Axes, p_from, p_to, eps=0.02, full_curve=True):
        if full_curve:
            margin = (p_to[0] - p_from[0]) / 2
            curve_nodes = np.asfortranarray(
                [
                    [
                        p_from[0] + eps,
                        p_from[0] + margin,
                        p_to[0] - margin,
                        p_to[0] - eps,
                    ],
                    [p_from[1], p_from[1], p_to[1], p_to[1]],
                ]
            )
            curve = Bezier.Curve(np.linspace(0, 1, 101), curve_nodes.T)
            (line,) = ax.plot(
                curve[:, 0], curve[:, 1], color="k", zorder=-4, linewidth=5
            )
            line.set_clip_on(False)
        else:
            bezier_from = (
                p_to[0]
                - self.plot_config["afx_module_mini"]["x_spacing"]
                + 2 * self.plot_config["afx_module_mini"]["lets"]["width"]
            )
            margin = (p_to[0] - bezier_from) / 2
            curve_nodes = np.asfortranarray(
                [
                    [
                        bezier_from,
                        bezier_from + margin,
                        p_to[0] - margin,
                        p_to[0] - eps,
                    ],
                    [p_from[1], p_from[1], p_to[1], p_to[1]],
                ]
            )
            curve = Bezier.Curve(np.linspace(0, 1, 101), curve_nodes.T)
            (line,) = ax.plot(
                [p_from[0]] + list(curve[:, 0]),
                [p_from[1]] + list(curve[:, 1]),
                color="k",
                linewidth=3,
                zorder=1,
            )
            line.set_clip_on(False)

    def draw_structure_connection(
        self, ax: Axes, G: Grafx, output_connection, node_from
    ):
        node_to = output_connection[1]
        config = output_connection[2]

        afx_type_from = G.nodes[node_from]["afx_type"]
        afx_type_to = G.nodes[node_to]["afx_type"]
        outlets = self.afx_config[afx_type_from]["outlets"]
        inlets = self.afx_config[afx_type_to]["inlets"]

        outlet_index = outlets.index(config["outlet"])
        inlet_index = inlets.index(config["inlet"])

        p_from = self.get_outlet_pos(
            G.nodes[node_from]["rel_xpos"],
            G.nodes[node_from]["rel_ypos"],
            outlet_index,
            for_edge=True,
        )
        p_to = self.get_inlet_pos(
            G.nodes[node_to]["rel_xpos"],
            G.nodes[node_to]["rel_ypos"],
            inlet_index,
            for_edge=True,
        )

        self.add_connection_curves(ax, [p_from, p_to])

    def draw_node_connection(
        self, ax: Axes, G: Grafx, output_connection, node_from
    ):
        node_to = output_connection[1]
        config = output_connection[2]

        if self.plot_mode in ["default", "big"]:
            out_channel, in_channel = config["outlet"], config["inlet"]  # ex) main, ir

            afx_type_from = G.nodes[node_from]["afx_type"]
            afx_type_to = G.nodes[node_to]["afx_type"]
            outlets = self.afx_config[afx_type_from]["outlets"]
            inlets = self.afx_config[afx_type_to]["inlets"]

            outlet_index = outlets.index(config["outlet"])
            inlet_index = inlets.index(config["inlet"])

            p_from = self.draw_node.get_outlet_pos(
                G.nodes[node_from]["rel_xpos"], 
                G.nodes[node_from]["rel_ypos"],
                outlet_index,
                for_edge=True,
            )
            p_to = self.draw_node.get_inlet_pos(
                G.nodes[node_to]["rel_xpos"],
                G.nodes[node_to]["rel_ypos"],
                inlet_index,
                for_edge=True,
            )
        else:  # mini
            p_from = (
                G.nodes[node_from]["rel_xpos"]
                + self.plot_config["node"]['width'][self.plot_mode],
                G.nodes[node_from]["rel_ypos"]
                + self.plot_config["node"]['height'][self.plot_mode] / 2,
            )
            p_to = (
                G.nodes[node_to]["rel_xpos"],
                G.nodes[node_to]["rel_ypos"]
                + self.plot_config["node"]['height'][self.plot_mode] / 2,
            )

        self.add_connection_curves(ax, [p_from, p_to])
