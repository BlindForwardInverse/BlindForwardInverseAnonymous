import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from omegaconf import OmegaConf
import numpy as np
from degrad_operator.grafx import Grafx


class DrawNode:
    def __init__(self,
                afx_config,
                plot_config,
                color_config,
                unit_config,
                synonyms,
                mini_synonyms
                ):        
        self.afx_config = afx_config
        self.plot_config = plot_config
        self.color_config = color_config
        self.unit_config = unit_config
        self.synonyms = synonyms
        self.mini_synonyms = mini_synonyms
        
        self.lfo_nodes = ["lfo", "stereo_lfo"]
        self.ir_nodes = ["rir", "micir", "noise"]
        self.half_nodes = self.lfo_nodes + self.ir_nodes
        self.compressor_nodes = ["compressor", "inverted_compressor"]
        
        self.set_plot_mode()
    
    def set_plot_mode(self, plot_mode="default"):
        self.plot_mode = plot_mode
        assert plot_mode in ["mini", "default", "big"]

    def get_inlet_pos(self, node_x, node_y, inlet_index, for_edge=False):
        inlet_x = node_x
        inlet_y = (
            self.plot_config['node']['let_offset_y'][self.plot_mode]
            + node_y
            + self.plot_config['node']['let_padding_y'][self.plot_mode]
            + self.plot_config['node']['let_interval_y'][self.plot_mode] * inlet_index
        )
        if for_edge:
            inlet_y = inlet_y + self.plot_config['node']['let_height'][self.plot_mode] / 2
        return (inlet_x, inlet_y)

    def get_outlet_pos(self, node_x, node_y, outlet_index, for_edge=False):
        outlet_x = node_x + self.plot_config['node']['width'][self.plot_mode]
        outlet_y = (
            self.plot_config['node']['let_offset_y'][self.plot_mode]
            + node_y
            + self.plot_config['node']['let_padding_y'][self.plot_mode]
            + self.plot_config['node']['let_interval_y'][self.plot_mode] * outlet_index
        )
        if for_edge:
            outlet_y = outlet_y + self.plot_config['node']['let_height'][self.plot_mode] / 2
        return (outlet_x, outlet_y)

    def draw_node(
        self,
        ax: Axes,
        x,
        y,
        w,
        h,
        name,
        inlets,
        outlets,
        font_size,
        face_color,
        edge_color,
        desc,
    ):
        if name in self.half_nodes:
            h_val = self.plot_config['node']['height'][self.plot_mode] / 2 # lfo_node_height
        else:
            h_val = h

        if self.plot_mode in ["mini", "default"]:
            # name text
            ax.text(
                x=x + self.plot_config['node']['width'][self.plot_mode] / 2,
                y=y + self.plot_config['node']['name_offset_y'][self.plot_mode]
                if name not in self.half_nodes
                else y + self.plot_config['node']['lfo_name_offset_y'][self.plot_mode],
                s=self.mini_synonyms[name] if self.plot_mode == "default" else name[0],
                ha="center",
                fontsize=font_size,
                zorder=7,
            )
            # main box
            patch = patches.FancyBboxPatch(
                (x, y),
                w,
                h_val,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=self.plot_config['edge']['line_thickness'][self.plot_mode],
                fill=True,
                zorder=6,
                boxstyle="round,pad=0.1,rounding_size=20",
            )
            patch.set_clip_on(False)
            ax.add_patch(patch)
        else:  # big
            # name text
            ax.text(
                x=x + self.plot_config['node']['width'][self.plot_mode] / 2,
                y=y + self.plot_config['node']['name_offset_y'][self.plot_mode],
                s=self.synonyms[name] if name in self.synonyms else name,
                ha="center",
                fontsize=font_size,
                zorder=7,
            )
            # main box
            patch = patches.FancyBboxPatch(
                (x, y),
                w,
                h_val,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=self.plot_config['edge']['line_thickness'][self.plot_mode],
                fill=True,
                zorder=6,
                boxstyle="round,pad=0.1,rounding_size=20",
            )

            # desc box
            patch2 = patches.FancyBboxPatch(
                (x, y + self.plot_config['node']['desc_offset_y'][self.plot_mode]),
                w,
                h_val - self.plot_config['node']['desc_offset_y'][self.plot_mode],
                facecolor="white",
                edgecolor=edge_color,
                linewidth=self.plot_config['edge']['line_thickness'][self.plot_mode],
                fill=True,
                zorder=7,
                boxstyle="round,pad=0.1,rounding_size=20",
            )

            # desc text
            for i, (k, v) in enumerate(desc.items()):
                if not isinstance(v, (str, np.ndarray)) :
                    ax.text(
                        x=x + self.plot_config['node']['desc_offset_x'][self.plot_mode],
                        y=y + self.plot_config['node']['desc_offset_y'][self.plot_mode] \
                            + 30 \
                            + i * self.plot_config['node']['desc_margin_y'][self.plot_mode],
                        s=f"{self.synonyms[k] if k in self.synonyms else k} : {v:.2f}",
                        fontsize=self.plot_config['ui']['small_font_size'][self.plot_mode],
                        zorder=8,
                    )

            patch.set_clip_on(False)
            patch2.set_clip_on(False)
            ax.add_patch(patch)
            ax.add_patch(patch2)

        if self.plot_mode in ["default", "big"]:
            # inlets and outlets, each max 4
            inlet_num = len([ele for ele in inlets if ele != "latent"])
            outlet_num = len([ele for ele in outlets if ele != "global"])

            # inlets
            for i in range(inlet_num):
                inlet_patch = patches.Rectangle(
                    self.get_inlet_pos(x, y, i),
                    self.plot_config['node']['let_width'][self.plot_mode] - self.plot_config['node']['let_offset_x'][self.plot_mode] * 2,
                    self.plot_config['node']['let_height'][self.plot_mode],
                    facecolor=self.color_config["colors"]["lets"]["inlet_color"],
                    edgecolor=self.color_config["colors"]["lets"]["inlet_color"],
                    linewidth=self.plot_config['edge']['line_thickness'][self.plot_mode],
                    fill=True,
                    zorder=2,
                )
                inlet_patch.set_clip_on(False)
                ax.add_patch(inlet_patch)

            # outlets
            for i in range(outlet_num):
                outlet_patch = patches.Rectangle(
                    self.get_outlet_pos(x, y, i),
                    self.plot_config['node']['let_width'][self.plot_mode],
                    self.plot_config['node']['let_height'][self.plot_mode],
                    facecolor=self.color_config["colors"]["lets"]["outlet_color"],
                    edgecolor=self.color_config["colors"]["lets"]["outlet_color"],
                    linewidth=self.plot_config['edge']['line_thickness'][self.plot_mode],
                    fill=True,
                    zorder=2,
                )
                outlet_patch.set_clip_on(False)
                ax.add_patch(outlet_patch)
