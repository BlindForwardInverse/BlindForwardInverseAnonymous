# from plot.plot_grafx import PlotGrafx
from networkx import topological_sort
from plot.plot_grafx import PlotGrafx
from render_grafx import RenderGrafx
from sampler.graph_sampler import GraphSampler
from sampler.simple_graphs import MonolithicGraph, SingleEffectGraph
import os; opj = os.path.join

from time import time
import numpy as np
import soundfile as sf
import jax
import jax.numpy as jnp
from pprint import pprint

if __name__ == "__main__":
    graph_plotter = PlotGrafx()
    renderer = RenderGrafx(mono_processing=True)
#     sampler = GraphSampler(verbose=False, randomize_params=True)
#     sampler = MonolithicGraph(randomize_params=True)
    sampler = SingleEffectGraph(randomize_params=False)
    save_dir = '/ssd3/doyo/plot_static_hard'
    audio_tar = '/ssd4/inverse_problem/speech/speech44k/daps_parsed/clean/m9_script5_clean_9_12.wav'
    audio, sr = sf.read(audio_tar)
    audio_len = len(audio) / sr
    print(audio_len)
    audio = jnp.array(audio)

    os.makedirs(save_dir, exist_ok=True)
    num_audio = 20
    pprint(sampler.get_supporting_afx())

    input_signal = {'speech': audio}
    sf.write(opj(save_dir, "input_dry.wav"), audio, sr)

    for i in range(num_audio):
        print(f"-----{i}-----")
        G = sampler(single_afx_name='add_noise', noise_mode='guitar', noise_intensity='hard')
        start_time = time()
        output = renderer(G, input_signal)
        output_time = time()
        
        print("Rendering time", round(output_time - start_time, 3))
        sf.write(opj(save_dir, f"wet_{i}.wav"), output, sr)

        graph_plotter.plot(
            G, plot_mode="default", name=opj(save_dir, f"grafx_plot_default_{i}.png")
        )
        graph_plotter.plot(
            G, plot_mode="big", name=opj(save_dir, f"grafx_plot_big_{i}.png")
        )
        graph_plotter.plot(
            G, plot_mode="mini", name=opj(save_dir, f"grafx_plot_mini_{i}.png")
        )
    # graph_plotter.plot_grafx(graph, "SingleInputGraph.png")
