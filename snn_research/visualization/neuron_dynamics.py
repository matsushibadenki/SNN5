# snn_research/visualization/neuron_dynamics.py
"""
Visualization tools for neuron dynamics.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class NeuronDynamicsRecorder:
    """ニューロンの状態を記録するクラス"""
    
    def __init__(self, max_timesteps: int = 100):
        self.max_timesteps = max_timesteps
        self.history: Dict[str, List[torch.Tensor]] = {
            'membrane': [],
            'threshold': [],
            'spikes': []
        }
    
    def record(self, membrane: torch.Tensor, threshold: Optional[torch.Tensor], spikes: torch.Tensor):
        """1タイムステップの状態を記録"""
        if len(self.history['membrane']) < self.max_timesteps:
            self.history['membrane'].append(membrane.detach().cpu())
            if threshold is not None:
                self.history['threshold'].append(threshold.detach().cpu())
            self.history['spikes'].append(spikes.detach().cpu())
    
    def clear(self):
        """記録をクリア"""
        for key in self.history:
            self.history[key].clear()


def plot_neuron_dynamics(
    history: Dict[str, List[torch.Tensor]], 
    neuron_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    膜電位とスパイクの時系列をプロット。
    """
    if not history['membrane']:
        raise ValueError("No data to plot")
    
    num_neurons_to_plot = min(10, history['membrane'][0].shape[-1])
    if neuron_indices is None:
        neuron_indices = list(range(num_neurons_to_plot))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    mem_values = torch.stack(history['membrane']).numpy()
    time_steps = mem_values.shape[0]
    
    for i in neuron_indices:
        axes[0].plot(mem_values[:, 0, i], label=f'Neuron {i}', alpha=0.7)
    
    if history['threshold']:
        threshold_values = torch.stack(history['threshold']).numpy()
        axes[0].plot(threshold_values[:, 0, neuron_indices[0]], 'k--', alpha=0.8, label='Threshold')
    else:
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='Base Threshold', alpha=0.5)
    
    axes[0].set_ylabel('Membrane Potential')
    axes[0].set_title('Neuron Membrane Dynamics')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    spike_values = torch.stack(history['spikes']).numpy()
    spike_times, spike_neurons = np.where(spike_values[:, 0, :] > 0.5)
    
    axes[1].scatter(spike_times, spike_neurons, s=5, c='black', marker='|')
    axes[1].set_ylabel('Neuron Index')
    axes[1].set_title('Spike Raster Plot')
    axes[1].set_ylim(-0.5, history['membrane'][0].shape[-1] - 0.5)
    axes[1].grid(True, alpha=0.3)
    
    spike_rate = spike_values[:, 0, :].mean(axis=1)
    axes[2].plot(spike_rate, color='blue', linewidth=2)
    axes[2].fill_between(range(time_steps), spike_rate, alpha=0.3)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Average Spike Rate')
    axes[2].set_title('Population Spike Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig