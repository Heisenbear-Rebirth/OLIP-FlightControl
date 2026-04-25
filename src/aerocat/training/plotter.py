import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

class MetricsPlotter:
    """
    Handles logging of training metrics and generating plots using Matplotlib.
    Supports saving and loading history for training resumption.
    Ported from AeroCat v17.2.
    """
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.history: Dict[str, List[float]] = {}
        os.makedirs(save_dir, exist_ok=True)
        
    def update(self, metrics: Dict[str, float], step: int = None):
        """
        Update history with new metrics.
        metrics: dict of {metric_name: value}
        step: global step (optional, used for x-axis)
        """
        for key, value in metrics.items():
            if key not in self.history:
                # New structure: dict with steps and values
                self.history[key] = {'steps': [], 'values': []}
            
            # Handle legacy format migration (list -> dict) on the fly if needed
            if isinstance(self.history[key], list):
                # Migrate to new format assuming continuous steps 0, 1, 2...
                self.history[key] = {
                    'steps': list(range(len(self.history[key]))),
                    'values': self.history[key]
                }

            # Handle JAX arrays or numpy scalars
            if hasattr(value, 'item'):
                val = value.item()
            else:
                val = float(value)
            
            # Determine step
            if step is None:
                # Auto-increment based on last step
                if self.history[key]['steps']:
                    s = self.history[key]['steps'][-1] + 1
                else:
                    s = 0
            else:
                s = step
                
            self.history[key]['steps'].append(s)
            self.history[key]['values'].append(val)

    def truncate(self, after_step: int):
        """
        Truncate history to remove any steps >= after_step.
        Useful when resuming from an earlier checkpoint to avoid overlapping plot lines.
        """
        truncated_count = 0
        for key in self.history:
            steps = self.history[key]['steps']
            values = self.history[key]['values']
            
            # Filter
            new_steps = []
            new_values = []
            for s, v in zip(steps, values):
                if s <= after_step:
                    new_steps.append(s)
                    new_values.append(v)
            
            diff = len(steps) - len(new_steps)
            if diff > 0:
                truncated_count += diff
                self.history[key]['steps'] = new_steps
                self.history[key]['values'] = new_values
                
        if truncated_count > 0:
            print(f"[*] Metrics: Truncated {truncated_count} future points (Step >= {after_step})")
            
    def plot(self, filename: str = "training_curves.png"):
        """
        Generate and save training curves.
        """
        if not self.history:
            return

        # Filter and migrate keys
        keys = []
        for k in self.history.keys():
            # Migrate legacy if encounterd
            if isinstance(self.history[k], list):
                self.history[k] = {
                    'steps': list(range(len(self.history[k]))),
                    'values': self.history[k]
                }
            if len(self.history[k]['values']) > 0:
                keys.append(k)
        
        keys.sort()
        num_metrics = len(keys)
        if num_metrics == 0:
            return

        # Create subplots
        cols = 2
        rows = (num_metrics + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, key in enumerate(keys):
            ax = axes[i]
            steps = np.array(self.history[key]['steps'])
            values = np.array(self.history[key]['values'])
            
            # Detect gaps to insert NaNs for plotting (creating blank spaces)
            if len(steps) > 1:
                # Calculate median interval
                intervals = np.diff(steps)
                median_interval = np.median(intervals)
                if median_interval == 0: median_interval = 1
                
                # Identify gaps: > 3x median
                gap_mask = intervals > (3 * median_interval)
                
                if np.any(gap_mask):
                    # Insert NaNs at gaps
                    plot_steps = []
                    plot_values = []
                    for j in range(len(steps)):
                        plot_steps.append(steps[j])
                        plot_values.append(values[j])
                        if j < len(steps) - 1 and gap_mask[j]:
                            plot_steps.append((steps[j] + steps[j+1]) / 2)
                            plot_values.append(np.nan)
                    steps_plot = np.array(plot_steps)
                    values_plot = np.array(plot_values)
                else:
                    steps_plot = steps
                    values_plot = values
            else:
                steps_plot = steps
                values_plot = values
            
            # Raw data
            ax.plot(steps_plot, values_plot, alpha=0.3, label='Raw')
            
            # Moving average (using valid data only)
            if len(values) > 5:
                window = min(len(values)//10 + 2, 50)
                if window > 1:
                    avg_values = np.convolve(values, np.ones(window)/window, mode='valid')
                    # Map convolution back to steps (centered or trailing)
                    # We use trailing (steps[window-1:])
                    valid_steps = steps[window-1:]
                    
                    # Also need to break trend line at gaps
                    # Simplify: just plot continuous trend for now, or apply same gap logic?
                    # Applying gap logic to trend is complex because convolution spans gaps.
                    # Just plot trend, it often bridges gaps which is fine for "Trend".
                    # BUT user asked for "blank".
                    # Let's trust "Raw" shows the blank clearly.
                    
                    ax.plot(valid_steps, avg_values, 'r-', linewidth=1.5, label='Trend')
            
            ax.set_title(key)
            ax.grid(True, alpha=0.3)
            if len(values) > 5:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(handles, labels, fontsize='small')

        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        try:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=100)
        except Exception as e:
            print(f"[!] Plot: Error saving: {e}")
        finally:
            plt.close(fig)

    def save(self, filepath: str = "metrics.json"):
        """Save history to file (JSON)"""
        import json
        save_path = os.path.join(self.save_dir, filepath) if not os.path.isabs(filepath) else filepath
            
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def load(self, filepath: str = "metrics.json"):
        """Load history from file (JSON)"""
        import json
        load_path = os.path.join(self.save_dir, filepath) if not os.path.isabs(filepath) else filepath
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'r') as f:
                    self.history = json.load(f)
                
                # Check migration
                for k, v in self.history.items():
                    if isinstance(v, list):
                         self.history[k] = {
                            'steps': list(range(len(v))),
                            'values': v
                        }
                        
                print(f"[+] Metrics: Loaded history from {load_path}")
            except Exception as e:
                print(f"[-] Metrics: Failed to load: {e}")
        else:
            pass
