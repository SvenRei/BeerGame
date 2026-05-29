import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_policy_comparison(results_dict, output_path="comparison_plot.png"):
    """
    results_dict: {'Sterman': [costs...], 'MAPPO': [costs...]}
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Flatten data for seaborn
    data = []
    for policy, costs in results_dict.items():
        for cost in costs:
            data.append({'Policy': policy, 'Total Cost': cost})
            
    # Create Violin Plot (shows distribution density + median)
    sns.violinplot(data=data, x='Policy', y='Total Cost', palette="muted", inner="quartile")
    
    plt.title("Supply Chain Policy Performance Comparison", fontsize=15)
    plt.ylabel("Total System Cost", fontsize=12)
    plt.xlabel("Policy Strategy", fontsize=12)
    
    # Save the figure for your paper
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")
    plt.show()

def plot_convergence(episode_costs, window=100, output_path="convergence_plot.png"):
    """
    episode_costs: List of costs recorded every episode during training.
    window: Number of episodes for the moving average (smoothing).
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Calculate moving average to smooth out the noise
    moving_avg = np.convolve(episode_costs, np.ones(window)/window, mode='valid')
    
    plt.plot(moving_avg, label='Moving Average (100 eps)', color='royalblue', linewidth=2)
    # Plot raw cost with high transparency
    plt.plot(episode_costs, alpha=0.1, color='gray', label='Raw Episode Cost')
    
    plt.title("MAPPO Learning Curve: System Cost Reduction", fontsize=15)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Total System Cost", fontsize=12)
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to {output_path}")
    plt.show()


    import matplotlib.pyplot as plt
import numpy as np

def plot_spider_chart(metrics_dict, labels):
    """
    metrics_dict: {'MAPPO': [0.9, 0.8, 0.95, 0.85], 'Sterman': [0.5, 0.4, 0.3, 0.6]}
    labels: ['Baseline Perf', 'Jitter Robustness', 'Black Swan Adaptation', 'Bullwhip Stability']
    """
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circle

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for policy, scores in metrics_dict.items():
        scores += scores[:1]
        ax.plot(angles, scores, label=policy, linewidth=2)
        ax.fill(angles, scores, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Supply Chain Robustness Profile")
    plt.show()

# Example Usage logic to be added to scripts/evaluate.py
# plot_policy_comparison({'Sterman': sterman_costs, 'MAPPO': mappo_costs})