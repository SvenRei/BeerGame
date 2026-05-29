import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data consolidated
data = {
    'Policy': ['Order-Up-To', 'Sterman Heuristic', 'MAPPO (Baseline)', 'MAPPO (Black Swan)'],
    'Mean Cost': [246908, 215806, 177019, 177088],
    'Std Dev': [45200, 38100, 15431, 14855]
}

df = pd.DataFrame(data)

def plot_performance_comparison(df):
    plt.figure(figsize=(10, 6))
    
    # 1. Map 'Policy' to 'hue' to fix the deprecation warning
    # 2. Use errorbar=None to disable Seaborn's automatic (and often wrong) 
    #    error calculation, then plot our own using plt.errorbar
    ax = sns.barplot(x='Policy', y='Mean Cost', data=df, hue='Policy', palette='viridis', legend=False)
    
    # Manually plot the error bars correctly for each bar
    # 'x' positions are 0, 1, 2, 3
    plt.errorbar(x=range(len(df)), y=df['Mean Cost'], yerr=df['Std Dev'], 
                 fmt='none', c='black', capsize=5, elinewidth=2)
    
    plt.title("Performance Benchmark: MAPPO vs. Classic Policies", fontsize=16)
    plt.ylabel("Total System Cost (Lower is Better)")
    plt.xlabel("Policy Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("performance_benchmark.png", dpi=300)
    print("Plot saved as performance_benchmark.png")
    plt.show()

plot_performance_comparison(df)