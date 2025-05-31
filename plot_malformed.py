import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Malformed counts for K-Shot
malformed_k_data = {
    'shots': ['0-shot', '1-shot', '3-shot', '5-shot'] * 6,
    'malformed': [
        0, 0, 0, 0,           # Latxa EU
        0, 0, 0, 1,           # Latxa EN
        510, 1, 1, 1,         # Llama EU
        165, 0, 1, 3,         # Llama EN
        102, 0, 2, 0,         # Qwen EU
        51, 0, 0, 0           # Qwen EN
    ],
    'config': (
        ['Latxa EU'] * 4 +
        ['Latxa EN'] * 4 +
        ['Llama EU'] * 4 +
        ['Llama EN'] * 4 +
        ['Qwen EU'] * 4 +
        ['Qwen EN'] * 4
    )
}

# Malformed counts for CoT
malformed_cot_data = {
    'shots': ['0-shot', '1-shot', '3-shot', '5-shot'] * 6,
    'malformed': [
        23, 0, 1, 1,          # Latxa EU
        10, 2, 9, 8,          # Latxa EN
        93, 13, 57, 66,       # Llama EU
        61, 1, 10, 5,         # Llama EN
        46, 7, 21, 28,        # Qwen EU
        11, 10, 14, 11        # Qwen EN
    ],
    'config': (
        ['Latxa EU'] * 4 +
        ['Latxa EN'] * 4 +
        ['Llama EU'] * 4 +
        ['Llama EN'] * 4 +
        ['Qwen EU'] * 4 +
        ['Qwen EN'] * 4
    )
}

# Convert to DataFrames
df_k_malformed = pd.DataFrame(malformed_k_data)
df_cot_malformed = pd.DataFrame(malformed_cot_data)

# Define consistent shot order
shot_order = ['0-shot', '1-shot', '3-shot', '5-shot']

# Apply categorical ordering
for df in (df_k_malformed, df_cot_malformed):
    df['shots'] = pd.Categorical(df['shots'], categories=shot_order, ordered=True)

# Create colorblind-friendly palette using seaborn
# Get the colorblind palette colors
colorblind_colors = sns.color_palette("colorblind")

# Map model base names to colors
model_base_colors = {
    'Latxa': colorblind_colors[0],  # Blue
    'Llama': colorblind_colors[1],  # Orange  
    'Qwen': colorblind_colors[2],   # Green
}

# Create color palette mapping each config to its model's color
color_palette = {}
for config in ['Latxa EU', 'Latxa EN', 'Llama EU', 'Llama EN', 'Qwen EU', 'Qwen EN']:
    model_base = config.split()[0]  # Extract 'Latxa', 'Llama', or 'Qwen'
    color_palette[config] = model_base_colors[model_base]

# Helper function to plot grouped bar chart
def plot_grouped_bars(df, filename, title):
    configs = df['config'].unique()
    n_models = len(configs)
    n_shots = len(shot_order)
    bar_width = 0.12
    group_gap = 0.3  # space between each shot group
    indices = np.arange(n_shots)
    
    # Set up figure
    plt.figure(figsize=(7, 4))
    
    # Set matplotlib parameters for thinner hatch lines
    plt.rcParams['hatch.linewidth'] = 0.4
    
    # Plot each model
    for i, config in enumerate(configs):
        subset = df[df['config'] == config].sort_values('shots')
        # Compute position offset
        offset = (i - n_models / 2) * bar_width + bar_width / 2
        
        # Add diagonal hatching for EN variants
        hatch_pattern = '/////' if 'EN' in config else None
        
        plt.bar(indices + offset, subset['malformed'], width=bar_width, 
                label=config, color=color_palette[config], hatch=hatch_pattern,
                edgecolor='black', linewidth=0.5)
    
    # Configure axes and labels
    plt.xticks(indices, shot_order)
    plt.xlabel('Number of Shots (k)')
    plt.ylabel('Number of Malformed Outputs')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    # plt.legend(title='Model + Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(title='Model + Language', loc='upper right', ncol=3, 
               columnspacing=1.0, handletextpad=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.close()

# Plot both K-Shot and CoT malformed output counts
plot_grouped_bars(df_k_malformed, 'images/malformed_k_shot.png', 'Baseline Malformed Outputs')
plot_grouped_bars(df_cot_malformed, 'images/malformed_cot.png', 'Chain-of-Thought Malformed Outputs')
