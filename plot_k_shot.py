import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for K-Shot evaluations
k_shot_data = {
    'shots': ['0-shot', '1-shot', '3-shot', '5-shot'] * 6,
    'accuracy': [
        0.5158, 0.6672, 0.6743, 0.6547,  # Latxa EU
        0.6275, 0.7789, 0.7663, 0.7609,  # Latxa EN
        0.3834, 0.5550, 0.5566, 0.5599,  # Llama EU
        0.6770, 0.8088, 0.8192, 0.8224,  # Llama EN
        0.5256, 0.5692, 0.5812, 0.5741,  # Qwen EU
        0.7838, 0.8099, 0.8393, 0.8551,  # Qwen EN
        
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

# Data for CoT evaluations
cot_data = {
    'shots': ['0-shot', '1-shot', '3-shot', '5-shot'] * 6,
    'accuracy': [
        0.6492, 0.6694, 0.6705, 0.6585,  # Latxa EU
        0.8077, 0.8001, 0.8034, 0.8312,  # Latxa EN
        0.5245, 0.5735, 0.5539, 0.5430,  # Llama EU
        0.7990, 0.8268, 0.8306, 0.8306,  # Llama EN
        0.6187, 0.6471, 0.6449, 0.6324,  # Qwen EU
        0.8981, 0.8905, 0.8932, 0.8943,  # Qwen EN
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
df_k = pd.DataFrame(k_shot_data)
df_cot = pd.DataFrame(cot_data)

# Ensure even spacing by treating shots as ordered categorical
shot_order = ['0-shot', '1-shot', '3-shot', '5-shot']
for df in (df_k, df_cot):
    df['shots'] = pd.Categorical(df['shots'], categories=shot_order, ordered=True)

# Determine common y-axis limits
all_accuracies = pd.concat([df_k['accuracy'], df_cot['accuracy']])
y_min = all_accuracies.min() - 0.05
y_max = all_accuracies.max() + 0.05

# Function to plot using seaborn and save with shared y-axis settings
def plot_and_save(df, filename, title):
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=df,
        x='shots',
        y='accuracy',
        hue='config',
        marker='o',
        palette='colorblind',
        linewidth=2.0,
    )
    plt.ylim(y_min, y_max)
    plt.xlabel('Number of Shots (k)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.minorticks_on()
    plt.grid(axis='y', which='major', color='grey', linestyle='--', linewidth=0.5)
    plt.legend(title='Model + Language', loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.close()

# Helper to filter and rename plots
def filter_and_plot(df, lang, eval_type):
    df_filtered = df[df['config'].str.endswith(lang)]
    filename = f'images/{eval_type.lower()}_{lang.lower()}_accuracy.png'
    if lang == 'EU':
        dataname='Basque PIQA'
    else:
        dataname='English PIQA'
    title = f'{eval_type} Accuracy for {dataname} Dataset'
    plot_and_save(df_filtered, filename, title)

# Generate 4 separate plots
for eval_df, eval_name in [(df_k, 'Baseline'), (df_cot, 'Chain-of-Thought')]:
    for lang in ['EU', 'EN']:
        filter_and_plot(eval_df, lang, eval_name)
