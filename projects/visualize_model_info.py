import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#4682B4',
    'dark': '#002147',
    'gray': '#708090',
    'red': '#C83232',
    'green': '#288C50',
    'amber': '#C88200',
    'purple': '#7A39BB'
}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_dataset_split(info, out_path):
    training = info['training']
    labels = ['Train', 'Test']
    values = [training['train_rows'], training['test_rows']]
    colors = [COLORS['blue'], COLORS['amber']]

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, _texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f'{p:.1f}%\n({int(round(p/100*sum(values))):,})',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_weight('bold')
    ax.set_title(f"Dataset Split ({training['total_rows']:,} total rows)")
    save_fig(fig, out_path)


def fig_insdn_metrics(info, out_path):
    models = info['all_models']
    names = list(models.keys())
    f1 = [models[n]['F1'] for n in names]
    acc = [models[n]['Accuracy'] for n in names]
    auc = [models[n]['AUC'] for n in names]

    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width, f1, width, label='F1', color=COLORS['blue'])
    b2 = ax.bar(x, acc, width, label='Accuracy', color=COLORS['green'])
    b3 = ax.bar(x + width, auc, width, label='AUC', color=COLORS['amber'])

    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel('Score')
    ax.set_title('InSDN Model Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha='right')
    ax.legend(ncols=3, frameon=False)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    save_fig(fig, out_path)


def fig_latency_composite(info, out_path):
    models = info['all_models']
    names = list(models.keys())
    latency = np.array([models[n]['lat_us'] for n in names], dtype=float)
    composite = np.array([models[n]['composite'] for n in names], dtype=float)

    latency_cost = np.log10(latency + 0.01)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = [COLORS['blue'], COLORS['green'], COLORS['amber'], COLORS['gray']]
    for i, name in enumerate(names):
        ax.scatter(latency_cost[i], composite[i], s=140, color=palette[i], edgecolor='black', linewidth=0.5)
        ax.text(latency_cost[i] + 0.015, composite[i] + 0.002, name, fontsize=9)

    ax.set_xlabel('Latency cost (log10 microseconds)')
    ax.set_ylabel('Composite score')
    ax.set_title('Latency vs Composite Score Tradeoff')
    ax.set_ylim(min(composite) - 0.02, max(composite) + 0.03)
    save_fig(fig, out_path)


def fig_mininet_metrics(info, out_path):
    models = info['mininet_models']
    names = list(models.keys())
    f1 = [models[n]['F1'] for n in names]
    acc = [models[n]['Accuracy'] for n in names]
    auc = [models[n]['AUC'] for n in names]

    x = np.arange(len(names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    b1 = ax.bar(x - width, f1, width, label='F1', color=COLORS['blue'])
    b2 = ax.bar(x, acc, width, label='Accuracy', color=COLORS['green'])
    b3 = ax.bar(x + width, auc, width, label='AUC', color=COLORS['amber'])

    ax.set_ylim(0.9985, 1.0005)
    ax.set_ylabel('Score')
    ax.set_title('Mininet-Calibrated Model Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(ncols=3, frameon=False)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.00003, f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    save_fig(fig, out_path)


def fig_feature_overview(info, out_path):
    features = info['feature_names']
    y = np.arange(len(features))
    vals = np.linspace(1, len(features), len(features))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(y, vals, color=COLORS['blue'])
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature index for controller pipeline')
    ax.set_title('OpenFlow-Compatible Features Used by the Model')
    ax.set_xlim(0, len(features) + 1)
    ax.set_xticks([])

    for i, bar in enumerate(bars, start=1):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'F{i}', va='center', fontsize=8)
    save_fig(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description='Generate presentation-ready charts from model_info.json')
    parser.add_argument('--json', default='model_info.json', help='Path to model_info.json')
    parser.add_argument('--out-dir', default='.', help='Directory for PNG output')
    args = parser.parse_args()

    info = load_json(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dataset_split(info, out_dir / 'fig_modelinfo_dataset_split.png')
    fig_insdn_metrics(info, out_dir / 'fig_modelinfo_insdn_metrics.png')
    fig_latency_composite(info, out_dir / 'fig_modelinfo_latency_composite.png')
    fig_mininet_metrics(info, out_dir / 'fig_modelinfo_mininet_metrics.png')
    fig_feature_overview(info, out_dir / 'fig_modelinfo_features.png')

    summary = {
        'winner': info.get('winner'),
        'winner_file': info.get('winner_file'),
        'decision_threshold': info.get('decision_threshold'),
        'generated_files': [
            'fig_modelinfo_dataset_split.png',
            'fig_modelinfo_insdn_metrics.png',
            'fig_modelinfo_latency_composite.png',
            'fig_modelinfo_mininet_metrics.png',
            'fig_modelinfo_features.png'
        ]
    }
    with open(out_dir / 'modelinfo_figures_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()