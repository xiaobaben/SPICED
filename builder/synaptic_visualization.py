import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_synaptic_net(nodes, args):
    prototypes = np.array([n.prototype for n in nodes.values()])
    connected_counts = np.array([n.connection_count() for n in nodes.values()])

    plt.figure(figsize=(24, 12))
    ax = plt.gca()

    tsne = TSNE(n_components=2, perplexity=15)
    X_tsne = tsne.fit_transform(prototypes)
    min_size = 40
    max_size = 400
    sizes = min_size + (max_size - min_size) * (
            connected_counts - np.min(connected_counts)) / (
                    np.max(connected_counts) - np.min(connected_counts))

    main_scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        s=sizes,
        c='steelblue',
        edgecolors='white',
        linewidths=1,
        zorder=2
    )

    line_style = {
        'linestyle': '--',
        'linewidth': 0.6,
        'alpha': 0.3,
        'color': 'steelblue'
    }

    for i, (name, node) in enumerate(nodes.items()):
        for target, (_, _) in node.connections.items():
            j = list(nodes.keys()).index(target.name)
            ax.plot([X_tsne[i, 0], X_tsne[j, 0]],
                    [X_tsne[i, 1], X_tsne[j, 1]],
                    **line_style,
                    zorder=1)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Subject Similarity Network Visualization\n'
                 '(Point Size Indicates Connection Degree)',
                 fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()

