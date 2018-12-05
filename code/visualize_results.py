import pandas as pd
import matplotlib.pyplot as plt

BOLD = r"$\bf{"
END = " }$"


def plot_clustering_results(type):
    df = pd.read_excel('data/clustering_results_' + type + '.xlsx')
    df = df.rename(columns={"ARI (Raw-Text)": "Raw Text",
                          'ARI (Event-Template)' : 'Event Template'})

    pd.concat(
        [df[c] for c in df.columns],
        axis=1).sort_values(by='Raw Text',ascending=False).plot.bar(rot=15,sort_columns=True)
    plt.xlabel('Algorithm')
    plt.ylabel('ARI Score')
    plt.title('Adjusted Rand Index (ARI) Scores per Algorithm,\nGrouped by Text Abstraction:\n' + BOLD  + type + END + ' Clusters')
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(fname='output/' +type + '.png')
    plt.show()

plot_clustering_results('Topic')
plot_clustering_results('SubTopic')