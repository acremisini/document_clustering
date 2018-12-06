import pandas as pd
import matplotlib.pyplot as plt

BOLD = r"$\bf{"
END = " }$"


def plot_clustering_results(type,read_from,fname,filter):
    plt.figure()
    df = pd.read_excel(read_from)
    if filter:
        df = df.rename(columns={"ARI (Raw-Text)": "Filtered Lemmatized Text",
                              'ARI (Event-Template)' : 'Event Template'})
        pd.concat(
            [df[c] for c in df.columns],
            axis=1).sort_values(by='Event Template',ascending=False).plot.bar(rot=15,sort_columns=True)
    else:
        df = df.rename(columns={"ARI (Raw-Text)": "Full Lemmatized Text",
                                'ARI (Event-Template)': 'Event Template'})
        pd.concat(
            [df[c] for c in df.columns],
            axis=1).sort_values(by='Full Lemmatized Text', ascending=False).plot.bar(rot=15, sort_columns=True)

    plt.xlabel('Algorithm')
    plt.ylabel('ARI Score')
    plt.title('Adjusted Rand Index (ARI) Scores per Algorithm,\nGrouped by Text Abstraction:\n' + BOLD  + type + END + ' Clusters')
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(fname='output/' +fname + '.png')
    plt.show()
