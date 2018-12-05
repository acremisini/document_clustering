
class Clustering_Utility():

    def unique_pairwise_no_diagonal(self, files):
        pairs = set()
        for f1 in files:
            for f2 in files:
                if f1 != f2:
                    pairs.add(frozenset((f1,f2)))
        return pairs