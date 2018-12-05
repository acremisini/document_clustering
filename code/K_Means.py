from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from time import time


class K_Means():


    def cluster(self,dataset,true_k,svd,vectorizer,n_components='',use_hashing=False,minibatch=False,verbose = False):
        if minibatch:
            km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=verbose)
        else:
            km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                        verbose=verbose)

        # print("Clustering sparse data with %s" % km)
        t0 = time()
        km.fit(dataset.X)
        # print("done in %0.3fs" % (time() - t0))

        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(dataset.Y, km.labels_))
        # print("Completeness: %0.3f" % metrics.completeness_score(dataset.Y, km.labels_))
        # print("V-measure: %0.3f" % metrics.v_measure_score(dataset.Y, km.labels_))
        # print("Adjusted Rand-Index: %.3f"
        #       % metrics.adjusted_rand_score(dataset.Y, km.labels_))
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(dataset.X, km.labels_, sample_size=1000))


        if not use_hashing:
            print("Top terms per cluster:")

            if n_components is not '':
                original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()
            for i in range(true_k):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind], end='')
                print()

        return metrics.adjusted_rand_score(dataset.Y, km.labels_)