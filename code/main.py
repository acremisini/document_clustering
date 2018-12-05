from code.ECBWrapper import ECBWrapper
from code import Globals
from code.CorpusVectorizer import CorpusVectorizer
from code.K_Means import K_Means
from scipy import stats
from sklearn.cluster import AffinityPropagation, SpectralClustering, Birch
from sklearn import metrics
import sys
from time import time
import pandas as pd

element_types = {0 : 'event_trigger',
                 1 : 'doc_template',
                 2 : 'event_participant',
                 3 : 'event_participant_location',
                 4 : 'participant',
                 5 : 'time',
                 6 : 'location',
                 7 : 'event_hum_participant',
                 8 : 'hum_participant',
                 9 : 'event_hum_participant_location',
                 10 : 'text'}

#wrapper for the ecb+ dataset
ecb_wrapper = ECBWrapper(Globals.ECB_DIR, topics=None, lemmatize=True)

'''
Step 1: 

Make 2 representations of the ECB+ corpus by representing the documents using:
            1. event template information
            2. raw text 
'''
datasets = {'Event_Template': ecb_wrapper.make_data_for_clustering(option=element_types[1], sub_topics=False),
            'Raw_Text': ecb_wrapper.make_data_for_clustering(option=element_types[10], sub_topics=False)
            }
'''
Step 2: 

Apply different clustering algorithms to each of the representations using the following algorithms:
    1. K-Means
    2. Affinity Propagation
    3. Mean Shift
    4. Spectral Clustering

'''

#do clustering
ari_results = dict()
n_clusters = dict()
algorithms = []
repetitions = 1
while repetitions > 0:
    repetitions = repetitions - 1
    for representation, data in datasets.items():
        '''
        Get corpus ready for scikit
        '''
        corpus = CorpusVectorizer(data)
        corpus.vectorize(data, use_idf = False, n_features = 10000)
        # dim_reducer = DimensionalityReducer()
        # dim_reducer.lsa_reduce(corpus.X,n_components=10)
        dim_reducer = None
        #########
        '''
        Do Clusterings:
        '''
        ##########
        '''
        1. K-means
        '''
        t0 = time()
        #### cluster
        k_means = K_Means()
        ari = k_means.cluster(corpus,43,dim_reducer,corpus.vectorizer,n_components='',use_hashing=True,minibatch=True,verbose=False)
        #### record results
        # ari
        algorithm = 'K-Means'
        algorithms.append(algorithm)
        print(algorithm + ' done in %0.3fs' %  (time() - t0) )
        if algorithm not in ari_results:
            ari_results[algorithm] = dict()
        if representation not in ari_results[algorithm]:
            ari_results[algorithm][representation] = []
        ari_results[algorithm][representation].append(ari)



        '''
        2. Affinity Propagation
        '''
        t0 = time()
        #### cluster
        clustering_obj = AffinityPropagation(damping=.95,convergence_iter=25).fit(corpus.X)
        ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        #### record results
        algorithm = 'Affinity-Propagation'
        algorithms.append(algorithm)
        print(algorithm + ' done in %0.3fs' % (time() - t0))
        # ari
        if algorithm not in ari_results:
            ari_results[algorithm] = dict()
        if representation not in ari_results[algorithm]:
            ari_results[algorithm][representation] = []
        ari_results[algorithm][representation].append(ari)


        # n_cluster error
        if algorithm not in n_clusters:
            n_clusters[algorithm] = dict()
        if representation not in n_clusters[algorithm]:
            n_clusters[algorithm][representation] = []
        n_clusters[algorithm][representation].append(len(set(clustering_obj.labels_)) - len(set(corpus.Y)))
        '''
        3. Mean Shift
        '''
        # t0 = time()
        # #### cluster
        # clustering_obj = MeanShift(bandwidth=2).fit(corpus.X.toarray())
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # #### record results
        # algorithm = 'Mean-Shift'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)
        # # n_cluster error
        # if algorithm not in n_clusters:
        #     n_clusters[algorithm] = dict()
        # if representation not in n_clusters[algorithm]:
        #     n_clusters[algorithm][representation] = []
        # n_clusters[algorithm][representation].append(len(set(corpus.Y)) - len(set(clustering_obj.labels_)))

        '''
        4. Spectral Clustering
        '''
        t0 = time()
        #### cluster
        clustering_obj = SpectralClustering(n_clusters=43,assign_labels="discretize",random_state=0).fit(corpus.X)
        ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        #### record results
        algorithm = 'Spectral-Clustering'
        algorithms.append(algorithm)
        print(algorithm + ' done in %0.3fs' % (time() - t0))
        # ari
        if algorithm not in ari_results:
            ari_results[algorithm] = dict()
        if representation not in ari_results[algorithm]:
            ari_results[algorithm][representation] = []
        ari_results[algorithm][representation].append(ari)

        '''
        5. Agglomerative Clustering (Ward Linkage)

        '''
        # t0 = time()
        # ### cluster
        # clustering_obj = AgglomerativeClustering(linkage='ward').fit(corpus.X.toarray())
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # ### record results
        # algorithm = 'Agglomerative-Clustering_Ward-Linkage'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)

        '''
        6. Agglomerative Clustering (Complete Linkage)

        '''
        # t0 = time()
        # ### cluster
        # clustering_obj = AgglomerativeClustering(linkage='ward').fit(corpus.X.toarray())
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # ### record results
        # algorithm = 'Agglomerative-Clustering_Complete-Linkage'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)

        '''
        7. Agglomerative Clustering (Average Linkage)
        '''
        # t0 = time()
        # ### cluster
        # clustering_obj = AgglomerativeClustering(linkage='ward').fit(corpus.X.toarray())
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # ### record results
        # algorithm = 'Agglomerative-Clustering_Average-Linkage'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)


        '''
        8. Agglomerative Clustering (Single Linkage)
        '''
        # t0 = time()
        # ### cluster
        # clustering_obj = AgglomerativeClustering(linkage='ward').fit(corpus.X.toarray())
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # ### record results
        # algorithm = 'Agglomerative-Clustering_Single-Linkage'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)


        '''
        9. DBSCAN
        '''
        # t0 = time()
        # ### cluster
        # clustering_obj = DBSCAN(eps=5, min_samples=10).fit(corpus.X)
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # #### record results
        # algorithm = 'DBSCAN'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)
        # # n_cluster error
        # if algorithm not in n_clusters:
        #     n_clusters[algorithm] = dict()
        # if representation not in n_clusters[algorithm]:
        #     n_clusters[algorithm][representation] = []
        # n_clusters[algorithm][representation].append(len(set(clustering_obj.labels_)) - len(set(corpus.Y)))

        '''
        10. Gaussian Mixture
        '''
        # t0 = time()
        # ### cluster
        # clustering_obj = GaussianMixture(n_components=43).fit(corpus.X.toarray())
        # ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        # #### record results
        # algorithm = 'Gaussian-Mixture'
        # algorithms.append(algorithm)
        # print(algorithm + ' done in %0.3fs' % (time() - t0))
        # # ari
        # if algorithm not in ari_results:
        #     ari_results[algorithm] = dict()
        # if representation not in ari_results[algorithm]:
        #     ari_results[algorithm][representation] = []
        # ari_results[algorithm][representation].append(ari)

        '''
        11. Birch
        '''
        t0 = time()
        ### cluster
        clustering_obj = Birch(n_clusters=43).fit(corpus.X)
        ari = metrics.adjusted_rand_score(corpus.Y, clustering_obj.labels_)
        #### record results
        algorithm = 'Birch'
        algorithms.append(algorithm)
        print(algorithm + ' done in %0.3fs' % (time() - t0))
        # ari
        if algorithm not in ari_results:
            ari_results[algorithm] = dict()
        if representation not in ari_results[algorithm]:
            ari_results[algorithm][representation] = []
        ari_results[algorithm][representation].append(ari)

result_table = pd.DataFrame(columns=['ARI (Raw-Text)',
                                     'ARI (Event-Template)',
                                     'Number of Clusters Error (Raw-Text)',
                                     'Number of Clusters Error (Event-Template)'])
for alg in ari_results.keys():
    row = pd.DataFrame(index={alg}, columns=['ARI (Raw-Text)',
                                                'ARI (Event-Template)',
                                                'Number of Clusters Error (Raw-Text)',
                                                'Number of Clusters Error (Event-Template)'])
    result_table = result_table.append(row)

#record avg. ari results

for alg in ari_results.keys():
    for representation in ari_results[alg]:
        row = []
        if 'Event' in representation:
            result_table.at[alg,'ARI (Event-Template)'] = "%.2f" % round(stats.describe(ari_results[alg][representation]).mean,2)
        if 'Raw' in representation:
            result_table.at[alg,'ARI (Raw-Text)'] = "%.2f" % round(stats.describe(ari_results[alg][representation]).mean,2)

#print avg. cluster error
for alg in n_clusters.keys():
    for representation in n_clusters[alg]:
        if 'Event' in representation:
            result_table.loc[alg,'Number of Clusters Error (Event-Template)'] = "%.2f" % round(stats.describe(n_clusters[alg][representation]).mean,2)
        if 'Raw' in representation:
            result_table.loc[alg,'Number of Clusters Error (Raw-Text)'] = "%.2f" % round(stats.describe(n_clusters[alg][representation]).mean,2)
result_table.fillna(value=0.0,inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result_table)

writer = pd.ExcelWriter('data/clustering_results_SubTopic.xlsx')
result_table.to_excel(writer,'Sheet1')
writer.save()

'''
Step 3:

Visualize the clustering results
'''

sys.exit(0)


'''
Step 4: 

Compute homogeneity metrics for the corpus using the different representations used above
for clustering

'''
#used to represent docs as vectors
# ecb_wrapper.compute_doc_template_set()

#descriptive statistics for the corpus
sim_metrics = SimilarityMetrics('data/ecb_doc_template_set.txt')
#if you want to use tfidf representations of documents
sim_metrics.make_tfidf_matrix(ecb_wrapper, element_type=element_types[element_choice])
#writes an excel file with avg. KL-divergence and Cosine Similarity per subtopic and topic
sim_metrics.compute_ecb_clustering_stats(ecb=ecb_wrapper, outfile_name=element_types[element_choice] + 'no_tfidf',element_type=element_types[element_choice],tfidf=False)




