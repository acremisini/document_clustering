from code.ECBWrapper import ECBWrapper
from code.SimilarityMetrics import  SimilarityMetrics
from code import Globals
from code.CorpusVectorizer import CorpusVectorizer
from code.K_Means import K_Means
from scipy import stats
from sklearn.cluster import AffinityPropagation, SpectralClustering, Birch
from sklearn import metrics
from time import time
import pandas as pd
from code.visualize_results import plot_clustering_results

#wrapper for the ecb+ dataset
ecb_wrapper = ECBWrapper(Globals.ECB_DIR, topics=None, lemmatize=True)
#class to calculate cluster stats
sim_metrics = SimilarityMetrics('data/ecb_term_set.txt')

#available text abstractions
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
do_pairwise = False

for do_subtopic in [True,False]:
    ####################################################################################
    ####### NOTE 1: if do_subtopic = True, num_topics = number of topics in ecb above*2
    #######       if False, num_topics = number of topics in ecb above
    #######       -----> if number of topics in ecb above = None, then num_topics = 86 if sub_topic, 43 o.w.
    #######       -----> running across the entire dataset takes quite a while. picking ~two random topics
    #######       -----> above is recommended to see code functioning. However, see note 2:
    ####### NOTE 2: if do_pairwise = False, only the clustering experiments are done (avoiding
    #######         n^2 performance for the pairwise homogeneity metrics), and this runs
    #######         in a reasonable (~ a few minutes) amount of time. This is the default
    #######         setting.
    ####################################################################################
    if do_subtopic:
        num_topics = 43
    else:
        num_topics = 86
    file_name = ''
    if do_subtopic:
        file_name = 'sub-topic'
    else:
        file_name = 'topic'

    #these are unaffected by filtering:
    #dataset
    event_template = ecb_wrapper.make_data_for_clustering(option=element_types[1], sub_topics=do_subtopic)
    if do_pairwise:
        # word count stats (for event)
        sim_metrics.get_word_count_stats(corpus=event_template, fname=file_name + '_event_word_counts.png',
                                         title='Summary Statistics of Word per Document,\nEvent Template Abstraction')

        # event abstraction, within and cross cluster
        if do_subtopic:
            sim_metrics.get_cluster_sim(ecb_wrapper, element_types[1], within=True, fname=file_name + '_sim_event_within.png',
                                        title='Within SubTopic Cosine Similarity,\nEvent Abstraction')
            sim_metrics.get_cluster_sim(ecb_wrapper, element_types[1], within=False, fname=file_name + '_sim_event_across.png',
                                        title='Cross SubTopic Cosine Similarity,\nEvent Abstraction')
        else:
            sim_metrics.get_cluster_sim(ecb_wrapper, element_types[1], within=True, fname=file_name + '_sim_event_within.png',
                                        title='Within Topic Cosine Similarity,\nEvent Abstraction')
            sim_metrics.get_cluster_sim(ecb_wrapper, element_types[1], within=False, fname=file_name + '_sim_event_across.png',
                                        title='Cross Topic Cosine Similarity,\nEvent Abstraction')

    for do_filter in [True,False]:
        file_name = ''
        if do_subtopic:
            file_name = 'sub-topic'
        else:
            file_name = 'topic'
        if do_filter:
            file_name += '_filter'
        else:
            file_name += '_no-filter'

        raw_text = ecb_wrapper.make_data_for_clustering(option=element_types[10], filter=do_filter,sub_topics=do_subtopic)

        '''
        Step 1: 
        
        Compute homogeneity metrics for the corpus using the different text abstractions
        
        '''
        if do_pairwise:
            #word count stats (for lemmatized text)
            if do_filter:
                sim_metrics.get_word_count_stats(corpus=raw_text,fname= file_name + '_text_word_counts.png',
                                                 title='Summary Statistics of Word per Document,\n Filtered Lemmatized Text Abstraction')
            else:
                sim_metrics.get_word_count_stats(corpus=raw_text, fname=file_name + '_text_word_counts.png',
                                                 title='Summary Statistics of Word per Document,\n Full Lemmatized Text Abstraction')

            #averaged homogeneity stats
            if do_subtopic:
                if do_filter:
                    # text abstraction, within and cross cluster
                    sim_metrics.get_cluster_sim(ecb_wrapper,element_types[10], within=True,fname='filtered_text_sims_within_ST.png',
                                                title='Within SubTopic Pairwise Cosine Similarity,\nFiltered Text Abstraction')
                    sim_metrics.get_cluster_sim(ecb_wrapper,element_types[10], within=False,fname='filtered_text_sims_across_ST.png',
                                                title='Cross SubTopic Pairwise Cosine Similarity,\nFiltered Text Abstraction')
                else:
                    # text abstraction, within and cross cluster
                    sim_metrics.get_cluster_sim(ecb_wrapper, element_types[10], within=True, fname='full_text_sims_within_ST.png',
                                                title='Within SubTopic Pairwise Cosine Similarity,\nFull Text Abstraction')
                    sim_metrics.get_cluster_sim(ecb_wrapper, element_types[10], within=False, fname='full_text_sims_across_ST.png',
                                                title='Cross SubTopic Pairwise Cosine Similarity,\nFull Text Abstraction')
            else:
                if do_filter:
                    # text abstraction, within and cross cluster
                    sim_metrics.get_cluster_sim(ecb_wrapper,element_types[10], within=True,fname='filtered_text_sims_within_T.png',
                                                title='Within Topic Pairwise Cosine Similarity,\nFiltered Text Abstraction')
                    sim_metrics.get_cluster_sim(ecb_wrapper,element_types[10], within=False,fname='filtered_text_sims_across_T.png',
                                                title='Cross Topic Pairwise Cosine Similarity,\nFiltered Text Abstraction')
                else:
                    # text abstraction, within and cross cluster
                    sim_metrics.get_cluster_sim(ecb_wrapper, element_types[10], within=True, fname='full_text_sims_within_T.png',
                                                title='Within Topic Pairwise Cosine Similarity,\nFull Text Abstraction')
                    sim_metrics.get_cluster_sim(ecb_wrapper, element_types[10], within=False, fname='full_text_sims_across_T.png',
                                                title='Cross Topic Pairwise Cosine Similarity,\nFull Text Abstraction')


        '''
        Step 2: 
        
        Make 2 representations of the ECB+ corpus by representing the documents using:
                    1. event template information
                    2. raw text 
        '''
        datasets = {'Event_Template': event_template,
                    'Raw_Text': raw_text
                    }
        '''
        Step 3: 
        
        Apply different clustering algorithms to each of the representations:
            1. K-Means
            2. Affinity Propagation
            3. Spectral Clustering
            4. Birch
        
        '''

        #do clustering
        ari_results = dict()
        algorithms = []

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
            ari = k_means.cluster(corpus,num_topics,dim_reducer,corpus.vectorizer,n_components='',use_hashing=True,minibatch=True,verbose=False)
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
            clustering_obj = SpectralClustering(n_clusters=num_topics,assign_labels="discretize",random_state=0).fit(corpus.X)
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
            clustering_obj = Birch(n_clusters=num_topics,threshold=0.25).fit(corpus.X)
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
                                             'ARI (Event-Template)'])
        for alg in ari_results.keys():
            row = pd.DataFrame(index={alg}, columns=['ARI (Raw-Text)',
                                                        'ARI (Event-Template)'])
            result_table = result_table.append(row)

        #record avg. ari results

        for alg in ari_results.keys():
            for representation in ari_results[alg]:
                row = []
                if 'Event' in representation:
                    result_table.at[alg,'ARI (Event-Template)'] = "%.2f" % round(stats.describe(ari_results[alg][representation]).mean,2)
                if 'Raw' in representation:
                    result_table.at[alg,'ARI (Raw-Text)'] = "%.2f" % round(stats.describe(ari_results[alg][representation]).mean,2)

        result_table.fillna(value=0.0,inplace=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(result_table)

        writer = pd.ExcelWriter('data/' + file_name + '.xlsx')
        result_table.to_excel(writer,'Sheet1')
        writer.save()

        '''
        Step 4:
        
        Visualize the clustering results
        '''
        if do_subtopic:
                plot_clustering_results('SubTopic', read_from='data/' + file_name + '.xlsx', fname='cluster_'+file_name,filter = do_filter)
        else:
            plot_clustering_results('Topic', read_from='data/' + file_name + '.xlsx',fname='cluster_'+file_name, filter=do_filter)



