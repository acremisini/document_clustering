from ECBWrapper import ECBWrapper
from SimilarityMetrics import SimilarityMetrics
import Globals
from CorpusVectorizer import CorpusVectorizer
from DimensionalityReducer import DimensionalityReducer
from K_Means import K_Means
from scipy import stats
import math
import sys

element_types = {0 : 'event_trigger',
                 1 : 'doc_template',
                 2 : 'event_participant',
                 3 : 'event_participant_location',
                 4 : 'participant',
                 5 : 'time',
                 6 : 'location',
                 7 : 'doc_template',
                 8 : 'event_hum_participant',
                 9 : 'hum_participant',
                 10 : 'event_hum_participant_location',
                 11 : 'text'}
element_choice = 7


#ecb wrapper class
ecb_wrapper = ECBWrapper(Globals.ECB_DIR, topics=None,lemmatize=True)

'''
Step 1: 

Make 3 representations of the ECB+ corpus, by representing the documents as:
            1. event template
            2. event 
            3. raw text
'''
template_dataset = ecb_wrapper.make_data_for_clustering(option=element_types[1], sub_topics=False)
text_dataset = ecb_wrapper.make_data_for_clustering(option=element_types[1], sub_topics=False)
event_dataset = ecb_wrapper.make_data_for_clustering(option=element_types[0], sub_topics=False)

datasets = {'template': template_dataset,
            'event': event_dataset,
            'text': text_dataset}
'''
Step 2: 

Apply different clustering algorithms to each of the representations using the following algorithms:
    1. K-Means
    2. 

'''
results = {'k-means' : {'template' : [],
                        'event' : [],
                        'text' : []}
           }
#do clustering
print('K-means:')
repetitions = 1000
while repetitions > 0:
    for name,data in datasets.items():

        corpus = CorpusVectorizer(data)
        corpus.vectorize(data, use_idf = False, n_features = 10000)
        # dim_reducer = DimensionalityReducer()
        # dim_reducer.lsa_reduce(corpus.X,n_components=10)
        dim_reducer = None

        k_means = K_Means()
        results['k-means'][name].append(k_means.cluster(corpus,43,dim_reducer,corpus.vectorizer,n_components='',use_hashing=True,minibatch=True,verbose=False))
        repetitions -= 1
for name in results['k-means'].keys():
    s = stats.describe(results['k-means'][name])
    print(name + '-based clustering, avg. adjusted rand-index: ' + str(s.mean) +  ', ' + 'std_dev: ' + str(math.sqrt(s.variance)))

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




