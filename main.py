from ECBWrapper import ECBWrapper
from SimilarityMetrics import SimilarityMetrics
import Globals
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
                 10 : 'event_hum_participant_location'}
element_choice = 7

#ecb wrapper class
ecb_wrapper = ECBWrapper(Globals.ECB_DIR, topics=['1', '11', '35'],lemmatize=True)
#used to represent docs as vectors
# ecb_wrapper.compute_doc_template_set()

#descriptive statistics for the corpus
sim_metrics = SimilarityMetrics('data/ecb_doc_template_set.txt')
#if you want to use tfidf representations of documents
sim_metrics.make_tfidf_matrix(ecb_wrapper, element_type=element_types[element_choice])
#writes an excel file with avg. KL-divergence and Cosine Similarity per subtopic and topic
sim_metrics.compute_ecb_clustering_stats(ecb=ecb_wrapper, outfile_name=element_types[element_choice] + 'no_tfidf',element_type=element_types[element_choice],tfidf=False)




