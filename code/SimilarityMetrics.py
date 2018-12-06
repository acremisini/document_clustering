from scipy import stats, spatial
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from code.Clustering_Utility import Clustering_Utility
import sys
import matplotlib.pyplot as plt
from pylab import text


class SimilarityMetrics():

    def __init__(self, term_set_path):
        self.terms = dict()
        with open(term_set_path) as f:
            i = 0
            for l in f.readlines():
                self.terms[l.replace('\n','')] = i
                i += 1
        self.tfidf_matrix = ''
        self.vectorizer = ''
        self.corpus = dict()
        self.path_to_corpus_idx = dict()

    def make_word_count_vector(self, txt):

        vec = [0] * len(self.terms)
        term_i = set()
        num_words = 0
        if isinstance(txt,list):
            for t in txt:
                if t in self.terms:
                    try:
                        vec[self.terms[t]] += 1
                    except:
                        print(t)
                        print(self.terms[t])
                        sys.exit(-1)
                    num_words += 1
                    term_i.add(self.terms[t])
            for i in term_i:
                vec[i] = vec[i] / num_words*1.0
        else:
            print(txt)
            sys.stderr>>'ERROR'
        return vec

    def get_tfidf_vector(self,path):
        return self.tfidf_matrix[self.path_to_corpus_idx[path]]

    def laplace_smooth(self, v, alpha = .001):
        d = len(self.terms)
        n = np.count_nonzero(v)
        for i in range(len(v)):
            v[i] = (v[i] + alpha)/(n + alpha*d)*1.0

        return v

    def divergence(self, v1, v2):
        return float(stats.entropy(v1,v2))

    def cos_sim(self,v1,v2):
        return 1 - spatial.distance.cosine(v1,v2)

    def make_tfidf_matrix(self, ecb, element_type):
        i = 0
        for f in ecb.all_files:
            self.corpus[i] = ' '.join(ecb.get_text(f, element_type=element_type))
            self.path_to_corpus_idx[f] = i
            i += 1
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus.values())

    def get_word_count_stats(self,corpus,fname, title):
        counts = pd.DataFrame({"Word Counts" : [len(doc) for doc in corpus[0]]})
        plt.figure()
        bp_dict = counts.boxplot(column="Word Counts",return_type='dict')
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1]  # top of median line
            # overlay median value
            text(x, y, counts.median()['Word Counts'],
                 horizontalalignment='left')  # draw above, centered

        plt.title(title)
        plt.ylabel('# Words per Document')
        plt.savefig(fname='output/'+fname)

    def get_cluster_sim(self,ecb,element_type,within,fname,title):
        # helper objects
        utils = Clustering_Utility()
        files_by_topic = ecb.get_files_by_topic()

        sims = []
        j = 0
        file_pairs = utils.unique_pairwise_no_diagonal(ecb.all_files)
        for p in file_pairs:
            pairs = list(p)
            if within:
                if ecb.get_topic_num(pairs[0])[0] == ecb.get_topic_num(pairs[1])[0]:
                    v0 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[0], element_type=element_type)))
                    v1 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[1], element_type=element_type)))
                    sims.append(self.cos_sim(v0, v1))
                if (round((j * 1.0 / len(file_pairs)),2) % .05 == 0):
                    print(str(round((j * 1.0 / len(file_pairs)), 2)) + '% ...')
                j += 1
            if not within:
                if ecb.get_topic_num(pairs[0])[0] is not ecb.get_topic_num(pairs[1])[0]:
                    v0 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[0], element_type=element_type)))
                    v1 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[1], element_type=element_type)))
                    sims.append(self.cos_sim(v0, v1))
                if (round((j * 1.0 / len(file_pairs)),2) % .05 == 0):
                    print(str(round((j * 1.0 / len(file_pairs)),2)) + '% ...')
                j += 1
        print('---')
        df = pd.DataFrame({'Cosine Similarity' : sims})
        plt.figure()
        bp_dict = df.boxplot(column="Cosine Similarity",return_type='dict')
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1]  # top of median line
            # overlay median value
            text(x, y, round(df.median()['Cosine Similarity'],2),
                 horizontalalignment='left')  # draw above, centered
        plt.title(title)
        plt.ylabel('Pairwise Cosine Similarity')
        plt.autoscale()
        plt.tight_layout()
        plt.savefig(fname='output/'+fname)





    def compute_ecb_clustering_stats(self, ecb, outfile_name, element_type,tfidf):
        #helper objects
        utils = Clustering_Utility()
        files_by_topic = ecb.get_files_by_topic()

        #get dict ready to write excel
        topic_stats = dict()
        #will store everything in here
        df = pd.DataFrame(
            columns=['topic', 'sub_1_num_docs', 'sub_1_avg_kl', 'sub_1_avg_cos_sim', 'sub_2_num_docs', 'sub_2_avg_kl',
                     'sub_2_avg_cos_sim', 'total_docs', 'avg_topic_kl', 'avg_topic_cos_sim'])

        #compute corpus level stats
        topic_kl = []
        topic_cos = []
        j = 0
        file_pairs = utils.unique_pairwise_no_diagonal(ecb.all_files)
        #compare all documents
        #only compare cross-topic documents
        # file_pairs = [list(pair) for pair in file_pairs]
        # file_pairs = [pair for pair in file_pairs if ecb.get_topic_num(pair[0])[0] != ecb.get_topic_num(pair[1])[0]]

        print('computing cross-topic similarities...')
        #general metrics, as if i knew nothing about the partitioning of the corpus
        for p in file_pairs:
            pairs = list(p)
            if not tfidf:
                v0 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[0],element_type=element_type)))
                v1 = self.laplace_smooth(
                    self.make_word_count_vector(ecb.get_text(pairs[1], element_type=element_type)))
            else:
                v0 = self.laplace_smooth(self.get_tfidf_vector(pairs[0]).todense().T)
                v1 = self.laplace_smooth(self.get_tfidf_vector(pairs[1]).todense().T)
            topic_kl.append(self.divergence(v0, v1))
            topic_cos.append(self.cos_sim(v0, v1))
            if (j % 100 == 0):
                print(str(j*1.0/len(file_pairs)))
            j += 1
        print('done')

        #record corpus level stats
        total_obs = len(ecb.all_files)
        topic_kl = stats.describe(topic_kl).mean
        topic_cos = stats.describe(topic_cos).mean
        row = ['corpus level', '-', '-', '-', '-', '-', '-', total_obs, topic_kl, topic_cos]
        df.loc[0] = row

        #compute topic level stats
        # for each topic
        for topic, topic_files in files_by_topic.items():
            if len(topic_files['1']) == len(topic_files['2']) == 0:
                continue
            if topic not in topic_stats:
                topic_stats[topic] = {'0': {'kl': [], 'cosine': [], 'num_files': 0},
                                  '1': {'kl': [], 'cosine': [], 'num_files': 0},
                                  '2': {'kl': [], 'cosine': [], 'num_files': 0}}

            j = 0
            print('computing topic ' + str(topic) + '...')
            file_pairs = utils.unique_pairwise_no_diagonal(topic_files['1'] + topic_files['2'])
            #across both subtopics
            for p in file_pairs:
                pairs = list(p)
                if not tfidf:
                    v0 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[0],element_type=element_type)))
                    v1 = self.laplace_smooth(
                        self.make_word_count_vector(ecb.get_text(pairs[1], element_type=element_type)))
                else:
                    v0 = self.laplace_smooth(self.get_tfidf_vector(pairs[0]).todense().T)
                    v1 = self.laplace_smooth(self.get_tfidf_vector(pairs[1]).todense().T)
                topic_stats[topic]['0']['kl'].append(self.divergence(v0, v1))
                topic_stats[topic]['0']['cosine'].append(self.cos_sim(v0, v1))
                topic_stats[topic]['0']['num_files'] = len(files_by_topic[topic]['1']) + len(files_by_topic[topic]['2'])
                if (j % 100 == 0):
                    print(str(j) + "/" + str(len(file_pairs)))
                j += 1
            print('done')

            #within each subtopic
            for sub_topic, sub_files in topic_files.items():
                print('computing sub-topic ' + str(sub_topic))
                file_pairs = utils.unique_pairwise_no_diagonal(sub_files)
                j = 0
                for f in file_pairs:
                    pairs = list(f)
                    if not tfidf:
                        v0 = self.laplace_smooth(self.make_word_count_vector(ecb.get_text(pairs[0],element_type=element_type)))
                        v1 = self.laplace_smooth(self.make_word_count_vector(
                            ecb.get_text(pairs[1], element_type=element_type)))
                    else:
                        v0 = self.laplace_smooth(self.get_tfidf_vector(pairs[0]).todense().T)
                        v1 = self.laplace_smooth(self.get_tfidf_vector(pairs[1]).todense().T)
                    topic_stats[topic][sub_topic]['kl'].append(self.divergence(v0, v1))
                    topic_stats[topic][sub_topic]['cosine'].append(self.cos_sim(v0, v1))
                    topic_stats[topic][sub_topic]['num_files'] = len(files_by_topic[topic][sub_topic])
                    if (j % 100 == 0):
                        print(str(j) + "/" + str(len(file_pairs)))
                    j += 1
                print('done')

        i = 1
        print('writing to excel...')
        for topic, topic_files in topic_stats.items():
            # sub-top 1
            sub = topic_files['1']
            num_obs_1 = len(sub['kl'])
            num_files_1 = sub['num_files']
            kl_1 = stats.describe(sub['kl']).mean
            cos_1 = stats.describe(sub['cosine']).mean
            # sub-top 2
            sub = topic_files['2']
            num_obs_2 = len(sub['kl'])
            num_files_2 = sub['num_files']
            kl_2 = stats.describe(sub['kl']).mean
            cos_2 = stats.describe(sub['cosine']).mean
            # aggregate
            total_obs = topic_stats[topic]['0']['num_files']
            topic_kl = stats.describe(topic_stats[topic]['0']['kl']).mean
            topic_cos = stats.describe(topic_stats[topic]['0']['cosine']).mean

            #put everything into a row
            row = [topic, num_files_1, kl_1, cos_1, num_files_2, kl_2, cos_2, total_obs, topic_kl, topic_cos]
            df.loc[i] = row
            i += 1

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(df)
        writer = pd.ExcelWriter('data/'+ outfile_name + '.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
        print('done')