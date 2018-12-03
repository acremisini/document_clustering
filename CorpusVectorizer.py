from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from time import time

class CorpusVectorizer():

    def __init__(self,dataset):
        self.dataset = dataset
        self.X = None
        self.Y = None
        self.vectorizer = None
        self.vectorizer = None


    def vectorize(self,use_hashing,use_idf,n_features):
        t0 = time()
        if use_hashing:
            if use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=n_features,
                                           stop_words='english', alternate_sign=False,
                                           norm=None, binary=False)
                self.vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                self.vectorizer = HashingVectorizer(n_features=n_features,
                                               stop_words='english',
                                               alternate_sign=False, norm='l2',
                                               binary=False)
        else:
            self.vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                         min_df=2, stop_words='english',
                                         use_idf=use_idf)
        self.X = self.vectorizer.fit_transform(self.dataset[0])
        self.Y = self.dataset[1]
        # print("done in %fs" % (time() - t0))
        # print("n_samples: %d, n_features: %d" % self.X.shape)
        # print()