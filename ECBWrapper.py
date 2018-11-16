from os.path import isfile, join
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import wordnet,stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

'''
Wrapper for the ECB+ corpus. 

    input:
    - dir_ : path to root directory of the corpus 
    - topics: a list of str that can subset the corpus
    
    class_vars:
    - all_files: flat list of all the files in the current subset
'''
class ECBWrapper():

    def __init__(self, dir_,lemmatize,topics=None):
        self.root_dir = dir_
        self.topics = topics
        # for root, subdirs, files in walk(dir_)
        self.all_files = []
        self.lemmatize = lemmatize

        for root, subdirs, files in os.walk(dir_):
            for f in files:
                if isfile(join(root,f)) and f.endswith('xml'):
                    if self.topics is not None:
                        if self.get_topic_num(join(root,f))[0] in self.topics:
                            self.all_files.append(join(root, f))
                    else:
                        self.all_files.append(join(root, f))

    '''
    Gives all paths in a given topic, sub-topic. 
    If sup_topic is not passed, all files in the topic are returned
    
        input:
        - topic: str of desired topic
        - sub_topic: srt of desired sub topic
        
        output: 
        - flat list of files
    '''
    def get_topic(self, topic, sub_topic = -1):
        files = []
        if sub_topic == -1:
            for f in self.all_files:
                if os.path.basename(f).split("_")[0] == str(topic):
                    files.append(f)
        else:
            for f in self.all_files:
                # ecb
                if sub_topic == 1:
                    if os.path.basename(f).split("_")[0] == str(topic) and not 'plus' in f:
                        files.append(f)
                # ecb+
                elif sub_topic == 2:
                     if os.path.basename(f).split("_")[0] == str(topic) and 'plus' in f:
                         files.append(f)
        return files

    '''
    Gives files for a subset of topics (includes sub-topic in a dictionary)
    
        input: 
        - topics: list to subset topics
        
        output: 
        - dictionary of dictionaries, where the first level key = topic, first level value = dict of 
        subtopics, with key = sub_topic, value = list of paths 
    '''
    def get_files_by_topic(self, topics= None):
        files_by_topic = dict()
        topic_nums = range(1,46)
        if topics is not None:
            topic_nums = topics
        # get all files in all topics, partitioned by suptopic
        for i in topic_nums:
            if i not in [15, 17]:
                files_by_topic[str(i)] = {'1': [], '2': []}
                # get all files in topic i
                topic_files = self.get_topic(topic=i)
                for f in topic_files:
                    top = self.get_topic_num(f)
                    files_by_topic[top[0]][top[1]].append(f)
        return files_by_topic

    '''
    Gives topic,subtopic for a given path 
    
        input: 
        - path: path you want topic number from 
        
        output: 
        - tuple, where tuple[0] = topic, tuple[1] = subtopic
    
    '''
    def get_topic_num(self, path):
        p = os.path.basename(path).split("_")
        top = p[0]
        sub = '2' if 'plus' in p[1] else '1'
        return (top,sub)

    '''
    Return different types of text representations of an ECB+ document 
    
        input: 
        - path: path of desired document
        - lemmatize: boolean asking for lemmatized/non-lemmatized text 
        - element_type: None gives full text, 
            * 'event_trigger' gives event triggers
            * 'doc_template' gives all ecb annotations (event triggers, human/non-human participans, times, locations)
            * 'events_and_participants' gives events and human/non-human participants
            * 'events_participants_locations' gives events, human/non-human participants and locations
            * 'participants' gives human/non-human participants
        
        output: 
        - a string (if element_type=None) or list of desired types of elements
    '''
    def get_text(self, path, element_type=None):
        #text
        if element_type is None:
            tree = ET.parse(path)
            root = tree.getroot()
            if self.lemmatize:
                lemmas = self.lemmatize_ecb_file(path)
                return ' '.join([lemmas[token.attrib['t_id']] for token in root.findall('token')])
            else:
                return ' '.join([self.only_alphanumeric(token.text) for token in root.findall('token')])
        #ecb+ annotations
        else:
            #this is a dictionary with a list of strings for every
            #ecb annotation (slot) in the document
            #comes with only alphanumeric chars
            elements = self.get_document_template(path)
            if element_type == 'event_trigger':
                return [term for term in [term_list for term_list in elements['ACTION']]]
            elif element_type == 'participant':
                return [term for term in [term_list for term_list in elements['HUM_PARTICIPANT'] + elements['NON_HUM_PARTICIPANT']]]
            elif element_type == 'time':
                return [term for term in [term_list for term_list in elements['TIME']]]
            elif element_type == 'location':
                return [term for term in [term_list for term_list in elements['LOCATION']]]
            elif element_type == 'doc_template':
                elements = elements['ACTION'] + elements['HUM_PARTICIPANT'] + elements['NON_HUM_PARTICIPANT'] + elements['LOCATION'] + elements['TIME']
                return [term for term in [term_list for term_list in elements]]
            elif element_type == 'event_hum_participant':
                return [term for term in [term_list for term_list in elements['ACTION'] + elements['HUM_PARTICIPANT']]]
            elif element_type == 'hum_participant':
                return [term for term in [term_list for term_list in elements['HUM_PARTICIPANT']]]
            elif element_type == 'event_hum_participant_location':
                return [term for term in [term_list for term_list in elements['ACTION'] + elements['HUM_PARTICIPANT'] + elements['LOCATION']]]

    '''
    Builds a document template for an ECB+ document. A document template includes all event components
    in a document. 
    
        input: 
        - path: path of desired ECB+ document
        - lemmatize: if you want to lemmatize elements in template or not
        
        output: 
        - a dictionary with key = slot, value = list of elements, with non-alphanumeric chars stripped
    '''
    def get_document_template(self,path):
        tree = ET.parse(path)
        root = tree.getroot()
        template = {"ACTION":[],
                    "HUM_PARTICIPANT":[],
                    "NON_HUM_PARTICIPANT":[],
                    "LOCATION":[],
                    "TIME":[]}
        lemmas = self.lemmatize_ecb_file(path)
        for markable in root.findall('Markables'):
            for child in markable:
                tags = [c.tag for c in child]
                if 'token_anchor' in tags:
                    if 'ACTION' in child.tag:
                        if self.lemmatize:
                            txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                        else:
                            txt = ' '.join(self.only_alphanumeric(self.get_token(path,c.attrib['t_id']).text) for c in child)
                        template['ACTION'].append(txt)
                    elif 'PART' in child.tag:
                        if 'NON_HUM' in child.tag:
                            if self.lemmatize:
                                txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                            else:
                                txt = ' '.join(
                                    self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                            template['NON_HUM_PARTICIPANT'].append(txt)
                        else:
                            if self.lemmatize:
                                txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                            else:
                                txt = ' '.join(
                                    self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                            template['HUM_PARTICIPANT'].append(txt)
                    elif 'LOC' in child.tag:
                        if self.lemmatize:
                            txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                        else:
                            txt = ' '.join(
                                self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                        template['LOCATION'].append(txt)
                    elif 'TIME' in child.tag:
                        if self.lemmatize:
                            txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                        else:
                            txt = ' '.join(
                                self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                        template['TIME'].append(txt)
        return template

    '''
    Lemmatizes every word in an ECB+ file. 

        input: 
        - path: path of file you want to lemmatize 

        output: 
        - a dictionary where key = ECB+ token id (t_id), value = lowercase lemma, with non-alphanumeric chars stripped
    '''
    def lemmatize_ecb_file(self, path):
        lemmas = dict()
        lemmatizer = WordNetLemmatizer()
        for k, v in self.get_all_sentences(path).items():
            pos_tag = nltk.pos_tag([self.only_alphanumeric(t.text) for t in v])
            for i in range(len(pos_tag)):
                lemmas[v[i].attrib['t_id']] = lemmatizer.lemmatize(pos_tag[i][0],
                                                                   self.get_wordnet_pos(pos_tag[i][1])).lower()
        return lemmas

    '''
    Maps a Treebank tag to a Wordnet tag

        input: 
        - treebank_tag: Treebank tag to be mapped to Wordnet tag

        output: 
        - a wordnet object with the corresponding POS tag
    '''
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    '''
    Returns an xml element representing the token in the path with given t_id
    
        input: 
        - path: path of desired document
        - t_id: token id (index, 1 based) of in ECB+ document
        
        output: 
        - xml element of that token 
    '''
    def get_token(self,path,t_id):
        tree = ET.parse(path)
        root = tree.getroot()
        for child in root.findall('token'):
            if child.attrib['t_id'] == str(t_id):
                return child

    '''
    Returns all sentences in an ECB+ document 
    
        input: 
        - path: path of desired document 
        
        output: 
        - dictionary with key = sentence id, value = list of tokens
    '''
    def get_all_sentences(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        sentences = dict()
        for child in root.findall('token'):
            if child.attrib['sentence'] not in sentences:
                sentences[child.attrib['sentence']] = []
            sentences[child.attrib['sentence']].append(child)
        return sentences

    '''
    Returns a sentence with a given sentence id
    
        input: 
        - path: path of desired ECB+ document
        - s_id: sentence id of desired sentence 
        
        output: 
        - list of tokens in that sentence
    '''
    def get_sentence(self,path,s_id):
        tree = ET.parse(path)
        root = tree.getroot()
        sentence = []
        for child in root.findall('token'):
            if child.attrib['sentence'] == str(s_id):
                sentence.append(child)
        return sentence

    '''
    Get xlm element representing that token 
    
        input: 
        - path: path of desired ECB+ document
        - t_id: token id
        
        output: 
        - xlm element of that token
    '''
    def get_token(self,path, t_id):
        tree = ET.parse(path)
        root = tree.getroot()
        for tok in root.findall('token'):
            if tok.attrib['t_id'] == t_id:
                return tok

    '''
    Computes a set of non-stopword words in corpus and saves in .txt file
    
        output:
        - set of terms, with non-alphanumeric chars stripped
    '''
    def compute_term_set(self):
        terms = set()
        for f in self.all_files:
            words = self.get_text(f).lower()
            words = filter(lambda x: x not in stopwords.words('english'), words)
            for w in words:
                terms.add(self.only_alphanumeric(w))
        with open('data/ecb_term_set.txt', 'w') as f:
            for item in terms:
                f.write("%s\n" % item)
        return terms

    '''
    Computes set of event triggers (could be multi-word) and saves in .txt file
    
        output: 
        - set of event triggers
    '''
    def compute_event_trigger_set(self):
        terms = set()
        for f in self.all_files:
            triggers = self.get_text(path=f,element_type='event_trigger')
            for t in triggers:
                terms.add(self.only_alphanumeric(t))
        with open('data/ecb_event_trigger_set.txt', 'w') as f:
            for item in terms:
                f.write("%s\n" % item)
        return terms

    '''
    Computes set of document template elements (could be multi word and contain non-alphanumeric characters)
    and saves in a .txt file
    
        output: 
        - set of document template elements
    '''
    def compute_doc_template_set(self):
        terms = set()
        for f in self.all_files:
            elements = self.get_document_template(f)
            for slot,term_list in elements.items():
                for t in term_list:
                    terms.add(self.only_alphanumeric(t))
        with open('data/ecb_doc_template_set.txt', 'w') as f:
            for item in terms:
                f.write("%s\n" % item)
        return terms
    '''
    Strip all non-alphanumeric chars from a string 
    
        input:
        - s: desired string
        
        output: 
        - same string w/o non-alphanumeric chars 
    '''
    def only_alphanumeric(self, s):
        #return re.sub(r'\W+', '', s)
        if len(s) > 1:
            return re.sub(r'\"', '',s)
        else:
            return s

