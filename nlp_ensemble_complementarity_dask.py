#!/usr/bin/env python
# coding: utf-8
'''
  Copyright (c) 2020 Regents of the University of Minnesota.
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
      http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
'''

#from combo_searcher_new import combo_searcher as cs
from combo_searcher_all import combo_searcher as cs
import gevent
from scipy import stats 
from scipy.stats import norm, mode
from scipy.stats.mstats import gmean
import pandas as pd
import numpy as np
import sparse as sp
import math
import time 
import functools as ft
import operator as op
from pathlib import Path
from itertools import combinations, product, permutations
from sqlalchemy.engine import create_engine
from datetime import datetime
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
from typing import List, Set, Tuple 
from sklearn.metrics import classification_report, confusion_matrix
from scipy import sparse
import statistics as s
from sqlite3 import connect
import joblib
from dask.distributed import Client
from sklearn.metrics import confusion_matrix

# If you have a remote cluster running Dask
# client = Client('tcp://scheduler-address:8786')

# If you want Dask to set itself up on your personal computer
#client = Client(processes=False)

# The cell below contains the configurable parameters to ensure that our ensemble explorer runs properaly on your machine. 
# Please read carfully through steps (1-11) before running the rest of the cells.


# STEP-1: CHOOSE YOUR CORPUS
# TODO: get working with list of corpora

# cross-system semantic union merge filter for cross system aggregations using custom system annotations file with corpus name and system name using 'ray_test':

# TODO: move to click param
#corpus = 'clinical_trial2'
#corpus = 'fairview'
#corpus = 'i2b2'
#corpus = 'fairview'
corpus = 'mipacq'
#corpus = 'medmentions'

# TODO: create config.py file
# STEP-2: CHOOSE YOUR DATA DIRECTORY; this is where output data will be saved on your machine
data_directory = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/' 

data_out = Path('/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/')

# TODO: move to click param
# STEP-3: CHOOSE WHICH SYSTEMS YOU'D LIKE TO EVALUATE AGAINST THE CORPUS REFERENCE SET
systems = ['biomedicus', 'clamp', 'ctakes', 'metamap', 'quick_umls']
#systems = ['biomedicus', 'clamp', 'ctakes', 'metamap']

# TODO: move to click param
# STEP-4: CHOOSE TYPE OF RUN:  
rtype = 7      # OPTIONS INCLUDE: 2->Ensemble; 3->Tests; 4 -> majority vote; 6 -> add hoc ensemble; 7 -> complementarity
               # The Ensemble can include the max system set ['ctakes','biomedicus','clamp','metamap','quick_umls']

# TODO: move to click param
# STEP-5: CHOOSE WHAT TYPE OF ANALYSIS YOU'D LIKE TO RUN ON THE CORPUS
analysis_type = 'entity' #options include 'entity', 'cui' OR 'full'

# TODO: create config.py file
# STEP-(6A): ENTER DETAILS FOR ACCESSING MANUAL ANNOTATION DATA
#database_type = 'mysql+pymysql' # We use mysql+pymql as default
#database_username = 'gms'
#database_password = 'nej123' 
#database_url = 'localhost' # HINT: use localhost if you're running database on your local machine
#database_name = 'medmentions' # concepts' # Enter database name
#database_name = 'medmentions' # Enter database name

def ref_data(corpus):
    return corpus + '_all' # Enter the table within the database where your reference data is stored

table_name = ref_data(corpus)

# TODO: move to click param
# STEP-(8A): FILTER BY SEMTYPE
filter_semtype = True #False #True #False #True #False #True #False #True #False #True #False #True#False #True#False #True 

# STEP-(6B): ENTER DETAILS FOR ACCESSING SYSTEM ANNOTATION DATA

def sys_data(corpus, analysis_type):
    if analysis_type == 'entity' and not filter_semtype:
        #return 'disambiguated_analytical_'+corpus+'.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
        return 'analytical_disambiguated_'+corpus+'_full.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
    elif analysis_type == 'entity' and filter_semtype:
        return 'analytical_disambiguated_'+corpus+'_st.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
        #return '/submission/new_ner/analytical_disambiguated_mipacq_Disorders,Sign_Symptom_1633978586.737513.csv'
    elif analysis_type in ('cui', 'full', 'entity'):
        return 'analytical_'+corpus+'_cui.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 

system_annotation = sys_data(corpus, analysis_type)

# STEP-7: CREATE A DB CONNECTION POOL
#engine_request = str(database_type)+'://'+database_username+':'+database_password+"@"+database_url+'/'+database_name
#engine = create_engine(engine_request, pool_pre_ping=True, pool_size=20, max_overflow=30)
#engine = engine_request
#engine = connect('data/medmentions.sqlite')


# TODO: create config.py file
# STEP-(8B): IF STEP-(8A) == True -> GET REFERENCE SEMTYPES

def ref_semtypes(filter_semtype, corpus):
    if filter_semtype:
        if corpus == 'fairview':
            semtypes = ['Drug', 'Finding', 'Anatomy', 'Procedure']
            #semtypes = ['Finding']
        elif corpus == 'i2b2':
            semtypes = ['test,treatment', 'problem']
            #semtypes = ['problem']
        elif corpus == 'mipacq':
            #semtypes = ['Anatomy', 'Procedures', 'Disorders,Sign_Symptom', 'Chemicals_and_drugs']
            semtypes = ['Disorders,Sign_Symptom']
        elif corpus == 'medmentions':
            semtypes = ['Anatomy', 'Disorders', 'Chemicals & Drugs', 'Procedures']

        return semtypes

semtypes = ref_semtypes(filter_semtype, corpus)

# STEP-9: Set data directory/table for source documents for vectorization
src_table = 'sofa'

# TODO: move to click param
# STEP-10: Specify match type from {'exact', 'overlap'}
run_type = 'overlap'

# STEP-11: Specify type of ensemble: merge or vote: used for file naming -> TODO: remove!
ensemble_type = 'vote'


def echo_config():
    print('corpus:', corpus)
    print('filter:', filter_semtype)
    print('systems:', systems)
    print('task:', analysis_type)
    print('semtypes:', semtypes)
    

#****** TODO 
'''
-> add majority vote to union for analysis_type = 'full', 'cui': done!
-> case for multiple labels on same/overlapping span/same system; disambiguate (order by score if exists and select random for ties): done!
-> port to command line: in process... 
-> rtype 1 -> dump expression_type = 'single': done! -> use count of operators; integrate combo_searcher to generate expressiopn
-> refactor vectorized_cooccurences ->  vectorized_cooccurences_test and vectorized_annotations
----------------------->
-> swap out confused with vectorized_cooccurences
-> better control for overlap/exact matching across all single systems and ensembles (e.g., eliminate run_type = for majority-vote control of type of match)
-> negation/polarity: in process...
-> still need to validate that all semtypes in corpus!
-> handle case where intersect merges are empty/any confusion matrix values are 0; specificallly on empty df in evaluate method: done!
-> case when system annotations empty from semtype filter; print as 0: done!
-> trim whitespace on CSV import -> done for semtypes
-> cross-system semantic union merge on aggregation
-> other modification, such as 'present'
-> clean up configuration process
-> allow iteration through all corpora and semtypes
-> optimize vecorization (remove confusion?)
-> clean up SemanticTypes
-> add non-UIMA solution for get_docs
-> confusion matrix for multiclass 
'''


def get_connect(corpus):
    if corpus == 'medmentions':
        return connect('data/medmentions.sqlite')
    else:
        return connect('data/all_corpora.sqlite')


# config class for analysis
class AnalysisConfig():
    """
    Configuration object:
    systems to use
    notes by corpus
    paths by output, gold and system location
    """
    def __init__(self):
        self = self    
        self.systems = systems
        self.data_dir = data_directory
    
    def corpus_config(self): 
        usys_data = system_annotation
        #ref_data = database_name+'.'+table_name
        ref_data = table_name
        return usys_data, ref_data


analysisConf =  AnalysisConfig()


class SemanticTypes(object):
    '''
    Filter semantic types based on: https://metamap.nlm.nih.gov/SemanticTypesAndGroups.shtml
    :params: semtypes list from corpus, system to query
    :return: list of equivalent system semtypes 
    '''
   
 
    def __init__(self, semtypes, corpus):
        self = self

        engine = get_connect(corpus)
        
        if corpus == 'medmentions':
            sql = "SELECT st.tui, abbreviation, clamp_name, ctakes_name FROM semantic_groups sg join semantic_types st on sg.tui = st.tui where group_name in ({})".format(', '.join(['?' for _ in semtypes]))  
        else:
            sql = "SELECT st.tui, abbreviation, clamp_name, ctakes_name FROM semantic_groups sg join semantic_types st on sg.tui = st.tui where " + corpus + "_name in ({})" .format(', '.join(['?' for _ in semtypes]))  
        
        stypes = pd.read_sql(sql, params=semtypes, con=engine) 
       
        if len(stypes['tui'].tolist()) > 0:
            self.biomedicus_types = set(stypes['tui'].tolist())
            self.qumls_types = set(stypes['tui'].tolist())
        
        else:
            self.biomedicus_types = None
            self.qumls_types = None

        if stypes['clamp_name'].dropna(inplace=True) or len(stypes['clamp_name'].tolist()) == 0 or len(set(stypes['clamp_name'].tolist()).intersection({'NULL', None})) > 0:
            self.clamp_types = None
        else:
            self.clamp_types = set(stypes['clamp_name'].tolist()[0].split(','))
         
        if stypes['ctakes_name'].dropna(inplace=True) or len(stypes['ctakes_name'].tolist()) == 0 or None in stypes['ctakes_name'].tolist():
            self.ctakes_types = None
        else:
            self.ctakes_types = set(stypes['ctakes_name'].tolist()[0].split(','))
        
        if len(stypes['abbreviation'].tolist()) > 0:
            self.metamap_types = set(stypes['abbreviation'].tolist())
        else:
            self.metamap_types = None
        
        self.reference_types =  set(semtypes)
    
    def get_system_type(self, system):  
        
        if system == 'biomedicus':
            semtypes = self.biomedicus_types
        elif system == 'ctakes':
            semtypes = self.ctakes_types
        elif system == 'clamp':
            semtypes = self.clamp_types
        elif system == 'metamap':
            semtypes = self.metamap_types
        elif system == 'quick_umls':
            semtypes = self.qumls_types
        elif system == 'reference':
            semtypes = self.reference_types
            
        return semtypes
    
#print(SemanticTypes(['Drug'], corpus).get_system_type('biomedicus'))
#print(SemanticTypes(['Drug'], corpus).get_system_type('quick_umls'))
#print(SemanticTypes(['drug'], corpus).get_system_type('clamp'))
#print(SemanticTypes(['Anatomy'], 'mipacq').get_system_type('ctakes'))


#semtypes = ['test,treatment']
#semtypes = 'drug,drug::drug_name,drug::drug_dose,dietary_supplement::dietary_supplement_name,dietary_supplement::dietary_supplement_dose'
#semtypes =  'demographics::age,demographics::sex,demographics::race_ethnicity,demographics::bmi,demographics::weight'
#corpus = 'clinical_trial'
#sys = 'quick_umls'

# is semantic type in particular system
def system_semtype_check(sys, semtype, corpus):
    st = SemanticTypes([semtype], corpus).get_system_type(sys)
    if st:
        return sys
    else:
        return None

#print(system_semtype_check(sys, semtypes, corpus))

class Metrics(object):
    """
    metrics class:
    returns an instance with confusion matrix metrics
    """
    def __init__(self, system_only, gold_only, gold_system_match, system_n, neither = 0): # neither: no sys or manual annotation

        self = self    
        self.system_only = system_only
        self.gold_only = gold_only
        self.gold_system_match = gold_system_match
        self.system_n = system_n
        self.neither = neither
        
    def get_confusion_metrics(self, corpus = None, test = False):
        
        """
        compute confusion matrix measures, as per  
        https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
        """

        TP = self.gold_system_match
        FP = self.system_only
        FN = self.gold_only
        
        TM = TP/math.sqrt(self.system_n) # TigMetric
       
        if not test:
            
            if corpus == 'casi':
                recall = TP/(TP + FN)
                precision = TP/(TP + FP)
                F = 2*(precision*recall)/(precision + recall)
            else:
                if self.neither == 0:
                    confusion = [[0, self.system_only],[self.gold_only,self.gold_system_match]]
                else:
                    confusion = [[self.neither, self.system_only],[self.gold_only,self.gold_system_match]]
                c = np.asarray(confusion)
                
                if TP != 0 or FP != 0:
                    precision = TP/(TP+FP)
                else:
                    precision = 0
                
                if TP != 0 or FN != 0:
                    recall = TP/(TP+FN)
                else:
                    recall = 0
                
                if precision + recall != 0:
                    F = 2*(precision*recall)/(precision + recall)
                else:
                    F = 0
    
        else:
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            F = 2*(precision*recall)/(precision + recall)
        
        # Tignanelli Metric
        if FN == 0:
            TP_FN_R = TP
        elif FN > 0:
            TP_FN_R = TP/FN
 
        return F, recall, precision, TP, FP, FN, TP_FN_R, TM

# https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


def df_to_set(df, analysis_type = 'entity', df_type = 'sys', corpus = None):
    
    # get values for creation of series of type tuple
    if 'entity' in analysis_type: 
        if corpus == 'casi':
            arg = df.case, df.overlap
        else:    
            arg = df.begin, df.end, df.case
            
    elif 'cui' in analysis_type:
        arg = df.value, df.case
    elif 'full' in analysis_type:
        arg = df.begin, df.end, df.value, df.case
    
    return set(list(zip(*arg)))


# use for exact matches
#%%cython 
def get_cooccurences(ref, sys, analysis_type: str, corpus: str):
    """
    get cooccurences between system and reference; exact match
    """
    # cooccurences
    class Cooccurences(object):
        
        def __init__(self):
            self.ref_system_match = 0
            self.ref_only = 0
            self.system_only = 0
            self.system_n = 0
            self.ref_n = 0
            self.matches = set()
            self.false_negatives = set()
            self.corpus = corpus

    c = Cooccurences()
    
    if c.corpus != 'casi':
        if analysis_type in ['cui', 'full']:
            sys = sys.rename(index=str, columns={"note_id": "case", "cui": "value"})
            # do not overestimate FP
            sys = sys[~sys['value'].isnull()] 
            ref = ref[~ref['value'].isnull()]
        
        if 'entity' in analysis_type: 
            sys = sys.rename(index=str, columns={"note_id": "case"})
            cols_to_keep = ['begin', 'end', 'case']
        elif 'cui' in analysis_type: 
            cols_to_keep = ['value', 'case']
        elif 'full' in analysis_type: 
            cols_to_keep = ['begin', 'end', 'value', 'case']
        
        sys = sys[cols_to_keep].drop_duplicates()
        ref = ref[cols_to_keep].drop_duplicates()
        # matches via inner join
        tp = pd.merge(sys, ref, how = 'inner', left_on=cols_to_keep, right_on = cols_to_keep) 
        # reference-only via left outer join
        fn = pd.merge(ref, sys, how = 'left', left_on=cols_to_keep, right_on = cols_to_keep, indicator=True) 
        fn = fn[fn["_merge"] == 'left_only']

        tp = tp[cols_to_keep]
        fn = fn[cols_to_keep]

        # use for metrics 
        c.matches = c.matches.union(df_to_set(tp, analysis_type, 'ref'))
        c.false_negatives = c.false_negatives.union(df_to_set(fn, analysis_type, 'ref'))
        c.ref_system_match = len(c.matches)
        c.system_only = len(sys) - len(c.matches) # fp
        c.system_n = len(sys)
        c.ref_n = len(ref)
        c.ref_only = len(c.false_negatives)
        
    else:
        sql = "select `case` from test.amia_2019_analytical_v where overlap = 1 and `system` = %(sys.name)s"  
        tp = pd.read_sql(sql, params={"sys.name":sys.name}, con=engine)
        
        sql = "select `case` from test.amia_2019_analytical_v where (overlap = 0 or overlap is null) and `system` = %(sys.name)s"  
        fn = pd.read_sql(sql, params={"sys.name":sys.name}, con=engine)
        
        c.matches = df_to_set(tp, 'entity', 'sys', 'casi')
        c.fn = df_to_set(fn, 'entity', 'sys', 'casi')
        c.ref_system_match = len(c.matches)
        c.system_only = len(sys) - len(c.matches)
        c.system_n = len(tp) + len(fn)
        c.ref_n = len(tp) + len(fn)
        c.ref_only = len(fn)
        
    # sanity check
    if len(ref) - c.ref_system_match < 0:
        print('Error: ref_system_match > len(ref)!')
    if len(ref) != c.ref_system_match + c.ref_only:
        print('Error: ref count mismatch!', len(ref), c.ref_system_match, c.ref_only)
   
    return c 

#Relaxed matches using vectorization:

# https://stackoverflow.com/questions/49210801/python3-pass-lists-to-function-with-functools-lru-cache
def listToTuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result
    return wrapper


def flatten_list(l):
    return [item for sublist in l for item in sublist]


#@listToTuple
#@ft.lru_cache(maxsize=None)
#%load_ext cython
#get_ipython().run_line_magic('load_ext', 'cython')
def label_vector(doc: int, ann: List[int], labels: List[str]) -> np.array:

    v = np.zeros(doc)
    labels = list(labels)
    for (i, lab) in enumerate(labels):
        i += 1  # 0 is reserved for no label
        idxs = [np.arange(a.begin, a.end) for a in ann if a.label == lab]
        idxs = [j for mask in idxs for j in mask]
        v[idxs] = i 

    return v

# confusion matrix elements for vectorized annotation set binary classification
# https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/

#get_ipython().run_line_magic('load_ext', 'cython')
#import numpy as np
def confused(pred, true):

    not_true = np.logical_not(true)
    not_predicted = np.logical_not(pred)
    TP = np.sum(np.logical_and(true, pred))
    TN = np.sum(np.logical_and(not_true, not_predicted))
    FP = np.sum(np.logical_and(not_true, pred))
    FN = np.sum(np.logical_and(true, not_predicted))

    #FN = len(pred) - (TP+TN+FP)
    
    return TP, TN, FP, FN


@ft.lru_cache(maxsize=None)
def get_labels(analysis_type, corpus, filter_semtype, semtype=None):
    
    if analysis_type == 'entity':
        labels = ["concept"]
    elif analysis_type in ['cui', 'full']:
        ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
        labels = list(set(ann["label"].tolist()))
    
    return labels

#@ft.lru_cache(maxsize=None)
def vectorized_annotations(ann, analysis_type, labels):
    
    docs = get_docs(corpus)
    out = []
    
    #if analysis_type != 'cui':
    #    ann1 = list(ann.itertuples(index=False))
    
    for k, v in docs.items():
        if analysis_type != 'cui':
            a1 = list(ann.loc[ann.case == k].itertuples(index=False))
            #a1 = [i for i in ann1 if i.case == k]
            a = label_vector(v, a1, labels)
            out.append(a)
        else:
            a = ann.loc[ann.case == docs[n][0]]['label'].tolist()
            x = [1 if x in a else 0 for x in labels]
            out.append(x)

    return out

#@ft.lru_cache(maxsize=None)
def vectorized_cooccurences(r: object, analysis_type: str, corpus: str, filter_semtype, semtype = None) -> np.int64:
    docs = get_docs(corpus)
        
    sys = get_sys_ann(analysis_type, r)
    labels = get_labels(analysis_type, corpus, filter_semtype, semtype)

    sys2 = []
    s2 = []

    #if analysis_type != 'cui':
    #    s = list(sys.itertuples(index=False))
    
    for k, v in docs.items():
        if analysis_type != 'cui':
            s1 = list(sys.loc[sys.case == k].itertuples(index=False))
            #s1 = [i for i in s if i.case==k] # list(sys.loc[sys.case == docs[n][0]].itertuples(index=False))
            sys1 = label_vector(v, s1, labels)
            sys2.append(sys1)
        else:
            s = sys.loc[sys.case == docs[n][0]]['label'].tolist()
            x = [1 if x in s else 0 for x in labels]
            s2.append(x)

    a2 = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)
            
    if analysis_type != 'cui': #binary and multiclass
        s2 = np.array(flatten_list(sys2))
        
        if analysis_type == 'full':
            report = classification_report(a2, s2, output_dict=True)
            macro_precision =  report['macro avg']['precision'] 
            macro_recall = report['macro avg']['recall']    
            macro_f1 = report['macro avg']['f1-score']
            return ((0, 0, 0, 0), (macro_precision, macro_recall, macro_f1))
        else:
            #TN, FP, FN, TP = confusion_matrix(a2, s2).ravel()

            TP, TN, FP, FN = confused(sp.COO(s2), sp.COO(a2))
            #TP, TN, FP, FN = confused(s2, a2)
            return ((TP, TN, FP, FN), (0, 0, 0))
                    
    else: # multilabel/multiclass
        x_sparse = sparse.csr_matrix(a2)
        y_sparse = sparse.csr_matrix(s2)
        report = classification_report(x_sparse, y_sparse, output_dict=True)
        macro_precision =  report['macro avg']['precision'] 
        macro_recall = report['macro avg']['recall']    
        macro_f1 = report['macro avg']['f1-score']
        return ((0, 0, 0, 0), (macro_precision, macro_recall, macro_f1))
                                       

# http://www.lrec-conf.org/proceedings/lrec2016/pdf/105_Paper.pdf        
#%load_ext cython
#import numpy as numpy
#import sparse as sp
def vectorized_complementarity(r: object, analysis_type: str, corpus: str, filter_semtype, semtype = None) -> np.int64:
    docs = get_docs(corpus)
    
    out = pd.DataFrame()
    
    sysA = r.sysA
    sysB = r.sysB
    sysA = sysA.rename(columns={"note_id": "case"})
    sysB = sysB.rename(columns={"note_id": "case"})

    sysA["label"] = 'concept'
    sysB["label"] = 'concept'
    cols_to_keep = ['begin', 'end', 'case', 'label']

    sysA = sysA[cols_to_keep]
    sysB = sysB[cols_to_keep]

    labels = ["concept"]

    sys_a2 = []
    sys_b2 = []
    sys_ab2 = []
    s_a2 = []
    s_b2 = []
    sys_ab1_ab3 = []
    diff_left = []
    diff_right = []
    #diff_left_ref = []
    #diff_right_ref = []

    #a = list(sysA.itertuples(index=False))
    #b = list(sysB.itertuples(index=False))

    ref = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)
    #ref_df = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)

    # test optimization
    '''
    inter = sysA.merge(sysB, on=['case', 'begin', 'end'])

    leftA = sysA.merge(sysB, how='left', on=['case', 'begin', 'end'], indicator=True)
    leftB = sysB.merge(sysA, how='left', on=['case', 'begin', 'end'], indicator=True)

    leftA = leftA.loc[leftA["_merge"] == 'left_only']
    leftB = leftB.loc[leftB["_merge"] == 'left_only']

    symmetric = pd.concat([leftA, leftB, inter])
    inter['label'] = 'concept'
    symmetric['label'] = 'concept'
    
    labels = get_labels(analysis_type, corpus, filter_semtype, semtype)
    
    s_a2 = vectorized_annotations(sysA, analysis_type, labels)
    s_b2 = vectorized_annotations(sysB, analysis_type, labels)
    sys_inter = vectorized_annotations(inter, analysis_type, labels)
    sys_symmetric = vectorized_annotations(symmetric, analysis_type, labels)

    s_a2 = np.concatenate(s_a2).ravel()
    s_b2 = np.concatenate(s_b2).ravel()
    sys_inter = np.concatenate(sys_inter).ravel()
    sys_symmetric = np.concatenate(sys_symmetric).ravel()
 
    _, _, FP, _ = confused(sp.COO(sys_inter), sp.COO(ref))
    _, _, _, FN = confused(sp.COO(sys_symmetric), sp.COO(ref))

    _, _, aFP, aFN = confused(sp.COO(s_a2), sp.COO(ref))
    _, _, bFP, bFN = confused(sp.COO(s_b2), sp.COO(ref))

    '''
   
    l = 0
    for k, v in docs.items():

        l+=v

        # get for Aright/Awrong and Bright/Bwrong
        s_a = list(sysA.loc[sysA.case == k].itertuples(index=False))
        s_b = list(sysB.loc[sysB.case == k].itertuples(index=False)) 
        #ref_l = list(ref_df.loc[ref_df.case == k].itertuples(index=False)) 
        #s_a1 = [i for i in a if i.case==k]##list(sysA.loc[sysA.case == docs[n][0]].itertuples(index=False))
        #s_b1 = [i for i in b if i.case==k]# list(sysB.loc[sysB.case == docs[n][0]].itertuples(index=False))
        sys_a1 = label_vector(v, s_a, labels)
        sys_b1 = label_vector(v, s_b, labels)

        sys_a2.append(sys_a1)
        sys_b2.append(sys_b1)

        # intersected list this will give all positive values 
        # NB: intersection only gives positive labels, 
        # since systems do not annotate for negative class
        s_ab1 = list(set(s_a).intersection(set(s_b)))

        sys_ab1 = label_vector(v, s_ab1, labels)
        sys_ab2.append(sys_ab1)
        
        # in one set or other but not both for all negative values
        # to account for all negative values in intersection, need set of all possitives
        s_ab2 = list(set(s_a).symmetric_difference(set(s_b)))
        s_ab1_ab2 = list(set(s_ab1).union(set(s_ab2)))

        sys_ab1_ab2 = label_vector(v, s_ab1_ab2, labels)
        sys_ab1_ab3.append(sys_ab1_ab2)
       
        # a not in b, etc.
        difference_a = list(set(s_a).difference(set(s_b)))
        difference_b = list(set(s_b).difference(set(s_a)))

        diff_a = label_vector(v, difference_a, labels)
        diff_b = label_vector(v, difference_b, labels)

        diff_left.append(diff_a)
        diff_right.append(diff_b)
       
        '''
        difference_left_ref = label_vector(v, list((set(ref_l).difference(set(difference_b)).union(set(difference_a)))), labels)
        difference_right_ref = label_vector(v, list((set(ref_l).difference(set(difference_a)).union(set(difference_b)))), labels)

        diff_left_ref.append(difference_left_ref)
        diff_right_ref.append(difference_right_ref)
        '''
    
    # right/wrong for A and B
    s_a2 = np.concatenate(sys_a2).ravel()
    s_b2 = np.concatenate(sys_b2).ravel()

    inter = np.concatenate(sys_ab2).ravel()
    symmetric_all = np.concatenate(sys_ab1_ab3).ravel()

    difference_A = np.concatenate(diff_left).ravel()
    difference_B = np.concatenate(diff_right).ravel()

    #diff_left_ = np.concatenate(diff_left_ref).ravel()
    #diff_right_ = np.concatenate(diff_right_ref).ravel()

    TP, _, FP, _ = confused(sp.COO(inter), sp.COO(ref))
    _, TN, _, FN = confused(sp.COO(symmetric_all), sp.COO(ref))

    aTP, aTN, aFP, aFN = confused(sp.COO(s_a2), sp.COO(ref))
    bTP, bTN, bFP, bFN = confused(sp.COO(s_b2), sp.COO(ref))

    #lrTP, lrTN, lrFP, lrFN = confused(sp.COO(sp.COO(inter)), sp.COO(ref))

    lTP, _, lFP, _ = confused(sp.COO(difference_A), sp.COO(ref))
    rTP, _, rFP, _ = confused(sp.COO(difference_B), sp.COO(ref))
    #_, lTN, _, lFN = confused(sp.COO(diff_left_), sp.COO(ref))
    #_, rTN, _, rFN = confused(sp.COO(diff_right_), sp.COO(ref))

    b_over_a, a_over_b, mean_comp = complementarity_measures(FN, FP, aFN, aFP, bFN, bFP)

    b_over_a['systems_comp'] = str((r.nameB, r.nameA))
    #b_over_a['B'] = r.nameB    
    #b_over_a['A'] = r.nameA

    a_over_b['system_comp'] = str((r.nameA, r.nameB))
    #a_over_b['B'] = r.nameA    
    #a_over_b['A'] = r.nameB

    mean_comp['system'] = 'mean_comp(' + r.nameA + ',' + r.nameB + ')'

    frames = [out, pd.DataFrame(b_over_a, index=[0])]
    out = pd.concat(frames, ignore_index=True, sort=False) 
    
    frames = [out, pd.DataFrame(a_over_b, index=[0])]
    out = pd.concat(frames, ignore_index=True, sort=False) 

    frames = [out, pd.DataFrame(mean_comp, index=[0])]
    out = pd.concat(frames, ignore_index=True, sort=False) 

    out['length'] = l
    out['a_only_TP'] = lTP
    out['a_only_FP'] = lFP
    #out['lFN'] = lFN
    #out['lTN'] = lTN
    out['b_only_TP'] = rTP
    out['b_only_FP'] = rFP
    #out['rFN'] = rFN
    #out['rTN'] = rTN
    out['abTP'] = TP
    out['abFP'] = FP
    #out['abFN'] = FN
    #out['abTN'] = TN
    out['aTP'] = aTP
    out['aTN'] = aTN
    out['aFP'] = aFP
    out['aFN'] = aFN
    out['bTP'] = bTP
    out['bTN'] = bTN
    out['bFP'] = bFP
    out['bFN'] = bFN

    return out
        
def complementarity_measures(FN, FP, aFN, aFP, bFN, bFP):
    
    compA = 1 - (FN+FP)/(aFN + aFP)
    compB = 1 - (FN+FP)/(bFN + bFP)
    meanComp = s.mean([compA, compB])
   
    r_compA = 1 - (FP)/(aFP)
    r_compB = 1 - (FP)/(bFP)
    meanR = s.mean([r_compA, r_compB])
    
    p_compA = 1 - (FN)/(aFN)
    p_compB = 1 - (FN)/(bFN)
    meanP = s.mean([p_compA, p_compB])
    
    f1_compA = 2*(p_compA*r_compA)/(p_compA + r_compA)
    f1_compB = 2*(p_compB*r_compB)/(p_compB + r_compB)
    meanF1 = s.mean([f1_compA, f1_compB])
    
    if FN > aFN:
        print('a bad n:', FN, aFN)
    if FN > bFN:
        print('b bad n:', FN, bFN)
    if FP > aFP:
        print('a bad p:', FP, aFP)
    if FP > bFP:
        print('b bad p:', FP, bFP)
    if FP > min([aFP, bFP]):
        print('both bad p:', FP, aFP, bFP)
    
    b_over_a = {'test': 'COMP(A, B)', 'max_prop_error_reduction': compA, 'p_comp': p_compA, 'r_comp': r_compA, 'F1-score_comp': f1_compA}
    a_over_b = {'test': 'COMP(B, A)', 'max_prop_error_reduction': compB, 'p_comp': p_compB, 'r_comp': r_compB, 'F1-score_comp': f1_compB}

    mean_complementarity = {'test': 'mean(COMP(B, A),COMP(A, B))', 'max_prop_error_reduction': meanComp, 'mean p_comp': meanP, 'mean r_comp': meanR, 'mean F1-score_comp': meanF1}

    return b_over_a, a_over_b, mean_complementarity


#@ft.lru_cache(maxsize=None)
def cm_dict(ref_only: int, system_only: int, ref_system_match: int, system_n: int, ref_n: int) -> dict:
    """
    Generate dictionary of confusion matrix params and measures
    :params: ref_only, system_only, reference_system_match -> sets
    matches, system_n, reference_n -> counts
    :return: dictionary object
    """

    if ref_only + ref_system_match != ref_n:
        print('ERROR!')
        
    # get evaluation metrics
    F, recall, precision, TP, FP, FN, TP_FN_R, TM  = Metrics(system_only, ref_only, ref_system_match, system_n).get_confusion_metrics()

    d = {
         'F1': F, 
         'precision': precision, 
         'recall': recall, 
         'TP': TP, 
         'FN': FN, 
         'FP': FP, 
         'TP/FN': TP_FN_R,
         'n_gold': ref_n, 
         'n_sys': system_n, 
         'TM': TM
    }

    # generate confidence intervals

    [recall, dr, r_lower_bound, r_upper_bound] = normal_approximation_binomial_confidence_interval(TP, TP + FN)
    [precision, dp, p_lower_bound, p_upper_bound] = normal_approximation_binomial_confidence_interval(TP, TP + FP)
    [f, df, f_lower_bound, f_upper_bound] = f1_score_confidence_interval(recall, precision, dr, dp)

    d['r_upper_bound'] = r_upper_bound
    d['r_lower_bound'] = r_lower_bound

    d['p_upper_bound'] = p_upper_bound
    d['p_lower_bound'] = p_lower_bound

    d['f_upper_bound'] = f_upper_bound
    d['f_lower_bound'] = f_lower_bound

    
    if system_n - FP != TP:
        print('inconsistent system n!')

    return d


@ft.lru_cache(maxsize=None)
def get_metric_data(analysis_type: str, corpus: str):
  
    engine = get_connect(corpus)

    usys_file, ref_table = AnalysisConfig().corpus_config()
    #systems = AnalysisConfig().systems
   
    if corpus != 'medmentions':
        sys_ann = pd.read_csv(analysisConf.data_dir + usys_file, dtype={'note_id': str})
    else:
        sys_ann = pd.read_csv(analysisConf.data_dir + usys_file)
        sys_ann['note_id'] = pd.to_numeric(sys_ann['note_id'])

    sys_ann = sys_ann.rename(columns={"semtype": "semtypes"})
    
    sql = "SELECT start, end, file, semtype FROM " + ref_table #+ " where semtype in('Anatomy', 'Chemicals_and_drugs')"a
    
    ref_ann = pd.read_sql(sql, con=engine)
    ref_ann = ref_ann.drop_duplicates()

    if corpus == 'medmentions':
        cases = set(ref_ann["file"].tolist())
        cases = [int(i) for i in cases]
        ref_ann['file'] = pd.to_numeric(ref_ann['file'])
        sys_ann['note_id'] = pd.to_numeric(sys_ann['note_id'])
        sys_ann = sys_ann.loc[sys_ann.note_id.isin(cases)]

    sys_ann = sys_ann[['begin', 'end', 'score', 'note_id', 'semtypes', 'system']]
    sys_ann = sys_ann.drop_duplicates()

    #ref_ann, _ = reduce_mem_usage(ref_ann)
    #sys_ann, _ = reduce_mem_usage(sys_ann)
 
    
    return ref_ann, sys_ann


def geometric_mean(metrics):
    """
    1. Get rank average of F1, TP/FN, TM
        http://www.datasciencemadesimple.com/rank-dataframe-python-pandas-min-max-dense-rank-group/
        attps://stackoverflow.com/questions/46686315/in-pandas-how-to-create-a-new-column-with-a-rank-according-to-the-mean-values-o?rq=1
    2. Take geomean of rank averages
        https://stackoverflow.com/questions/42436577/geometric-mean-applied-on-row
    """
    
    data = pd.DataFrame() 

    metrics['F1 rank']=metrics['F1'].rank(ascending=0,method='average')
    metrics['TP/FN rank']=metrics['TP/FN'].rank(ascending=0,method='average')
    metrics['TM rank']=metrics['TM'].rank(ascending=0,method='average')
    metrics['Gmean'] = gmean(metrics.iloc[:,-3:],axis=1)

    return metrics  


# confidence intervals
# https://github.com/sousanunes/confidence_intervals.git

def normal_approximation_binomial_confidence_interval(s, n, confidence_level=.95):
    '''Computes the binomial confidence interval of the probability of a success s, 
    based on the sample of n observations. The normal approximation is used,
    appropriate when n is equal to or greater than 30 observations.
    The confidence level is between 0 and 1, with default 0.95.
    Returns [p_estimate, interval_range, lower_bound, upper_bound].
    For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book.'''

    p_estimate = (1.0 * s) / n

    interval_range = norm.interval(confidence_level)[1] * np.sqrt( (p_estimate * (1-p_estimate))/n )

    return p_estimate, interval_range, p_estimate - interval_range, p_estimate + interval_range


def f1_score_confidence_interval(r, p, dr, dp):
    '''Computes the confidence interval for the F1-score measure of classification performance
    based on the values of recall (r), precision (p), and their respective confidence
    interval ranges, or absolute uncertainty, about the recall (dr) and the precision (dp).
    Disclaimer: I derived the formula myself based on f(r,p) = 2rp / (r+p).
    Nobody has revised my computation. Feedback appreciated!'''

    f1_score = (2.0 * r * p) / (r + p)

    left_side = np.abs( (2.0 * r * p) / (r + p) )

    right_side = np.sqrt( np.power(dr/r, 2.0) + np.power(dp/p, 2.0) + ((np.power(dr, 2.0)+np.power(dp, 2.0)) / np.power(r + p, 2.0)) )

    interval_range = left_side * right_side

    return f1_score, interval_range, f1_score - interval_range, f1_score + interval_range


def generate_metrics(analysis_type: str, corpus: str, filter_semtype, semtype = None):
    start = time.time()

    systems = AnalysisConfig().systems
    metrics = pd.DataFrame()

    __, sys_ann = get_metric_data(analysis_type, corpus)
    c = None
    
    for sys in systems:
       
        if filter_semtype and semtype:
            ref_ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
        else:
            ref_ann = get_ref_ann(analysis_type, corpus, filter_semtype)
            
        system_annotations = sys_ann[sys_ann['system'] == sys].copy()

        if filter_semtype:
            st = SemanticTypes([semtype], corpus).get_system_type(sys)

            if st: 
                #system_annotations = sys_ann[sys_ann['semtypes'].isin(st)].copy()
                system_annotations = sys_ann[sys_ann['semtypes'].isin(st)]
        else:
            #system_annotations = sys_ann.copy()
            system_annotations = sys_ann

        if (filter_semtype and st) or filter_semtype is False:
            #system = system_annotations.copy()
            system = system_annotations

            if sys == 'quick_umls':
                system = system[system.score.astype(float) >= .8]

            if sys == 'metamap':
                system = system.fillna(0)
                system = system[system.score.abs().astype(int) >= 800]

            system = system.drop_duplicates()

            ref_ann = ref_ann.rename(index=str, columns={"start": "begin", "file": "case"})
            c = get_cooccurences(ref_ann, system, analysis_type, corpus) # get matches, FN, etc.

            if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
                # get dictionary of confusion matrix metrics
                d = cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n)
                d['system'] = sys

                data = pd.DataFrame(d,  index=[0])
                metrics = pd.concat([metrics, data], ignore_index=True)
                metrics.drop_duplicates(keep='last', inplace=True)
            else:
                print("NO EXACT MATCHES FOR", sys)
            elapsed = (time.time() - start)
            print("elapsed:", sys, elapsed)
   
    if c:
        elapsed = (time.time() - start)
        print(geometric_mean(metrics))

        now = datetime.now()
        timestamp = datetime.timestamp(now)

        file_name = 'metrics_'

        metrics.to_csv(analysisConf.data_dir + corpus + '_' + file_name + analysis_type + '_' + str(timestamp) + '.csv')

        print("total elapsed time:", elapsed) 

# https://stackoverflow.com/questions/44414313/how-to-add-complementary-intervals-in-pandas-dataframe
def complement(df, corpus, system, filter_semtype, semtype=None):
    docs=get_docs(corpus)

    out = pd.DataFrame()

    if filter_semtype:
        a = df.loc[df.semtypes.isin(semtype)]
   
    for k, v in docs.items():
        b = a.loc[a.note_id==k]
        
        d = pd.DataFrame({"begin":[0] + sorted(pd.concat([b.begin, b.end+1])), "end": sorted(pd.concat([b.begin-1, b.end]))+[v]})
        d = d.loc[d.end>d.begin]
        
        e = b.merge(d, how='right', on=['begin', 'end'],indicator=True)
        e = e.loc[e._merge=="right_only"]
        e['note_id'] = k
        e['system'] = system
        
        #if k=='0001112285':
        #    print(e)
        
        del e['_merge']
        out = pd.concat([out, e])
    
    return out
        

@ft.lru_cache(maxsize=None)
def get_sys_data(system: str, analysis_type: str, corpus: str, filter_semtype, semtype = None) -> pd.DataFrame:
   
    _, data = get_metric_data(analysis_type, corpus)

    negate = False
    if 'prime' in system:
        negate = True
        if 'quick' not in system:
            system = system.split('_')[0]
        else:
            system = '_'.join(system.split('_')[0:2])

    out = data.loc[data.system == system]

    
    #if system == 'metamap':
    #out = disambiguate(out)

    if system == 'quick_umls':
        out = out.loc[out.score.astype(float) >= 0.8]
    
    if system == 'metamap':
        out = out.loc[out.score.abs().astype(int) >= 800]

    
    if filter_semtype:
        st = SemanticTypes([semtype], corpus).get_system_type(system)
        print(system, ' ST:', st)
    
    if negate:
        if filter_semtype:
            out = complement(out, corpus, system, filter_semtype, st)
        else:
            out = complement(out, corpus, system, filter_semtype)
    
    if corpus == 'casi':
        cols_to_keep = ['case', 'overlap'] 
        out = out[cols_to_keep].drop_duplicates()
        return out
        
    else:

        if not negate:

            if filter_semtype:
                out = out.loc[out.semtypes.isin(st)]
            else:
                out = out.loc[out.system == system]
            
        if 'entity' in analysis_type:
            cols_to_keep = ['begin', 'end', 'note_id', 'system']
        elif 'cui' in analysis_type:
            cols_to_keep = ['cui', 'note_id', 'system']
        elif 'full' in analysis_type:
            cols_to_keep = ['begin', 'end', 'cui', 'note_id', 'system']
            
        if analysis_type in ['cui','full']:
            out = out.loc[out.cui.str.startswith("C") == True]

        out = out[cols_to_keep]
    
        return out.drop_duplicates(subset=cols_to_keep)

#GENERATE merges

# disambiguate multiple labeled CUIS on span for union 
def disambiguate(df):
    df['length'] = (df.end - df.begin).abs()
    
    cases = set(df['note_id'].tolist())
    
    data = []

    for case in cases:
        
        test = df.loc[df.note_id == case].copy()
        
        for row in test.itertuples():

            iix = pd.IntervalIndex.from_arrays(test.begin, test.end, closed='neither')
            span_range = pd.Interval(row.begin, row.end)
            fx = test[iix.overlaps(span_range)].copy()

            maxLength = fx['length'].max()
            minLength = fx['length'].min()
            maxScore = abs(float(fx['score'].max()))
            minScore = abs(float(fx['score'].min()))

            if len(fx) > 1: 

                # if longer span exists, use as tie-breaker else use score
                if maxLength > minLength:
                    fx = fx[fx['length'] == fx['length'].max()]
                elif maxScore > minScore:
                    fx = fx[fx['score'] == maxScore]

            data.append(fx)

    out = pd.concat(data, axis=0)
  
    # Remaining ties: randomly reindex to keep random row when dropping duplicates: https://gist.github.com/cadrev/6b91985a1660f26c2742
    out.reset_index(inplace=True)
    out = out.reindex(np.random.permutation(out.index))
    #out = out.drop_duplicates(['begin', 'end', 'note_id', 'length', 'cui'])
    out = out.drop_duplicates(['begin', 'end', 'note_id', 'length'])
    
    return out  

# majority vote -> plurality for entity only, with ties winning
def vote(df, systems):
   
    key = ["begin", "end", "case"]
    n = len(systems) // 2

    mask = df.groupby(key)["system"].count() >= n

    out = df.set_index(key)[mask] \
            .reset_index() \
            .drop(columns="system") \
            .drop_duplicates()
    
    return out 


@ft.lru_cache(maxsize=None)
def process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype = None):

    """
    Recursively evaluate parse tree, 
    with check for existence before build
       :param sentence: to process
       :return class of merged annotations, boolean operated system df 
    """
    
    class Results(object):
        def __init__(self):
            self.results = set()
            self.system_merges = pd.DataFrame()
            self.ref = np.array(list()) # return empty array 
            
    r = Results()
    
    if 'entity' in analysis_type and corpus != 'casi': 
        cols_to_keep = ['begin', 'end', 'note_id'] # entity only
    elif 'full' in analysis_type: 
        cols_to_keep = ['cui', 'begin', 'end', 'note_id'] # entity only
    elif 'cui' in analysis_type:
        cols_to_keep = ['cui', 'note_id'] # entity only
    elif corpus == 'casi':
        cols_to_keep = ['case', 'overlap']
    
    def evaluate(parseTree):
        oper = {'&': op.and_, '|': op.or_, '^': op.xor}
       
        if parseTree:
            leftC = gevent.spawn(evaluate, parseTree.getLeftChild())
            rightC = gevent.spawn(evaluate, parseTree.getRightChild())
            
            if leftC.get() is not None and rightC.get() is not None:
                system_query = pd.DataFrame()
                fn = oper[parseTree.getRootVal()]

                if isinstance(leftC.get(), str):
                    # get system as leaf node 
                    if filter_semtype:
                        left_sys = get_sys_data(leftC.get(), analysis_type, corpus, filter_semtype, semtype)
                    else:
                        left_sys = get_sys_data(leftC.get(), analysis_type, corpus, filter_semtype)
                
                elif isinstance(leftC.get(), pd.DataFrame):
                    l_sys = leftC.get()
                
                if isinstance(rightC.get(), str):

                    # get system as leaf node
                    if filter_semtype:
                        right_sys = get_sys_data(rightC.get(), analysis_type, corpus, filter_semtype, semtype)
                    else:
                        right_sys = get_sys_data(rightC.get(), analysis_type, corpus, filter_semtype)
                    
                elif isinstance(rightC.get(), pd.DataFrame):
                    r_sys = rightC.get()
                    
                if fn == op.or_:
                    if isinstance(leftC.get(), str) and isinstance(rightC.get(), str):
                        frames = [left_sys, right_sys]

                    elif isinstance(leftC.get(), str) and isinstance(rightC.get(), pd.DataFrame):
                        frames = [left_sys, r_sys]

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), str):
                        frames = [l_sys, right_sys]

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), pd.DataFrame):
                        frames = [l_sys, r_sys]
                    
                    df = pd.concat(frames,  ignore_index=True)

                if fn == op.and_:
                    if isinstance(leftC.get(), str) and isinstance(rightC.get(), str):
                        if not left_sys.empty and not right_sys.empty:
                            df = left_sys.merge(right_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), str) and isinstance(rightC.get(), pd.DataFrame):
                        if not left_sys.empty and not r_sys.empty:
                            df = left_sys.merge(r_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), str):
                        if not l_sys.empty and not right_sys.empty:
                            df = l_sys.merge(right_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), pd.DataFrame):
                        if not l_sys.empty and not r_sys.empty:
                            df = l_sys.merge(r_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                if fn == op.xor:

                    if isinstance(leftC.get(), str) and isinstance(rightC.get(), str):
                        if not left_sys.empty and not right_sys.empty:
                            df = left_sys.merge(right_sys, on=cols_to_keep, how='outer', indicator=True)
                            df = df[cols_to_keep].loc[(df._merge=='left_only')|(df._merge=='right_only')].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), str) and isinstance(rightC.get(), pd.DataFrame):
                        if not left_sys.empty and not r_sys.empty:
                            df = left_sys.merge(r_sys, on=cols_to_keep, how='outer', indicator=True)
                            df = df[cols_to_keep].loc[(df._merge=='left_only')|(df._merge=='right_only')].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), str):
                        if not l_sys.empty and not right_sys.empty:
                            df = l_sys.merge(right_sys, on=cols_to_keep, how='outer', indicator=True)
                            df = df[cols_to_keep].loc[(df._merge=='left_only')|(df._merge=='right_only')].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), pd.DataFrame):
                        if not l_sys.empty and not r_sys.empty:
                            df = l_sys.merge(r_sys, on=cols_to_keep, how='outer', indicator=True)
                            df = df[cols_to_keep].loc[(df._merge=='left_only')|(df._merge=='right_only')].drop_duplicates(subset=cols_to_keep)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                # get combined system results
                r.system_merges = df
                
                if len(df) > 0:
                    system_query = system_query.append(df)
                else:
                    print('wtf!')
                    
                return system_query
            else:
                return parseTree.getRootVal()
    
    if sentence.n_or > 0 or sentence.n_and > 0 or sentence.x_or > 0:
        evaluate(pt)  
    
    # trivial case
    elif sentence.n_or == 0 and sentence.n_and == 0 and sentence.x_or == 0:
      
        if filter_semtype:
            r.system_merges = get_sys_data((sentence.sentence), analysis_type, corpus, filter_semtype, semtype)
        else:
            r.system_merges = get_sys_data((sentence.sentence), analysis_type, corpus, filter_semtype)
        
    return r


class Results(object):
    def __init__(self):
        self.ref = np.array(list())
        self.sys = np.array(list())
        self.df = pd.DataFrame()
        self.labels = []

"""
Incoming Boolean sentences are parsed into a binary tree.

Test expressions to parse:
sentence = '((((A&B)|C)|D)&E)'
sentence = '(E&(D|(C|(A&B))))'
sentence = '(((A|(B&C))|(D&(E&F)))|(H&I))'

"""
# build parse tree from passed sentence using grammatical rules of Boolean logic
@ft.lru_cache(maxsize=None)
def buildParseTree(fpexp):
    """
       Iteratively build parse tree from passed sentence using grammatical rules of Boolean logic
       :param fpexp: sentence to parse
       :return eTree: parse tree representation
       Incoming Boolean sentences are parsed into a binary tree.
       Test expressions to parse:
       sentence = '(A&B)'
       sentence = '(A|B)'
       sentence = '((A|B)&C)'
       
    """
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree

    for i in fplist:

        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in ['&', '|', '^', ')']:
            currentTree.setRootVal(i)
            parent = pStack.pop()
            currentTree = parent
        elif i in ['&', '|', '^']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError

    l = []

    out=eTree.preorder(l)

    return eTree

def build_sentence(sentence):
    sentence = sentence.replace('(~(A))','A_prime'). \
            replace('(~(B))','B_prime'). \
            replace('(~(C))','C_prime'). \
            replace('(~(D))','D_prime'). \
            replace('(~(E))','E_prime'). \
            replace('~(A)', 'A_prime'). \
            replace('~(B)', 'B_prime'). \
            replace('~(C)','C_prime'). \
            replace('~(D)','D_prime'). \
            replace('~(E)','E_prime'). \
            replace('(A)','A'). \
            replace('(B)','B'). \
            replace('(C)','C'). \
            replace('(D)','D'). \
            replace('(E)','E')

    for c in permutations(['A','B','C','D','E'],2):
        if c[0]+'&'+c[1] in sentence:
            sentence=sentence.replace('(('+c[0]+'&'+c[1]+'))','('+c[0]+'&'+c[1]+')')
        if c[0]+'|'+c[1] in sentence:
            sentence=sentence.replace('(('+c[0]+'|'+c[1]+'))','('+c[0]+'|'+c[1]+')')
    
    return  sentence.replace('A','biomedicus'). \
                replace('B','clamp'). \
                replace('C','ctakes'). \
                replace('D','metamap'). \
                replace('E','quick_umls') 

def clean_sentence(sentence):
    sentence = sentence.replace('(~(A))','A_prime'). \
            replace('(~(B))','B_prime'). \
            replace('(~(C))','C_prime'). \
            replace('(~(D))','D_prime'). \
            replace('(~(E))','E_prime'). \
            replace('~(A)', 'A_prime'). \
            replace('~(B)', 'B_prime'). \
            replace('~(C)','C_prime'). \
            replace('~(D)','D_prime'). \
            replace('~(E)','E_prime'). \
            replace('(A)','A'). \
            replace('(B)','B'). \
            replace('(C)','C'). \
            replace('(D)','D'). \
            replace('(E)','E')

    for c in permutations(['A','B','C','D','E'],2):
        if c[0]+'&'+c[1] in sentence:
            sentence=sentence.replace('(('+c[0]+'&'+c[1]+'))','('+c[0]+'&'+c[1]+')')
        print(c[0]+'|'+c[1])
        if c[0]+'|'+c[1] in sentence:
            sentence=sentence.replace('(('+c[0]+'|'+c[1]+'))','('+c[0]+'|'+c[1]+')')
    
    return sentence

@ft.lru_cache(maxsize=None)
def make_parse_tree(payload):
    """
    Ensure data to create tree are in correct form
    :param sentence: sentence to preprocess
    :return pt, parse tree graph
            sentence, processed sentence to build tree
            a: order
    """
    def preprocess_sentence(sentence):
        #sentence = build_sentence(sentence)

        # prepare statement for case when a boolean AND/OR is given
        sentence = sentence.replace('(', ' ( '). \
                replace(')', ' ) '). \
                replace('&', ' & '). \
                replace('|', ' | '). \
                replace('^', ' ^ '). \
                replace('  ', ' ')
        return sentence

    print(payload)
    sentence = preprocess_sentence((payload))
    print('Processing sentence:', sentence)
    
    pt = buildParseTree(sentence)
    
    return pt

class Sentence(object):
    '''
    Details about boolean expression -> number operators and expression
    '''
    def __init__(self, sentence):
        self = self
        self.n_and = sentence.count('&')
        self.n_or = sentence.count('|')
        self.x_or = sentence.count('^')
        self.sentence = sentence

@ft.lru_cache(maxsize=None)
def get_docs(corpus):
   
    engine = get_connect(corpus)
    
    # KLUDGE!!!
    if corpus == 'ray_test':
        corpus = 'fairview'
    
    if corpus == "medmentions":
        sql = 'select distinct note_id, len_doc from sofas where test=1 and corpus = (?) order by note_id'
    else:
        sql = 'select distinct note_id, sofa from sofas where corpus = (?) order by note_id'
    
    df = pd.read_sql(sql, params={corpus, }, con=engine)
    df.drop_duplicates()

    if corpus != "medmentions":
        df['len_doc'] = df['sofa'].apply(len)
    else:
        df['note_id'] = pd.to_numeric(df['note_id'])
    
    subset = df[['note_id', 'len_doc']]
    return subset.set_index('note_id')['len_doc'].to_dict()


def set_labels(analysis_type, df):
    if analysis_type == 'entity':   
        df["label"] = 'concept'
    elif analysis_type in ['cui','full']:
        df["label"] = df["value"]
    
    return df

@ft.lru_cache(maxsize=None)
def get_ref_ann(analysis_type, corpus, filter_semtype, semtype = None):
    
    if filter_semtype:
        if ',' in semtype:
            semtype = semtype.split(',')
        else:
            semtype = [semtype]
        
    ann, _ = get_metric_data(analysis_type, corpus)
    ann = ann.rename(columns={"start": "begin", "file": "case", "semgroup": "semtype"})
    
    if filter_semtype:
        ann = ann.loc[ann.semtype.isin(semtype)]
    
    ann = set_labels(analysis_type, ann)
        
    if analysis_type == 'entity':
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'cui':
        cols_to_keep = ['case', 'label']
    elif analysis_type == 'full':
        cols_to_keep = ['begin', 'end', 'case', 'label']
        
    return ann[cols_to_keep]


#@ft.lru_cache(maxsize=None)
def get_sys_ann(analysis_type, r):
    sys = r.system_merges

    sys = sys.rename(columns={"note_id": "case", "cui": "value"})
    
    sys = set_labels(analysis_type, sys)
    
    if analysis_type in ['entity', 'full']:
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'cui':
        cols_to_keep = ['case', 'label']
    
    if len(sys) == 0:
        sys = pd.DataFrame(columns=cols_to_keep)
        print('WTF!')

    return sys[cols_to_keep]


@ft.lru_cache(maxsize=None)
def get_metrics(boolean_expression: str, analysis_type: str, corpus: str, run_type: str, filter_semtype, semtype = None):
    """
    Traverse binary parse tree representation of Boolean sentence
        :params: boolean expression in form of '(<annotator_engine_name1><boolean operator><annotator_engine_name2>)'
                 analysis_type (string value of: 'entity', 'cui', 'full') used to filter set of reference and system annotations 
        :return: dictionary with values needed for confusion matrix
    """     
    
    sentence = Sentence(boolean_expression)   
    pt = make_parse_tree(sentence.sentence)
    
    if filter_semtype:
        r = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype)
    else:
        r = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype)
        
    # vectorize merges using i-o labeling
    if run_type == 'overlap':
        if filter_semtype:
            ((TP, TN, FP, FN),(macro_p,macro_r,macro_f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype, semtype)
        else:
            ((TP, TN, FP, FN),(macro_p,macro_r,macro_f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype)
        
        # TODO: validate against ann1/sys1 where val = 1
        # total by number chars
        system_n = TP + FP
        reference_n = TP + FN

        if analysis_type == 'entity':
            d = cm_dict(FN, FP, TP, system_n, reference_n)
            
        else:
            d = {}
            d['F1'] = 0
            d['precision'] = 0 
            d['recall'] = 0
            d['TP/FN'] = 0
            d['TM'] = 0
            
        d['TN'] = TN
        d['macro_p'] = macro_p
        d['macro_r'] = macro_r
        d['macro_f1'] = macro_f1
        
        # return full metrics
        return d

    elif run_type == 'exact':
        # total by number spans
        
        if filter_semtype:
            ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
        else: 
            ann = get_ref_ann(analysis_type, corpus, filter_semtype)
        
        c = get_cooccurences(ann, r.system_merges, analysis_type, corpus) # get matches, FN, etc.

        if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
            # get dictionary of confusion matrix metrics
            d = cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n)
        else:
            d = None
            
        return d


# get list of systems with a semantic type in grouping
def get_valid_systems(systems, semtype):
    return [system_semtype_check(sys, semtype, corpus) for sys in systems if system_semtype_check(sys, semtype, corpus)]


# permute system combinations and evaluate system merges for performance
def run_ensemble(systems, analysis_type, corpus, filter_semtype, semtype=None):
    metrics = pd.DataFrame()
    
    # pass single system to evaluate
    #if expression_type == 'single':

    if len(systems) == 1: 
        
        for system in systems:
            if filter_semtype:
                d = get_metrics(system, analysis_type, corpus, run_type, filter_semtype, semtype)
            else:
                d = get_metrics(system, analysis_type, corpus, run_type, filter_semtype)
            d['merge'] = system
            d['n_terms'] = 1

            frames = [metrics, pd.DataFrame(d, index=[0])]
            metrics = pd.concat(frames, ignore_index=True, sort=False)

    else:
        expressions = get_ensemble_combos(systems)
               
        for e in expressions:
            if filter_semtype:
                d = get_metrics(e, analysis_type, corpus, run_type, filter_semtype, semtype)
            else:
                d = get_metrics(e, analysis_type, corpus, run_type, filter_semtype)

            d['merge'] = e
            n = e.count('&') + e.count('|') + 1 
            d['n_terms'] = n
            
            frames = [metrics, pd.DataFrame(d, index=[0])]
            metrics = pd.concat(frames, ignore_index=True, sort=False) 
        
    return metrics

# write to file
def generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype, semtype = None):
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    
    file_name = corpus + '_all_'
   
    # drop exact matches:
    metrics = metrics.drop_duplicates()
    
    if ensemble_type == 'merge':
        metrics = metrics.sort_values(by=['n_terms', 'merge'])
        file_name += 'merge_'
    elif ensemble_type == 'vote':
        file_name += 'vote_'
    
    #metrics = metrics.drop_duplicates(subset=['TP', 'FN', 'FP', 'n_sys', 'precision', 'recall', 'F', 'TM', 'TP/FN', 'TM', 'n_terms'])

    file = file_name + analysis_type + '_' + run_type +'_'
    
    if filter_semtype:
        file += semtype
        
    if ensemble_type != 'vote':
        geometric_mean(metrics).to_csv(analysisConf.data_dir + file + str(timestamp) + '.csv')
        print(geometric_mean(metrics))
    
    
# control ensemble run
def ensemble_control(systems, analysis_type, corpus, run_type, filter_semtype, semtypes=None):
    if filter_semtype:
        for semtype in semtypes:
            test = get_valid_systems(systems, semtype)
            print('SYSTEMS FOR SEMTYPE', semtype, 'ARE', test)
            metrics = run_ensemble(test, analysis_type, corpus, filter_semtype, semtype)
            generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype, semtype)
    else:
        metrics = run_ensemble(systems, analysis_type, corpus, filter_semtype)
        generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype)


# ad hoc query for performance evaluation
def get_merge_data(boolean_expression: str, analysis_type: str, corpus: str, run_type: str, filter_semtype, metrics = True, semtype = None):
    """
    Traverse binary parse tree representation of Boolean sentence
        :params: boolean expression in form of '(<annotator_engine_name1><boolean operator><annotator_engine_name2>)'
                 analysis_type (string value of: 'entity', 'cui', 'full') used to filter set of reference and system annotations 
        :return: dictionary with values needed for confusion matrix
    """


    sentence = Sentence(build_sentence(boolean_expression))   

    # kludge for deMorgan eqiuvalence
    if '~' in sentence.sentence:
        d =  {}
        d['F1'] = 0
        d['precision'] = 0
        d['recall'] = 0
        return d

    if sentence.n_and > 0 or sentence.n_or > 0 or sentence.x_or > 0:
        sentence.sentence = '(' + sentence.sentence + ')'

    pt = make_parse_tree(sentence.sentence)

    results = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype)

    #print(results.system_merges)

    if metrics:
        if run_type == 'overlap':
            if filter_semtype:
                ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(results, analysis_type, corpus, filter_semtype, semtype)
            else:
                 ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(results, analysis_type, corpus, filter_semtype)

            # TODO: validate against ann1/sys1 where val = 1
            # total by number chars
            system_n = TP + FP
            reference_n = TP + FN

            d = cm_dict(FN, FP, TP, system_n, reference_n)
            d['TN'] = TN
            d['corpus'] = corpus

            print(d)
            
        elif run_type == 'exact':
            c = get_cooccurences(ann, r.system_merges, analysis_type, corpus) # get matches, FN, etc.

            if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
                # get dictionary of confusion matrix metrics
                d = cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n)

                print('cm', d)
        else:
            pass
        
        # get matched data from merge
        return d # merge_eval(reference_only, system_only, reference_system_match, system_n, reference_n)

    else:
        return results.system_merges

@ft.lru_cache(maxsize=None)
def get_reference_vector(analysis_type, corpus, filter_semtype, semtype = None):
    ref = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
    
    if 'entity' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif 'cui' in analysis_type: 
        cols_to_keep = ['case', 'label']
    elif 'full' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'case', 'label']
     
    labels = get_labels(analysis_type, corpus, filter_semtype, semtype)
    
    ref = ref[cols_to_keep].drop_duplicates(subset=cols_to_keep)
    test = vectorized_annotations(ref, analysis_type, labels)
    
    if analysis_type != 'cui':
        ref = np.asarray(flatten_list(test)) 
    else: 
        ref = np.asarray(test)

    return ref

def get_majority_sys(systems, analysis_type, corpus, filter_semtype, semtype):
    if 'entity' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'case', 'label', 'system']
    elif 'cui' in analysis_type: 
        cols_to_keep = ['case', 'label']
    elif 'full' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'case', 'label']

    out = pd.DataFrame()
  
    for system in systems:
        print(system, semtype)
        sys_ann = get_sys_data(system, analysis_type, corpus, filter_semtype, semtype)
        df = sys_ann.copy()
        df = df.dropna()
        
        df = df.rename(index=str, columns={"note_id": "case", "cui": "value"})
        df = set_labels(analysis_type, df)
     
        frames = [out, df]
        out = pd.concat(frames)
        out = out[cols_to_keep]

    return vote(out, systems) 

    
def majority_overlap_vote_out(ref, vote, corpus, semtype = None):   
    class Results(object):
        def __init__(self):
            self.ref = np.array(list())
            self.system_merges = pd.DataFrame()

    r = Results()
    
    r.ref = ref
    r.system_merges = vote
   
    ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype, semtype)
    
    if analysis_type == 'entity':
        system_n = TP + FP
        reference_n = TP + FN

        d = cm_dict(FN, FP, TP, system_n, reference_n)

        d['TN'] = TN
        d['corpus'] = corpus
        print(d)
    
    else:
        d = {}
        d['precision'] = p
        d['recall'] = r
        d['F1'] = f1
    
    metrics = pd.DataFrame(d, index=[0])
    
    return metrics
    
# control vote run
def majority_vote(test, analysis_type, corpus, run_type, filter_semtype, semtype = None):

    if filter_semtype:
        
        #for semtype in semtypes:
        #test = get_valid_systems(systems, semtype)
        print('SYSYEMS FOR SEMTYPE', semtype, 'ARE', test)
        
        if run_type == 'overlap' and len(test) > 1:
            ref = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)
            vote = get_majority_sys(test, analysis_type, corpus, filter_semtype, semtype)
            #labels = get_labels(analysis_type, corpus, filter_semtype, semtype)
    
            out = majority_overlap_vote_out(ref, vote, corpus, semtype)
       
            out['semgroup'] = semtype
            out['systems'] = ','.join(test)
            #generate_ensemble_metrics(out, analysis_type, corpus, ensemble_type, filter_semtype, semtype)
            #frames = [metrics, out]
            #metrics = pd.concat(frames, ignore_index=True, sort=False)
                
    else:
        if run_type == 'overlap':
            ref = get_reference_vector(analysis_type, corpus, filter_semtype)
            vote = get_majority_sys(systems, analysis_type, corpus, filter_semtype)
            #labels = get_labels(analysis_type, corpus, filter_semtype)
        
            out = majority_overlap_vote_out(ref, vote, corpus)
            
        out['systems'] = ','.join(systems)
        #generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype)
    
    return out

#http://stackoverflow.com/questions/4284991/parsing-nested-parentheses-in-python-grab-content-by-level
def parenthetic_contents(string):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), string[start + 1: i])

# get all baby ensembles for given input
def get_baby_ensembles(sentence):
    comp=[]
    separator=["^","|","&"]
    for l in list(parenthetic_contents(sentence)):
        if len(l[1])==3:
            print('a', l[1][0], l[1][-1])
            comp.append((l[1][0], l[1][-1], l[1][1], l[0]))
            #comp.append((l[1][0], l[1]))
            #comp.append((l[1][-1], l[1]))
        elif l[1][-1] != ')' and len(l[1]) > 3:
            if l[1][-2] in separator:

                test = l[1][0:-2] + l[1][-2].replace(l[1][-2], " ") + l[1][-1:]
                print('b', l[1][-1:], l[1][0:-2])
                comp.append((l[1][-1:], l[1][0:-2], l[1][-2], l[0]))
        else:
            #for i in list(parenthetic_contents(l[1])):
            print('c', [i for i in list(parenthetic_contents(l[1])) if i[0]==0])
            a='(' + [i for i in list(parenthetic_contents(l[1])) if i[0]==0][1][1] + ')'
            b='(' + [i for i in list(parenthetic_contents(l[1])) if i[0]==0][0][1] + ')'
            idx_a=sentence.index(a)
            idx_b=sentence.index(b)
            if a < b:
                idx = idx_b
            else:
                idx = idx_a
            comp.append((a, b, sentence[idx-1:idx], (l[0])))

    return comp

# generate complementarity measu<F3>res for given input
def ad_hoc_complementarity(sentence, mtype):

    comp = get_baby_ensembles(sentence[1])
    # generate complementarity 
    for c in comp:
        # no semtype filtering
        main(None, c, mtype, sentence[0], sentence[1], True, False)
        #main('Disorders,Sign_Symptom', c, True, True)

# generate complementarity measu<F3>res for given input
def ad_hoc_complementarity_st(sentence, mtype, st):

    comp = get_baby_ensembles(sentence[1])
    for c in comp:
        # no semtype filtering
        main(st, c, mtype, sentence[0], sentence[1], True, True)
        #main('Disorders,Sign_Symptom', c, True, True)

def ad_hoc_complementarity_test(st, sentence):

    main(st, sentence, 'F1', .9, 'A^C', True, True)
        #main('Disorders,Sign_Symptom', c, True, True)


def get_ensemble_combos_measure_st():
    
    print(semtypes, systems)

    for semtype in semtypes:

        test = get_valid_systems(systems, semtype)
        replacements = {'biomedicus':'A', 'clamp':'B', 'ctakes':'C', 'metamap':'D', 'quick_umls':'E'}
        replacer = replacements.get  # For faster gets.

        print([replacer(n, n) for n in test])

        get_ensemble_combos_measure([replacer(n, n) for n in test], filter_semtype, semtype)

def get_ann_vectors(analysis_type, corpus, filter_semtype, semtype):
    class Results(object):
        def __init__(self):
            self.system_merges = pd.DataFrame()

    r = Results()

    # TODO for system and reference data annotation sets:
    ann_dict = {}
    labels = get_labels(analysis_type, corpus, filter_semtype, semtype)

    system_annotations = 'analytical_disambiguated_i2b2_full.csv'
    gs_ann = 'i2b2_gs.csv'
    gs_len = 'i2b2_doc_len.csv'
    cols_to_keep = ['begin', 'end', 'case', 'label']

    docs = get_docs(corpus)

    ########### GS vector here:
    #ref_ann=pd.read_csv(data_dir + gs_ann, dtype={'file': str})
    #ref_ann = ref_ann.rename(columns={"start": "begin", "file": "case"}) # `note_id` may be `file`
    #ref_ann = set_labels(ref_ann)
    #ref_ann = ref_ann[cols_to_keep]

    ann_dict['reference'] = list(get_reference_vector(analysis_type, corpus, filter_semtype, semtype))
    #ann_dict['reference'] = vectorized_annotations(ref_ann, 'entity', labels)
    ann_dict['reference'] = [int(i) for i in ann_dict['reference']]
    ########### system vectors here:

    test = get_valid_systems(systems, semtype)

    for system in test:

        sys = get_sys_data(system, analysis_type, corpus, filter_semtype, semtype)

        r.system_merges = sys
        docs = get_docs(corpus)

        #r.system_merges = sys.loc[sys.system==sysstem]
        sys = get_sys_ann(analysis_type, r)

        sys2 = []
        s2 = []

        #if analysis_type != 'cui':
        #    s = list(sys.itertuples(index=False))
        
        for k, v in docs.items():
            s1 = list(sys.loc[sys.case == k].itertuples(index=False))
            #s1 = [i for i in s if i.case==k] # list(sys.loc[sys.case == docs[n][0]].itertuples(index=False))
            sys1 = label_vector(v, s1, labels)
            sys2.append(sys1)
        
        ann_dict[system] = flatten_list(sys2)
        ann_dict[system] = [int(i) for i in ann_dict[system]]

    return ann_dict

def get_ensemble_combos_measure_no_st():
    
    print(systems)

    #for semtype in semtypes:

    replacements = {'biomedicus':'A', 'clamp':'B', 'ctakes':'C', 'metamap':'D', 'quick_umls':'E'}
    replacer = replacements.get  # For faster gets.

    print([replacer(n, n) for n in systems])

    get_ensemble_combos_measure([replacer(n, n) for n in systems], filter_semtype)

def get_ensemble_combos_measure(systems, filter_semtype, semtype=None):
# get a ensemble combinations

    def length_score(expr):
        return len(expr)

    def wrap_ad_hoc_measure():
         def retval(expr):
             return ad_hoc_measure(expr, analysis_type, corpus, 'F1', filter_semtype, semtype)
         return retval

    def get_result():
        result = cs.get_best_ensembles(score_method=wrap_ad_hoc_measure(),
                        names=systems,
                        used_binops=['&', '|'],
                        used_unops=['~'],
                        minimum_increase=0.1)

        ''''
        result = cs.get_best_ensembles(score_method=wrap_ad_hoc_measure(),
                        names=systems,
                        operators=['&', '|'],
                        minimum_increase=-1) # fv all: 0.06 '((A|B)&C)'
                                           # mipacq: '((((D|A)&E)&B)|C)'
        '''

        return result

    return [r[0] for r in get_result()]

def get_ensemble_combos(systems=['biomedicus', 'clamp', 'ctakes', 'metamap', 'quick_umls']): 
    # get a ensemble combinations

    def length_score(expr):
        return len(expr)

    def get_result():
        result = cs.get_best_ensembles(score_method=length_score, 
                           names=systems,
                           operators=['&', '|'],
                           order=None,
                           minimum_increase=0)

        return result
    
    return [r[0] for r in get_result()]


def get_ensemble_pairs(systems=['biomedicus', 'clamp', 'ctakes', 'metamap', 'quick_umls']): 
    results = get_ensemble_combos(systems)

    # return True if number of operators found in expression
    def n_operators(x, y, n, m, o):

        s1 = sum([x.count(o[0]), x.count(o[1])])
        s2 = sum([y.count(o[0]), y.count(o[1])])
        if s1 == n and s2 == m:
          return True
        else:
          return False
    
    # return length of overlap
    def overlap(expr1, expr2, operators):

        s1 = expr1
        s2 = expr2

        for o in operators:
            s1 = s1.replace(o, ' ')
            s2 = s2.replace(o, ' ')

        s1 = s1.replace('(', '').replace(')','').split()
        s2 = s2.replace('(', '').replace(')','').split()

        return len(set(s1).intersection(set(s2)))

    operators = ['&', '|']

    # parse out into all paired combos for comparison
    test = [r for r in results if sum([r.count('|'), r.count('&')]) < len(systems)-1]
    out = list(combinations(test, 2))

    return [o for i in range(len(systems)) for j in range(len(systems)) if i + j < len(systems)  
            for o in out if n_operators(o[0], o[1], i, j, operators) and overlap(o[0], o[1], operators) == 0]

# use with combo_searcher
def ad_hoc_measure(statement, analysis_type, corpus, measure, filter_semtype, semtype = None):
   
    # kludge in get_merge_data
    #if statement[0]=='~' and '^' in statement:
    #    print('cannot:', statement)
    #    return 0

    d = get_merge_data(statement, analysis_type, corpus, run_type, filter_semtype, True, semtype)

    if filter_semtype:
        d['semtype'] = semtype

    now = datetime.now()
    now = now.strftime("%m-%d-%Y")
 
    if filter_semtype:
        file_name = corpus + '_' + measure + '_st_' + now + '.csv'
    else:
        file_name = corpus + '_' + measure + '_' + now + '.csv'

    if measure in ['F1', 'precision', 'recall']:
        
        d['system'] = statement
        df = pd.DataFrame(d,  index=[0])
        
        file_path = Path(data_out / file_name)
        if file_path.exists():
            df.to_csv(file_path, header=False, mode='a')
        else:
            df.to_csv(file_path, header=True, mode='w')

        #with open(data_out / file_name, 'a') as f:

            #d['system'] = statement
            #df = pd.DataFrame(d,  index=[0])
            
            #df.to_csv(data_out / file_name, mode='a', header=f.tell()==0)
        
        return d[measure]
    else:
        print('Invalid measure!')

def ad_hoc_sys(statement, analysis_type, corpus, filter_semtype, metrics = False, semtype = None):
    sys = get_merge_data(statement, analysis_type, corpus, run_type, filter_semtype, metrics, semtype)

    return sys


#def main_test(systems, corpus, filter_semtype, semtypes = None):
def main_test():

    out = pd.DataFrame()
    for semtype in semtypes:
        for i in range(1, len(systems) + 1):
            for s in combinations(systems, i):
                test = get_valid_systems(s, semtype)

                print('SYSYEMS FOR SEMTYPE', semtype, 'ARE', test)
                if len(test) == 1:
                   d = ad_hoc_sys(test[0], analysis_type, corpus, filter_semtype, True, semtype)
                   metrics = pd.DataFrame(d, index=[0])

                   metrics['systems'] = test[0]
                   metrics['semgroup'] = semtype
                    
                elif len(test) > 1: 
                    if filter_semtype:
                        metrics = majority_vote(test, analysis_type, corpus, run_type, filter_semtype, semtype)
                    else:
                        metrics = majority_vote(test, analysis_type, corpus, run_type, filter_semtype)
                        
                frames = [out, metrics]
                out = pd.concat(frames, ignore_index=True, sort=False)
                
        now = datetime.now()
        timestamp = datetime.timestamp(now)


    file = corpus + '_vote_' + analysis_type + '_' + str(filter_semtype) + '_' + str(timestamp) +'.csv'
    out.drop_duplicates().to_csv(data_out / file)

def get_ensemble_combos(systems=['A','B','C','D','E']):
# get a ensemble combinations

    def length_score(expr):
        return len(expr)

    def wrap_ad_hoc_measure():
         def retval(expr):
             return ad_hoc_measure(expr, 'entity', 'fairview', 'F1', True, 'Finding')
         print(retval)
         return retval

    def get_result():
        result = cs.get_best_ensembles(score_method=wrap_ad_hoc_measure(),
                        names=systems,
                        used_binops=[andop, orop],
                        used_unops=[notop],
                        minimum_increase=-1)

        return result

    return [r[0] for r in get_result()]

    
def main(semtype, c, measure_type=None, measure=None, sentence=None, out_file=False, filter_semtype=True):
   
    #now = datetime.now()
    #timestamp = datetime.timestamp(now)
    #file_out = 'complement_' + corpus + '_filter_semtype_' + str(filter_semtype) + '_.csv'
    class Results(object):
        def __init__(self):
            self.sysA = pd.DataFrame() 
            self.sysB = pd.DataFrame() 
            self.nameA = ''
            self.nameB = ''
        
    r = Results()

    #for semtype in semtypes:
   # test = get_valid_systems(systems, semtype)
    #expressions = get_ensemble_pairs(test)

    #print('SYSTEMS FOR SEMTYPE', semtype)
        

    #for c in expressions:

    print('complementarity between:', c)
    
    r.nameA = c[0]
    r.nameB = c[1]

    r.sysA = ad_hoc_sys(c[0], analysis_type, corpus, filter_semtype, False, semtype) # c[0]
    r.sysB = ad_hoc_sys(c[1], analysis_type, corpus, filter_semtype, False, semtype) # c[1]

    out = vectorized_complementarity(r, analysis_type, corpus, filter_semtype, semtype)


    # get individual left and right evaluation metric prior to merging them
    left_ = ad_hoc_sys(c[0], analysis_type, corpus, filter_semtype, True, semtype)
    right_ = ad_hoc_sys(c[1], analysis_type, corpus, filter_semtype, True, semtype)
  
    left_['merge'] = c[0] 
    left_['p_ci'] = str((left_['p_lower_bound'], left_['p_upper_bound'])) 
    left_['r_ci'] = str((left_['r_lower_bound'], left_['r_upper_bound']))  
    left_['f_ci'] = str((left_['f_lower_bound'], left_['f_upper_bound'])) 

    right_['merge'] = c[1]
    right_['p_ci'] = str((right_['p_lower_bound'], right_['p_upper_bound'])) 
    right_['r_ci'] = str((right_['r_lower_bound'], right_['r_upper_bound']))  
    right_['f_ci'] = str((right_['f_lower_bound'], right_['f_upper_bound'])) 

    out['n_ref'] = left_['n_gold']
    out['TP_left'] = left_['TP'] 
    out['TN_left'] = left_['TN'] 
    out['FP_left'] = left_['FP'] 
    out['FN_left'] = left_['FN'] 
    out['n_sys_left'] = left_['n_sys']
    out['precision_left'] = left_['precision']
    out['recall_left'] = left_['recall']
    out['f1_left'] = left_['F1']
    out['merge_left'] = left_['merge']
    out['p_ci_left'] = left_['p_ci']
    out['r_ci_left'] = left_['r_ci']
    out['f_ci_left'] = left_['f_ci']

    out['TP_right'] = right_['TP'] 
    out['TN_right'] = right_['TN'] 
    out['FP_right'] = right_['FP'] 
    out['FN_right'] = right_['FN'] 
    out['n_sys_right'] = right_['n_sys']
    out['precision_right'] = right_['precision']
    out['recall_right'] = right_['recall']
    out['f1_right'] = right_['F1']
    out['merge_right'] = right_['merge']
    out['p_ci_right'] = right_['p_ci']
    out['r_ci_right'] = right_['r_ci']
    out['f_ci_right'] = right_['f_ci']
    # --> get standard metrics

    print('(' + c[0] + '&' + c[1] + ')')
    print('(' + c[0] + '|' + c[1] + ')')
    print('(' + c[0] + '^' + c[1] + ')')

    statement = '('  + c[0] + '&' +  c[1] + ')'

    and_ = ad_hoc_sys(statement, analysis_type, corpus, filter_semtype, True, semtype)

    and_['merge'] = statement
    n = statement.count('&') + statement.count('|') + statement.count('^') + 1 
    and_['n_terms'] = n
    and_['p_ci'] = str((and_['p_lower_bound'], and_['p_upper_bound'])) 
    and_['r_ci'] = str((and_['r_lower_bound'], and_['r_upper_bound']))  
    and_['f_ci'] = str((and_['f_lower_bound'], and_['f_upper_bound'])) 

    statement = '(' + c[0] + '|' + c[1] + ')'

    or_ = ad_hoc_sys(statement, analysis_type, corpus, filter_semtype, True, semtype)

    or_['merge'] = statement
    n = statement.count('&') + statement.count('|') + statement.count('^') + 1 
    or_['n_terms'] = n
    or_['p_ci'] = str((or_['p_lower_bound'], or_['p_upper_bound'])) 
    or_['r_ci'] = str((or_['r_lower_bound'], or_['r_upper_bound']))  
    or_['f_ci'] = str((or_['f_lower_bound'], or_['f_upper_bound'])) 

    statement = '(' + c[0] + '^' + c[1] + ')'

    xor_ = ad_hoc_sys(statement, analysis_type, corpus, filter_semtype, True, semtype)

    xor_['merge'] = statement
    n = statement.count('&') + statement.count('|') + statement.count('^') + 1 
    xor_['n_terms'] = n
    xor_['p_ci'] = str((xor_['p_lower_bound'], xor_['p_upper_bound'])) 
    xor_['r_ci'] = str((xor_['r_lower_bound'], xor_['r_upper_bound']))  
    xor_['f_ci'] = str((xor_['f_lower_bound'], xor_['f_upper_bound'])) 
    # --> end standard metrics

    out['semgroup'] = semtype

    out['precision_and'] = and_['precision']
    out['recall_and'] = and_['recall']
    out['f1_and'] = and_['F1']
    out['merge_and'] = and_['merge']
    out['p_ci_and'] = and_['p_ci']
    out['r_ci_and'] = and_['r_ci']
    out['f_ci_and'] = and_['f_ci']
    out['TP_and'] = and_['TP'] 
    out['TN_and'] = and_['TN'] 
    out['FP_and'] = and_['FP'] 
    out['FN_and'] = and_['FN'] 
    out['n_sys_and'] = and_['n_sys']
    
    out['precision_or'] = or_['precision']
    out['recall_or'] = or_['recall']
    out['f1_or'] = or_['F1']
    out['merge_or'] = or_['merge']
    out['p_ci_or'] = or_['p_ci']
    out['r_ci_or'] = or_['r_ci']
    out['f_ci_or'] = or_['f_ci']
    out['TP_or'] = or_['TP'] 
    out['TN_or'] = or_['TN'] 
    out['FP_or'] = or_['FP'] 
    out['FN_or'] = or_['FN'] 
    out['n_sys_or'] = or_['n_sys']

    out['precision_xor'] = xor_['precision']
    out['recall_xor'] = xor_['recall']
    out['f1_xor'] = xor_['F1']
    out['merge_xor'] = xor_['merge']
    out['p_ci_xor'] = xor_['p_ci']
    out['r_ci_xor'] = xor_['r_ci']
    out['f_ci_xor'] = xor_['f_ci']
    out['TP_xor'] = xor_['TP'] 
    out['TN_xor'] = xor_['TN'] 
    out['FP_xor'] = xor_['FP'] 
    out['FN_xor'] = xor_['FN'] 
    out['n_sys_xor'] = xor_['n_sys']
    
    out['nterms'] = n
    out['sentence'] = sentence
    out['moi'] = measure

    out['mtype'] = measure_type
    if measure_type == 'F1':
        out['max_baby_measure'] = out[['f1_and','f1_or','f1_xor']].apply(max, axis=1)
        out['min_baby_measure'] = out[['f1_and','f1_or','f1_xor']].apply(min, axis=1)
    elif measure_type == 'F1':
        out['max_baby_measure'] = out[['precision_and','precision_or','precision_xor']].apply(max, axis=1)
        out['min_baby_measure'] = out[['precision_and','precision_or','precision_xor']].apply(min, axis=1)
    else:    
        out['max_baby_measure'] = out[['recall_and','recall_or','recall_xor']].apply(max, axis=1)
        out['min_baby_measure'] = out[['recall_and','recall_or','recall_xor']].apply(min, axis=1)
    
    out['order'] = c[3]
    out['operator'] = c[2]
    # write to file 

    #if not filter_semtype:
    now = datetime.now()
    now = now.strftime("%m-%d-%Y")
    
    #file_out = 'complement_' + corpus + '_filter_semtype_' + str(filter_semtype) + '_' + str(now) +'.csv'
    file_out = 'complement_' + corpus + '_filter_semtype_' + str(filter_semtype) + '_1234.csv'

    if out_file:
        with open(data_out / file_out, 'a') as f:
            out.to_csv(f, header=f.tell()==0)
    else:
        return out

if __name__ == '__main__':

    run_ = 'vote'

    start = time.time()

    now = datetime.now()
    timestamp = datetime.timestamp(now)

    if run_ == 'comp':
        parallel = True

        file_out = 'complement_' + corpus + '_filter_semtype_' + str(filter_semtype) + '_' + str(timestamp) +'.csv'
        
        '''
        class Results(object):
            def __init__(self):
                self.sysA = pd.DataFrame() 
                self.sysB = pd.DataFrame() 
                self.nameA = ''
                self.nameB = ''
            
        r = Results()
        '''

        if parallel:
        
            with joblib.parallel_backend('dask'):

                joblib.Parallel(verbose=100)(joblib.delayed(main)(semtype, c) for semtype in semtypes for c in get_ensemble_pairs(get_valid_systems(systems, semtype)))

        else:
            for semtype in semtypes:
                test = get_valid_systems(systems, semtype)
                expressions = get_ensemble_pairs(test)
    
                print('NP -> SYSTEMS FOR SEMTYPE', semtype, 'ARE', test)

                for c in expressions:
                    print('complementarity between:', c)
                    main(semtype, c)

    elif run_ == 'vote':
        get_ipython().run_line_magic('prun', 'main_test()')
         #%lprun -f vote main_test()
         #get_ipython().run_line_magic('lprun -f vote', 'main_test()')


    elapsed = (time.time() - start)
    print('elapsed:', elapsed)

'''
for sys in systems:
    ref_ann[sys]=0
    for row in ref_ann.itertuples():
        iix = pd.IntervalIndex.from_arrays(df.begin, df.end, closed='neither')
        span_range = pd.Interval(row.start, row.end)
        fx = df[iix.overlaps(span_range)].copy()
        if len(fx) > 0:
            for sys in systems:
                if sys in fx.system.values:
                    ref_ann.loc[row.Index,sys] = 1


# generate complementarity:
import pandas as pd

measures = ['F1', 'precision', 'recall']
data=pd.read_csv('/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/mipacq_F1_10-03-2021.csv')
for measure in measures:
    merges=data.sort_values(measure, ascending=False)
    merges['measure_system'] = merges[[measure, 'system']].apply(tuple, axis=1)
    sentences=merges.head(1)['measure_system'].to_list()
    for sentence in sentences:
        nec.ad_hoc_complementarity(sentence, measure)


measures = ['F1', 'precision', 'recall']
data=pd.read_csv('/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/fairview_F1_st_10-08-2021.csv')
for measure in measures:
    groups=set(data.semtype.tolist())
    for group in groups: 
        df = data.loc[data.semtype==group]
        merges=df.sort_values(measure, ascending=False)
        merges['measure_system'] = merges[[measure, 'system']].apply(tuple, axis=1)
        sentences=merges.head(1)['measure_system'].to_list()
        for sentence in sentences:
            nec.ad_hoc_complementarity_st(sentence, measure, group)


# get data for monotinicity
# https://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity

import itertools
import operator
import numpy as np 
import pandas as pd

def monotone_increasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.le, pairs))

def monotone_decreasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.ge, pairs))

def monotone(lst):
    return monotone_increasing(lst) or monotone_decreasing(lst)



data=pd.read_csv('/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/complement_mipacq_filter_semtype_False_10-12-2021.csv')
measures = ['f1', 'precision', 'recall']

for mtype in measures[0:1]: 
    sentences=set(data.loc[data.mtype==mtype.upper()].sentence.tolist())

    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right','merge_left', 'precision_left', 'recall_left',
           'f1_left', 'f1-score', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 'sentence', 'order', 'operator', 'merge_left', 'merge_right']

    monotonic = []
    increase = []
    decrease = []
    nonmono = []

    m=0
    n=0

    for s in list(sentences)[0:1]:
        test=data.loc[(data.sentence==s)&(data.mtype==mtype.upper())]
        test[mtype + '-score'] = np.where(test['operator']=='&', test[mtype + '_and'], test[mtype + '_or'])
        #print(test[cols_to_keep].sort_values(['order','f1-score'], ascending=False))
        t=test[cols_to_keep].sort_values(['order','f1-score'], ascending=False)
        o=set(t.order.to_list())
        scores=[]
        #if t.moi.values[0] not in scores:
        #    scores.append(t.moi.values[0])
        for i in o:
            #print(t.loc[t.order==i])
            u=t.loc[t.order==i]
            #if len(u['merge_left'].values[0][0]) == 1:
            #    print(s,'left', u['merge_left'].values[0][0])
            #    scores.append(u[mtype+'_left'].values[0])

            #if len(u['merge_right'].values[0][0]) == 1:
            #    print('right', u['merge_right'].values[0][0])
            #    scores.append(u[mtype+'_right'].values[0])


            scores.append(u[mtype + '-score'].values[0])
            print(scores, u[mtype + '-score'].values[0])
    
        print('monotonic', monotone(scores[::-1]))
        print('monotonic increasing', monotone_increasing(scores[::-1]))
        print('monotonic decreasing', monotone_decreasing(scores[::-1]))

        if monotone(scores[::-1]):
            m+=1
        else:
            n+=1

        if monotone_increasing(scores[::-1]):
            increase.append(1)
        if monotone_decreasing(scores[::-1]):
            decrease.append(1)
    
    monotonic.append(m)
    nonmono.append(n)
    index = ['snail', 'pig'] 
    df = pd.DataFrame({'mono': monotone, 'nonmono': nonmono}, index=index)
            

'''
