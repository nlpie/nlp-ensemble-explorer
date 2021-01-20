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

import click
from combo_searcher_new import combo_searcher as cs
import importlib as i
import gevent
from scipy import stats 
from scipy.stats import norm, mode
from scipy.stats.mstats import gmean
import random
import pandas as pd
import numpy as np
import sparse as sp
import math
import pymysql
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

import multiprocessing as mp
from joblib import Parallel, delayed

%load_ext Cython
# The cell below contains the configurable parameters to ensure that our ensemble explorer runs properaly on your machine. 
# Please read carfully through steps (1-11) before running the rest of the cells.


# STEP-1: CHOOSE YOUR CORPUS
# TODO: get working with list of corpora
#corpora = ['mipacq','i2b2','fairview'] #options for concept extraction include 'fairview', 'mipacq' OR 'i2b2'

# cross-system semantic union merge filter for cross system aggregations using custom system annotations file with corpus name and system name using 'ray_test':

# TODO: move to click param
# need to add semantic type filrering when reading in sys_data
#corpus = 'ray_test'
#corpus = 'clinical_trial2'
#corpus = 'fairview'
#corpus = 'i2b2'
#corpus = 'mipacq'
corpus = 'medmentions'

# TODO: create config.py file
# STEP-2: CHOOSE YOUR DATA DIRECTORY; this is where output data will be saved on your machine
data_directory = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/' 

data_out = Path('/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/')

# TODO: move to click param
# STEP-3: CHOOSE WHICH SYSTEMS YOU'D LIKE TO EVALUATE AGAINST THE CORPUS REFERENCE SET
systems = ['biomedicus', 'clamp', 'ctakes', 'metamap', 'quick_umls']

# TODO: move to click param
# STEP-4: CHOOSE TYPE OF RUN:  
rtype = 7      # OPTIONS INCLUDE: 2->Ensemble; 3->Tests; 4 -> majority vote; 6 -> add hoc ensemble; 7 -> complementarity
               # The Ensemble can include the max system set ['ctakes','biomedicus','clamp','metamap','quick_umls']

# TODO: move to click param
# STEP-5: CHOOSE WHAT TYPE OF ANALYSIS YOU'D LIKE TO RUN ON THE CORPUS
analysis_type = 'entity' #options include 'entity', 'cui' OR 'full'

# TODO: create config.py file
# STEP-(6A): ENTER DETAILS FOR ACCESSING MANUAL ANNOTATION DATA
database_type = 'mysql+pymysql' # We use mysql+pymql as default
database_username = 'gms'
database_password = 'nej123' 
database_url = 'localhost' # HINT: use localhost if you're running database on your local machine
database_name = 'medmentions' # concepts' # Enter database name
#database_name = 'medmentions' # Enter database name

def ref_data(corpus):
    return corpus + '_all' # Enter the table within the database where your reference data is stored

table_name = ref_data(corpus)

# STEP-(6B): ENTER DETAILS FOR ACCESSING SYSTEM ANNOTATION DATA

def sys_data(corpus, analysis_type):
    if analysis_type == 'entity':
        return 'analytical_'+corpus+'.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
    elif analysis_type in ('cui', 'full', 'entity'):
        return 'analytical_'+corpus+'_cui.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
        
system_annotation = sys_data(corpus, analysis_type)

# STEP-7: CREATE A DB CONNECTION POOL
engine_request = str(database_type)+'://'+database_username+':'+database_password+"@"+database_url+'/'+database_name
engine = create_engine(engine_request, pool_pre_ping=True, pool_size=20, max_overflow=30)
#engine = engine_request


# TODO: move to click param
# STEP-(8A): FILTER BY SEMTYPE
filter_semtype = True 

# TODO: create config.py file
# STEP-(8B): IF STEP-(8A) == True -> GET REFERENCE SEMTYPES

def ref_semtypes(filter_semtype, corpus):
    if filter_semtype:
        if corpus == 'fairview':
            semtypes = ['Drug', 'Finding', 'Anatomy', 'Procedure']
        elif corpus == 'i2b2':
            semtypes = ['test,treatment', 'problem']
        elif corpus == 'mipacq':
            semtypes =  ['Anatomy'] #['Procedures', 'Disorders,Sign_Symptom', 'Anatomy', 'Chemicals_and_drugs']
        elif corpus in ['clinical_trial', 'clinical_trial2']:
            semtypes = ['drug,drug::drug_name,drug::drug_dose,dietary_sppplement::dietary_seeelement_name,dietary_supplement::dietary_supplement_dose',
                        'temporal_measurement,qualifier,measurement',
                        'device',
                        'condition,observation', 
                        'demographics::age,demographics::sex,demographics::race_ethnicity,demographics::bmi,demographics::weight',
                        'diet',
                        'measurement,qualifier',
                        'procedure,observation']
        elif corpus == 'medmentions':
            semtypes = ['Procedures'] #, 'Anatomy', 'Disorders', 'Chemicals & Drugs']

        return semtypes

semtypes = ref_semtypes(filter_semtype, corpus)

# STEP-9: Set data directory/table for source documents for vectorization
src_table = 'sofa'

# TODO: move to click param
# STEP-10: Specify match type from {'exact', 'overlap'}
run_type = 'overlap'

# STEP-11: Specify type of ensemble: merge or vote: used for file naming -> TODO: remove!
ensemble_type = 'merge'

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
        ref_data = database_name+'.'+table_name
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

        if corpus == 'medmentions':
            sql = "SELECT st.tui, abbreviation, clamp_name, ctakes_name FROM concepts.semantic_groups sg join concepts.semantic_types st on sg.tui = st.tui where group_name in ({})"               .format(', '.join(['%s' for _ in semtypes]))  
        elif corpus == 'clinical_trial2':
            sql = "SELECT st.tui, abbreviation, clamp_name, ctakes_name, biomedicus_name FROM clinical_trial.semantic_groups sg join semantic_types st on sg.tui = st.tui where " + corpus + "_name in ({})"                .format(', '.join(['%s' for _ in semtypes]))  
        else:
            sql = "SELECT st.tui, abbreviation, clamp_name, ctakes_name FROM concepts.semantic_groups sg join concepts.semantic_types st on sg.tui = st.tui where " + corpus + "_name in ({})"               .format(', '.join(['%s' for _ in semtypes]))  
        
        stypes = pd.read_sql(sql, params=[semtypes], con=engine) 
       
        if len(stypes['tui'].tolist()) > 0:
            self.biomedicus_types = set(stypes['tui'].tolist())
            self.qumls_types = set(stypes['tui'].tolist())
        
        else:
            self.biomedicus_types = None
            self.qumls_types = None

        
        if stypes['clamp_name'].dropna(inplace=True) or len(stypes['clamp_name'].tolist()) == 0 or None in stypes['clamp_name'].tolist():
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
def label_vector(doc: int, ann: List[int], labels: List[str]) -> np.array:

    v = np.zeros(doc, dtype=np.uint8)
    labels = list(labels)
    
    for (i, lab) in enumerate(labels):
        i += 1  # 0 is reserved for no label
        idxs = [np.arange(a.begin, a.end, dtype=np.int16) for a in ann if a.label == lab]
        idxs = [j for mask in idxs for j in mask]
        v[idxs] = i 

    return v

# confusion matrix elements for vectorized annotation set binary classification
# https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/
#%%cython 
#%load_ext cython
#import numpy as np
def confused(pred, true):

    predicted_true, predicted_false = pred == 1, pred == 0
    true_true, true_false = true == 1, true == 0

    not_true = np.logical_not(true)
    not_predicted = np.logical_not(pred)
    TP = np.sum(np.logical_and(true, pred))
    TN = np.sum(np.logical_and(not_true, not_predicted))
    FP = np.sum(np.logical_and(not_true, pred))

    FN = len(pred) - (TP+TN+FP)
    
    return TP, TN, FP, FN

@ft.lru_cache(maxsize=None)
def get_labels(analysis_type, corpus, filter_semtype, semtype = None):
    
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
    
    if analysis_type != 'cui':
        ann1 = list(ann.itertuples(index=False))
    
    #for n in range(len(docs)):
    for k, v in docs.items():
        if analysis_type != 'cui':
            #a1 = [i for i in ann1 if i.case == docs[n][0]]
            #a1 = [i for i in ann1 if i.case == docs[n]['note_id']]
            a1 = [i for i in ann1 if i.case == k]
            #a = label_vector(docs[n][1], a1, labels)
            #a = label_vector(docs[n]['len_doc'], a1, labels)
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
    
    sys2 = list()
    s2 = list()

    if analysis_type != 'cui':
        s = list(sys.itertuples(index=False))
    

    #for n in range(len(docs)):
    for k, v in docs.items():

        if analysis_type != 'cui':
            #s1 = [i for i in s if i.case==docs[n][0]] # list(sys.loc[sys.case == docs[n][0]].itertuples(index=False))
            #sys1 = label_vector(docs[n][1], s1, labels)
            s1 = [i for i in s if i.case==k] # list(sys.loc[sys.case == docs[n][0]].itertuples(index=False))
            sys1 = label_vector(v, s1, labels)
            #s1 = [i for i in s if i.case==docs[n]['note_id']] # list(sys.loc[sys.case == docs[n][0]].itertuples(index=False))
            #sys1 = label_vector(docs[n]['len_doc'], s1, labels)
            sys2.append(sys1)
        else:
            s = sys.loc[sys.case == docs[n][0]]['label'].tolist()
            x = [1 if x in s else 0 for x in labels]
            s2.append(x)

    a2 = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)
            
    if analysis_type != 'cui': #binary and multiclass
        s2 = np.array(flatten_list(sys2), dtype=np.uint8)
        
        if analysis_type == 'full':
            report = classification_report(a2, s2, output_dict=True)
            macro_precision =  report['macro avg']['precision'] 
            macro_recall = report['macro avg']['recall']    
            macro_f1 = report['macro avg']['f1-score']
            return ((0, 0, 0, 0), (macro_precision, macro_recall, macro_f1))
        else:
            #TN, FP, FN, TP = confusion_matrix(a2, s2).ravel()

            TP, TN, FP, FN = confused(sp.COO(s2), sp.COO(a2))
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
def vectorized_complementarity(r: object, analysis_type: str, corpus: str, c: tuple, filter_semtype, semtype = None) -> np.int64:
    docs = get_docs(corpus)
    
    out = pd.DataFrame()
    
    if filter_semtype:
        ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
    else: 
        ann = get_ref_ann(analysis_type, corpus, filter_semtype)
    
    sysA = r.sysA
    sysB = r.sysB
    sysA = sysA.rename(columns={"note_id": "case"})
    sysB = sysB.rename(columns={"note_id": "case"})

    if analysis_type == 'entity':
        sysA["label"] = 'concept'
        sysB["label"] = 'concept'
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'full':
        sysA["label"] = sysA["cui"]
        sysB["label"] = sysB["cui"]
        cols_to_keep = ['begin', 'end', 'case', 'value', 'label']
    elif analysis_type == 'cui':
        sysA["label"] = sysA["cui"]
        sysB["label"] = sysA["cui"]
        cols_to_keep = ['case', 'cui', 'label']

    sysA = sysA[cols_to_keep]
    sysB = sysB[cols_to_keep]

    if analysis_type == 'entity':
        labels = ["concept"]
    elif analysis_type in ['cui', 'full']:
        labels = list(set(ann["value"].tolist()))

    sys_a2 = list()
    sys_b2 = list()
    sys_ab2 = list()
    ann2 = list()
    s_a2 = list()
    s_b2 = list()
    sys_ab1_ab3 = list()

    a2 = list()
    
    cvals = list()
    
    a = list(sysA.itertuples(index=False))
    b = list(sysB.itertuples(index=False))

    a2 = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)

    for k, v in docs.items():

        # get for Aright/Awrong and Bright/Bwrong
        s_a1 = [i for i in a if i.case==k]##list(sysA.loc[sysA.case == docs[n][0]].itertuples(index=False))
        s_b1 = [i for i in b if i.case==k]# list(sysB.loc[sysB.case == docs[n][0]].itertuples(index=False))
        sys_a1 = label_vector(v, s_a1, labels)
        sys_b1 = label_vector(v, s_b1, labels)

        sys_a2.append(sys_a1)
        sys_b2.append(sys_b1)

        # intersected list this will give positive values 
        # NB: intersection only gives positive labels, 
        # since systems do not annotate for negative class
        s_ab1 = list(set(s_a1).intersection(set(s_b1)))

        sys_ab1 = label_vector(v, s_ab1, labels)
        sys_ab2.append(sys_ab1)
        
        # in one set or other but not both for negative values
        # NB: FN is inherently antisymetric for FP
        s_ab2 = list(set(s_a1).symmetric_difference(set(s_b1)))
        s_ab1_ab2 = list(set(s_ab1).union(set(s_ab2)))
        
        sys_ab1_ab2 = label_vector(v, s_ab1_ab2, labels)
        sys_ab1_ab3.append(sys_ab1_ab2)

    # right/wrong for A and B
    s_a2 = np.concatenate(sys_a2).ravel()
    s_b2 = np.concatenate(sys_b2).ravel()
    
    sys_ab3 = np.concatenate(sys_ab2).ravel()
    sys_ab1_ab4 = np.concatenate(sys_ab1_ab3).ravel()

    _, _, FP, _ = confused(sp.COO(sys_ab3), sp.COO(a2))
    _, _, _, FN = confused(sp.COO(sys_ab1_ab4), sp.COO(a2))

    _, _, aFP, aFN = confused(sp.COO(s_a2), sp.COO(a2))
    _, _, bFP, bFN = confused(sp.COO(s_b2), sp.COO(a2))

    b_over_a, a_over_b, mean_comp = complementarity_measures(FN, FP, aFN, aFP, bFN, bFP)

    b_over_a['system'] = str((r.nameB, r.nameA))
    b_over_a['B'] = r.nameB    
    b_over_a['A'] = r.nameA

    a_over_b['system'] = str((r.nameA, r.nameB))
    a_over_b['B'] = r.nameA    
    a_over_b['A'] = r.nameB

    mean_comp['system'] = 'mean_comp(' + r.nameA + ',' + r.nameB + ')'

    frames = [out, pd.DataFrame(b_over_a, index=[0])]
    out = pd.concat(frames, ignore_index=True, sort=False) 
    
    frames = [out, pd.DataFrame(a_over_b, index=[0])]
    out = pd.concat(frames, ignore_index=True, sort=False) 

    frames = [out, pd.DataFrame(mean_comp, index=[0])]
    out = pd.concat(frames, ignore_index=True, sort=False) 

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
    
    b_over_a = {'test': 'COMP(A, B)', 'max_prop_error_reduction': compA, 'p': p_compA, 'r': r_compA, 'F1-score': f1_compA}
    a_over_b = {'test': 'COMP(B, A)', 'max_prop_error_reduction': compB, 'p': p_compB, 'r': r_compB, 'F1-score': f1_compB}

    mean_complementarity = {'test': 'mean(COMP(B, A),COMP(A, B))', 'max_prop_error_reduction': meanComp, 'mean p': meanP, 'mean r': meanR, 'mean F1-score': meanF1}

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
    
    if system_n - FP != TP:
        print('inconsistent system n!')

    return d


@ft.lru_cache(maxsize=None)
def get_metric_data(analysis_type: str, corpus: str):
   
    usys_file, ref_table = AnalysisConfig().corpus_config()
    #systems = AnalysisConfig().systems
   
    if corpus != 'medmentiopns':
        sys_ann = pd.read_csv(analysisConf.data_dir + usys_file, dtype={'note_id': str})
    else:
        sys_ann = pd.read_csv(analysisConf.data_dir + usys_file)
        sys_ann['note_id'] = pd.to_numeric(sys_ann['note_id'])

    sys_ann = sys_ann.rename(columns={"semtype": "semtypes"})
    
    sql = "SELECT * FROM " + ref_table #+ " where semtype in('Anatomy', 'Chemicals_and_drugs')"a
    
    ref_ann = pd.read_sql(sql, con=engine)

    if corpus == 'medmentions':
        cases = set(ref_ann["file"].tolist())
        cases = [int(i) for i in cases]
        ref_ann['file'] = pd.to_numeric(ref_ann['file'])
        sys_ann['note_id'] = pd.to_numeric(sys_ann['note_id'])
        sys_ann = sys_ann.loc[sys_ann.note_id.isin(cases)]

    sys_ann = sys_ann.drop_duplicates()

    ref_ann, _ = reduce_mem_usage(ref_ann)
    sys_ann, _ = reduce_mem_usage(sys_ann)
 
    
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


@ft.lru_cache(maxsize=None)
def get_sys_data(system: str, analysis_type: str, corpus: str, filter_semtype, semtype = None) -> pd.DataFrame:
   
    _, data = get_metric_data(analysis_type, corpus)
    
    out = data.loc[data.system == system]
    
    if filter_semtype:
        st = SemanticTypes([semtype], corpus).get_system_type(system)
        print(system, ' ST:', st)
    
    if corpus == 'casi':
        cols_to_keep = ['case', 'overlap'] 
        out = out[cols_to_keep].drop_duplicates()
        return out
        
    else:
        if filter_semtype:
            out = out.loc[out.semtypes.isin(st)]
            
        else:
            out = out.loc[out.system == system]
            
        if system == 'quick_umls':
            out = out.loc[(out.score.astype(float) >= 0.8) & ((out.type == 'concept_jaccard_score_False')|(out.type=='concept'))]
        
        if system == 'metamap':
            out = out.loc[out.score.abs().astype(int) >= 800]
            
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
    
    for case in cases:
        i = 0
        data = []
        out = pd.DataFrame()
        
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
                if i%5000 == 0:
                    print('iteration:', i)

                # if longer span exists, use as tie-breaker else use score
                if maxLength > minLength:
                    fx = fx[fx['length'] == fx['length'].max()]
                elif maxScore > minScore:
                    fx = fx[fx['score'] == maxScore]

            i += 1
            data.append(fx)

        out = pd.concat(data, axis=0)
   
    # Remaining ties: randomly reindex to keep random row when dropping duplicates: https://gist.github.com/cadrev/6b91985a1660f26c2742
    out.reset_index(inplace=True)
    out = out.reindex(np.random.permutation(out.index))
    out = out.drop_duplicates(['begin', 'end', 'note_id', 'length', 'cui'])
    
    return out  

# majority vote -> plurality for entity only, witth tties winning
def vote(df, systems):
   
    df = df.drop_duplicates(subset=['begin', 'end', 'case', 'system'])
    cases = set(df['case'].tolist())
    
    data = []
    out = pd.DataFrame()
    
    for case in cases:
        i = 0
        
        test = df.loc[df.case == case].copy()
        
        for row in test.itertuples():

            fx = test.loc[(test.begin == row.begin) & (test.end == row.end)].copy()

            n = int(len(systems)/2)

            fx = fx[fx.system.isin(systems)]
           
            fx = fx.drop_duplicates()
 
            if len(set(fx.system.tolist()))>n:
                data.append(fx)
             
    out = pd.concat(data, axis=0)
   
    out = out.drop_duplicates(subset=['begin', 'end', 'case'])
    
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
        oper = {'&': op.and_, '|': op.or_}
        
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
                    
#                     if analysis_type == 'full':
#                         df = disambiguate(df)

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
                
                # get combined system results
                r.system_merges = df
                
                if len(df) > 0:
                    system_query = system_query.append(df)
                else:
                    print('wtf!')
                    
                return system_query
            else:
                return parseTree.getRootVal()
    
    if sentence.n_or > 0 or sentence.n_and > 0:
        evaluate(pt)  
    
    # trivial case
    elif sentence.n_or == 0 and sentence.n_and == 0:
        
        if filter_semtype:
            r.system_merges = get_sys_data(sentence.sentence, analysis_type, corpus, filter_semtype, semtype)
        else:
            r.system_merges = get_sys_data(sentence.sentence, analysis_type, corpus, filter_semtype)
        
    return r


class Results(object):
    def __init__(self):
        self.ref = np.array(list(), dtype=np.uint8)
        self.sys = np.array(list(), dtype=np.uint8)
        self.df = pd.DataFrame()
        self.labels = list()

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
        elif i not in ['&', '|', ')']:
            currentTree.setRootVal(i)
            parent = pStack.pop()
            currentTree = parent
        elif i in ['&', '|']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError

    return eTree

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
        # prepare statement for case when a boolean AND/OR is given
        sentence = payload.replace('(', ' ( ').replace(')', ' ) ').replace('&', ' & ').replace('|', ' | ').replace('  ', ' ')
        return sentence

    sentence = preprocess_sentence(payload)
    print('Processing sentence:', sentence)
    
    pt = buildParseTree(sentence)
    #pt.postorder() 
    
    return pt

class Sentence(object):
    '''
    Details about boolean expression -> number operators and expression
    '''
    def __init__(self, sentence):
        self = self
        self.n_and = sentence.count('&')
        self.n_or = sentence.count('|')
        self.sentence = sentence

@ft.lru_cache(maxsize=None)
def get_docs(corpus):
    
    # KLUDGE!!!
    if corpus == 'ray_test':
        corpus = 'fairview'
    
    if corpus == "medmentions":
        sql = 'select distinct note_id, len_doc from medmentions.sofas where test=1 and corpus = %(corpus)s order by note_id'
    else:
        sql = 'select distinct note_id, sofa from sofas where corpus = %(corpus)s order by note_id'
    
    df = pd.read_sql(sql, params={"corpus": corpus}, con=engine)
    df.drop_duplicates()

    if corpus != "medmentions":
        df['len_doc'] = df['sofa'].apply(len)
    else:
        df['note_id'] = pd.to_numeric(df['note_id'])
    
    subset = df[['note_id', 'len_doc']]
    docs = subset.set_index('note_id')['len_doc'].to_dict()
    return docs

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
    ann = ann[cols_to_keep]
    
    return ann

#@ft.lru_cache(maxsize=None)
def get_sys_ann(analysis_type, r):
    sys = r.system_merges

    sys = sys.rename(columns={"note_id": "case", "cui": "value"})
    
    sys = set_labels(analysis_type, sys)
    
    if analysis_type == 'entity':
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'full':
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'cui':
        cols_to_keep = ['case', 'label']
    
    sys = sys[cols_to_keep]
    return sys

@ft.lru_cache(maxsize=None)
def get_metrics(boolean_expression: str, analysis_type: str, corpus: str, run_type: str, filter_semtype, semtype = None):
    """
    Traverse binary parse tree representation of Boolean sentence
        :params: boolean expression in form of '(<annotator_engine_name1><boolean operator><annotator_engine_name2>)'
                 analysis_type (string value of: 'entity', 'cui', 'full') used to filter set of reference and system annotations 
        :return: dictionary with values needed for confusion matrix
    """     
            
    results = Results()
    
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
            
             # get CI:
            [recall, dr, r_lower_bound, r_upper_bound] = normal_approximation_binomial_confidence_interval(TP, TP + FN)
            [precision, dp, p_lower_bound, p_upper_bound] = normal_approximation_binomial_confidence_interval(TP, TP + FP)
            [f, df, f_lower_bound, f_upper_bound] = f1_score_confidence_interval(recall, precision, dr, dp)

            d['r_upper_bound'] = r_upper_bound
            d['r_lower_bound'] = r_lower_bound

            d['p_upper_bound'] = p_upper_bound
            d['p_lower_bound'] = p_lower_bound

            d['f_upper_bound'] = f_upper_bound
            d['f_lower_bound'] = f_lower_bound

        else:
            d = dict()
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
def run_ensemble(systems, analysis_type, corpus, filter_semtype, semtype = None):
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
            print(e)
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
    if filter_semtype:
        ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
    else: 
        ann = get_ref_ann(analysis_type, corpus, filter_semtype)


    sentence = Sentence(boolean_expression)   

    pt = make_parse_tree(sentence.sentence)

    results = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype)

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
    ref_ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)

    df = ref_ann
    
    if 'entity' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif 'cui' in analysis_type: 
        cols_to_keep = ['case', 'label']
    elif 'full' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'case', 'label']
     
    labels = get_labels(analysis_type, corpus, filter_semtype, semtype)
    
    df = df.drop_duplicates(subset=cols_to_keep)
    ref = df[cols_to_keep]
    
    test = vectorized_annotations(ref, analysis_type, labels)
    
    if analysis_type != 'cui':
        ref =  np.asarray(flatten_list(test), dtype=np.uint8) 
    else: 
        ref =  np.asarray(test, dtype=np.int16)

    return ref

def get_majority_sys(systems, analysis_type, corpus, filter_semtype, semtype):
    
    d = {}
    
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
        #TP, TN, FP, FN = confused(vote, ref)
        system_n = TP + FP
        reference_n = TP + FN

        d = cm_dict(FN, FP, TP, system_n, reference_n)

        # get CI:
        [r, dr, r_lower_bound, r_upper_bound] = normal_approximation_binomial_confidence_interval(TP, TP + FN)
        [p, dp, p_lower_bound, p_upper_bound] = normal_approximation_binomial_confidence_interval(TP, TP + FP)
        [f, df, f_lower_bound, f_upper_bound] = f1_score_confidence_interval(r, p, dr, dp)

        d['r_upper_bound'] = r_upper_bound
        d['r_lower_bound'] = r_lower_bound

        d['p_upper_bound'] = p_upper_bound
        d['p_lower_bound'] = p_lower_bound

        d['f_upper_bound'] = f_upper_bound
        d['f_lower_bound'] = f_lower_bound

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
def majority_vote(systems, analysis_type, corpus, run_type, filter_semtype, semtypes = None):
    print(semtypes, systems)

    if filter_semtype:
        
        metrics = pd.DataFrame()
        for semtype in semtypes:
            test = get_valid_systems(systems, semtype)
            print('SYSYEMS FOR SEMTYPE', semtype, 'ARE', test)
            
            if run_type == 'overlap' and len(test) > 1:
                ref = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)
                vote = get_majority_sys(test, analysis_type, corpus, filter_semtype, semtype)
                labels = get_labels(analysis_type, corpus, filter_semtype, semtype)
        
                out = majority_overlap_vote_out(ref, vote, corpus, semtype)
           
            if len(test) > 1:
                out['semgroup'] = semtype
                out['systems'] = ','.join(test)
                generate_ensemble_metrics(out, analysis_type, corpus, ensemble_type, filter_semtype, semtype)
                frames = [metrics, out]
                metrics = pd.concat(frames, ignore_index=True, sort=False)
                
    else:
        if run_type == 'overlap':
            ref = get_reference_vector(analysis_type, corpus, filter_semtype)
            vote = get_majority_sys(systems, analysis_type, corpus, filter_semtype)
            labels = get_labels(analysis_type, corpus, filter_semtype)
        
            metrics = majority_overlap_vote_out(ref, vote, corpus)
            
        metrics['systems'] = ','.join(systems)
        generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype)
    
    print(metrics)
    
    return metrics


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
        #expressions = [o for i in range(len(systems) - 2) for j in range(len(systems) - 2)  if i + j < len(systems) - 1 for o in out if n_operators(o[0], o[1], i, j, operators) and overlap(o[0], o[1], operators) == 0]

        return [o for i in range(len(systems)) for j in range(len(systems)) if i + j < len(systems)  
                for o in out if n_operators(o[0], o[1], i, j, operators) and overlap(o[0], o[1], operators) == 0]


# use with combo_searcher
def ad_hoc_measure(statement, analysis_type, corpus, measure, filter_semtype, semtype = None):
    d = get_merge_data(statement, analysis_type, corpus, run_type, filter_semtype, True, semtype)

    if measure in ['F1', 'precision', 'recall']:
        return d[measure]
    else:
        print('Invalid measure!')

def ad_hoc_sys(statement, analysis_type, corpus, metrics = False, semtype = None):
    sys = get_merge_data(statement, analysis_type, corpus, run_type, filter_semtype, metrics, semtype)

    return sys

def main():
    '''
        Control for:

        corpora: i2b2, mipacq, fv017
        analyses: entity only (exact span), cui by document, full (aka (entity and cui on exaact span/exact cui)
        systems: ctakes, biomedicus, clamp, metamap, quick_umls
        
        TODO -> Vectorization (entity only) -> done; (all) -> done:
                add switch for use of TN on single system performance evaluations -> done
                add switch for overlap matching versus exact span -> done
             -> Other tasks besides concept extraction
        
    ''' 
    analysisConf =  AnalysisConfig()
    print(analysisConf.systems, analysisConf.corpus_config())
   
    if (rtype == 2):
        print('run_type:', run_type)
        if filter_semtype:
            print(semtypes)
            ensemble_control(analysisConf.systems, analysis_type, corpus, run_type, filter_semtype, semtypes)
        else:
            ensemble_control(analysisConf.systems, analysis_type, corpus, run_type, filter_semtype)

    elif (rtype == 4):
        
        out = pd.DataFrame()
        for i in range(2, len(systems) + 1):
            for s in combinations(systems, i):
                print(s)
                if filter_semtype:
                    metrics = majority_vote(s, analysis_type, corpus, run_type, filter_semtype, semtypes)
                else:
                    metrics = majority_vote(s, analysis_type, corpus, run_type, filter_semtype)
                    
                frames = [out, metrics]
                out = pd.concat(frames, ignore_index=True, sort=False)
                
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        
        file = corpus + '_vote_' + analysis_type + '_' + str(filter_semtype) + '_' + str(timestamp) +'.csv'
        out.to_csv(data_out / file)
        
    elif (rtype == 6): # 5 system merges with evaluation
        
        statement = '(ctakes&biomedicus)'

        ad_hoc_sys(statement, analysis_type, corpus)
    
    elif (rtype == 7): # complementarity ala http://www.lrec-conf.org/proceedings/lrec2016/pdf/105_Paper.pdf
       
        class Results(object):
            def __init__(self):
                self.sysA = pd.DataFrame() 
                self.sysB = pd.DataFrame() 
                self.nameA = ''
                self.nameB = ''
            
        r = Results()

        def ad_hoc(analysis_type, corpus, systems):
           
            df = pd.DataFrame()
            if filter_semtype:
                for semtype in semtypes:
                    test = get_valid_systems(systems, semtype)
                    expressions = get_ensemble_pairs(test)
                    print('SYSTEMS FOR SEMTYPE', semtype, 'ARE', test)

                    for c in expressions:
                        print('complementarity between:', c)
                        
                        r.nameA = c[0]
                        r.nameB = c[1]

                        r.sysA = ad_hoc_sys(c[0], analysis_type, corpus, False, semtype) # c[0]
                        r.sysB = ad_hoc_sys(c[1], analysis_type, corpus, False, semtype) # c[1]

                        # --> get standard metrics
                    
                        print('(' + c[0] + '&' + c[1] + ')')
                        print('(' + c[0] + '|' + c[1] + ')')

                        statement = '(' + c[0] + '&' + c[1] + ')'

                        and_ = ad_hoc_sys(statement, analysis_type, corpus, True, semtype)

                        and_['merge'] = statement
                        n = statement.count('&') + statement.count('|') + 1 
                        and_['n_terms'] = n

                        statement = '(' + c[0] + '|' + c[1] + ')'

                        or_ = ad_hoc_sys(statement, analysis_type, corpus, True, semtype)

                        or_['merge'] = statement
                        n = statement.count('&') + statement.count('|') + 1 
                        or_['n_terms'] = n

                        # --> end standard metrics

                        out = vectorized_complementarity(r, analysis_type, corpus, c, filter_semtype, semtype)
                        out['semgroup'] = semtype

                        out['precision_and'] = and_['precision']
                        out['recall_and'] = and_['recall']
                        out['F1_and'] = and_['F1']
                        out['merge_and'] = and_['merge']
                        
                        out['precision_or'] = or_['precision']
                        out['recall_or'] = or_['recall']
                        out['F1_or'] = or_['F1']
                        out['merge_or'] = or_['merge']
                        out['nterms'] = n

                        frames = [df, out]
                        df = pd.concat(frames, ignore_index=True, sort=False) 


            else:
                expressions = get_ensemble_pairs(systems)

                print('total pairs', len(expressions))
               
                for c in expressions:
                    print('complementarity between:', c)

                    r.nameA = c[0]
                    r.nameB = c[1]

                    r.sysA = ad_hoc_sys(c[0], analysis_type, corpus) # c[0]
                    r.sysB = ad_hoc_sys(c[1], analysis_type, corpus) # c[1]

                    # --> get standard metrics
                    
                    print('(' + c[0] + '&' + c[1] + ')')
                    print('(' + c[0] + '|' + c[1] + ')')

                    statement = '(' + c[0] + '&' + c[1] + ')'

                    and_ = ad_hoc_sys(statement, analysis_type, corpus, True)

                    and_['merge'] = statement
                    n = statement.count('&') + statement.count('|') + 1 
                    and_['n_terms'] = n

                    statement = '(' + c[0] + '|' + c[1] + ')'

                    or_ = ad_hoc_sys(statement, analysis_type, corpus, True)

                    or_['merge'] = statement
                    n = statement.count('&') + statement.count('|') + 1 
                    or_['n_terms'] = n

                    # --> end staandard metrics


                    out = vectorized_complementarity(r, analysis_type, corpus, c, filter_semtype)
                    
                    out['precision_and'] = and_['precision']
                    out['recall_and'] = and_['recall']
                    out['F1_and'] = and_['F1']
                    out['merge_and'] = and_['merge']
                    
                    out['precision_or'] = or_['precision']
                    out['recall_or'] = or_['recall']
                    out['F1_or'] = or_['F1']
                    out['merge_or'] = or_['merge']
                    out['nterms'] = n

                   
                    out['semgroup'] = 'All groups'
                    frames = [df, out]
                    df = pd.concat(frames, ignore_index=True, sort=False) 
            
            now = datetime.now()
            timestamp = datetime.timestamp(now)
               
            file = 'complement_' + corpus + '_filter_semtype_' + str(filter_semtype) + '_' + str(timestamp) +'.csv'
            df.to_csv(data_out / file)
            print(df)
        
        
        ad_hoc(analysis_type, corpus, systems)

if __name__ == '__main__':
    # %load_ext memory_profiler
    get_ipython().run_line_magic('prun', 'main()')
    #%lprun -f vectorized_complementarity -f label_vector main()
    print('done!')


'''
if __name__ == '__main__':

    @click.group()
    def analyze():
        pass

    @analyze.command()
    @click.option('-c', '--corpus', 'corpus', default='fairview', help='Select corpus for analysis: (i2b2), (mipacq), (fairview)', type=click.STRING)
    @click.option('-t', '--task', 'analysis_type', default='entity', help='Select analysis task: (entity), (cui), (full)', type=click.STRING)
    @click.option('-f', '--filter', 'filter_semtype', default=False, help='Filter task for semantic group: (True), (False)', type=click.STRING)
    @click.option('-m', '--match_type', 'run_type', default='overlap', help='Exact or overlapping match: (overlap), (exact)', type=click.STRING)
    def ensemble(corpus):
        """ Analyze ensemble """
        analysisConf =  AnalysisConfig()
        if corpus is None:
            exit(1)
        systems = ['ctakes','biomedicus','clamp','metamap','quick_umls']

        print('run_type:', run_type)
        print('Running ', corpus, analysis_type) 
        
        if filter_semtype:
            print(semtypes)
            ensemble_control(analysisConf.systems, analysis_type, corpus, run_type, filter_semtype, semtypes)
        else:
            ensemble_control(analysisConf.systems, analysis_type, corpus, run_type, filter_semtype)

        start = time.perf_counter()
        run_ensemble(systems, analysis_type, corpus)
        elapsed = (time.perf_counter() - start)
        print('elapsed:', elapsed)

    @analyze.command()
    @click.option('-s', '--systems', 'systems', default='biomedicus', help='Select corpus for analysis: (i2b2), (mipacq), (casi)', type=click.STRING)
    def tests(systems):
        """ Analyze ensemble """
        if systems is None:
            exit(1)
        print('Running ', systems) 

    analyze()
    
    #get_options()
    #main()
'''

