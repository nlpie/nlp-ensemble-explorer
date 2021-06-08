#!/usr/bin/env python

"""
  Copyright (c) 2021 Regents of the University of Minnesota.
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
      http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""


import gevent
import random
import pandas as pd
import numpy as np
import math
import time 
import functools as ft
import glob, os, sys   
import operator as op
import shelve
#import ipywidgets as widgets
#from ipywidgets import interact, interact_manual
from pathlib import Path
from itertools import combinations, product, permutations
from sqlalchemy.engine import create_engine
from datetime import datetime
from ast import literal_eval
from scipy import stats  
from scipy.stats.mstats import gmean
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
from collections import defaultdict
import collections
from typing import List, Set, Tuple 
#from sklearn.metrics import classification_report, confusion_matrix
from scipy import sparse



# STEP-1: CHOOSE YOUR CORPUS
# TODO: get working with list of corpora
#corpora = ['mipacq','i2b2','fairview'] #options for concept extraction include 'fairview', 'mipacq' OR 'i2b2'

# cross-system semantic union merge filter for cross system aggregations using custom system annotations file with corpus name and system name using 'ray_test': 
# need to add semantic type filrering when reading in sys_data
#corpus = 'ray_test'
#corpus = 'clinical_trial2'
corpus = 'fairview'
#corpora = ['i2b2','fairview']

# STEP-2: CHOOSE YOUR DATA DIRECTORY; this is where output data will be saved on your machine
data_directory = '/mnt/DataResearch/DataStageData/ed_provider_notes/output/' 

# STEP-3: CHOOSE WHICH SYSTEMS YOU'D LIKE TO EVALUATE AGAINST THE CORPUS REFERENCE SET
#systems = ['biomedicus', 'clamp', 'ctakes', 'metamap', 'quick_umls']
#systems = ['biomedicus', 'clamp', 'metamap', 'quick_umls']
#systems = ['biomedicus', 'quick_umls']
#systems = ['biomedicus', 'ctakes', 'quick_umls']
systems = ['biomedicus', 'clamp', 'ctakes', 'metamap']
#systems = ['biomedicus', 'clamp']
#systems = ['ctakes', 'quick_umls', 'biomedicus', 'metamap']
#systems = ['biomedicus', 'metamap']
#systems = ['ray_test']
#systems = ['metamap']

# STEP-4: CHOOSE TYPE OF RUN
rtype = 6      # OPTIONS INCLUDE: 1->Single systems; 2->Ensemble; 3->Tests; 4 -> majority vote 
               # The Ensemble can include the max system set ['ctakes','biomedicus','clamp','metamap','quick_umls']
    
# STEP-5: CHOOSE WHAT TYPE OF ANALYSIS YOU'D LIKE TO RUN ON THE CORPUS
analysis_type = 'full' #options include 'entity', 'cui' OR 'full'

# STEP-(6A): ENTER DETAILS FOR ACCESSING MANUAL ANNOTATION DATA
database_type = 'postgresql+psycopg2' # We use mysql+pymql as default
database_username = 'username'
database_password = 'pw' 
database_url = 'host_name' # HINT: use localhost if you're running database on your local machine
database_name = 'covid-19' # Enter database name

def ref_data(corpus):
    return corpus + '_all' # Enter the table within the database where your reference data is stored

table_name = ref_data(corpus)

# STEP-(6B): ENTER DETAILS FOR ACCESSING SYSTEM ANNOTATION DATA

# TODO: snarf up file automatically
def sys_data(corpus, analysis_type):
    if analysis_type == 'entity':
        return 'analytical_'+corpus+'.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
    elif analysis_type in ('cui', 'full'):
        return 'analytical_fairview_cui_filtered_by_semtype_test_1606061637.230739.csv' # 'analytical_'+corpus+'_cui.csv' # OPTIONS include 'analytical_cui_mipacq_concepts.csv' OR 'analytical_cui_i2b2_concepts.csv' 
        
system_annotation = sys_data(corpus, analysis_type)

# STEP-7: CREATE A DB CONNECTION POOL
engine_request = str(database_type)+'://'+database_username+':'+database_password+"@"+database_url+'/'+database_name
engine = create_engine(engine_request, pool_pre_ping=True, pool_size=20, max_overflow=30)

# STEP-(8A): FILTER BY SEMTYPE
filter_semtype = True #False

# STEP-(8B): IF STEP-(8A) == True -> GET REFERENCE SEMTYPES

def ref_semtypes(filter_semtype, corpus):
    if filter_semtype:
        if corpus == 'fairview':
            semtypes = ['Disorders']
        else: pass
        
        return semtypes

semtypes = ref_semtypes(filter_semtype, corpus)

# STEP-9: Set data directory/table for source documents for vectorization
src_table = 'sofa'

# STEP-10: Specificy match type from {'exact', 'overlap', 'cui' -> kludge for majority}
run_type = 'overlap'

# for clinical trial, measurement/temoral are single system since no overlap for intersect
# STEP-11: Specify expression type for run (TODO: run all at once; make less kludgey)
expression_type = 'nested' #'nested_with_singleton' # type of merge expression: nested ((A&B)|C), paired ((A&B)|(C&D)), nested_with_singleton ((A&B)|((C&D)|E)) 
# -> NB: len(systems) for pair must be >= 4, and for nested_with_singleton == 5; single-> skip merges

# STEP-12: Specify type of ensemble: merge or vote
ensemble_type = 'merge' 

# ****** TODO 
# -> add majority vote to union for analysis_type = 'full'
# -> case for multiple labels on same/overlapping span/same system; disambiguate (order by score if exists and select random for ties): done!
# -> port to command line 
# ----------------------->
# -> still need to validate that all semtypes in corpus!
# -> handle case where intersect merges are empty/any confusion matrix values are 0; specificallly on empty df in evaluate method: done!
# -> case when system annotations empty from semtype filter; print as 0: done!
# -> trim whitespace on CSV import -> done for semtypes
# -> eliminate rtype = 1 for expression_type = 'single'
# -> cross-system semantic union merge on aggregation
# -> negation: testing
# -> other modification, such as 'present'
# -> clean up configuration process
# -> allow iteration through all corpora and semtypes
# -> optimize vecorization (remove confusion?)

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
        
        sql = "SELECT st.tui, abbreviation, clamp_name, ctakes_name FROM semantic_groups sg join semantic_types st on sg.tui = st.tui where group_name in ({})"           .format(', '.join(['%s' for _ in semtypes]))  
        
        stypes = pd.read_sql(sql, params=[semtypes], con=engine) 
       
        if len(stypes['tui'].tolist()) > 0:
            self.biomedicus_types = set(stypes['tui'].tolist())
            self.qumls_types = set(stypes['tui'].tolist())
        
        else:
            self.biomedicus_types = None
            self.qumls_types = None
        
        if stypes['clamp_name'].dropna(inplace=True) or len(stypes['clamp_name']) == 0:
            self.clamp_types = None
        else:
            self.clamp_types = set(stypes['clamp_name'].tolist()[0].split(','))
         
        if stypes['ctakes_name'].dropna(inplace=True) or len(stypes['ctakes_name']) == 0:
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
    
def system_semtype_check(sys, semtype, corpus):
    st = SemanticTypes([semtype], corpus).get_system_type(sys)
    if st:
        return sys
    else:
        return None

# annotation class for systems
class AnnotationSystems():
    """   
    System annotations of interest for UMLS concept extraction
    NB: ctakes combines all "mentions" annotation types
    
    """
    def __init__(self):
        
        """ 
        annotation base types
        """   
        
        self.biomedicus_types = ["biomedicus.v2.UmlsConcept"]
        self.clamp_types = ["edu.uth.clamp.nlp.typesystem.ClampNameEntityUIMA"]
        self.ctakes_types = ["ctakes_mentions"]
        self.metamap_types = ["org.metamap.uima.ts.Candidate"]
        self.qumls_types = ["concept_jaccard_score_False"]
       
    def get_system_type(self, system):
        
        """
        return system types
        """
        
        if system == "biomedicus":
            view = "Analysis"
        else:
            view = "_InitialView"

        if system == 'biomedicus':
            types = self.biomedicus_types

        elif system == 'clamp':
            types = self.clamp_types

        elif system == 'ctakes':
            types = self.ctakes_types

        elif system == 'metamap':
            types = self.metamap_types
        
        elif system == "quick_umls":
            types = self.qumls_types
            
        return types, view
    
annSys = AnnotationSystems()


get_ipython().run_line_magic('reload_ext', 'Cython')

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




def get_cooccurences(ref, sys, analysis_type: str, corpus: str):
    """
    get cooccurences between system and reference; exact match; TODO: add relaxed -> done in single system evals during ensemble run
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


def label_vector(doc: str, ann: List[int], labels: List[str]) -> np.array:

    v = np.zeros(doc)
    labels = list(labels)
    
    for (i, lab) in enumerate(labels):
        i += 1  # 0 is reserved for no label
        idxs = [np.arange(a.begin, a.end) for a in ann if a.label == lab]
        idxs = [j for mask in idxs for j in mask]
        v[idxs] = i

    return v

# test confusion matrix elements for vectorized annotation set; includes TN
# https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/

def confused(sys1, ann1):
    TP = np.sum(np.logical_and(ann1 > 0, sys1 == ann1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(ann1 == 0, sys1 == ann1))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(sys1 > 0, sys1 != ann1))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(ann1 > 0, sys1 == 0))
    
    return TP, TN, FP, FN

@ft.lru_cache(maxsize=None)
def vectorized_cooccurences(r: object, analysis_type: str, corpus: str, filter_semtype, semtype = None) -> np.int64:
    docs = get_docs(corpus)
    
    if filter_semtype:
        ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
    else: 
        ann = get_ref_ann(analysis_type, corpus, filter_semtype)
        
    sys = get_sys_ann(analysis_type, r)
    
    #cvals = []
    if analysis_type == 'entity':
        labels = ["concept"]
    elif analysis_type in ['cui', 'full']:
        labels = list(set(ann["value"].tolist()))
    
    sys2 = list()
    ann2 = list()
    s2 = list()
    a2 = list()
    
    for n in range(len(docs)):
        
        if analysis_type != 'cui':
            a1 = list(ann.loc[ann["case"] == docs[n][0]].itertuples(index=False))
            s1 = list(sys.loc[sys["case"] == docs[n][0]].itertuples(index=False))
            ann1 = label_vector(docs[n][1], a1, labels)
            sys1 = label_vector(docs[n][1], s1, labels)

            #TP, TN, FP, FN = confused(sys1, ann1)
            #cvals.append([TP, TN, FP, FN])
            sys2.append(list(sys1))
            ann2.append(list(ann1))

        else:
            a = ann.loc[ann["case"] == docs[n][0]]['label'].tolist()
            s = sys.loc[sys["case"] == docs[n][0]]['label'].tolist()
            x = [1 if x in a else 0 for x in labels]
            y = [1 if x in s else 0 for x in labels]
#             x_sparse = sparse.csr_matrix(x)
#             y_sparse = sparse.csr_matrix(y)
            s2.append(y)
            a2.append(x)
           
            
            #a1 = list(ann.loc[ann["case"] == docs[n][0]].itertuples(index=False))
            #s1 = list(sys.loc[sys["case"] == docs[n][0]].itertuples(index=False))
   
    if analysis_type != 'cui':
        a2 = [item for sublist in ann2 for item in sublist]
        s2 = [item for sublist in sys2 for item in sublist]
        report = classification_report(a2, s2, output_dict=True)
        #print('classification:', report)
        macro_precision =  report['macro avg']['precision'] 
        macro_recall = report['macro avg']['recall']    
        macro_f1 = report['macro avg']['f1-score']
        TN, FP, FN, TP = confusion_matrix(a2, s2).ravel()
        
        #return (np.sum(cvals, axis=0), (macro_precision, macro_recall, macro_f1))
        return ((TP, TN, FP, FN), (macro_precision, macro_recall, macro_f1))
    else:
        x_sparse = sparse.csr_matrix(a2)
        y_sparse = sparse.csr_matrix(s2)
        report = classification_report(x_sparse, y_sparse, output_dict=True)
        macro_precision =  report['macro avg']['precision'] 
        macro_recall = report['macro avg']['recall']    
        macro_f1 = report['macro avg']['f1-score']
        #print((macro_precision, macro_recall, macro_f1))
        return ((0, 0, 0, 0), (macro_precision, macro_recall, macro_f1))
                                       

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
#          'F1': F[1], 
#          'precision': precision[1], 
#          'recall': recall[1], 
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
    systems = AnalysisConfig().systems
    
    sys_ann = pd.read_csv(analysisConf.data_dir + usys_file, dtype={'note_id': str})
    
#     sql = "SELECT * FROM " + ref_table #+ " where semtype in('Anatomy', 'Chemicals_and_drugs')" 
    
#     ref_ann = pd.read_sql(sql, con=engine)
    sys_ann = sys_ann.drop_duplicates()
    ref_ann = None
    
    return ref_ann, sys_ann


import pandas as pd
from scipy import stats
from scipy.stats.mstats import gmean

def geometric_mean(metrics):
    """
    1. Get rank average of F1, TP/FN, TM
        http://www.datasciencemadesimple.com/rank-dataframe-python-pandas-min-max-dense-rank-group/
        https://stackoverflow.com/questions/46686315/in-pandas-how-to-create-a-new-column-with-a-rank-according-to-the-mean-values-o?rq=1
    2. Take geomean of rank averages
        https://stackoverflow.com/questions/42436577/geometric-mean-applied-on-row
    """
    
    data = pd.DataFrame() 

    metrics['F1 rank']=metrics['F1'].rank(ascending=0,method='average')
    metrics['TP/FN rank']=metrics['TP/FN'].rank(ascending=0,method='average')
    metrics['TM rank']=metrics['TM'].rank(ascending=0,method='average')
    metrics['Gmean'] = gmean(metrics.iloc[:,-3:],axis=1)

    return metrics  

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
                system_annotations = sys_ann[sys_ann['semtypes'].isin(st)].copy()
        else:
            system_annotations = sys_ann.copy()

        if (filter_semtype and st) or filter_semtype is False:
            system = system_annotations.copy()

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
def get_ref_n(analysis_type: str, corpus: str, filter_semtype: str) -> int:
    
    ref_ann, _ = get_metric_data(analysis_type, corpus)
    
    if filter_semtype:
        ref_ann = ref_ann[ref_ann['semtype'].isin(SemanticTypes(semtypes, corpus).get_system_type('reference'))]
            
    if corpus == 'casi':
        return len(ref_ann)
        
    else:
        # do not overestimate fn
        if 'entity' in analysis_type:
            ref_ann = ref_ann[['start', 'end', 'file']].drop_duplicates()
        elif 'cui' in analysis_type:
            ref_ann = ref_ann[['value', 'file']].drop_duplicates()
        elif 'full' in analysis_type:
            ref_ann = ref_ann[['start', 'end', 'value', 'file']].drop_duplicates()
        else:
            pass

        ref_n = len(ref_ann.drop_duplicates())

        return ref_n
    
@ft.lru_cache(maxsize=None)
def get_sys_data(system: str, analysis_type: str, corpus: str, filter_semtype, semtype = None) -> pd.DataFrame:
   
    _, data = get_metric_data(analysis_type, corpus)
    
    out = data[data['system'] == system].copy()
    
    if filter_semtype:
        st = SemanticTypes([semtype], corpus).get_system_type(system)
        print(system, 'st', st)
    
    if corpus == 'casi':
        cols_to_keep = ['case', 'overlap'] 
        out = out[cols_to_keep].drop_duplicates()
        return out
        
    else:
        if filter_semtype:
            out = out[out['semtype'].isin(st)].copy()
            
        else:
            out = out[out['system']== system].copy()
            
        if system == 'quick_umls':
            out = out[(out.score.astype(float) >= 0.8) & (out["type"] == 'concept')]
            # fix for leading space on semantic type field
            out = out.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
            out['semtype'] = out['semtype'].str.strip()
        
        if system == 'metamap':
            out = out[out.score.abs().astype(int) >= 800]
            
        if 'entity' in analysis_type:
            cols_to_keep = ['begin', 'end', 'note_id']
        elif 'cui' in analysis_type:
            cols_to_keep = ['cui', 'note_id']
        elif 'full' in analysis_type:
            cols_to_keep = ['begin', 'end', 'cui', 'note_id', 'polarity']

        out = out[cols_to_keep]
        
        return out.drop_duplicates()


class SetTotals(object):
    """ 
    returns an instance with merged match set numbers using either union or intersection of elements in set 
    """
    def __init__(self, ref_n, sys_n, match_set):

        self = self    
        self.ref_ann = ref_n
        self.sys_n = sys_n
        self.match_set = match_set

    def get_ref_sys(self):

        ref_only = self.ref_ann - len(self.match_set)
        sys_only = self.sys_n - len(self.match_set)

        return ref_only, sys_only, len(self.match_set), self.match_set



def union_vote(arg):
    arg['length'] = (arg.end - arg.begin).abs()
    
    df = arg[['begin', 'end', 'note_id', 'cui', 'length', 'polarity']].copy()
    df.sort_values(by=['note_id','begin'],inplace=True)
    df = df.drop_duplicates(['begin', 'end', 'note_id', 'cui', 'polarity'])
    
    cases = set(df['note_id'].tolist())
    data = []
    out = pd.DataFrame()
    
    for case in cases:
        test = df[df['note_id']==case].copy()
        
        for row in test.itertuples():

            iix = pd.IntervalIndex.from_arrays(test.begin, test.end, closed='neither')
            span_range = pd.Interval(row.begin, row.end)
            fx = test[iix.overlaps(span_range)].copy()

            maxLength = fx['length'].max()
            minLength = fx['length'].min()

            if len(fx) > 1: 
                #if longer span exists use as tie-breaker
                if maxLength > minLength:
                    fx = fx[fx['length'] == fx['length'].max()]

            data.append(fx)

    out = pd.concat(data, axis=0)
   
    # Remaining ties on span with same or different CUIs
    # randomly reindex to keep random selected row when dropping duplicates: https://gist.github.com/cadrev/6b91985a1660f26c2742
    out.reset_index(inplace=True)
    out = out.reindex(np.random.permutation(out.index))
    
    return out.drop_duplicates(['begin', 'end', 'note_id', 'polarity']) #out.drop('length', axis=1, inplace=True) 



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
            
    r = Results()
    
    if 'entity' in analysis_type and corpus != 'casi': 
        cols_to_keep = ['begin', 'end', 'note_id', 'polarity'] # entity only
    elif 'full' in analysis_type: 
        cols_to_keep = ['cui', 'begin', 'end', 'note_id','polarity'] # entity only
        join_cols = ['cui', 'begin', 'end', 'note_id', 'polarity']
    elif 'cui' in analysis_type:
        cols_to_keep = ['cui', 'note_id', 'polarity'] # entity only
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
                    
                    if analysis_type == 'full':
                        df = union_vote(df)

                if fn == op.and_:
                    
                    if isinstance(leftC.get(), str) and isinstance(rightC.get(), str):
                        if not left_sys.empty and not right_sys.empty:
                            df = left_sys.merge(right_sys, on=join_cols, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=join_cols)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), str) and isinstance(rightC.get(), pd.DataFrame):
                        if not left_sys.empty and not r_sys.empty:
                            df = left_sys.merge(r_sys, on=join_cols, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=join_cols)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), str):
                        if not l_sys.empty and not right_sys.empty:
                            df = l_sys.merge(right_sys, on=join_cols, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=join_cols)
                        else:
                            df = pd.DataFrame(columns=cols_to_keep)

                    elif isinstance(leftC.get(), pd.DataFrame) and isinstance(rightC.get(), pd.DataFrame):
                        if not l_sys.empty and not r_sys.empty:
                            df = l_sys.merge(r_sys, on=join_cols, how='inner')
                            df = df[cols_to_keep].drop_duplicates(subset=join_cols)
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



"""
Incoming Boolean sentences are parsed into a binary tree.

Test expressions to parse:

sentence = '((((A&B)|C)|D)&E)'

sentence = '(E&(D|(C|(A&B))))'

sentence = '(((A|(B&C))|(D&(E&F)))|(H&I))'

"""
# build parse tree from passed sentence using grammatical rules of Boolean logic
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
        sentence = payload.replace('(', ' ( ').             replace(')', ' ) ').             replace('&', ' & ').             replace('|', ' | ').             replace('  ', ' ')
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
        
    sql = 'select distinct note_id, sofa from sofas where corpus = %(corpus)s order by note_id'
    df = pd.read_sql(sql, params={"corpus":corpus}, con=engine)
    df.drop_duplicates()
    df['len_doc'] = df['sofa'].apply(len)
    
    subset = df[['note_id', 'len_doc']]
    docs = [tuple(x) for x in subset.to_numpy()]
    
    return docs

@ft.lru_cache(maxsize=None)
def get_ref_ann(analysis_type, corpus, filter_semtype, semtype = None):
    
    if filter_semtype:
        if ',' in semtype:
            semtype = semtype.split(',')
        else:
            semtype = [semtype]
        
    ann, _ = get_metric_data(analysis_type, corpus)
    ann = ann.rename(index=str, columns={"start": "begin", "file": "case"})
    
    if filter_semtype:
        ann = ann[ann['semtype'].isin(semtype)]
    if analysis_type == 'entity':   
        ann["label"] = 'concept'
    elif analysis_type in ['cui','full']:
        ann["label"] = ann["value"]
    
    if analysis_type == 'entity':
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'cui':
        cols_to_keep = ['value', 'case', 'label']
    elif analysis_type == 'full':
        cols_to_keep = ['begin', 'end', 'value', 'case', 'label']
    ann = ann[cols_to_keep]
    
    return ann

@ft.lru_cache(maxsize=None)
def get_sys_ann(analysis_type, r):
    sys = r.system_merges   
    
    sys = sys.rename(index=str, columns={"note_id": "case"})
    if analysis_type == 'entity':
        sys["label"] = 'concept'
        cols_to_keep = ['begin', 'end', 'case', 'label']
    elif analysis_type == 'full':
        sys["label"] = sys["cui"]
        cols_to_keep = ['begin', 'end', 'case', 'value', 'label']
    elif analysis_type == 'cui':
        sys["label"] = sys["cui"]
        cols_to_keep = ['case', 'cui', 'label']
    
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
    
    sentence = Sentence(boolean_expression)   
    pt = make_parse_tree(sentence.sentence)
    
    if filter_semtype:
        r = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype)
    else:
        r = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype)
        
    # vectorize merges using i-o labeling
    if run_type == 'overlap':
        if filter_semtype:
             ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype, semtype)
        else:
            ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype)
        
        print('results:',((TP, TN, FP, FN),(p,r,f1)))
        # TODO: validate against ann1/sys1 where val = 1
        # total by number chars
        system_n = TP + FP
        reference_n = TP + FN

        if analysis_type != 'cui':
            d = cm_dict(FN, FP, TP, system_n, reference_n)
        else:
            d = dict()
            d['F1'] = 0
            d['precision'] = 0 
            d['recall'] = 0
            d['TP/FN'] = 0
            d['TM'] = 0
            
        d['TN'] = TN
        d['macro_p'] = p
        d['macro_r'] = r
        d['macro_f1'] = f1
        
        
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



# generate all combinations of given list of annotators:
def partly_unordered_permutations(lst, k):
    elems = set(lst)
    for c in combinations(lst, k):
        for d in permutations(elems - set(c)):
            yield c + d
            
def expressions(l, n):
    for (operations, *operands), operators in product(
            combinations(l, n), product(('&', '|'), repeat=n - 1)):
        for operation in zip(operators, operands):
            operations = [operations, *operation]
        yield operations

# get list of systems with a semantic type in grouping
def get_valid_systems(systems, semtype):
    test = []
    for sys in systems:
        st = system_semtype_check(sys, semtype, corpus)
        if st:
            test.append(sys)

    return test

# permute system combinations and evaluate system merges for performance
def run_ensemble(systems, analysis_type, corpus, filter_semtype, expression_type, semtype = None):
    metrics = pd.DataFrame()
    
    # pass single system to evaluate
    if expression_type == 'single':
        for system in systems:
            if filter_semtype:
                d = get_metrics(system, analysis_type, corpus, run_type, filter_semtype, semtype)
            else:
                d = get_metrics(system, analysis_type, corpus, run_type, filter_semtype)
            d['merge'] = system
            d['n_terms'] = 1

            frames = [metrics, pd.DataFrame(d, index=[0])]
            metrics = pd.concat(frames, ignore_index=True, sort=False) 
    
    elif expression_type == 'nested':
        for l in partly_unordered_permutations(systems, 2):
            print('processing merge combo:', l)
            for i in range(1, len(l)+1):
                test = list(expressions(l, i))
                for t in test:
                    if i > 1:
                        # format Boolean sentence for parse tree 
                        t = '(' + " ".join(str(x) for x in t).replace('[','(').replace(']',')').replace("'","").replace(",","").replace(" ","") + ')'

                    if filter_semtype:
                        d = get_metrics(t, analysis_type, corpus, run_type, filter_semtype, semtype)
                    else:
                        d = get_metrics(t, analysis_type, corpus, run_type, filter_semtype)

                    d['merge'] = t
                    d['n_terms'] = i

                    frames = [metrics, pd.DataFrame(d, index=[0])]
                    metrics = pd.concat(frames, ignore_index=True, sort=False) 
                    
    elif expression_type == 'nested_with_singleton' and len(systems) == 5:
        # form (((a&b)|c)&(d|e))
        
        nested = list(expressions(systems, 3))
        test = list(expressions(systems, 2))
        to_do_terms = []
    
        for n in nested:
            # format Boolean sentence for parse tree 
            n = '(' + " ".join(str(x) for x in n).replace('[','(').replace(']',')').replace("'","").replace(",","").replace(" ","") + ')'

            for t in test:
                t = '(' + " ".join(str(x) for x in t).replace('[','(').replace(']',')').replace("'","").replace(",","").replace(" ","") + ')'

                new_and = '(' + n +'&'+ t + ')'
                new_or = '(' + n +'|'+ t + ')'

                if new_and.count('biomedicus') != 2 and new_and.count('clamp') != 2 and new_and.count('ctakes') != 2 and new_and.count('metamap') != 2 and new_and.count('quick_umls') != 2:

                    if new_and.count('&') != 4 and new_or.count('|') != 4:
                        #print(new_and)
                        #print(new_or)
                        to_do_terms.append(new_or)
                        to_do_terms.append(new_and)
        
        print('nested_with_singleton', len(to_do_terms))
        for term in to_do_terms:
            if filter_semtype:
                d = get_metrics(term, analysis_type, corpus, run_type, filter_semtype, semtype)
            else:
                d = get_metrics(term, analysis_type, corpus, run_type, filter_semtype)
                
            n = term.count('&')
            m = term.count('|')
            d['merge'] = term
            d['n_terms'] = m + n + 1

            frames = [metrics, pd.DataFrame(d, index=[0])]
            metrics = pd.concat(frames, ignore_index=True, sort=False) 
                        
    elif expression_type == 'paired':
        m = list(expressions(systems, 2))
        test = list(expressions(m, 2))

        to_do_terms = []
        for t in test:
            # format Boolean sentence for parse tree 
            t = '(' + " ".join(str(x) for x in t).replace('[','(').replace(']',')').replace("'","").replace(",","").replace(" ","") + ')'
            if t.count('biomedicus') != 2 and t.count('clamp') != 2 and t.count('ctakes') != 2 and t.count('metamap') != 2 and t.count('quick_umls') != 2:
                if t.count('&') != 3 and t.count('|') != 3:
                    to_do_terms.append(t)
                    if len(systems) == 5:
                        for i in systems:
                            if i not in t:
                                #print('('+t+'&'+i+')')
                                #print('('+t+'|'+i+')')
                                new_and = '('+t+'&'+i+')'
                                new_or = '('+t+'|'+i+')'
                                to_do_terms.append(new_and)
                                to_do_terms.append(new_or)
                            
        print('paired', len(to_do_terms))
        for term in to_do_terms:
            if filter_semtype:
                d = get_metrics(term, analysis_type, corpus, run_type, filter_semtype, semtype)
            else:
                d = get_metrics(term, analysis_type, corpus, run_type, filter_semtype)
                
            n = term.count('&')
            m = term.count('|')
            d['merge'] = term
            d['n_terms'] = m + n + 1

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
        file_name += '_'
    
    #metrics = metrics.drop_duplicates(subset=['TP', 'FN', 'FP', 'n_sys', 'precision', 'recall', 'F', 'TM', 'TP/FN', 'TM', 'n_terms'])

    file = file_name + analysis_type + '_' + run_type +'_'
    
    if filter_semtype:
        file += semtype
        
    
    geometric_mean(metrics).to_csv(analysisConf.data_dir + file + str(timestamp) + '.csv')
    print(geometric_mean(metrics))
    
# control ensemble run
def ensemble_control(systems, analysis_type, corpus, run_type, filter_semtype, semtypes = None):
    if filter_semtype:
        for semtype in semtypes:
            test = get_valid_systems(systems, semtype)
            print('SYSTEMS FOR SEMTYPE', semtype, 'ARE', test)
            metrics = run_ensemble(test, analysis_type, corpus, filter_semtype, expression_type, semtype)
            if (expression_type == 'nested_with_singleton' and len(test) == 5) or expression_type in ['nested', 'paired', 'single']:
                generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype, semtype)
    else:
        metrics = run_ensemble(systems, analysis_type, corpus, filter_semtype, expression_type)
        generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype)


# ad hoc query for performance evaluation
def get_merge_data(boolean_expression: str, analysis_type: str, corpus: str, run_type: str, filter_semtype, semtype = None):
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

    r = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype)

    if run_type == 'overlap' and rtype != 6:
        if filter_semtype:
             ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype, semtype)
        else:
             ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype)

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
    return r.system_merges # merge_eval(reference_only, system_only, reference_system_match, system_n, reference_n)



# ad hoc query for extraction
def get_sys_merge(boolean_expression: str, analysis_type: str, corpus: str, run_type: str, filter_semtype, semtype = None):
    """
    Traverse binary parse tree representation of Boolean sentence
        :params: boolean expression in form of '(<annotator_engine_name1><boolean operator><annotator_engine_name2>)'
                 analysis_type (string value of: 'entity', 'cui', 'full') used to filter set of reference and system annotations 
        :return: dictionary with values needed for confusion matrix
    """
#     if filter_semtype:
#         ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)
#     else: 
#         ann = get_ref_ann(analysis_type, corpus, filter_semtype)
    
    sentence = Sentence(boolean_expression)   

    pt = make_parse_tree(sentence.sentence)

    for semtype in semtypes:
            test = get_valid_systems(systems, semtype)
            r = process_sentence(pt, sentence, analysis_type, corpus, filter_semtype, semtype)

#     if run_type == 'overlap' and rtype != 6:
#         if filter_semtype:
#              ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype, semtype)
#         else:
#              ((TP, TN, FP, FN),(p,r,f1)) = vectorized_cooccurences(r, analysis_type, corpus, filter_semtype)

#         # TODO: validate against ann1/sys1 where val = 1
#         # total by number chars
#         system_n = TP + FP
#         reference_n = TP + FN

#         d = cm_dict(FN, FP, TP, system_n, reference_n)
#         print(d)
        
#     elif run_type == 'exact':
#         c = get_cooccurences(ann, r.system_merges, analysis_type, corpus) # get matches, FN, etc.

#         if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
#             # get dictionary of confusion matrix metrics
#             d = cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n)

#             print('cm', d)
#     else:
#         pass
    
    # get matched data from merge
    return r.system_merges # merge_eval(reference_only, system_only, reference_system_match, system_n, reference_n)



# majority vote 
def vectorized_annotations(ann):
    
    docs = get_docs(corpus)
    labels = ["concept"]
    out= []
    
    for n in range(len(docs)):
        a1 = list(ann.loc[ann["case"] == docs[n][0]].itertuples(index=False))
        a = label_vector(docs[n][1], a1, labels)
        out.append(a)

    return out

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def get_reference_vector(analysis_type, corpus, filter_semtype, semtype = None):
    ref_ann = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)

    df = ref_ann.copy()
    df = df.drop_duplicates(subset=['begin','end','case'])
    df['label'] = 'concept'

    cols_to_keep = ['begin', 'end', 'case', 'label']
    ref = df[cols_to_keep].copy()
    test = vectorized_annotations(ref)
    ref =  np.asarray(flatten_list(test), dtype=np.int32) 

    return ref

def majority_overlap_sys(systems, analysis_type, corpus, filter_semtype, semtype = None):
    
    d = {}
    cols_to_keep = ['begin', 'end', 'case', 'label']
    sys_test = []
    
    for system in systems:
        sys_ann = get_sys_data(system, analysis_type, corpus, filter_semtype, semtype)
        df = sys_ann.copy()
        df['label'] = 'concept'
        df = df.rename(index=str, columns={"note_id": "case"})
        sys = df[df['system']==system][cols_to_keep].copy()
        test = vectorized_annotations(sys)
        d[system] = flatten_list(test) 
        sys_test.append(d[system])

    output = sum(np.array(sys_test))
    
    n = int(len(systems) / 2)
    #print(n)
    if ((len(systems) % 2) != 0):
        vote = np.where(output > n, 1, 0)
    else:
        vote = np.where(output > n, 1, 
         (np.where(output == n, random.randint(0, 1), 0)))
        
    return vote

def majority_overlap_vote_out(ref, vote, corpus):    
    TP, TN, FP, FN = confused(ref, vote)
    print(TP, TN, FP, FN)
    system_n = TP + FP
    reference_n = TP + FN

    d = cm_dict(FN, FP, TP, system_n, reference_n)

    d['TN'] = TN
    d['corpus'] = corpus
    print(d)
    
    metrics = pd.DataFrame(d, index=[0])
    
    return metrics

# control vote run
def majority_vote(systems, analysis_type, corpus, run_type, filter_semtype, semtypes = None):
    print(semtypes, systems)
    if filter_semtype:
        for semtype in semtypes:
            test = get_valid_systems(systems, semtype)
            print('SYSYEMS FOR SEMTYPE', semtype, 'ARE', test)
            
            if run_type == 'overlap':
                ref = get_reference_vector(analysis_type, corpus, filter_semtype, semtype)
                vote = majority_overlap_sys(test, analysis_type, corpus, filter_semtype, semtype)
                metrics = majority_overlap_vote_out(ref, vote, corpus)
                #generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype, semtype)
            elif run_type == 'exact':
                sys = majority_exact_sys(test, analysis_type, corpus, filter_semtype, semtype)
                d = majority_exact_vote_out(sys, filter_semtype, semtype)
                metrics = pd.DataFrame(d, index=[0])
            elif run_type == 'cui':
                sys = majority_cui_sys(test, analysis_type, corpus, filter_semtype, semtype)
                d = majority_cui_vote_out(sys, filter_semtype, semtype)
                metrics = pd.DataFrame(d, index=[0])
           
            metrics['systems'] = ','.join(test)
            generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype, semtype)
                
    else:
        if run_type == 'overlap':
            ref = get_reference_vector(analysis_type, corpus, filter_semtype)
            vote = majority_overlap_sys(systems, analysis_type, corpus, filter_semtype)
            metrics = majority_overlap_vote_out(ref, vote, corpus)
            
        elif run_type == 'exact':
            sys = majority_exact_sys(systems, analysis_type, corpus, filter_semtype)
            d = majority_exact_vote_out(sys, filter_semtype)
            metrics = pd.DataFrame(d, index=[0])
            
        elif run_type == 'cui':
            sys = majority_cui_sys(systems, analysis_type, corpus, filter_semtype)
            d = majority_cui_vote_out(sys, filter_semtype)
            metrics = pd.DataFrame(d, index=[0])
            
        metrics['systems'] = ','.join(systems)
        generate_ensemble_metrics(metrics, analysis_type, corpus, ensemble_type, filter_semtype)
    
    print(metrics)
    
def majority_cui_sys(systems, analysis_type, corpus, filter_semtype, semtype = None):
   
    cols_to_keep = ['cui', 'note_id', 'system']
    
    df = pd.DataFrame()
    for system in systems:
        if filter_semtype:
            sys = get_sys_data(system, analysis_type, corpus, filter_semtype, semtype)
        else:
            sys = get_sys_data(system, analysis_type, corpus, filter_semtype)
            
        sys = sys[sys['system'] == system][cols_to_keep].drop_duplicates()
        
        frames = [df, sys]
        df = pd.concat(frames)
        
    return df

def majority_cui_vote_out(sys, filter_semtype, semtype = None):
    
    sys = sys.astype(str)
    sys['value_cui'] = list(zip(sys.cui, sys.note_id.astype(str)))
    sys['count'] = sys.groupby(['value_cui'])['value_cui'].transform('count')

    n = int(len(systems) / 2)
    if ((len(systems) % 2) != 0):
        sys = sys[sys['count'] > n]
    else:
        # https://stackoverflow.com/questions/23330654/update-a-dataframe-in-pandas-while-iterating-row-by-row
        for i in sys.index:
            if sys.at[i, 'count'] == n:
                sys.at[i, 'count'] = random.choice([1,len(systems)])
        sys = sys[sys['count'] > n]

    sys = sys.drop_duplicates(subset=['value_cui', 'cui', 'note_id'])
    ref = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)

    c = get_cooccurences(ref, sys, analysis_type, corpus) # get matches, FN, etc.

    if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
        # get dictionary of confusion matrix metrics
        print(cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n))
        return cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n)
    

def majority_exact_sys(systems, analysis_type, corpus, filter_semtype, semtype = None):
   
    cols_to_keep = ['begin', 'end', 'note_id', 'system']
    
    df = pd.DataFrame()
    for system in systems:
        if filter_semtype:
            sys = get_sys_data(system, analysis_type, corpus, filter_semtype, semtype)
        else:
            sys = get_sys_data(system, analysis_type, corpus, filter_semtype)
            
        sys = sys[sys['system'] == system][cols_to_keep].drop_duplicates()
        
        frames = [df, sys]
        df = pd.concat(frames)
        
    return df
        
def majority_exact_vote_out(sys, filter_semtype, semtype = None):
    sys['span'] = list(zip(sys.begin, sys.end, sys.note_id.astype(str)))
    sys['count'] = sys.groupby(['span'])['span'].transform('count')

    n = int(len(systems) / 2)
    if ((len(systems) % 2) != 0):
        sys = sys[sys['count'] > n]
    else:
        # https://stackoverflow.com/questions/23330654/update-a-dataframe-in-pandas-while-iterating-row-by-row
        for i in sys.index:
            if sys.at[i, 'count'] == n:
                sys.at[i, 'count'] = random.choice([1,len(systems)])
        sys = sys[sys['count'] > n]

    sys = sys.drop_duplicates(subset=['span', 'begin', 'end', 'note_id'])
    ref = get_ref_ann(analysis_type, corpus, filter_semtype, semtype)

    c = get_cooccurences(ref, sys, analysis_type, corpus) # get matches, FN, etc.

    if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
        # get dictionary of confusion matrix metrics
        print(cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n))
        return cm_dict(c.ref_only, c.system_only, c.ref_system_match, c.system_n, c.ref_n)
    


#%%time
def main():
    '''
        corpora: i2b2, mipacq, fv017
        analyses: entity only (exact span), cui by document, full (aka (entity and cui on exaact span/exact cui)
        systems: ctakes, biomedicus, clamp, metamap, quick_umls
        
        TODO -> Vectorization (entity only) -> done:
                add switch for use of TN on single system performance evaluations -> done
                add switch for overlap matching versus exact span -> done
             -> Other tasks besides concept extraction
        
    ''' 
    analysisConf =  AnalysisConfig()
    print(analysisConf.systems, analysisConf.corpus_config())
    
    if (rtype == 1):
        print(semtypes, systems)
        if filter_semtype:
            for semtype in semtypes:
                test = get_valid_systems(systems, semtype)
                print('SYSYEMS FOR SEMTYPE', semtype, 'ARE', test)
                generate_metrics(analysis_type, corpus, filter_semtype, semtype)
            
        else:
            generate_metrics(analysis_type, corpus, filter_semtype)
        
    elif (rtype == 2):
        print('run_type:', run_type)
        if filter_semtype:
            print(semtypes)
            ensemble_control(analysisConf.systems, analysis_type, corpus, run_type, filter_semtype, semtypes)
        else:
            ensemble_control(analysisConf.systems, analysis_type, corpus, run_type, filter_semtype)
    elif (rtype == 3):
        t = ['concept_jaccard_score_false']
        test_systems(analysis_type, analysisConf.systems, corpus)  
        test_count(analysis_type, corpus)
        test_ensemble(analysis_type, corpus)
    elif (rtype == 4):
        if filter_semtype:
            majority_vote(systems, analysis_type, corpus, run_type, filter_semtype, semtypes)
        else:
            majority_vote(systems, analysis_type, corpus, run_type, filter_semtype)
    elif (rtype == 5):
        
        # control filter_semtype in get_sys_data, get_ref_n and generate_metrics. TODO consolidate. 
        # # run single ad hoc statement
        statement = '((ctakes&biomedicus)|metamap)'

        def ad_hoc(analysis_type, corpus, statement):
            sys = get_merge_data(statement, analysis_type, corpus, run_type, filter_semtype)
            sys = sys.rename(index=str, columns={"note_id": "case"})
            sys['label'] = 'concept'

            ref = get_reference_vector(analysis_type, corpus, filter_semtype)
            sys = vectorized_annotations(sys)
            sys = np.asarray(flatten_list(list(sys)), dtype=np.int32)

            return ref, sys

        ref, sys = ad_hoc(analysis_type, corpus, statement)

    elif (rtype == 6): # 5 w/o evaluation
        
        statement = '(((biomedicus&ctakes)&metamap)|clamp)' #'((((biomedicus&ctakes)&metamap)&quick_umls)|clamp)' #'(ctakes|biomedicus)' #((((A∧C)∧D)∧E)∨B)->for covid pipeline

        def ad_hoc(analysis_type, corpus, statement):
            print(semtypes)
            for semtype in semtypes:
                sys = get_sys_merge(statement, analysis_type, corpus, run_type, filter_semtype, semtype)
                sys = sys.rename(index=str, columns={"note_id": "case"})

            return sys

        sys = ad_hoc(analysis_type, corpus, statement).sort_values(by=['case', 'begin'])
        #sys.drop_duplicates(['cui', 'case', 'polarity'],inplace=True)
        sys.to_csv(data_directory + 'test_new_no_dedup.csv')
       
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        test = sys.copy()
        test.drop(['begin','end'], axis=1, inplace=True) 
        test.to_csv(data_directory + 'ensemble_' + str(timestamp) + '.csv')

if __name__ == '__main__':
    get_ipython().run_line_magic('prun', 'main()')
    #main()
    print('done!')
    pass


