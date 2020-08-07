# In[76]:


'''
  Copyright (c) 2019 Regents of the University of Minnesota.
 
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


# In[77]:
import time
import click
import numpy as np
import shelve
import pandas as pd
import numpy as np
import math
#import pymysql
import time 
import functools as ft
import glob   
import operator as op
import shelve
from pandas.api.types import is_numeric_dtype
from itertools import combinations, product
from sqlalchemy.engine import create_engine
from datetime import datetime
from ast import literal_eval
from scipy import stats  
from scipy.stats.mstats import gmean
from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree
from collections import defaultdict
from typing import List, Set, Tuple 
from config import analysis_type, engine, data_dir, dir_out, systems, system_annotations, reference_annotations, database_name, database_type, database_username, database_password, database_url
from typesystems import Annotations 
    
#engine = create_engine('mysql+pymysql://gms:nej123@localhost/test', pool_pre_ping=True, pool_size=20, max_overflow=30)


# In[78]:


# config class for analysis
class AnalysisConfig(object):
    """
    Configuration object:
    notes by test, full per corpus
    paths by output, gold and system location
    """
    def __init__(self):
        self = self    
       
        self.systems = systems      
        self.data_dir = data_dir
    
    def corpus_config(self):
        
        usys_data = system_annotations
        ref_data = database_name+'.'+reference_annotations

        return usys_data, ref_data
        

# In[79]:

analysisConf =  AnalysisConfig()

# annotation class for UIMA systems
class AnnotationSystems(object):
    """   
    CAS XMI Annotations of interest
    
    """
    def __init__(self):
        
        """ 
        annotation base types
        """   
        
        self.biomedicus_dir = "biomedicus_out/"
        self.biomedicus_types = ["biomedicus.v2.UmlsConcept"]
                                  #"biomedicus.v2.Negated"
                                 #"biomedicus.v2.Acronym",
                                 #"biomedicus.v2.DictionaryTerm",
                                 #"biomedicus.v2.Historical"]
        
        self.clamp_dir = "clamp_out/"
        self.clamp_types = ["edu.uth.clamp.nlp.typesystem.ClampNameEntityUIMA"]
                             #"org.apache.ctakes.typesystem.type.syntax.ConllDependencyNode",
                             #"edu.uth.clamp.nlp.typesystem.ClampRelationUIMA"]    
        
        self.ctakes_dir = "ctakes_out/"
        self.ctakes_types = ['ctakes_mentions']#"org.apache.ctakes.typesystem.type.textspan.Sentence",
                             #"org.apache.ctakes.typesystem.type.textsem.DiseaseDisorderMention",
                             #"org.apache.ctakes.typesystem.type.textsem.MedicationMention",
                             #"org.apache.ctakes.typesystem.type.textsem.ProcedureMention",
                             #"org.apache.ctakes.typesystem.type.refsem.UmlsConcept",
                             #"org.apache.ctakes.typesystem.type.textsem.SignSymptomMention",
                             #"org.apache.ctakes.typesystem.type.textsem.AnatomicalSiteMention"]
                             #"org.apache.ctakes.typesystem.type.textsem.MeasurementAnnotation",
                             #"org.apache.ctakes.typesystem.type.textsem.EventMention",
                             #"org.apache.ctakes.typesystem.type.textsem.EntityMention",
                             #"org.apache.ctakes.typesystem.type.textsem.Predicate",
                             #"org.apache.ctakes.typesystem.type.syntax.WordToken"]
        
        self.metamap_dir = "metamap_out/"
        self.metamap_types = [#"org.metamap.uima.ts.Utterance",
                              #"org.metamap.uima.ts.Span",
                              #"org.metamap.uima.ts.Phrase"]
                              "org.metamap.uima.ts.Candidate"]
                              #"org.metamap.uima.ts.CuiConcept",
                              #"org.metamap.uima.ts.Negation"]
                
        self.quick_umls_types = [#'concept']#,
                                #'concept_cosine_length_false',
                                #'concept_cosine_length_true',
                                #'concept_cosine_score_false',
                                #'concept_cosine_score_true',
                                #'concept_dice_length_false',
                                #'concept_dice_length_true',
                                #'concept_dice_score_false',
                                #'concept_dice_score_true',
                                #'concept_jaccard_length_false',
                                #'concept_jaccard_length_true',
                                'concept_jaccard_score_False']
                                #'concept_jaccard_score_true']
                
        
        '''

        self.biomedicus_dir = "biomedicus_out/"
        self.biomedicus_types = [#"biomedicus.v2.UmlsConcept"]
                                  #"biomedicus.v2.Negated"
                                 "biomedicus.v2.Acronym",
                                 "biomedicus.v2.DictionaryTerm",
                                 "biomedicus.v2.Historical"]
        
        
        self.clamp_dir = "clamp_out/"
        #self.clamp_types = [#"edu.uth.clamp.nlp.typesystem.ClampNameEntityUIMA"]
                             #"org.apache.ctakes.typesystem.type.syntax.ConllDependencyNode",
                             #"edu.uth.clamp.nlp.typesystem.ClampRelationUIMA"]
        
        
        self.ctakes_dir = "ctakes_out/"
        self.ctakes_types = ["org.apache.ctakes.typesystem.type.textspan.Sentence",
                             #"org.apache.ctakes.typesystem.type.textsem.DiseaseDisorderMention",
                             #"org.apache.ctakes.typesystem.type.textsem.MedicationMention",
                             #"org.apache.ctakes.typesystem.type.textsem.ProcedureMention",
                             #"org.apache.ctakes.typesystem.type.refsem.UmlsConcept",
                             #"org.apache.ctakes.typesystem.type.textsem.SignSymptomMention",
                             #"org.apache.ctakes.typesystem.type.textsem.AnatomicalSiteMention"]
                             #"org.apache.ctakes.typesystem.type.textsem.MeasurementAnnotation",
                             #"org.apache.ctakes.typesystem.type.textsem.EventMention",
                             #"org.apache.ctakes.typesystem.type.textsem.EntityMention",
                             "org.apache.ctakes.typesystem.type.textsem.Predicate",
                             "org.apache.ctakes.typesystem.type.syntax.WordToken"]
        
        self.metamap_dir = "metamap_out/"
        self.metamap_types = ["org.metamap.uima.ts.Utterance",
                              "org.metamap.uima.ts.Span",
                              "org.metamap.uima.ts.Phrase"]
                              #"org.metamap.uima.ts.Candidate"]
                              #"org.metamap.uima.ts.CuiConcept",
                              #"org.metamap.uima.ts.Negation"]
                              
        '''
       
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
            output = self.biomedicus_dir

        elif system == 'clamp':
            types = self.clamp_types
            output = self.clamp_dir

        elif system == 'ctakes':
            types = self.ctakes_types
            output = self.ctakes_dir

        elif system == 'metamap':
            types = self.metamap_types
            output = self.metamap_dir
        
        elif system == "quick_umls":
            types = self.quick_umls_types
            output = None
            
        return types, view, output
    
annSys = AnnotationSystems()


# In[80]:

# In[81]:


# In[82]:


def get_notes(analysis_type: str, corpus: str) -> List[str]:
    
    if 'test' in analysis_type:
        # test set of notes
        if corpus == 'mipacq':
            notes = ['522412787',
             '617637585',
             '3307880735-8',
             '9080688558',
             '618370565',
             '573718188',
             '534584',
             '60891',
             '62620',
             '616172834']
            
        elif corpus == 'i2b2':
            print('TODO')
        
        print('TEST NOTES!')
     
    else:
        
        if corpus == 'mipacq':
        # these did not meet the minimal criteria for parsing
            notes = ["0595040941-0",
                    "0778429553-0",
                    "1014681675",
                    "2889522952-2",
                    "3080383448-5",
                    "3300000926-3",
                    "3360037185-3",
                    "3580973392",
                    "3627629462-3",
                    "4323116051-4",
                    "477704053-4",
                    "528317073",
                    "531702602",
                    "534061073",
                    "54832076",
                    "5643725437-6",
                    "5944412090-5",
                    "6613169476-6",
                    "7261075903-7",
                    "7504944368-7",
                    "7999462393-7",
                    "8131081430",
                    "8171084310",
                    "8193787896",
                    "8295055184-8",
                    "8823185307-8"]
            
        elif corpus == 'i2b2':
            # these notes were not processed 
            notes = ['0081', 
                     '0401']

        else:
            notes = None
            
    return notes# training_notes


# In[ ]:

# In[ ]:

# In[83]:

class Metrics(object):
    """
    metrics class:
    returns an instance with confusion matrix metrics
    """
    def __init__(self, system_only, gold_only, gold_system_match, system_n, neither = 0):

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
                recall = np.diag(c) / np.sum(c, axis = 1)
                precision = np.diag(c) / np.sum(c, axis = 0)
                F = 2*(precision*recall)/(precision + recall)
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


# In[84]:


def write_out(name: str, analysis_type: str, c: object):
   
    """
    write matching and reference-only sets to file for ease in merging combinations
    """
    
    # write output to file
    dir_out = analysisConf.data_dir + 'single_system_out/'
    with open(dir_out + name + '_' + analysis_type + '_' + c.corpus + '_matches.txt', 'w') as f:
        for item in list(c.matches):
            f.write("%s\n" % str(item))

    # write to file
    with open(dir_out + name + '_' + analysis_type + '_' + c.corpus + '_ref_only.txt', 'w') as f:
        for item in list(c.false_negatives):
            f.write("%s\n" % str(item))


# In[ ]:

# In[85]:


def label_vector(doc: str, ann: List[int], labels: List[str]) -> np.array:

    #print(ann, doc, labels)

    v = np.zeros(doc)
    labels = list(labels)
    
    for (i, lab) in enumerate(labels):
        i += 1  # 0 is reserved for no label
        idxs = [np.arange(a.begin, a.end) for a in ann if a.label == lab]
            
        idxs = [j for mask in idxs for j in mask]
        v[idxs] = i

    return v

# test confusion matrix elements for vectorized annotation set; includes TN
def confused(sys1, ann1):
    TP = np.sum(np.logical_and(sys1 >= 1, ann1 == sys1 ))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(sys1 == 0, ann1 == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(sys1 >= 1, ann1 == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(sys1 == 0, ann1 >= 1))
    
    return TP, TN, FP, FN


def get_cooccurences(ref, sys, analysis_type: str, corpus: str, single_sys = True, name = None):
    """
    get coocurences between system and reference; exact match; TODO: add relaxed
    """
    # test cooccurences
    class Coocurences(object):
        
        def __init__(self):
            self.ref_system_match = 0
            self.ref_only = 0
            self.system_only = 0
            self.system_n = 0
            self.ref_n = 0
            self.matches = set()
            self.false_negatives = set()
            self.corpus = corpus
            #self.cases = set(ref["file"].tolist()) # cases to label 

    c = Coocurences()
    
    # test for converting to vectorization and i-o labeling
    def test_io():
        test = c.cases
        if analysis_type == 'entity':
            docs = [(x, len(open("/Users/gms/development/nlp/nlpie/data/amicus-u01/i2b2/source_data/test_data/" + x + ".txt", 'r').read())) for x in test]
        elif analysis_type == 'full':
            docs = [(x, len(open("/Users/gms/development/nlp/nlpie/data/amicus-u01/mipacq/source_data/source/" + x + ".source", 'r').read())) for x in test]

        ann = ref.copy()
        ann = ann.rename(index=str, columns={"start": "begin", "file": "case"}).copy()
        cols_to_keep = ['begin', 'end', 'case', 'label']
        if analysis_type == 'entity':
            labels = ["concept"]
            ann["label"] = 'concept'
            ann = ann[cols_to_keep].copy()
        elif analysis_type == 'full':  
            ann["label"] = ann["value"]
            sys["label"] = sys["cui"]
            labels = set(ref['value'].tolist())
            print('labels', len(set(labels)))

        sys_ = sys.rename(index=str, columns={"note_id": "case"}).copy()
        
        # need for enttity-only
        if analysis_type == 'entity':
            sys_["label"] = 'concept'
        
        sys_ = sys_[cols_to_keep]
       
        tp = []
        tn = []
        fp = []
        fn = []
        cvals = []
        out = []
        t = []
        d = defaultdict(list)
        
        for n in range(len(docs)):
            a1 = [i for i in ann[ann["case"] == docs[n][0]].copy().itertuples(index=False)]
            s1 = [i for i in sys_[sys_["case"] == docs[n][0]].copy().itertuples(index=False)]

            ann1 = label_vector(docs[n][1], a1, labels)
            sys1 = label_vector(docs[n][1], s1, labels)
            
            TP, TN, FP, FN = confused(sys1, ann1)
            cvals.append([TP, TN, FP, FN])
            
                 
            d['sys'].append(list([int(i) for i in sys1]))
            d['oracle'].append(list([int(i) for i in ann1]))
            d['case'].append(docs[n][0])
            
            '''
            print("tn:", np.intersect1d(np.where(ann1 == 0)[0], np.where(sys1 == 0)[0]),  
                  "tp:", np.intersect1d(np.where(ann1 == 1)[0], np.where(sys1 == 1)[0]), 
                  "fn:", np.intersect1d(np.where(ann1 == 1)[0], np.where(sys1 == 0)[0]), 
                  "fp:", np.intersect1d(np.where(ann1 == 0)[0], np.where(sys1 == 1)[0]))
            '''
        d['labels'] = labels
        
        corp = shelve.open('/Users/gms/Desktop/' + sys.name + '.dat')
        
        for k in d:
            corp[k] = d[k]
        
        corp.close()
       
        return cvals
    
    '''
    TP, TN, FP, FN = np.sum(test_io(), axis=0)
    F, recall, precision, TP, FP, FN, TP_FN_R, TM = Metrics(FP, FN, TP, len(sys), TN).get_confusion_metrics() #no TN
    print('test_io():', TP, TN, FP, FN, F, recall, precision)
    
    '''
    # non-vectorized:
    
    if corpus != 'casi':
        if 'entity' in analysis_type and single_sys: # mipacq n -> 16793
            cols_to_keep = ['begin', 'end', 'note_id']
            sys = sys[cols_to_keep].drop_duplicates()
            ref = ref[['start', 'end', 'file']].drop_duplicates()
            sys.name = name
        elif 'cui' in analysis_type and single_sys: # mipacq n -> 10799
            cols_to_keep = ['cui', 'note_id']
            sys = sys[cols_to_keep].drop_duplicates()
            # do not overestimate FP
            sys = sys[~sys['cui'].isnull()] 
            ref = ref[['value', 'file']].drop_duplicates()
            ref = ref[~ref['value'].isnull()]
            sys.name = name
        elif 'full' in analysis_type and single_sys: # mipacq n -> 17393
            cols_to_keep = ['begin', 'end', 'cui', 'note_id']
            sys = sys[cols_to_keep].drop_duplicates()
            sys = sys[~sys['cui'].isnull()]
            ref = ref[['start', 'end', 'value', 'file']].drop_duplicates()
            ref = ref[~ref['value'].isnull()]
            sys.name = name

        # matches via inner join
        matches = pd.merge(sys, ref, how = 'inner', left_on=['begin','end','note_id'], right_on = ['start','end','file']) 
        # reference-only via left outer join
        fn = pd.merge(ref, sys, how = 'left', left_on=['start','end','file'], right_on = ['begin','end','note_id']) 

        fn = fn[fn['begin'].isnull()] # get as outer join with no match

        if 'entity' in analysis_type and single_sys:
            cols_to_keep = ['start', 'end', 'file']
        else:
            cols_to_keep = ['start', 'end', 'value', 'file']


        matches = matches[cols_to_keep]
        fn = fn[cols_to_keep]

        # use for metrics 
        c.matches = c.matches.union(df_to_set(matches, analysis_type, 'ref'))
        c.false_negatives = c.false_negatives.union(df_to_set(fn, analysis_type, 'ref'))
        c.ref_system_match = len(c.matches)
        c.system_only = len(sys) - len(c.matches)
        c.system_n = len(sys)
        c.ref_n = len(ref)
        c.ref_only = len(c.false_negatives)
        
    else:
        #matches = df_to_set(pd.read_sql("select `case` from test.amia_2019_analytical_v where overlap = 1;", con=engine), 'entity', 'sys', 'casi')
        
        
        sql = "select `case` from test.amia_2019_analytical_v where overlap = 1 and `system` = %(sys.name)s"  
        #ref_ann = pd.read_sql(sql, params={"training_notes":training_notes}, con=engine)
        
        matches = pd.read_sql(sql, params={"sys.name":sys.name}, con=engine)
        
        sql = "select `case` from test.amia_2019_analytical_v where (overlap = 0 or overlap is null) and `system` = %(sys.name)s"  
        #ref_ann = pd.read_sql(sql, params={"training_notes":training_notes}, con=engine)
        
        fn = pd.read_sql(sql, params={"sys.name":sys.name}, con=engine)
        
        c.matches = df_to_set(matches, 'entity', 'sys', 'casi')
        c.fn = df_to_set(fn, 'entity', 'sys', 'casi')
        c.ref_system_match = len(c.matches)
        c.system_only = len(sys) - len(c.matches)
        c.system_n = len(matches) + len(fn)
        c.ref_n = len(matches) + len(fn)
        c.ref_only = len(fn)
        
        print('cooc', c.ref_system_match, c.system_only, c.ref_n, c.ref_n, c.ref_only)
        
    # sanity check
    if len(ref) - c.ref_system_match < 0:
        print('Error: ref_system_match > len(ref)!')
    if len(ref) != c.ref_system_match + c.ref_only:
        print('Error: ref count mismatch!')
   
    # save TP/FN
    if single_sys and corpus != 'casi':
        print(analysis_type)
        write_out(sys.name, analysis_type, c)
    return c 


# In[86]:


# merging test for i-o labeled data
# load shelve
def read_shelve():
        corp = shelve.open('/Users/gms/Desktop/test.dat')
        #print(corp['case'])
        
        return corp
        
test = read_shelve()


t0 = np.array(test['oracle'][3][0:750])
t1 = np.array(test['oracle'][5][0:750])

#l0 = list(t0)
#l1 = list(t1)

def test_merge_vector(test):
    # get sample for testing
    for case in test['case'][3:5]:
        for i in range(len(test['case'][3:5])):
            if i == 3:
                t0 = test['oracle'][3][0:750]
            else:
                t1 = test['oracle'][4][0:750]

            #print('case:', case, test['sys'][i], test['oracle'][i], confused(np.array(test['sys'][i]), np.array(test['oracle'][i])))
        #print(t0, t1)

    t0 = np.array(test['oracle'][3][0:750])
    t1 = np.array(test['oracle'][5][0:750])

    l0 = list(t0)
    l1 = list(t1)
    
    #l0 = [0, 4, 1, 4, 4, 0, 0, 0, 8, 0, 0] 
    #l1 = [0, 1, 4, 4, 0, 0, 0, 0, 8, 8, 8]

    def intersection(lst1, lst2): 
        out = list()
        if isinstance(lst1, set) and isinstance(lst2, set):
            out = (set(lst1) & set(lst2))
        elif isinstance(lst1, set) and isinstance(lst2, np.int64):
            out = (set(lst1) & set([lst2]))
        elif isinstance(lst1, np.int64) and isinstance(lst2, set):
            out = (set([lst1]) & set(lst2))
        elif isinstance(lst1, np.int64) and isinstance(lst2, np.int64):
            out = (set([lst1]) & set([lst2]))
        #if len(out) > 1:
        return out
        #elif len(out) == 1:
        #    return out[0]
        #else:
        #    return 0

    def union(lst1, lst2): 
        out = list()
        if isinstance(lst1, set) and isinstance(lst2, set):
            out = set(lst1) | set(lst2)
        elif isinstance(lst1, set) and isinstance(lst2, np.int64):
            out = set(lst1) | set([lst2])
        elif isinstance(lst1, np.int64) and isinstance(lst2, set):
            out = set([lst1]) | set(lst2)
        elif isinstance(lst1, np.int64) and isinstance(lst2, np.int64):
            out = set([lst1]) | set([lst2])
        #if len(out) == 1:
        #    #out = out[0]
        return out

    # union and intersect
    def umerges(l0, l1):
        #un = [0]*len(l0)
        #for i in range(len(l0)):
        #    un[i] = union(l0[i], l1[i])

        return [union(l0[i], l1[i]) for i in range(len(l0))]

    
    x = umerges(l0, l1)

    #l2 = [1, {1, 4}, {3}, {2, 4}, {1}, 0, 2, 3, {0, 8}, {1, 8}]
    
    #print(umerges(x, l2))
    
    def imerges(l0, l1):
        #inter = [0]*len(l0)
        #for i in range(len(l0)):
        

        return [intersection(l0[i], l1[i]) for i in range(len(l0))]
    
    
    '''
    union = [
        ( [set(x) | set(y)] if isinstance(x, list) and isinstance(y, list)
          else [set(x) | set([y])] if isinstance(x, list) and isinstance(y, int)
          else [set([x]) | set(y)] if isinstance(x, int) and isinstance(y, list)
          else [set([x]) | set([y])])

         for x, y in zip(l0, l1)
    ]

    # unpack map object
    #*y, = list(map(list, zip(*union)))
    #%timeit list(map(list, zip(*union)))

    intersection = [
        ( [set(x) & set(y)] if isinstance(x, list) and isinstance(y, list)
          else [set(x) & set([y])] if isinstance(x, list) and isinstance(y, int)
          else [set([x]) & set(y)] if isinstance(x, int) and isinstance(y, list)
          else [set([x]) & set([y])])
          for x, y in zip(l0, l1)

    ]

    #*x, = list(map(list, zip(*intersection)))
    #%timeit list(map(list, zip(*intersection)))
    '''
# test_merge_vector(test)


# In[87]:


test = read_shelve()
t0 = np.array(test['oracle'][3][0:750])
t1 = np.array(test['oracle'][5][0:750])

l0 = list(t0)
l1 = list(t1)


def imerge(l0, l1):

    return [
        ( [
            set(x) & set(y)] if isinstance(x, list) and  isinstance(y, list)
            else [set(x) & set([y])] if isinstance(x, list) and  isinstance(y, np.int64)
            else [set(x) & y] if isinstance(x, list) and isinstance(y, set)
            else [x & y] if isinstance(x, set) and isinstance(y, set)
            else [x & set(y)] if isinstance(x, set) and isinstance(y, list)
            else [x & set([y])] if isinstance(x, set) and isinstance(y, np.int64)
            else [set([x]) & set(y)] if isinstance(x, np.int64) and  isinstance(y, list)
            else [set([x]) & y] if isinstance(x, np.int64) and isinstance(y, set)
            else [set([x]) & set([y])])
        for x, y in zip(l0, l1)
    ]

def umerge(l0, l1):

    return [
        ( [
            set(x) | set(y)] if isinstance(x, list) and  isinstance(y, list)
            else [set(x) | set([y])] if isinstance(x, list) and  isinstance(y, np.int64)
            else [set(x) | y] if isinstance(x, list) and isinstance(y, set)
            else [x | y] if isinstance(x, set) and isinstance(y, set)
            else [x | set(y)] if isinstance(x, set) and isinstance(y, list)
            else [x | set([y])] if isinstance(x, set) and isinstance(y, np.int64)
            else [set([x]) | y] if isinstance(x, np.int64) and isinstance(y, set)
            else [set([x]) | set(y)] if isinstance(x, np.int64) and  isinstance(y, list)
            else [set([x]) | set([y])])
        for x, y in zip(l0, l1)
    ]

'''
start = time.perf_counter()
z = *map(list, zip(*umerge(l0, l1))),
#print(*map(list, zip(*imerge(l0, l1))),)
elapsed = (time.perf_counter() - start)
print('time 1:', elapsed)
%timeit  *map(list, zip(*umerge(l0, l1))),

*map(list, zip(*umerge(z[0], l1))),


start = time.perf_counter()
z = *map(list, zip(*imerge(l0, l1))),
elapsed = (time.perf_counter() - start)
print('time 2:', elapsed)
%timeit *map(list, zip(*imerge(l0, l1))),

%timeit *map(list, zip(*imerge(z[0], l1))),
'''


# In[88]:

def get_metric_data(training_notes: List[str], analysis_type: str, corpus: str):
    engine_request = str(database_type)+'://'+database_username+':'+database_password+"@"+database_url+'/'+database_name
    engine = create_engine(engine_request, pool_pre_ping=True, pool_size=20, max_overflow=30)
   
    usys_file, ref_table = AnalysisConfig().corpus_config()
    systems = AnalysisConfig().systems
    
    sys_ann = pd.read_csv(analysisConf.data_dir + usys_file, dtype={'note_id': str})
    
    if 'test' not in analysis_type:
        if corpus != 'fairview':
            sql = "SELECT * FROM " + ref_table + " where file not in %(training_notes)s"  
            sys_ann = sys_ann[~sys_ann['note_id'].isin(training_notes)]
        else:
            sql = "SELECT * FROM " + ref_table  
            sys_ann = sys_ann
            
        
    else:
        sql = "SELECT * FROM " + ref_table + " where file in %(training_notes)s"  
        sys_ann = sys_ann[sys_ann['note_id'].isin(training_notes)]
    
    ref_ann = pd.read_sql(sql, params={"training_notes":training_notes}, con=engine)
    sys_ann = sys_ann.drop_duplicates()
    
    return ref_ann, sys_ann

# In[89]:

def geometric_mean(metrics):
    """
    1. Get rank average of F1, TP/FN, TM
        http://www.datasciencemadesimple.com/rank-dataframe-python-pandas-min-max-dense-rank-group/
        https://stackoverflow.com/questions/46686315/in-pandas-how-to-create-a-new-column-with-a-rank-according-to-the-mean-values-o?rq=1
    2. Take geomean of 2.
        https://stackoverflow.com/questions/42436577/geometric-mean-applied-on-row
    """
    
    data = pd.DataFrame() 

    metrics['F1 rank']=metrics['F'].rank(ascending=0,method='average')
    metrics['TP/FN rank']=metrics['TP/FN'].rank(ascending=0,method='average')
    metrics['TM rank']=metrics['TM'].rank(ascending=0,method='average')
    metrics['Gmean'] = gmean(metrics.iloc[:,-3:],axis=1)

    return metrics  

# In[90]:


def generate_metrics(analysis_type: str, corpus: str, single_sys = None):
    start = time.time()

    systems = AnalysisConfig().systems
    metrics = pd.DataFrame()

    training_notes = get_notes(analysis_type, corpus)
    ref_ann, sys_ann = get_metric_data(training_notes, analysis_type, corpus)
    
    for sys in systems:
            types, _, _ = AnnotationSystems().get_system_type(sys) # system types for iterable
            for t in types:
                print(t)
                system = pd.DataFrame()
                
                system_annotations = sys_ann.copy()
                
                system = system_annotations[system_annotations['type'] == str(t)]
            
                if sys == 'quick_umls':
                    system = system[system.score.astype(float) >= 0.75]
            
                system = system.drop_duplicates()
                system.name = sys
                
                c = get_cooccurences(ref_ann, system, analysis_type, corpus, True, system.name) # get matches, FN, etc.
                
                print(c.ref_n, c.ref_only, c.system_n, c.system_only, c.ref_system_match)
                
            if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
                F, recall, precision, TP, FP, FN, TP_FN_R, TM = Metrics(c.system_only, c.ref_only, c.ref_system_match, c.system_n).get_confusion_metrics(corpus)
                
                if corpus == 'casi':
                    if sys == 'biomedicus':
                        t = 'biomedicus.v2.Acronym'
                        
                    d = {'system': sys, 
                         'type': t, 
                         'F': F, 
                         'precision': precision, 
                         'recall': recall, 
                         'FN': FN, 
                         'TP/FN': TP_FN_R,
                         'n_gold': c.ref_n, 
                         'n_sys': c.system_n, 
                         'TM': TM}
                else:
                    d = {'system': sys, 
                         'type': t, 
                         'F': F[1], 
                         'precision': precision[1], 
                         'recall': recall[1], 
                         'TP': TP, 
                         'FN': FN, 
                         'FP': FP, 
                         'TP/FN': TP_FN_R,
                         'n_gold': c.ref_n, 
                         'n_sys': c.system_n, 
                         'TM': TM}

                data = pd.DataFrame(d,  index=[0])
                metrics = pd.concat([metrics, data], ignore_index=True)
                metrics.drop_duplicates(keep='last', inplace=True)
            else:
                print("NO EXACT MATCHES FOR", t)
            elapsed = (time.time() - start)
            print("elapsed:", sys, elapsed)
     
    elapsed = (time.time() - start)
    print(geometric_mean(metrics))
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    
    if single_sys is None:
        file_name = 'metrics_'
    
    metrics.to_csv(analysisConf.data_dir + corpus + '_' + file_name + analysis_type + '_' + str(timestamp) + '.csv')
    
    print("total elapsed time:", elapsed) 

# use to iterate through mm scores
def generate_metrics_mm(analysis_type: str, corpus: str, single_sys = None):
    start = time.time()
    #systems = ["biomedicus","ctakes","metamap","clamp","quick_umls"]
    systems = AnalysisConfig().systems
    #systems = ["quick_umls"]
    metrics = pd.DataFrame()

    training_notes = get_notes(analysis_type, corpus)
    ref_ann, sys_ann = get_metric_data(training_notes, analysis_type, corpus)
    
    sys_ann = sys_ann[(sys_ann.score.notnull()) & (sys_ann['system'] == 'metamap')]
    sys_ann = sys_ann[['begin', 'end', 'note_id', 'system', 'score']].drop_duplicates()
    sys_ann.score = sys_ann.score.astype(int)
    
    for sys in systems:
        types, _, _ = AnnotationSystems().get_system_type(sys) # system types for iterable
        for t in types:
            print(t)

            for i in range(500, 1050, 50): 

                sys_ann = sys_ann[(sys_ann["score"] >= i)].copy()

                sys_ann.name = sys + str(i)

                c = get_cooccurences(ref_ann, sys_ann, analysis_type, corpus, True, sys_ann.name) # get matches, FN, etc.

                print(c.ref_n, c.ref_only, c.system_n, c.system_only, c.ref_system_match)

                #print(i, len(system))

                if c.ref_system_match > 0: # compute confusion matrix metrics and write to dictionary -> df
                    F, recall, precision, TP, FP, FN, TP_FN_R, TM = Metrics(c.system_only, c.ref_only, c.ref_system_match, c.system_n).get_confusion_metrics()
                    d = {'system': sys + '_score_' + str(i), 
                         'type': t, 
                         'F': F[1], 
                         'precision': precision[1], 
                         'recall': recall[1], 
                         'TP': TP, 
                         'FN': FN, 
                         'FP': FP, 
                         'TP/FN': TP_FN_R,
                         'n_gold': c.ref_n, 
                         'n_sys': c.system_n, 
                         'TM': TM}

                    data = pd.DataFrame(d,  index=[0])
                    metrics = pd.concat([metrics, data], ignore_index=True)
                    metrics.drop_duplicates(keep='last', inplace=True)
                else:
                    print("NO EXACT MATCHES FOR", t)
                elapsed = (time.time() - start)
                print("elapsed:", sys, elapsed)
     
    elapsed = (time.time() - start)
    print(geometric_mean(metrics))
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    # UIMA or QuickUMLS
    if single_sys is None:
        file_name = 'mm_metrics_'
    metrics.to_csv(analysisConf.data_dir + corpus + '_' + file_name + analysis_type + '_' + str(timestamp) + '.csv')
    
    print("total elapsed time:", elapsed) 


# In[91]:


# read in system matches from file

def get_ref_n(analysis_type: str, corpus) -> int:
    
    training_notes = get_notes(analysis_type, corpus)
    ref_ann, _ = get_metric_data(training_notes, analysis_type, corpus)
    
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

def get_sys_data(system: str, analysis_type: str, corpus: str) -> int: 
   
    training_notes = get_notes(analysis_type, corpus)
    _, data = get_metric_data(training_notes, analysis_type, corpus)
    
    out = data[data['system'] == system].copy()
    
    if corpus == 'casi':
        cols_to_keep = ['case', 'overlap'] 
        #cols_to_keep = ['case', 'begin', 'end'] 
        out = out[cols_to_keep].drop_duplicates()
        
        return out
        
    else:
        out = data[data['system']== system].copy()

        if system == 'quick_umls':
            out = out[(out.score.astype(float) >= 0.75) & (out["type"] == 'concept_jaccard_score_False')]

        if 'entity' in analysis_type:
            cols_to_keep = ['begin', 'end', 'note_id']
        elif 'cui' in analysis_type:
            cols_to_keep = ['cui', 'note_id']
        elif 'full' in analysis_type:
            cols_to_keep = ['begin', 'end', 'cui', 'note_id']

        out = out[cols_to_keep]

        return out.drop_duplicates()

def get_system_matches(system: str, analysis_type: str, corpus: str):
   
    if corpus == 'casi':
        
        sql = "select `case`, overlap from test.amia_2019_cases where overlap = 1 and `system` = %(system)s"  
        #ref_ann = pd.read_sql(sql, params={"training_notes":training_notes}, con=engine)
        
        data_matches = df_to_set(pd.read_sql(sql, params={"system":system}, con=engine), 'entity', 'sys', 'casi')
        
        sql = "select `case`, overlap from test.amia_2019_cases where (overlap = 0 or overlap is null) and `system` = %(system)s"  
        #ref_ann = pd.read_sql(sql, params={"training_notes":training_notes}, con=engine)
        
        data_fn = df_to_set(pd.read_sql(sql, params={"system":system}, con=engine), 'entity', 'sys', 'casi')
        
    else:
        
        dir_test = analysisConf.data_dir + 'single_system_out/'

        file = dir_test + system + '_' + analysis_type + '_' + corpus + '_matches.txt'
        data_matches = set(literal_eval(line.strip()) for line in open(file))

        file = dir_test + system + '_' + analysis_type + '_' + corpus + '_ref_only.txt'
        data_fn = set(literal_eval(line.strip()) for line in open(file)) #{ f for f in file.readlines() }

    return data_matches, data_fn


# Code to generate QuickUMLS system annotations (must run from shell):
# 
# import os, glob
# from client import get_quickumls_client
# from quickumls import QuickUMLS
# import pandas as pd
# 
# directory_to_parse = '/Users/gms/development/nlp/nlpie/data/amicus-u01/mipacq/data_in/'
# quickumls_fp = '/Users/gms/development/nlp/engines_misc_tools/QuickUMLS/data/'
# os.chdir(directory_to_parse)
# 
# #similarity = ['dice', 'cosine', 'jaccard', 'overlap']
# similarity = ['jaccard']
# overlapping_criteria = ['score', 'length']
# 
# for s in similarity:
#     for o in overlapping_criteria:
#         #matcher = get_quickumls_client(similarity_name)
#         matcher = QuickUMLS(quickumls_fp=quickumls_fp, overlapping_criteria, threshold=0.7, window=5, similarity_name=s)
#         test = pd.DataFrame()
#         for fname in glob.glob(directory_to_parse + '*.txt'):
#             t = os.path.basename(fname)
#             u = t.split('.')[0]
#             with open(directory_to_parse + u + '.txt') as f:
#                 f1 = f.read()
#                 out = matcher.match(f1, best_match=True, ignore_syntax=False)
#                 for i in out:
#                     i[0]['note_id'] = u
#                     frames = [ test, pd.DataFrame(i[0], index = [0]) ]
#                     test = pd.concat(frames, ignore_index=True)
#         test['system'] = 'quick_umls'
#         test['similarity'] = s
#         test['overlap'] = o
#         test['type'] = 'concept'
#         test['note_id'] = u
#         testt['best_match'] = 'true'
#         temp = test.rename(columns={'start': 'begin'}).copy()
#         print(temp.tail())
# 
#         temp.to_csv('/Users/gms/development/nlp/nlpie/data/amicus-u01/output/qumls.csv', mode='a', header=False)

# GENERATE merges

# In[92]:


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


# In[93]:


#def merge_eval(ref_only: int, system_only: int, ref_system_match: int, matches, system_n: int, ref_n: int):
def merge_eval(ref_only: int, system_only: int, ref_system_match: int, system_n: int, ref_n: int, corpus = None) -> dict:
    """
    Generate confusion matrix params
    :params: ref_only, system_only, reference_system_match -> sets
    matches, system_n, reference_n -> counts
    :return: dictionary object
    
    """

    if ref_only + ref_system_match != ref_n:
        print('ERROR!')

    # get evaluation metrics
    d = {}
    
    F, recall, precision, TP, FP, FN, TP_FN_R, TM  = Metrics(system_only, ref_only, ref_system_match, system_n).get_confusion_metrics()

    if corpus == 'casi':
        d = {'F': F, 
             'precision': precision, 
             'recall': recall, 
             'TP': TP, 
             'FN': FN, 
             'TP/FN': TP_FN_R,
             'n_gold': ref_n, 
             'n_sys': system_n, 
             'TM': TM}
    else:
        d = {'F': F[1], 
             'precision': precision[1], 
             'recall': recall[1], 
             'TP': TP, 
             'FN': FN, 
             'FP': FP, 
             'TP/FN': TP_FN_R,
             'n_gold': ref_n, 
             'n_sys': system_n, 
             'TM': TM}
#     d = {
#          'F': F[1], 
#          'precision': precision[1], 
#          'recall': recall[1], 
#          'TP': TP, 
#          'FN': FN, 
#          'FP': FP, 
#          'TP/FN': TP_FN_R,
#          'n_gold': ref_n, 
#          'n_sys': system_n, 
#          'TM': TM
#     }
    
    
    if system_n - FP != TP:
        print('inconsistent system n!')

    return d


# QUERY TO VALIDATE qumls system counts
# select count(*), type from (select cui, begin, end, note_id, type from
# (SELECT distinct *
#  FROM test.qumls_cui
# where note_id not in ("0595040941-0",
#                             "0778429553-0",
#                             "1014681675",
#                             "2889522952-2",
#                             "3080383448-5",
#                             "3300000926-3",
#                             "3360037185-3",
#                             "3580973392",
#                             "3627629462-3",
#                             "4323116051-4",
#                             "477704053-4",
#                             "528317073",
#                             "531702602",
#                             "534061073",
#                             "54832076",
#                             "5643725437-6",
#                             "5944412090-5",
#                             "6613169476-6",
#                             "7261075903-7",
#                             "7504944368-7",
#                             "7999462393-7",
#                             "8131081430",
#                             "8171084310",
#                             "8193787896",
#                             "8295055184-8",
#                             "8823185307-8") 
#                             and similarity >= 0.8 ) t
# group by cui, begin, end, note_id, type) t
# group by type;

# In[94]:



def process_sentence(pt, sentence, analysis_type, corpus):
    """
    Recursively evaluate parse tree, 
    with check for existence before build
       :param sentence: to process
       :return class of merged annotations, boolean operated system df 
    """
    
    class Results(object):
        def __init__(self):
            self.results = set()
            #self.operations = []
            self.system_merges = pd.DataFrame()
            
    r = Results()
    
    if 'entity' in analysis_type and corpus != 'casi': 
        cols_to_keep = ['begin', 'end', 'note_id'] # entity only
    elif 'full' in analysis_type: 
        cols_to_keep = ['cui', 'begin', 'end', 'note_id'] # entity only
    elif 'cui' in analysis_type:
        cols_to_keep = ['cui', 'note_id'] # entity only
    elif corpus == 'casi':
        cols_to_keep = ['case', 'overlap']
        #cols_to_keep = ['case', 'begin', 'end']
    
    def evaluate(parseTree):
        oper = {'&': op.and_, '|': op.or_}
        
        if parseTree:
            leftC = evaluate(parseTree.getLeftChild())
            rightC = evaluate(parseTree.getRightChild())
            
            if leftC and rightC:
                query = set()
                system_query = pd.DataFrame()
                fn = oper[parseTree.getRootVal()]
                
                if isinstance(leftC, str):
                    
                    # get system as leaf node 
                    left, _ = get_system_matches(leftC, analysis_type, corpus)
                    left_sys = get_sys_data(leftC, analysis_type, corpus)
                
                elif isinstance(leftC, tuple):
                    left = leftC[0]
                    l_sys = leftC[1]
                
                if isinstance(rightC, str):
                    
                    # get system as leaf node
                    right, _ = get_system_matches(rightC, analysis_type, corpus)
                    right_sys = get_sys_data(rightC, analysis_type, corpus)
                    
                elif isinstance(rightC, tuple):
                    right = rightC[0]
                    r_sys = rightC[1]
                    
                # create match set based on boolean operation
                match_set = fn(left, right)
               
                if corpus != 'casi':
                    if fn == op.or_:
                        r.results = r.results.union(match_set)

                        if isinstance(leftC, str) and isinstance(rightC, str):
                            frames = [left_sys, right_sys]
                            df = pd.concat(frames,  ignore_index=True)
                            #df = left_sys.append(right_sys)
                            df = df[cols_to_keep].drop_duplicates(cols_to_keep)

                        elif isinstance(leftC, str) and isinstance(rightC, tuple):
                            frames = [left_sys, r_sys]
                            df = pd.concat(frames,  ignore_index=True)
                            #df = left_sys.append(r_sys)
                            df = df[cols_to_keep].drop_duplicates(cols_to_keep)

                        elif isinstance(leftC, tuple) and isinstance(rightC, str):
                            frames = [l_sys, right_sys]
                            df = pd.concat(frames,  ignore_index=True)
                            #df = right_sys.append(l_sys)
                            df = df[cols_to_keep].drop_duplicates(cols_to_keep)

                        elif isinstance(leftC, tuple) and isinstance(rightC, tuple):
                            frames = [l_sys, r_sys]
                            df = pd.concat(frames,  ignore_index=True)
                            #df = l_sys.append(r_sys)
                            df = df[cols_to_keep].drop_duplicates(cols_to_keep)

                    if fn == op.and_:
                        if len(r.results) == 0:
                            r.results = match_set
                        r.results = r.results.intersection(match_set)

                        if isinstance(leftC, str) and isinstance(rightC, str):
                            df = left_sys.merge(right_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates()

                        elif isinstance(leftC, str) and isinstance(rightC, tuple):
                            df = left_sys.merge(r_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates()

                        elif isinstance(leftC, tuple) and isinstance(rightC, str):
                            df = l_sys.merge(right_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates()

                        elif isinstance(leftC, tuple) and isinstance(rightC, tuple):
                            df = l_sys.merge(r_sys, on=cols_to_keep, how='inner')
                            df = df[cols_to_keep].drop_duplicates()
                else:
                    if fn == op.or_:
                        r.results = r.results.union(match_set)

                        if isinstance(leftC, str) and isinstance(rightC, str):
                            df = left_sys.append(right_sys)
                            df = df.drop_duplicates()

                        elif isinstance(leftC, str) and isinstance(rightC, tuple):
                            df = left_sys.append(r_sys)
                            df = df.drop_duplicates()

                        elif isinstance(leftC, tuple) and isinstance(rightC, str):
                            df = right_sys.append(l_sys)
                            df = df.drop_duplicates()

                        elif isinstance(leftC, tuple) and isinstance(rightC, tuple):
                            df = l_sys.append(r_sys)
                            df = df.drop_duplicates()

                    if fn == op.and_:
                        if len(r.results) == 0:
                            r.results = match_set
                        r.results = r.results.intersection(match_set)

                        if isinstance(leftC, str) and isinstance(rightC, str):
                            df = left_sys.merge(right_sys, on=cols_to_keep, how='inner')
                            df = df.drop_duplicates()

                        elif isinstance(leftC, str) and isinstance(rightC, tuple):
                            df = left_sys.merge(r_sys, on=cols_to_keep, how='inner')
                            df = df.drop_duplicates()

                        elif isinstance(leftC, tuple) and isinstance(rightC, str):
                            df = l_sys.merge(right_sys, on=cols_to_keep, how='inner')
                            df = df.drop_duplicates()

                        elif isinstance(leftC, tuple) and isinstance(rightC, tuple):
                            df = l_sys.merge(r_sys, on=cols_to_keep, how='inner')
                            df = df.drop_duplicates()
                
                # get matched results
                query.update(r.results)
                
                # get combined system results
                r.system_merges = df
                
                if len(df) > 0:
                    system_query = system_query.append(df)
                else:
                    print('wtf!')
                    
                return query, system_query
            else:
                return parseTree.getRootVal()
    
    if sentence.n_or > 0 or sentence.n_and > 0:
        evaluate(pt)  
    
    # trivial case
    elif sentence.n_or == 0 and sentence.n_and == 0:
        r.results, _ = get_system_matches(sentence.sentence, analysis_type, corpus)
        r.system_merges = get_sys_data(sentence.sentence, analysis_type, corpus)
        print('trivial:', sentence.sentence, len(r.results), len(r.system_merges))
    
    return r

# In[95]:


"""
Incoming Boolean sentences are parsed into a binary tree.

Test expressions to parse:

sentence = '((((A&B)|C)|D)&E)'

sentence = '(E&(D|(C|(A&B))))'

sentence = '(((A|(B&C))|(D&(E&F)))|(H&I))'

"""
# build parse tree from passed sentence
# using grammatical rules of Boolean logic
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
    Ensure data to create tree are in standard form
    :param sentence: sentence to preprocess
    :return pt, parse tree graph
            sentence, processed sentence to build tree
            a: order
    """
    def preprocess_sentence(sentence):
        # prepare statement for case when a boolean AND/OR is given
        sentence = (payload.replace('(', ' ( ').             
                replace(')', ' ) ').             
                replace('&', ' & ').             
                replace('|', ' | ').             
                replace('  ', ' '))

        return sentence

    sentence = preprocess_sentence(payload)
    print(sentence)
    
    pt = buildParseTree(sentence)
    #pt.postorder() 
    
    return pt

class Sentence(object):

    def __init__(self, sentence):
        self = self
        self.n_and = sentence.count('&')
        self.n_or = sentence.count('|')
        self.sentence = sentence
    
def get_metrics(boolean_expression: str, analysis_type: str, corpus: str):
    """
    Traverse binary parse tree representation of Boolean sentence
        :params: boolean expression in form of '(<annotator_engine_name1><boolean operator><annotator_engine_name2>)'
                 analysis_type (string value of: 'test', 'entity', 'cui', 'full') used to filter set of reference and system annotations 
        :return: dictionary with values needed for confusion matrix
    """
    sentence = Sentence(boolean_expression)   

    pt = make_parse_tree(sentence.sentence)

    r = process_sentence(pt, sentence, analysis_type, corpus)
    
    print('len sys merges:', len(r.system_merges))
    system_n = len(r.system_merges)
    reference_n = get_ref_n(analysis_type, corpus)

    reference_only, system_only, reference_system_match, match_set = SetTotals(reference_n, system_n, r.results).get_ref_sys()

    # get overall TP/TF and various other counts for running confusion matrix metric analysis
    return merge_eval(reference_only, system_only, reference_system_match, system_n, reference_n, corpus)


# In[96]:


# generate all combinations of given list of annotators:
def expressions(l, n):
    for (operations, *operands), operators in product(
            combinations(l, n), product(('&', '|'), repeat=n - 1)):
        for operation in zip(operators, operands):
            operations = [operations, *operation]
        yield operations

def run_ensemble(l, analysis_type, corpus):

    metrics = pd.DataFrame()

    for i in range(1, len(l)+1): # change lower bound to get number of terms; TODO -> order counts for more than 2-terms
        test = list(expressions(l, i))
        for t  in test:
            if i > 1:
                # format Boolean sentence for parse tree 
                t = '(' + " ".join(str(x) for x in t).replace('[','(').replace(']',')').replace("'","").replace(",","").replace(" ","") + ')'

            d = get_metrics(t, analysis_type, corpus)
            d['merge'] = t
            frames = [metrics, pd.DataFrame(d, index=[0]) ]
            metrics = pd.concat(frames, ignore_index=True, sort=False) 
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    
    file_name = corpus + '_all_merge_metrics_'
        
    geometric_mean(metrics).to_csv(analysisConf.data_dir + file_name + analysis_type + '_' + str(timestamp) + '.csv')
    print(geometric_mean(metrics))


# In[97]:


def df_to_set(df, analysis_type = 'entity', df_type = 'sys', corpus = None):
    
    #print(df[0:10])
    
    # get values for creation of series of type tuple
    if 'entity' in analysis_type: 
        if corpus == 'casi':
            arg = df.case, df.overlap
        else:    
            if df_type == 'sys':
                arg = df.begin, df.end, df.note_id
            else:
                arg = df.start, df.end, df.file
            
    elif 'cui' in analysis_type:
        if df_type == 'sys':
            arg = df.cui, df.note_id
        else:
            arg = df.value, df.file
    elif 'full' in analysis_type:
        if df_type == 'sys':
            arg = df.begin, df.end, df.cui, df.note_id
        else:
            arg = df.start, df.end, df.value, df.file
    
    return set(list(zip(*arg)))
#     if corpus == 'casi':
#         return set(arg)
#     else:
#         return set(list(zip(*arg)))


# In[98]:


#TESTS -> ensemble:
def test_match_consistency(matches, ref_only, ref_n, sys):
    """test for reference only/match set consistency:
        params: match, system and reference only sets"""
   
    print('len', len(sys), len(matches), len(matches.union(sys)), len(matches.intersection(sys)))
    assert len(matches.union(ref_only)) == ref_n, 'Reference annotation mismatch union'
    assert len(matches.intersection(sys)) == len(matches), 'System annotation mismatch intersect'
    assert len(matches.union(sys)) == len(sys), 'System annotation mismatch union'
    assert len(matches.intersection(ref_only)) == 0, 'Reference annotation mismatch intersect'

def test_systems(analysis_type, systems, corpus):
    sys = df_to_set(get_sys_data(systems[0], analysis_type, corpus), analysis_type)
    test_match_consistency(*get_system_matches(systems[0], analysis_type, corpus), get_ref_n(analysis_type), sys)
    print('Match consistency:', len(sys),get_ref_n(analysis_type))

def test_metrics(ref, sys_m, match_m):
    test = True
    reference_n = len(ref)
    system_n = len(sys_m)

    print('Test metrics:', type(reference_n), type(system_n), type(match_m))

    reference_only, system_only, reference_system_match, match_set = SetTotals(reference_n, system_n, match_m).get_ref_sys()
    F, recall, precision, _, _, _, _, _ = Metrics(system_only, reference_only, reference_system_match, system_n).get_confusion_metrics()
    F_, recall_, precision_, _, _, _, _, _ = Metrics(system_only, reference_only, reference_system_match, system_n).get_confusion_metrics(test)

    assert F[1] == F_, 'F1 issue'
    assert recall[1] == recall_, 'recall issue'
    assert precision[1] == precision_, 'precision issue'
    print(F[1], F_)
    print(recall[1], recall_)
    print(precision[1], precision_)

def test_count(analysis_type, corpus):
    # test match counts:
    ctakes, _ = get_system_matches('ctakes', analysis_type, corpus)
    clamp, _ = get_system_matches('clamp', analysis_type, corpus)
    b9, _ = get_system_matches('biomedicus', analysis_type, corpus)
    mm, _ = get_system_matches('metamap', analysis_type, corpus)

    print('count:', len(mm.intersection(b9.intersection(clamp.intersection(ctakes)))))
    
def test_ensemble(analysis_type, corpus):
    
    print('ensemble:')
    # Get mixed system_n
    training_notes = get_notes(analysis_type, corpus)
    ref_ann, data = get_metric_data(training_notes, analysis_type, corpus)

    names = ['ctakes', 'biomedicus', 'metamap', 'clamp']
    if 'entity' in analysis_type: 
        cols_to_keep = ['begin', 'end', 'note_id']
    elif 'cui' in analysis_type:
        cols_to_keep = ['cui', 'note_id']
    elif 'full' in analysis_type:
        cols_to_keep = ['begin', 'end', 'cui', 'note_id']

    biomedicus = data[data["system"]=='biomedicus'][cols_to_keep].copy()
    ctakes = data[data["system"]=='ctakes'][cols_to_keep].copy()
    clamp = data[data["system"]=='clamp'][cols_to_keep].copy()
    metamap = data[data["system"]=='metamap'][cols_to_keep].copy()
    quickumls = data[data["system"]=='quick_umls'][cols_to_keep].copy()

    print('systems:', len(biomedicus), len(clamp), len(ctakes), len(metamap), len(quickumls))

    b9 = set()
    cl = set()
    ct = set()
    mm = set()
    qu = set()

    b9 = df_to_set(get_sys_data('biomedicus', analysis_type, corpus), analysis_type)
    print(len(b9))

    ct = df_to_set(get_sys_data('ctakes', analysis_type, corpus), analysis_type)
    print(len(ct))

    cl = df_to_set(get_sys_data('clamp', analysis_type, corpus), analysis_type)
    print(len(cl))

    mm = df_to_set(get_sys_data('metamap', analysis_type, corpus), analysis_type)
    print(len(mm))

    qu = df_to_set(get_sys_data('quick_umls', analysis_type, corpus), analysis_type)
    print(len(qu))
    
    print('various merges:')
    print(len(b9), len(cl), len(ct), len(mm), len(qu))
    print(len(mm.intersection(b9.intersection(cl.intersection(ct)))))
    print(len(mm.union(b9.intersection(cl.intersection(ct)))))
    print(len(mm.union(b9.union(cl.intersection(ct)))))
    print(len(mm.union(b9.union(cl.union(ct)))))
    print(len(b9.intersection(ct)))

    sys_m = b9.intersection(ct.intersection(qu))
    print('sys_m:', len(sys_m))

    # Get match merges:
    ct, _ = get_system_matches('ctakes', analysis_type, corpus)
    cl, _ = get_system_matches('clamp', analysis_type, corpus)
    b9, _ = get_system_matches('biomedicus', analysis_type, corpus)
    mm, _ = get_system_matches('metamap', analysis_type, corpus)
    qu, _ = get_system_matches('quick_umls', analysis_type, corpus)

    match_m = b9.intersection(ct.intersection(qu))
    print('match_m:', len(match_m))
    # reference df to set
    if 'entity' in analysis_type: 
        cols_to_keep = ['end', 'start','file']
    elif 'cui' in analysis_type:
        cols_to_keep = ['value','file']
    elif 'full' in analysis_type:
        cols_to_keep = ['end', 'start', 'value','file']

    ref = df_to_set(ref_ann[cols_to_keep], analysis_type, 'ref')

    print('ref:', len(ref))

    # test difference:
    print('FP:', len(sys_m - match_m), len(sys_m - ref))
    assert len(sys_m - match_m) == len(sys_m - ref), 'FP mismatch'
    print('FN:', len(ref - match_m), len(ref - sys_m))
    assert len(ref - match_m) == len(ref - sys_m), 'FN mismatch'
    
    test_metrics(ref, sys_m, match_m)


# In[99]:
def partly_unordered_permutations(lst, k):
    elems = set(lst)
    for c in combinations(lst, k):
        for d in permutations(elems - set(c)):
            yield c + d

def main():

   
    #options()

    rtype = int(input("Run: 1->Single systems; 2->Ensemble; 3->Tests; 4-> MM Test"))

    start = time.perf_counter()
   
    '''
        corpora: i2b2, mipacq, fv017
        analyses: entity only (exact span), cui by document, full (aka (entity and cui on exaact span/exact cui)
                  NB: add "_test" using mipacq to egnerate small test sample 
        systems: ctakes, biomedicus, clamp, metamap, quick_umls
        
        TODO -> Vectorization (entity only and full):
                add switch for use of TN on single system performance evaluations 
                add switch for overlap matching versus exact span
             -> Other tasks besides concept extraction
             -> Use of https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        
    ''' 
    #corpus = 'i2b2'
    #corpus = 'mipacq'
    corpus = "casi"
    analysis_type = 'entity'
    #analysis_type = 'full'
    analysisConf =  AnalysisConfig()
    print(analysisConf.systems, analysisConf.corpus_config(corpus))
    
    if (rtype == 1):
        generate_metrics(analysis_type, corpus)
    elif (rtype == 2):
        #l = ['ctakes','biomedicus','clamp','metamap','quick_umls']
        l = ['metamap', 'clamp', 'biomedicus']
        run_ensemble(l, analysis_type, corpus) 
    elif (rtype == 3):
        systems = ['biomedicus']
        t = ['concept_jaccard_score_false']
        test_systems(analysis_type, systems, corpus)  
        test_count(analysis_type, corpus)
        test_ensemble(analysis_type, corpus)
    elif (rtype == 4):
        generate_metrics_test(analysis_type, corpus)

if __name__ == '__main__':

    @click.group()
    def analyze():
        pass

    @analyze.command()
    @click.option('-c', '--corpus', 'corpus', default='i2b2', help='Select corpus for analysis: (i2b2), (mipacq), (casi)', type=click.STRING)
    #@click.option('-t', '--task', 'task', default='entity', help='Select analysis task: entity, cui, both', type=click.STRING)
    def single_system(corpus, task):
        """ Analyze single system """
        if corpus is None:
            exit(1)
        print('Running ', corpus, task) 
    
    @analyze.command()
    @click.option('-c', '--corpus', 'corpus', default='fairview', help='Select corpus for analysis: (i2b2), (mipacq), (casi)', type=click.STRING)
    #@click.option('-t', '--task', 'analysis_type', default='entity', help='Select analysis task: entity, cui, both', type=click.STRING)
    def ensemble(corpus):
        """ Analyze ensemble """
        analysisConf =  AnalysisConfig()
        if corpus is None:
            exit(1)
        systems = ['ctakes','biomedicus','clamp','metamap','quick_umls']
        print('Running ', corpus, analysis_type) 
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

# In[100]:


