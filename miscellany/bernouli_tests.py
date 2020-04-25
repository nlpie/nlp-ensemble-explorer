#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math 
import pymysql
import time 
import functools as ft
import glob, os   
import operator as op
import shelve
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from pandas.api.types import is_numeric_dtype
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
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

data_directory = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/' 
engine = create_engine('mysql+pymysql://gms:nej123@localhost/concepts', pool_pre_ping=True)


# In[70]:


# In[4]:


# confidence intervals
import numpy as np
from scipy.stats import norm

# Requires numpy and scipy.stats
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

# recall_successes = 42
# recall_obs = 63

# [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(recall_successes, recall_obs)


# In[66]:


# one off ss
'''   F1  precision    recall     TP     FN     FP     TP/FN  n_gold  0  0.718201   0.637617  0.822101  91887  19884  52223
TP	FN	FP
106875	31880	64609
'''

tp = 12125
tp = 91887
fn = 10622
fn = 19884
recall_obs = tp + fn
fp = 107509
fp = 52223
precision_obs = tp + fp

[r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
[p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
[f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)

#print(round(f_upper_bound, 3),round(f_lower_bound, 3))

tp = 106875
fn = 31880
recall_obs = tp + fn
fp = 64609
precision_obs = tp + fp 

[r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
[p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
[f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)

#print(round(f_upper_bound, 3),round(f_lower_bound, 3))


# In[72]:


# get ci for single system for table 2 -> TEST
import pandas as pd
input_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/'

file = 'single_system_summary_new.csv'

# change metric here

m_labels = ['F1', 'precision', 'recall']
corpora = ['fairview', 'i2b2', 'mipacq']
semtypes = ['Anatomy',
            'Findings',
            'Chemicals&Drugs',
            'Procedures',
            'all']

print('Single system significance within corpus by semtype, across systems:')
for corpus in corpora:
    for st in semtypes:
        print('CORPUS:', corpus, st)
        data = pd.read_csv(input_dir + file)
        data = data[data['corpus']==corpus]
        data = data[data['semtypes'] == st]
        if not data.empty:
            for m_label in m_labels:
                metric = list()
                ci = list()

                # entire collection:
                for row in data.itertuples():
                    #print(row.TP, row.FN, row.FP)
                    tp = row.TP
                    fn = row.FN
                    recall_obs = tp + fn 
                    fp = row.FP
                    precision_obs = tp + fp

                    [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
                    [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
                    [f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
                    if m_label == 'F1':
                        m = row.F1
                        ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.system, row.corpus, row.semtypes, row.F1))
                    elif m_label == 'precision':
                        m = row.precision
                        ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.system, row.corpus, row.semtypes, row.precision))
                    elif m_label == 'recall':
                        m = row.recall
                        ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.system, row.corpus, row.semtypes, row.recall))

                    metric.append(m)

                # SS for max F1
                M = max(metric) 

                c_i = None
                for c in ci:
                    if M == c[5]:
                        c_i = (c[0], c[1])

                print('st max:', m_label, corpus)
                for c in ci:
                    if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                        print(round(M, 3), c)

    #             ## SS wrt "All groups"
    #             c_i = None
    #             for c in ci:
    #                 if 'all' == c[4]:
    #                     c_i = (c[0], c[1])

    #             print('st all:')
    #             for c in ci:
    #             #     if c[0] <= F <= c[1]:
    #                 if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
    #                     print(round(M, 3), c)

            print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')
                        


# In[73]:


# get ci for single system for table 2
import pandas as pd
input_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/'

file = 'single_system_summary_new.csv'

# change metric here

print('Single system significance within corpus by max metric and all groups within system:')

corpora = ['fairview', 'i2b2', 'mipacq']
m_labels = ['F1', 'precision', 'recall']
systems = ['biomedicus','clamp','ctakes','metamap','quick_umls']

for corpus in corpora:
    for sys in systems:
        print('CORPUS:', corpus)
        for m_label in m_labels:
            df = pd.read_csv(input_dir + file)
            df = df[df['corpus']==corpus]
            df = df[df['system']==sys]

            metric = list()
            ci = list()
            # entire collection:
            for row in df.itertuples():
                #print(row.TP, row.FN, row.FP)
                tp = row.TP
                fn = row.FN
                recall_obs = tp + fn 
                fp = row.FP
                precision_obs = tp + fp

                [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
                [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
                [f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
                if m_label == 'F1':
                    m = row.F1
                    ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.system, row.corpus, row.semtypes, row.F1))
                elif m_label == 'precision':
                    m = row.precision
                    ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.system, row.corpus, row.semtypes, row.precision))
                elif m_label == 'recall':
                    m = row.recall
                    ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.system, row.corpus, row.semtypes, row.recall))
                metric.append(m)

            # SS for max F1
            M = max(metric) 

            c_i = None
            for c in ci:
                if M == c[5]:
                    c_i = (c[0], c[1])

            print('st max:', m_label, corpus)
            for c in ci:
                if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                    print(round(M, 3), c)

            ## SS wrt "All groups"
            c_i = None
            for c in ci:
                if 'all' == c[4]:
                    c_i = (c[0], c[1])

            print('st all:')
            for c in ci:
                if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                    print(round(M, 3), c)
        print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[74]:


# get ci for single system for table 2
import pandas as pd
input_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/'

file = 'single_system_summary_new.csv'

# change metric here

print('Single system significance within corpus by max metric and all groups across systems:')

corpora = ['fairview', 'i2b2', 'mipacq']
m_labels = ['F1', 'precision', 'recall']

for corpus in corpora:
    print('CORPUS:', corpus)
    for m_label in m_labels:
        df = pd.read_csv(input_dir + file)
        df = df[df['corpus']==corpus]

        metric = list()
        ci = list()
        # entire collection:
        for row in df.itertuples():
            #print(row.TP, row.FN, row.FP)
            tp = row.TP
            fn = row.FN
            recall_obs = tp + fn 
            fp = row.FP
            precision_obs = tp + fp

            [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
            [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
            [f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
            if m_label == 'F1':
                m = row.F1
                ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.system, row.corpus, row.semtypes, row.F1))
            elif m_label == 'precision':
                m = row.precision
                ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.system, row.corpus, row.semtypes, row.precision))
            elif m_label == 'recall':
                m = row.recall
                ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.system, row.corpus, row.semtypes, row.recall))
            metric.append(m)

        # SS for max F1
        M = max(metric) 

        c_i = None
        for c in ci:
            if M == c[5]:
                c_i = (c[0], c[1])

        print('st max:', m_label, corpus)
        for c in ci:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)

        ## SS wrt "All groups"
        c_i = None
        for c in ci:
            if 'all' == c[4]:
                c_i = (c[0], c[1])

        print('st all:')
        for c in ci:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)
    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[75]:


df = pd.read_csv(input_dir + file)

semtypes = ['Anatomy',
            'Chemicals&Drugs',
            'Findings',
            'Procedures',
            'all'] 

m_labels = ['F1', 'precision', 'recall']
print('-----------------')
print('Single system significance across biased st:')
for s in semtypes:
    for m_label in m_labels:
        metric = list()
        ci = list()

        # change metric here
        df = pd.read_csv(input_dir + file)
        df = df[df['semtypes'] == s]

        for row in df.itertuples():
            #print(row.TP, row.FN, row.FP)
            tp = row.TP
            fn = row.FN
            recall_obs = tp + fn 
            fp = row.FP
            precision_obs = tp + fp
            [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
            [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
            [f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)

            if m_label == 'F1':
                m = row.F1
                ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.system, row.corpus, row.semtypes, row.F1))
            elif m_label == 'precision':
                m = row.precision
                ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.system, row.corpus, row.semtypes, row.precision))
            elif m_label == 'recall':
                m = row.recall
                ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.system, row.corpus, row.semtypes, row.recall))
            metric.append(m)

        M = max(metric) 

        c_i = None
        for c in ci:
            if M == c[5]:
                c_i = (c[0], c[1])

        print('st max:', m_label, s)
        for c in ci:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)
    
    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')

print('Single system significance across st minus biased systems:')
for s in semtypes:
    for m_label in m_labels:
        metric = list()
        ci = list()
        df = pd.read_csv(input_dir + file)

        df = df[df['semtypes'] == s]

        for row in df.itertuples():
            #print(row.TP, row.FN, row.FP)
            tp = row.TP
            fn = row.FN
            recall_obs = tp + fn 
            fp = row.FP
            precision_obs = tp + fp
            [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
            [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
            [f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)

            if (row.corpus == 'fairview') or (row.system != 'clamp' and row.corpus == 'i2b2') or (row.system not in ['biomedicus', 'ctakes'] and row.corpus == 'mipacq'):
                if m_label == 'F1':
                    m = row.F1
                    ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.system, row.corpus, row.semtypes, row.F1))
                elif m_label == 'precision':
                    m = row.precision
                    ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.system, row.corpus, row.semtypes, row.precision))
                elif m_label == 'recall':
                    m = row.recall
                    ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.system, row.corpus, row.semtypes, row.recall))
                metric.append(m)

        print(max(metric))
        M = max(metric) 

        c_i = None
        for c in ci:
            if M == c[5]:
                c_i = (c[0], c[1])

        print('st max:', m_label, s)
        for c in ci:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)
                
    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[76]:


# by corpus/semtype all ensembles, including single sys 

input_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/overlap/combined/analysis/'

m_labels = ['F', 'precision', 'recall']
print('Within corpus/st ensembles:')
for file in glob.glob(input_dir + '*.csv'):
    df = pd.read_csv(file)
    df = df.drop_duplicates(subset=['F', 'precision', 'recall'])
    for m_label in m_labels:
        print(m_label,':', file)
        metric = list()
        ci = list()
        for row in df.itertuples():
            #print(row.TP, row.FN, row.FP)
            tp = row.TP
            fn = row.FN
            recall_obs = tp + fn 
            fp = row.FP
            precision_obs = tp + fp

            [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
            [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
            [f, df1, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
            if ('fairview' in file) or ('clamp' not in row.merge and 'i2b2' in file) or (('biomedicus' not in row.merge and 'ctakes' not in row.merge) and 'mipacq' in file):
                if m_label == 'F':
                    m = row.F
                    ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.merge, row.F))
                elif m_label == 'precision':
                    m = row.precision
                    ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.merge, row.precision))
                elif m_label == 'recall':
                    m = row.recall
                    ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.merge, row.recall))

                metric.append(m)

        M = max(metric)  

        c_i = None
        for c in ci:
            if M == c[3]:
                c_i = (c[0], c[1])

        for c in ci:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)


        print('--------------')
        
    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[78]:


# by max merges within corpus, across corpora(?)

data_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/'
file = 'max_merge_summary_new.xlsx'

corpora = ['fairview', 'i2b2', 'mipacq']
m_labels = ['F1', 'precision', 'recall']

print('Within corpus significance max merges:')
for corpus in corpora:
    print('CORPUS:', corpus)

    for m_label in m_labels:
        
        if m_label == 'F1':
            sheet_name='max F-score'
        elif m_label == 'precision':
            sheet_name='max precision'
        elif m_label == 'recall':
            sheet_name='max recall'
        
        df = pd.read_excel(open(data_dir + file, 'rb'), sheet_name=sheet_name)
        df = df[df['corpus'] == corpus]
        metric = list()
        ci = list()
        # entire collection:
        for row in df.itertuples():

            tp = row.TP
            fn = row.FN
            recall_obs = tp + fn 
            fp = row.FP
            precision_obs = tp + fp

            [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
            [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
            [f, df1, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
            if m_label == 'F1':
                m = row.F1
                ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.F1, row.merge, row.corpus, row.semtypes))
            elif m_label == 'precision':
                m = row.precision
                ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.precision, row.merge, row.corpus, row.semtypes))
            elif m_label == 'recall':
                m = row.recall
                ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.recall, row.merge, row.corpus, row.semtypes))
            metric.append(m)

        M = max(metric)  

        c_i = None
        for c in ci:
            #print(c)
            if M == c[2]:
                c_i = (c[0], c[1])

        print('st max:', m_label, corpus)
        for c in ci:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)

        ## SS wrt "All groups"
        c_i = None
        for c in ci:
            if 'all' == c[5]:
                c_i = (c[0], c[1])

        print('st all:')
        for c in ci:
        #     if c[0] <= F <= c[1]:
            if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
                print(round(M, 3), c)
                
    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[64]:


# by max merges within corpus, across corpora(?)

data_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/'
file = 'max_merge_summary_new_mipacq.xlsx'

m_labels = ['F1', 'precision', 'recall']

print('Within corpus significance max merges unbiased mipacq:')

for m_label in m_labels:
    if m_label == 'F1':
        sheet_name='max F-score'
    elif m_label == 'precision':
        sheet_name='max precision'
    elif m_label == 'recall':
        sheet_name='max recall'

    df = pd.read_excel(open(data_dir + file, 'rb'), sheet_name=sheet_name)
    metric = list()
    ci = list()
    
    # entire collection:
    for row in df.itertuples():

        tp = row.TP
        fn = row.FN
        recall_obs = tp + fn 
        fp = row.FP
        precision_obs = tp + fp

        [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
        [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
        [f, df1, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
        if m_label == 'F1':
            m = row.F1
            ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.F1, row.merge, row.corpus, row.semtypes))
        elif m_label == 'precision':
            m = row.precision
            ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.precision, row.merge, row.corpus, row.semtypes))
        elif m_label == 'recall':
            m = row.recall
            ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.recall, row.merge, row.corpus, row.semtypes))
        metric.append(m)

    M = max(metric)  

    c_i = None
    for c in ci:
        #print(c)
        if M == c[2]:
            c_i = (c[0], c[1])

    print('st max:', m_label, corpus)
    for c in ci:
        if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
            print(round(M, 3), c)

    ## SS wrt "All groups"
    c_i = None
    for c in ci:
        if 'all' == c[5]:
            c_i = (c[0], c[1])

    print('st all:')
    for c in ci:
    #     if c[0] <= F <= c[1]:
        if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
            print(round(M, 3), c)

    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[79]:



# by max merges within corpus, across corpora(?)

data_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/'
file = 'max_merge_summary_new_i2b2.xlsx'

m_labels = ['F1', 'precision', 'recall']

print('Within corpus significance max merges unbiased i2b2:')

for m_label in m_labels:
    if m_label == 'F1':
        sheet_name='max F-score'
    elif m_label == 'precision':
        sheet_name='max precision'
    elif m_label == 'recall':
        sheet_name='max recall'

    df = pd.read_excel(open(data_dir + file, 'rb'), sheet_name=sheet_name)
    metric = list()
    ci = list()
    # entire collection:
    for row in df.itertuples():

        tp = row.TP
        fn = row.FN
        recall_obs = tp + fn 
        fp = row.FP
        precision_obs = tp + fp

        [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
        [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
        [f, df1, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)
        if m_label == 'F1':
            m = row.F1
            ci.append((round(f_upper_bound, 3),round(f_lower_bound, 3), row.F1, row.merge, row.corpus, row.semtypes))
        elif m_label == 'precision':
            m = row.precision
            ci.append((round(p_upper_bound, 3),round(p_lower_bound, 3), row.precision, row.merge, row.corpus, row.semtypes))
        elif m_label == 'recall':
            m = row.recall
            ci.append((round(r_upper_bound, 3),round(r_lower_bound, 3), row.recall, row.merge, row.corpus, row.semtypes))
        metric.append(m)

    M = max(metric)  

    c_i = None
    for c in ci:
        #print(c)
        if M == c[2]:
            c_i = (c[0], c[1])

    print('st max:', m_label, corpus)
    for c in ci:
        if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
            print(round(M, 3), c)

    ## SS wrt "All groups"
    c_i = None
    for c in ci:
        if 'all' == c[5]:
            c_i = (c[0], c[1])

    print('st all:')
    for c in ci:
    #     if c[0] <= F <= c[1]:
        if (c_i[0] <= c[0] and c_i[1] > c[0]) or (c_i[0] >= c[0] and  c_i[0] < c[1]):
            print(round(M, 3), c)

    print('-----------------')

print('-----------------')
print('-----------------')
print('-----------------')
print('-----------------')


# In[ ]:




