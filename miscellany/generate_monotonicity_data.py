import pandas as pd
import copy
import itertools
import operator
import numpy as np
pd.options.display.float_format = '{:.5f}'.format

def monotone_increasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.lt, pairs))

def monotone_decreasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.gt, pairs))

def monotone(lst):
    return monotone_increasing(lst) or monotone_decreasing(lst)

data_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/new_ner/complementarity_top_100/'
corpora = ['fairview', 'i2b2', 'mipacq']

# Create monotoncity summary and top N sub-ensemble decomposition summary

# All groups by corpora
def get_corpora_all(data_dir, corpora, top=None):

    df = pd.DataFrame()
    mm = pd.DataFrame()
    
    for corpus in corpora:

        d1=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_False_10-25-2021.csv')
        d2=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_False_1234.csv')
        d3=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_False_12345.csv')
        
        data = d1# pd.concat([d1, d2, d3])
        
        #data=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_False_10-25-2021.csv')

        data['mtype'] = data["mtype"].str.lower()
        measures = ['precision', 'recall', 'f1']

        monotonic_p = []
        monotonic_r = []
        monotonic_f = []
        increase = []
        decrease = []
        nonmono_p = []
        nonmono_r = []
        nonmono_f = []
        increase_p = []
        decrease_p = []
        increase_r = []
        decrease_r = []
        increase_f = []
        decrease_f = []
        comp = {}

        for mtype in measures:

            #if mtype == 'f1':
            #    mtype = mtype.upper()
            
            if top:
                # get top one
                top_n = data.loc[data.mtype==mtype].sort_values('moi', ascending=False)
                sentences=set(top_n.loc[top_n.mtype==mtype].sentence.tolist())
            
            else:
                sentences=set(data.loc[data.mtype==mtype].sentence.tolist())

            if mtype == 'f1':
                cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                        'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                        'f1-score', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                        'sentence', 'order', 'operator', 'merge_left', 'merge_right',
                        'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp',
                        'max_baby_measure', 'min_baby_measure', 'nterms',
                        'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']

            elif mtype == 'precision':
                cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                        'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                        'precision', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                        'sentence', 'order', 'operator', 'merge_left', 'merge_right',
                        'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp',
                        'max_baby_measure', 'min_baby_measure', 'nterms',
                        'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']


            elif mtype == 'recall':
                cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                        'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                        'recall', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                        'sentence', 'order', 'operator', 'merge_left', 'merge_right',
                        'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp',
                        'max_baby_measure', 'min_baby_measure', 'nterms',
                        'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']


            m=0
            n=0
            pos=0
            neg=0

            for s in list(sentences):

                measure = mtype.lower()
                if mtype == 'f1':
                    measure = mtype + '-score'
                
                test=data.loc[(data.sentence==s)&(data.mtype==mtype)]

                test[measure] = np.where(test['operator']=='&', test[mtype + '_and'], test[mtype + '_or'])

                cols = ['sentence', 'corpus', 'group', 'mtype',
                        'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp', 
                        'moi', 'order', 'monotonicity', 'nterms',
                        'max_baby_measure', 'min_baby_measure', 'max_score', 'min_score', measure, 
                        'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']
                
                t=test[cols_to_keep].sort_values(['order', measure], ascending=False)
                t['corpus'] = corpus
                t['group'] = 'all'
                t['monotonicity'] = ''
                t['max_score'] = ''
                t['min_score'] = ''
                o=set(t.order.to_list())
                scores=[]
                for i in o:
                    u=t.loc[(t.order==i)&(t.p_comp.notnull())]
                    vals = list(set(u[measure].tolist()))
                    vals.sort(reverse=True)
                    u=u[cols]
                    mm = pd.concat([mm, u])
                    for score in vals:                       
                        scores.append(score)

                mtnicity = ''
                if monotone(scores[::-1]):
                    m+=1
                else:
                    n+=1
                    mtnicity = 'n'

                if monotone_increasing(scores[::-1]):
                    pos+=1
                    mtnicity = 'i'
                elif monotone_decreasing(scores[::-1]):
                    neg+=1
                    mtnicity = 'd'
                
                mask = (mm['sentence'] == s) & (mm.mtype==mtype) & (mm.corpus==corpus)

                mm['monotonicity'] = np.where(mask, mtnicity, mm.monotonicity)
                mm['max_score'] = np.where(mask, max(scores), mm.max_score)
                mm['min_score'] = np.where(mask, min(scores), mm.min_score)

            if mtype == 'precision':
                monotonic_p.append(m)
                nonmono_p.append(n)
                increase_p.append(pos) 
                decrease_p.append(neg) 
            elif mtype == 'recall':
                monotonic_r.append(m)
                nonmono_r.append(n)
                increase_r.append(pos) 
                decrease_r.append(neg) 
            elif mtype == 'f1':
                monotonic_f.append(m)
                nonmono_f.append(n)
                increase_f.append(pos) 
                decrease_f.append(neg) 

        out = pd.DataFrame({'corpus': corpus, 'group': 'all', #'comp': comp,
                            'monotonic p': monotonic_p, 'non p': nonmono_p, 'increase p': increase_p, 'decrease p': decrease_p, 
                            'monotonic r': monotonic_r, 'non r': nonmono_r, 'increase r': increase_r, 'decrease r': decrease_r,
                            'monotonic f1': monotonic_f, 'non f1': nonmono_f, 'increase f1': increase_f, 'decrease f1': decrease_f})
        df = pd.concat([df,out])
    
    return df, mm


############ semgroups across all corpora
def get_corpora_sg(data_dir, corpora, top=None):
    dict_of_df = {}
    df_list = []

    df = pd.DataFrame()
    mm = pd.DataFrame()

    for corpus in corpora:

        d1=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_True_10-25-2021.csv')
        d2=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_True_1234.csv')
        d3=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_True_12345.csv')
        
        data=d1 #pd.concat([d1, d2, d3])

        #data=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_True_10-25-2021.csv')

        data['mtype'] = data["mtype"].str.lower()
        measures = ['precision', 'recall', 'f1']

        data['semgroup'] = data['semgroup'].str.replace('test,treatment','Procedures')
        if corpus == 'fairview':
            data['semgroup'] = data['semgroup'].str.replace('Procedure','Procedures')

        data['semgroup'] = data['semgroup'].str.replace('Drug','Chemicals & Drugs')
        data['semgroup'] = data['semgroup'].str.replace('Chemicals_and_drugs','Chemicals & Drugs')

        data['semgroup'] = data['semgroup'].str.replace('problem','Disorders')
        data['semgroup'] = data['semgroup'].str.replace('Disorders,Sign_Symptom','Disorders')
        data['semgroup'] = data['semgroup'].str.replace('Finding','Disorders')

        groups = set(data.semgroup.tolist())

        for group in groups:
            key_name = group #need corpus name?
            dict_of_df[key_name] = copy.deepcopy(data)

            monotonic_p = []
            monotonic_r = []
            monotonic_f = []
            increase = []
            decrease = []
            nonmono_p = []
            nonmono_r = []
            nonmono_f = []
            increase_p = []
            decrease_p = []
            increase_r = []
            decrease_r = []
            increase_f = []
            decrease_f = []

            for mtype in measures:

                if top:
                    # get top one
                    top_n = dict_of_df[group].loc[(dict_of_df[group].mtype==mtype)&(dict_of_df[group].semgroup==group)].sort_values('moi', ascending=False)

                    sentences=set(top_n.loc[(top_n.mtype==mtype)&(top_n.semgroup==group)].sentence.tolist())
                else:
                    #sentences=set(data.loc[(data.mtype==mtype)&(data.semgroup==group)].sentence.tolist())
                    sentences=set(dict_of_df[group].loc[(dict_of_df[group].mtype==mtype)&(dict_of_df[group].semgroup==group)].sentence.tolist()) 

                
                if mtype == 'f1':
                    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                            'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                            'f1-score', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                            'sentence', 'order', 'operator', 'merge_left', 'merge_right',
                            'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp',
                            'max_baby_measure', 'min_baby_measure', 'nterms',
                            'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']


                elif mtype == 'precision':
                    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                            'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                            'precision', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                            'sentence', 'order', 'operator', 'merge_left', 'merge_right',
                            'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp',
                            'max_baby_measure', 'min_baby_measure', 'nterms',
                            'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']

                elif mtype == 'recall':
                    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                            'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                            'recall', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                            'sentence', 'order', 'operator', 'merge_left', 'merge_right',
                            'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp',
                            'max_baby_measure', 'min_baby_measure', 'nterms',
                            'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']
                

                m=0
                n=0
                pos=0
                neg=0

                for s in list(sentences):

                    measure = mtype.lower()
                    if mtype == 'f1':
                        measure = mtype + '-score'
                   
                    test=dict_of_df[group].loc[(dict_of_df[group].sentence==s)&(dict_of_df[group].mtype==mtype)&(dict_of_df[group].semgroup==group)]
                    
                    test[measure] = np.where(test['operator']=='&', test[mtype + '_and'], test[mtype + '_or'])

                    cols = ['sentence', 'corpus', 'group', 'mtype',
                            'max_prop_error_reduction', 'p_comp', 'r_comp', 'F1-score_comp', 
                            'moi', 'order', 'monotonicity', 'nterms',
                            'max_baby_measure', 'min_baby_measure', 'max_score', 'min_score', measure,
                            'TP_left', 'FP_left', 'FN_left', 'TP_right', 'FP_right', 'FN_right']
                    
                    t=test[cols_to_keep].sort_values(['order', measure], ascending=False)
                    t['corpus'] = corpus
                    t['group'] = group
                    t['monotonicity'] = ''
                    t['max_score'] = ''
                    t['min_score'] = ''
                    o=set(t.order.to_list())
                    scores=[]
                    for i in o:
                        u=t.loc[(t.order==i)&(t.p_comp.notnull())]
                        vals = list(set(u[measure].tolist()))
                        vals.sort(reverse=True)
                        u=u[cols]
                        mm = pd.concat([mm, u])
                        for score in vals:                       
                            #scores.append(u[measure].values[0])
                            scores.append(score)
                   
                    mtnicity = ''
                    if monotone(scores[::-1]):
                        m+=1
                    else:
                        n+=1
                        mtnicity = 'n'

                    if monotone_increasing(scores[::-1]):
                        pos+=1
                        mtnicity = 'i'
                    elif monotone_decreasing(scores[::-1]):
                        neg+=1
                        mtnicity = 'd'
                
                    mask = (mm['sentence'] == s) & (mm.mtype==mtype) & (mm.corpus==corpus) & (mm.group==group)

                    mm['monotonicity'] = np.where(mask, mtnicity, mm.monotonicity)
                    mm['max_score'] = np.where(mask, max(scores), mm.max_score)
                    mm['min_score'] = np.where(mask, min(scores), mm.min_score)

                if mtype == 'precision':
                    monotonic_p.append(m)
                    nonmono_p.append(n)
                    increase_p.append(pos) 
                    decrease_p.append(neg) 
                elif mtype == 'recall':
                    monotonic_r.append(m)
                    nonmono_r.append(n)
                    increase_r.append(pos) 
                    decrease_r.append(neg) 
                elif mtype == 'f1':
                    monotonic_f.append(m)
                    nonmono_f.append(n)
                    increase_f.append(pos) 
                    decrease_f.append(neg)         

            print({'corpus': corpus, 'group': group, 
                'monotonic p': monotonic_p, 'non p': nonmono_p, 'increase p': increase_p, 'decrease p': decrease_p, 
                'monotonic r': monotonic_r, 'non r': nonmono_r, 'increase r': increase_r, 'decrease r': decrease_r,
                'monotonic f1': monotonic_f, 'non f1': nonmono_f, 'increase f1': increase_f, 'decrease f1': decrease_f})

            
            out = pd.DataFrame({'corpus': corpus, 'group': group, 
                'monotonic p': monotonic_p, 'non p': nonmono_p, 'increase p': increase_p, 'decrease p': decrease_p, 
                'monotonic r': monotonic_r, 'non r': nonmono_r, 'increase r': increase_r, 'decrease r': decrease_r,
                'monotonic f1': monotonic_f, 'non f1': nonmono_f, 'increase f1': increase_f, 'decrease f1': decrease_f})
            df = pd.concat([df,out])
 

    return df, mm


df_all, mm_all = get_corpora_all(data_dir, corpora)
df_sg, mm_sg = get_corpora_sg(data_dir, corpora)

df = pd.concat( [df_all, df_sg] )
mm = pd.concat( [mm_all, mm_sg] )

mm.to_csv(data_dir+'mm_500.csv')

