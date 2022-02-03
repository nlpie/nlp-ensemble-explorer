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

#data = pd.concat([get_corpora_aGll(data_dir, corpora), get_corpora_sg(data_dir, corpora)])

#semgroups = sorted(list(set(data.group.tolist())))

### ->>>>>>>>>>>>>>>>>> plot monotonicity

df_all, mm_all = get_corpora_all(data_dir, corpora)
df_sg, mm_sg = get_corpora_sg(data_dir, corpora)

df = pd.concat( [df_all, df_sg] )
#mm = pd.concat( [mm_all, mm_sg] )

mm.to_csv(data_dir+'mm_500.csv')


#######################

#######################


#================> Analytical set
#mm=pd.read_csv('/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/new_ner/complementarity_top_100/mm_500.csv')
mm=pd.read_csv('/Users/gms/Desktop/ensembling/mm_500.csv')
mm.reset_index(drop=True, inplace=True)

mm = mm.rename(columns={"F1-score_comp": "f1_comp"})

semgroups = sorted(list(set(mm.group.tolist())))
corpora = ['fairview', 'i2b2', 'mipacq']
measures = ['precision', 'recall', 'f1']
cols_to_keep=['max_prop_error_reduction', 'p_comp', 'r_comp', 'f1_comp', 'nterms']

def get_analytic_set(mm, mt, group = 'Disorders'):

    mm['max_f1_comp'] = mm.groupby(['order', 'group', 'corpus', 'mtype', 'sentence'])['f1_comp'].transform('max')
    mm['min_f1_comp'] = mm.groupby(['order', 'group', 'corpus', 'mtype', 'sentence'])['f1_comp'].transform('min')
    mm['max_p_comp'] = mm.groupby(['order', 'group', 'corpus', 'mtype', 'sentence'])['p_comp'].transform('max')
    mm['min_p_comp'] = mm.groupby(['order', 'group', 'corpus', 'mtype', 'sentence'])['p_comp'].transform('min')
    mm['max_r_comp'] = mm.groupby(['order', 'group', 'corpus', 'mtype', 'sentence'])['r_comp'].transform('max')
    mm['min_r_comp'] = mm.groupby(['order', 'group', 'corpus', 'mtype', 'sentence'])['r_comp'].transform('min')
    mm['n_sub_ens'] = mm.groupby(['corpus', 'mtype', 'group','sentence'])['sentence'].transform('count')/2

    # imbalance
    mm['comp_diff_p_r'] = (mm['p_comp'] - mm['r_comp']).abs() 
    mm['comp_diff_p_r_sum'] = mm.groupby(['group', 'corpus', 'mtype', 'sentence'])['comp_diff_p_r'].transform('sum')
    mm['comp_diff_p_r_mean'] = mm['comp_diff_p_r_sum']/(2*mm['n_sub_ens'])

    mm['count_diff_p_n_left'] = (mm['FP_left'] - mm['FN_left']).abs() 
    mm['count_diff_p_n_right'] = (mm['FP_right'] - mm['FN_right']).abs() 
    mm['count_p_n'] = mm['count_diff_p_n_left'] + mm['count_diff_p_n_right'] 
    mm['count_p_n_sum'] = mm.groupby(['group', 'corpus', 'mtype', 'sentence'])['count_p_n'].transform('sum')/2
    mm['count_p_n_mean'] = mm['count_p_n_sum']/mm['n_sub_ens']

    mm['comp_diff_p_r_sq'] =  mm['comp_diff_p_r']*mm['comp_diff_p_r']
    mm['comp_diff_p_r_sq_sum'] = mm.groupby(['group', 'corpus', 'mtype', 'sentence'])['comp_diff_p_r_sq'].transform('sum')
    mm['comp_diff_p_r_mean_mse'] = (mm['comp_diff_p_r_sum']/(2*mm['n_sub_ens'])).pow(1./2)

    mm['count_p_n_sq'] =  mm['count_p_n']*mm['count_p_n']
    mm['count_p_n_sq_sum'] = mm.groupby(['group', 'corpus', 'mtype', 'sentence'])['count_p_n_sq'].transform('sum')
    mm['count_p_n_mean_mse'] = (mm['count_p_n_sq_sum']/mm['n_sub_ens']).pow(1./2)

    test=mm.sort_index().drop_duplicates(subset=['sentence', 'precision', 'recall', 'f1-score', 'corpus', 'mtype', 'group', 'max_score', 'min_score', 'f1_comp', 'max_f1_comp', 'min_f1_comp', 'max_p_comp', 'min_p_comp', 'max_r_comp', 'min_r_comp'])
    
    test['diff_p']=test['max_p_comp']-test['min_p_comp']
    test['diff_r']=test['max_r_comp']-test['min_r_comp']
    test['diff_f1']=test['max_f1_comp']-test['min_f1_comp']
    
    test['moi_score_range'] = test['max_score'] - test['min_score']
   
    df=test.loc[(test.mtype==mt)&(test.group==group)]
    
    return df
    
def get_out(mm, group, measures):
    out = pd.DataFrame()
    for mtype in measures:
        print(mtype)
        out=pd.concat([out, get_analytic_set(mm, mtype, group)])

    # number of sub-ensembles in decomposition
    out['n_sub_ens']=out.groupby(['corpus', 'mtype', 'group','sentence'])['sentence'].transform('count')/2
    
    # MSE of cummulative comp gain
    out['diff_f1_sq'] = out['diff_f1']*out['diff_f1']
    out['diff_f1_sq_sum'] = out.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_f1_sq'].transform('sum')/2
    out['diff_f1_mse']=(out['diff_f1_sq_sum'] / out['n_sub_ens']).pow(1./2)
    
    out['diff_p_sq'] = out['diff_p']*out['diff_p']
    out['diff_p_sq_sum'] = out.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_p_sq'].transform('sum')/2
    out['diff_p_mse']=(out['diff_p_sq_sum'] / out['n_sub_ens']).pow(1./2)

    out['diff_r_sq'] = out['diff_r']*out['diff_r']
    out['diff_r_sq_sum'] = out.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_r_sq'].transform('sum')/2
    out['diff_r_mse']=(out['diff_r_sq_sum'] / out['n_sub_ens']).pow(1./2)

    out['moi_score_range']=out.moi_score_range.astype('float64')
    out = out.rename(columns={"mtype": "measure"})
    out['measure'] = out.measure.str.replace('f1', 'F1-score')

    out['corpus'] = out.corpus.str.replace('fairview', 'Fairview').str.replace('mipacq', 'MiPACQ')
    out['group'] = out.group.str.replace('all', 'All groups')
    out['monotonicity'] = out.monotonicity.str.replace('n', 'non').str.replace('i', 'increasing').str.replace('d', 'decreasing')

    return out.drop_duplicates(subset=['sentence', 'corpus', 'measure', 'group', 'monotonicity'])

          
### GET df for analysis!!!!!!!!!!!!!!!!!!!!!!1

groups = ['all', 'Disorders']
#groups = ['Anatomy', 'Chemicals & Drugs', 'Disorders', 'Procedures', 'all']


df = pd.DataFrame()
for group in groups:
    out=get_out(mm, group, ['f1']).sort_values('moi', ascending=False)

    test=out[(out.corpus=='Fairview')]
    t=test.drop_duplicates('sentence')

    cols=['corpus', 'group', 'moi', 'moi_score_range', 'monotonicity', 'mono']

    cols+=['max_diff', 'min_diff', 'comp_f1_range', 
        'comp_f1_gain_mse', 'comp_f1_gain_mse_sq', 'comp_f1_range_sq', 
        'comp_p_range', 'comp_p_gain_mse', 'comp_p_gain_mse_sq', 
        'comp_r_range', 'comp_r_gain_mse', 'comp_r_gain_mse_sq',
        'comp_diff_p_r_mean', 'count_p_n_mean', 'comp_diff_p_r_mean_mse', 'sentence']

    # mean centered:
    t["mono"] = t["monotonicity"].astype("category").cat.codes
    t['diff_mse_sq'] = t['diff_mse']*t['diff_mse']  
    t['diff_sq'] = t['diff_']*t['diff_']  
    t['diff_p_mse_sq'] = t['diff_p_mse']*t['diff_p_mse']  
    t['diff_r_mse_sq'] = t['diff_r_mse']*t['diff_r_mse']  

    t = t.rename(columns={"diff_": "comp_f1_range", 
        "score_diff": "moi_score_range", 
        "diff_sq": "comp_f1_range_sq",
        "diff_mse": "comp_f1_gain_mse", 
        "diff_mse_sq": "comp_f1_gain_mse_sq", 
        "diff_p_mse": "comp_p_gain_mse",
        "diff_p_mse_sq": "comp_p_gain_mse_sq", 
        "diff_p_": "comp_p_range",
        "diff_r_": "comp_r_range",
        "diff_r_mse": "comp_r_gain_mse",
        "diff_r_mse_sq": "comp_r_gain_mse_sq",
        "count_p_n_mean_mse": "count_p_n_imb_mse",
        "comp_diff_p_r_mean_mse": "comp_p_r_imb_mse"})


    # mean center
    '''
    u = t[['max_diff', 'min_diff', 'nterms', 'comp_f1_range', 
        'comp_f1_gain_mse', 'comp_f1_gain_mse_sq', 'comp_f1_range_sq', 
        'comp_p_gain_mse', 'comp_p_gain_mse_sq', 'comp_r_gain_mse', 'comp_r_gain_mse_sq']].apply(lambda x: x-x.mean()).merge(t[cols], how='inner', left_index=True, right_index=True)
    '''

    # normalize

    u = t[['count_p_n_imb_mse', 'nterms']].apply(lambda x: (x-x.min())/ (x.max() - x.min()), axis=0).merge(t[cols], how='inner', left_index=True, right_index=True)
    
    #df = pd.concat([t[cols], df])
    df = pd.concat([u, df])

"""
############### Analysis

# Get correlations between moi_score_diff and comp measure range
from scipy.stats import pearsonr
import numpy as np

semgroups = sorted(list(set(mm.group.tolist())))
corpora = ['fairview', 'i2b2', 'mipacq']
measures = ['precision', 'recall', 'f1']

pr=pd.DataFrame()
for group in semgroups:
    for mt in measures:

        for corpus in corpora:
            out={}
            print(corpus, mt, group)
            
            out['corpus'] = corpus
            out['measure'] = mt
            out['group'] = group

            df = get_analytic_set(mm, mt, group)

            df = df.loc[df.corpus==corpus]
            #print(df.head())
            #sns.lmplot('diff_', 'score_diff', data=tt,  col='corpus', hue = 'monotonicity', truncate=False, scatter_kws={"marker": "D", "s": 20})

            test_plot=df.drop_duplicates(subset=['sentence', 'corpus', 'mtype'])
            test_plot['score_diff']=test_plot.score_diff.astype('float64')
            test_plot['add_comp'] = test_plot.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff'].transform('sum')/2
            test_plot['max_diff'] = test_plot.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff'].transform('max')

            # MSE of cummulative comp gain
            test_plot['diff_sq'] = test_plot['diff']*out['diff']
            test_plot['diff_sq_sum'] = test_plot.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_sq'].transform('sum')/2
            test_plot['diff_mse']=test_plot['diff_sq_sum'] / out['n_sub_ens']

            test_plot['diff_p_sq'] = test_plot['diff_p']*out['diff_p']
            test_plot['diff_p_sq_sum'] = test_plot.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_p_sq'].transform('sum')/2
            test_plot['diff_p_mse']=test_plot['diff_p_sq_sum'] / out['n_sub_ens']

            test_plot['diff_r_sq'] = test_plot['diff']*out['diff_r']
            test_plot['diff_r_sq_sum'] = test_plot.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_r_sq'].transform('sum')/2
            test_plot['diff_r_mse']=test_plot['diff_r_sq_sum'] / out['n_sub_ens']


            cols=['sentence', 'corpus', 'group', 'mtype', 
            'p_comp', 'r_comp', 'f1_comp',
            'moi', 'monotonicity',
            'max_f1_comp', 'min_f1_comp', 
            'max_p_comp','max_f1_comp_', 
            'diff','diff_', 'score_diff']
          
            tt=test_plot[test_plot.monotonicity.isin(['i'])]
            n_i = len(tt) 

            p1 = 'score_diff'
            p2 = 'diff_'
            if n_i > 2: 
                out['n_i'] = n_i 
                out['pearsonr_i']=pearsonr(tt[p1], tt[p2])[0]
                out['pearsonr_i_p']=pearsonr(tt[p1], tt[p2])[1]
                pass
            else:
                out['n_i'] = n_i 

            tt=test_plot[test_plot.monotonicity.isin(['n'])]
            n_n = len(tt) 
            if n_n > 2: 
                out['n_n'] = n_n 
                out['pearsonr_n']=pearsonr(tt[p1], tt[p2])[0]
                out['pearsonr_n_p']=pearsonr(tt[p1], tt[p2])[1]
                pass
            else:
                out['n_n'] = n_n 

            tt=test_plot[test_plot.monotonicity.isin(['d'])]
            n_d = len(tt) 
            if n_d > 2: 
                out['n_d'] = n_d 
                out['pearsonr_d']=pearsonr(tt[p1], tt[p2])[0]
                out['pearsonr_d_p']=pearsonr(tt[p1], tt[p2])[1]
                pass
            else:
                out['n_d'] = n_d 

            pr = pd.concat([pr, pd.DataFrame(out, index=[0])])

# person's r for diff_ versus moi_score_range_ 
pear=pd.read_csv('/Users/gms/Desktop/pr_500.csv')

pear['strength_n']=np.where((pear['n_n'] >= 30)& (pear['pearsonr_n'].abs()>=0.50)&(pear['pearsonr_n_p']<0.05), 'mod-strong', None)
pear['strength_i']=np.where((pear['n_i'] >= 30)& (pear['pearsonr_i'].abs()>=0.50)&(pear['pearsonr_i_p']<0.05), 'mod-strong', None)
pear['strength_d']=np.where((pear['n_d'] >= 30)& (pear['pearsonr_d'].abs()>=0.50)&(pear['pearsonr_d_p']<0.05), 'mod-strong', None)

pear['strength_n']=np.where((pear['n_n'] >= 30)& (pear['pearsonr_n'].abs()>=0.30)&(pear['pearsonr_n'].abs()<0.50)&(pear['pearsonr_n_p']<0.05), 'fair-mod',pear.strength_n)
pear['strength_i']=np.where((pear['n_i'] >= 30)& (pear['pearsonr_i'].abs()>=0.30)&(pear['pearsonr_i'].abs()<0.50)&(pear['pearsonr_i_p']<0.05), 'fair-mod',pear.strength_i)
pear['strength_d']=np.where((pear['n_d'] >= 30)& (pear['pearsonr_d'].abs()>=0.30)&(pear['pearsonr_d'].abs()<0.50)&(pear['pearsonr_d_p']<0.05), 'fair-mod',pear.strength_d)

pear['strength_i']=np.where((pear['n_i'] >= 30)& (pear['pearsonr_i'].abs()<0.30)&(pear['pearsonr_i'].abs()>0)&(pear['pearsonr_i_p']<0.05), 'poor-fair',pear.strength_i)
pear['strength_n']=np.where((pear['n_n'] >= 30)& (pear['pearsonr_n'].abs()<0.30)&(pear['pearsonr_n'].abs()>0)&(pear['pearsonr_n_p']<0.05), 'poor-fair',pear.strength_n)
pear['strength_d']=np.where((pear['n_d'] >= 30)& (pear['pearsonr_d'].abs()<0.30)&(pear['pearsonr_d'].abs()>0)&(pear['pearsonr_d_p']<0.05), 'poor-fair',pear.strength_d)

pear['n']=pear['n_i']+pear['n_d']+pear['n_n']

pear['sig_i']=np.where((pear.n_i < 30), 'N/A', 
np.where(pear.pearsonr_i_p>= 0.05, 'NS', 
np.where((pear.pearsonr_i_p<0.05)&(pear.pearsonr_i_p>=0.01), '*', 
np.where((pear.pearsonr_i_p<0.01)&(pear.pearsonr_i_p>=0.001), '**',  '***' ))))

pear['sig_n']=np.where((pear.n_n < 30), 'N/A', 
np.where(pear.pearsonr_n_p>= 0.05, 'NS', 
np.where((pear.pearsonr_n_p<0.05)&(pear.pearsonr_n_p>=0.01), '*', 
np.where((pear.pearsonr_n_p<0.01)&(pear.pearsonr_n_p>=0.001), '**',  '***' ))))

pear['sig_d']=np.where((pear.n_d < 30), 'N/A', 
np.where(pear.pearsonr_d_p>= 0.05, 'NS', 
np.where((pear.pearsonr_d_p<0.05)&(pear.pearsonr_d_p>=0.01), '*', 
np.where((pear.pearsonr_d_p<0.01)&(pear.pearsonr_d_p>=0.001), '**',  '***' ))))
            '''
            # =============>
            #test=mm.sort_index().drop_duplicates(subset=['sentence', 'precision', 'recall', 'f1-score', 'corpus', 'mtype', 'group', 'max_score', 'min_score', 'f1_comp', 'max_f1_comp', 'min_f1_comp', 'max_p_comp', 'min_p_comp', 'max_r_comp', 'min_r_comp'])
    

            #df=test.loc[(test.mtype=='f1')&(test.corpus=='fairview')&(test.group=='Disorders')]
            test_plot=df.drop_duplicates(subset=['sentence', 'corpus', 'mtype'])
            test_plot['score_diff']=test_plot.score_diff.astype('float64')
            tt=test_plot[test_plot.monotonicity.isin(['i', 'n', 'd'])]


            sns.lmplot('diff_', 'score_diff', data=test_plot, col='corpus', hue = 'monotonicity', row='mtype', truncate=False, scatter_kws={"marker": "D", "s": 20})

            plt.ylim(-.1, 1)
            plt.xlim(0, 1)

            plt.title("Top 100 Subensemble Decompositions - " + mt)
            plt.ylabel('Max(f1-score)-Min(f1-score)')
            plt.xlabel('Max(f1-score-comp)-Min(f1-score-comp)')
            '''


#----------> GLM

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
pandas2ri.activate()
#import rpy2.robjects.lib.ggplot2 as ggplot2
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('load_ext rpy2.ipython')

base = importr('base')
car = importr('car')
mctest = importr('mctest')

ipython.magic('R -i u')

#pmod = ro.r("pmod <- lm('moi ~ diff_comp_mse+factor(mono)+diff_comp+nterms', data=u)")

#pmod = ro.r("pmod <- lm('moi_score_diff ~ diff_comp_mse_sq+diff_comp+factor(mono)+nterms', data = u)")

pmod = ro.r("pmod <- lm('moi_score_diff ~ diff_comp_mse+diff_comp+diff_comp_sq+factor(mono)+nterms', data = u)")


print(base.summary(pmod))

vif = ro.r('car::vif(pmod)')

# ---------> corr

cols=['nterms', 'diff_comp', 'diff_comp_sq', 'diff_comp_mse', 'diff_comp_mse_sq', 'moi', 'moi_score_diff']

# Correlations ================> https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance


from scipy.stats import pearsonr
import numpy as np
rho = u[cols].corr()
pval = u[cols].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [0.001, 0.01,0.05] if x<=t]))
rho.round(2).astype(str) + p


def get_lm(mm, corpus, group, measures, measure='F1-score'):
    
    out=get_out(mm, group, measures)
    test=out[(out.corpus==corpus)&(out.measure==measure)]
    t=test.drop_duplicates('sentence')


    cols=['corpus', 'group', 'measure',  'moi', 'moi_score_diff', 'monotonicity', 'mono']
    # mean centered:
    t["mono"] = t["monotonicity"].astype("category").cat.codes
    t['diff_mse_sq'] = t['diff_mse']*t['diff_mse']  
    t['diff_sq'] = t['diff_']*t['diff_']  

    t = t.rename(columns={"diff_": "diff_comp", "score_diff": "moi_score_diff", "diff_sq": "diff_comp_sq",
        "diff_mse": "diff_comp_mse", "diff_mse_sq": "diff_comp_mse_sq"})
    u = t[['max_diff', 'min_diff', 'nterms', 'diff_comp', 'diff_comp_mse', 'diff_comp_mse_sq', 'diff_comp_sq']].apply(lambda x: x-x.mean()).merge(t[cols], how='inner', left_index=True, right_index=True)

    base = importr('base')
    car = importr('car')
    mctest = importr('mctest')
    
    ipython.magic('R -i u')

    #pmod = ro.r("pmod <- lm('moi ~ diff_comp_mse+factor(mono)+diff_comp+nterms', data=u)")

    #pmod = ro.r("pmod <- lm('moi_score_diff ~ diff_comp_mse_sq+diff_comp+factor(mono)+nterms', data = u)")

    pmod = ro.r("pmod <- lm('moi_score_diff ~ diff_comp_mse+diff_comp+diff_comp_sq+factor(mono)+nterms', data = u)")

    print(base.summary(pmod))

    vif = ro.r('car::vif(pmod)')

    print(vif)



########### PLOTS

# Monotonicity bar plots
semgroups = sorted(list(set(df.group.tolist())))
rows = [['monotonic p', 'non p', 'monotonic r', 'non r', 'monotonic f1', 'non f1'], 
        ['increase p', 'decrease p', 'increase r', 'decrease r', 'increase f1', 'decrease f1']]  # columns for each row of plots

for sg in semgroups:
    
    ix = semgroups.index(sg)

    t = df.loc[df.group==sg].copy() 
    #corpus = set(t.corpus.to_list())

    corpus = t.corpus.unique()  # unique corpus
    #idx = np.where(semgroups==sg)

    ix += 2

    ncols = len(set(t.corpus.to_list()))  # 3 columns for the example
    nrows = len(rows)  # 2 rows for the example

    # create a figure with 2 rows of 3 columns: axes is a 2x3 array of <AxesSubplot:>
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=(12, 10))

    # iterate through each plot row combined with a list from rows
    for axe, row in zip(axes, rows):
        # iterate through each plot column of the current row
        for i, ax in enumerate(axe):

            # select the data for each plot
            data = t.loc[t.group.eq(sg) & t.corpus.eq(corpus[i]), row]
            # plot the data with seaborn, which is easier to color the bars
            sns.barplot(data=data, ax=ax)

            # label row of subplot accordingly
            if 'monotonic p' in row:
                if corpus[i] == 'fairview':
                    
                    l2 = 'Fairview'
                    l1 ='(a) '

                elif corpus[i] == 'mipacq':
                    l2 = 'MiPACQ'
                    if ncols == 3:
                        l1 = '(c) '
                    else:
                        l1 = '(b) '
                elif corpus[i] == 'i2b2':
                    l2 = 'i2b2'
                    l1 = '(b) '
                
                ax.set_title(l1 + l2)
            
            else:
                if corpus[i] == 'fairview':
                    
                    if ncols == 3:
                        l1 ='(d)'
                    else:
                        l1 ='(c)'

                elif corpus[i] == 'mipacq':
                    if ncols == 3:
                        l1 = '(f)'
                    else:
                        l1 = '(d)'
                elif corpus[i] == 'i2b2':
                    l1 = '(e) '
    
                ax.set_title(l1)

            ax.tick_params(axis='x', labelrotation = 45)
    
    if sg == 'all':
        sg = 'All groups'

    # Defining custom 'xlim' and 'ylim' values.
    custom_ylim = (0, 80)
    custom_ylim = (0, 100)

    # Setting the values for all axes.
    plt.setp(axes, ylim=custom_ylim)
    fig.suptitle(sg)
    fig.tight_layout()
    #plt.show()
    plt.savefig('/users/gms/Desktop/'+sg+'_figure_'+str(ix)+'.png')


##### plot distributions


g=sns.lmplot('diff_', 'score_diff', data=test_plot, col='corpus', hue = 'monotonicity', row='measure', sharey=True, sharex=True, height=2.5,aspect=1.25, truncate=False, scatter_kws={"marker": "D", "s": 20})

#test_plot= test_plot[(test_plot.corpus=='Fairview')&(test_plot.measure=='F1-score')]
#g=sns.jointplot(x="diff_", y="score_diff", data=test_plot, kind="reg")

(
        g.set_axis_labels("Max-Min (comp measure)", "Max-Min (measure)")
        .set(xlim=(0, 1), ylim=(-.1, 0.8))
)
#(g.set(xlim=(0, 2.5), ylim=(-.1, 1)))
        

alpha = list('abcdefghijklmnopqrstuvwxyz')
axes = g.axes.flatten()

# ADJUST ALL AXES TITLES
for ax, letter in zip(axes, alpha[:len(axes)]):
    ttl = ax.get_title().split("|")[1].strip()   # GET CURRENT TITLE
    ax.set_title(f"({letter}) {ttl}")            # SET NEW TITLE

# ADJUST SELECT AXES Y LABELS
for i, m in zip(range(0, len(axes), 3), test_plot["measure"].unique()):
    tit='"Max-Min (' + m +')'
    axes[i].set_ylabel(tit)


# for order = 2, lowess
test_plot = test_plot.loc[test_plot.measure=='F1-score']

#### --> test!
g=sns.lmplot('diff_mse', 'score_diff', order=2, data=test_plot, col='corpus', hue = 'monotonicity', row='measure', sharey=True, sharex=False, height=2.5,aspect=1.25, truncate=False, scatter_kws={"marker": "D", "s": 20})

#test_plot= test_plot[(test_plot.corpus=='Fairview')&(test_plot.measure=='F1-score')]
#g=sns.jointplot(x="diff_", y="score_diff", data=test_plot, kind="reg")

(
        g.set_axis_labels("Max-Min (comp measure)", "Max-Min (measure)")
        .set(ylim=(-.1, 0.7)) # -> .75 procs ; 0.7 for all/lowess&order=2
)
#(g.set(xlim=(0, 2.5), ylim=(-.1, 1)))

alpha = list('abcdefghijklmnopqrstuvwxyz')
axes = g.axes.flatten()

# ADJUST ALL AXES TITLES
for ax, letter in zip(axes, alpha[:len(axes)]):
    ttl = ax.get_title().split("|")[1].strip()   # GET CURRENT TITLE
    ax.set_title(f"({letter}) {ttl}")            # SET NEW TITLE

# ADJUST SELECT AXES Y LABELS
for i, m in zip(range(0, len(axes), 3), test_plot["measure"].unique()):
    tit='"Max-Min (' + m +')'
    axes[i].set_ylabel(tit)

# disorders, procs, all
for i, ax in enumerate(g.axes.flat):
    if i % 3 == 0:
        ax.set_xlim(0, .6)
    elif i in [1, 4, 7]:
        ax.set_xlim(0, .9) # .8 -> all, procs
    else:
        ax.set_xlim(0, .8) # .6 -> procs

#drugs
for i, ax in enumerate(g.axes.flat):
    if i % 2 == 0:
        ax.set_xlim(0, .7) # .6 -> annatomy
    else:
        ax.set_xlim(0, .8) # .6 -> anatomy

# order 2: disorder
for i, ax in enumerate(g.axes.flat):
    if i == 0:
        ax.set_xlim(0, .4) # .5 -> all
    elif i == 1:
        ax.set_xlim(0, .8) # .6 -> all
    else:
        ax.set_xlim(0, .55) # .45 -> procs

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('All groups')

"""
