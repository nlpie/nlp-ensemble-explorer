import pandas as pd
import copy
import itertools
import operator
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [10, 6]
# Set up with a higher resolution screen (useful on Mac)
%config InlineBackend.figure_format = 'retina'
sns.set()

def monotone_increasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.lt, pairs))

def monotone_decreasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.gt, pairs))

def monotone(lst):
    return monotone_increasing(lst) or monotone_decreasing(lst)

data_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/new_ner/complementarity_top_100/'
#data_dir = '/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/submission/new_ner/complementarity_top_one/'
corpora = ['fairview', 'i2b2', 'mipacq']

def get_corpora_all(data_dir, corpora, top=None):

    df = pd.DataFrame()
    
    for corpus in corpora:

        data=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_False_10-25-2021.csv')
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

        for mtype in measures:

            if mtype == 'f1':
                mtype = mtype.upper()
            
            if top:
                # get top one
                top = data.loc[data.mtype==mtype].head(1)
                sentences=set(top.loc[top.mtype==mtype].sentence.tolist())
            
            else:
                sentences=set(data.loc[data.mtype==mtype].sentence.tolist())

            mtype = mtype.lower()
            
            if mtype == 'f1':
                cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                        'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                        'f1-score', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                        'sentence', 'order', 'operator', 'merge_left', 'merge_right']

            elif mtype == 'precision':
                cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                        'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                        'precision', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                        'sentence', 'order', 'operator', 'merge_left', 'merge_right']

            elif mtype == 'recall':
                cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                        'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                        'recall', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                        'sentence', 'order', 'operator', 'merge_left', 'merge_right']

            m=0
            n=0
            pos=0
            neg=0

            for s in list(sentences):

                measure = mtype.lower()
                if mtype == 'f1':
                    measure = mtype + '-score'
                    mtype = mtype.upper()

                
                test=data.loc[(data.sentence==s)&(data.mtype==mtype)]

                mtype = mtype.lower()
                
                test[measure] = np.where(test['operator']=='&', test[mtype + '_and'], test[mtype + '_or'])


                t=test[cols_to_keep].sort_values(['order', measure], ascending=False)
                o=set(t.order.to_list())
                scores=[]
                for i in o:
                    u=t.loc[t.order==i]
                    vals = list(set(u[measure].tolist()))
                    vals.sort(reverse=True)
                    for s in vals:                       
                        #scores.append(u[measure].values[0])
                        scores.append(s)

                if monotone(scores[::-1]):
                    m+=1
                else:
                    n+=1

                if monotone_increasing(scores[::-1]):
                    #increase.append(1)
                    pos+=1
                elif monotone_decreasing(scores[::-1]):
                    #decrease.append(1)
                    neg+=1

            if mtype == 'precision':
                monotonic_p.append(m)
                nonmono_p.append(n)
                #increase_p.append(sum(increase)) 
                #decrease_p.append(sum(decrease)) 
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

        out = pd.DataFrame({'corpus': corpus, 'group': 'all', 
            'monotonic p': monotonic_p, 'non p': nonmono_p, 'increase p': increase_p, 'decrease p': decrease_p, 
            'monotonic r': monotonic_r, 'non r': nonmono_r, 'increase r': increase_r, 'decrease r': decrease_r,
            'monotonic f1': monotonic_f, 'non f1': nonmono_f, 'increase f1': increase_f, 'decrease f1': decrease_f})
        df = pd.concat([df,out])
    
    return df


############ ssemgroups across all corpora
def get_corpora_sg(data_dir, corpora, top=None):
    dict_of_df = {}
    df_list = []

    df = pd.DataFrame()

    for corpus in corpora:

        data=pd.read_csv(data_dir + 'complement_'+ corpus +'_filter_semtype_True_10-25-2021.csv')
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

                if mtype == 'f1':
                    mtype = mtype.upper()

                if top:
                    # get top one
                    top = dict_of_df[group].loc[(dict_of_df[group].mtype==mtype)&(dict_of_df[group].semgroup==group)].head(1)

                    sentences=set(top.loc[(top.mtype==mtype)&(top.semgroup==group)].sentence.tolist()) 
                else:
                    #sentences=set(data.loc[(data.mtype==mtype)&(data.semgroup==group)].sentence.tolist())
                    sentences=set(dict_of_df[group].loc[(dict_of_df[group].mtype==mtype)&(dict_of_df[group].semgroup==group)].sentence.tolist()) 

                mtype = mtype.lower()
                
                if mtype == 'f1':
                    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                            'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                            'f1-score', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                            'sentence', 'order', 'operator', 'merge_left', 'merge_right']

                elif mtype == 'precision':
                    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                            'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                            'precision', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                            'sentence', 'order', 'operator', 'merge_left', 'merge_right']

                elif mtype == 'recall':
                    cols_to_keep=['mtype', 'moi', 'merge_right', 'precision_right', 'recall_right', 'f1_right',
                            'merge_left', 'precision_left', 'recall_left', 'f1_left', 
                            'recall', 'precision_or', 'recall_or', 'f1_or', 'precision_and', 'recall_and', 'f1_and', 
                            'sentence', 'order', 'operator', 'merge_left', 'merge_right']
                

                m=0
                n=0
                pos=0
                neg=0

                for s in list(sentences):

                    measure = mtype.lower()
                    if mtype == 'f1':
                        measure = mtype + '-score'
                        mtype = mtype.upper()
                   
                    test=dict_of_df[group].loc[(dict_of_df[group].sentence==s)&(dict_of_df[group].mtype==mtype)&(dict_of_df[group].semgroup==group)]
                    
                    mtype = mtype.lower()
                    
                    test[measure] = np.where(test['operator']=='&', test[mtype + '_and'], test[mtype + '_or'])


                    t=test[cols_to_keep].sort_values(['order', measure], ascending=False)

                    o=set(t.order.to_list())

                    scores=[]
                    for i in o:
                        u=t.loc[t.order==i]
                        vals = list(set(u[measure].tolist()))
                        vals.sort(reverse=True)
                        for s in vals:                       
                            #scores.append(u[measure].values[0])
                            scores.append(s)
                   
                    if monotone(scores[::-1]):
                        m+=1
                    else:
                        n+=1

                    if monotone_increasing(scores[::-1]):
                        #increase.append(1)
                        pos+=1
                    elif monotone_decreasing(scores[::-1]):
                        #decrease.append(1)
                        neg+=1

                if mtype == 'precision':
                    monotonic_p.append(m)
                    nonmono_p.append(n)
                    #increase_p.append(sum(increase)) 
                    #decrease_p.append(sum(decrease)) 
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
 

    return df

#data = pd.concat([get_corpora_all(data_dir, corpora), get_corpora_sg(data_dir, corpora)])

#semgroups = sorted(list(set(data.group.tolist())))

### ->>>>>>>>>>>>>>>>>>

df = pd.concat([get_corpora_all(data_dir, corpora), get_corpora_sg(data_dir, corpora)])
semgroups = sorted(list(set(df.group.tolist())))

#semgroups = df.group.unique()  # unique groups
#corpus = df.corpus.unique()  # unique corpus
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

    # Setting the values for all axes.
    plt.setp(axes, ylim=custom_ylim)
    fig.suptitle('Figure ' + str(ix) + ' ' + sg)
    fig.tight_layout()
    plt.show()

