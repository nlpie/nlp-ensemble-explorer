import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [10, 6]
# Set up with a higher resolution screen (useful on Mac)
%config InlineBackend.figure_format = 'retina'
sns.set()

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

            test=df.drop_duplicates(subset=['sentence', 'corpus', 'mtype'])
            test['score_diff']=test.score_diff.astype('float64')
            test['add_comp'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff'].transform('sum')/2
            test['max_diff'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff'].transform('max')

            # MSE of cummulative comp gain
            test['diff_sq'] = test['diff']*test['diff']
            test['diff_sq_sum'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_sq'].transform('sum')/2
            test['diff_mse']=(test['diff_sq_sum'] / test['n_sub_ens']).pow(1./2)

            test['diff_p_sq'] = test['diff_p']*test['diff_p']
            test['diff_p_sq_sum'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_p_sq'].transform('sum')/2
            test['diff_p_mse']=(test['diff_p_sq_sum'] / test['n_sub_ens']).pow(1./2)
            test['max_diff_p'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_p'].transform('max')

            test['diff_r_sq'] = test['diff_r']*test['diff_r']
            test['diff_r_sq_sum'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_r_sq'].transform('sum')/2
            test['diff_r_mse']=(test['diff_r_sq_sum'] / test['n_sub_ens']).pow(1./2)
            test['max_diff_r'] = test.groupby(['corpus', 'sentence', 'mtype', 'group'])['diff_r'].transform('max')

            '''
            cols=['sentence', 'corpus', 'group', 'mtype', 
            'p_comp', 'r_comp', 'f1_comp',
            'moi', 'monotonicity',
            'max_f1_comp', 'min_f1_comp', 
            'max_p_comp','max_f1_comp_', 
            'diff','diff_', 'score_diff']
            '''
          
            p1 = 'score_diff'
            #p2 = 'diff_'
            
            tt=test[test.monotonicity.isin(['i'])]
            n_i = len(tt) 
            if mt == 'precision':
                p2 = 'diff_p_sq'
                p2 = 'max_diff_p'
                p2 = 'diff_p_mse'
            elif mt == 'recall':
                p2 = 'diff_r_sq'
                p2 = 'max_diff_r'
                p2 = 'diff_r_mse'
            else:
                p2 = 'diff_sq'
                p2 = 'max_diff'
                p2 = 'diff_mse'

            if n_i > 2: 
                out['n_i'] = n_i 
                out['pearsonr_i']=pearsonr(tt[p1], tt[p2])[0]
                out['pearsonr_i_p']=pearsonr(tt[p1], tt[p2])[1]
                pass
            else:
                out['n_i'] = n_i 

            tt=test[test.monotonicity.isin(['n'])]
            n_n = len(tt) 
            if n_n > 2: 
                out['n_n'] = n_n 
                out['pearsonr_n']=pearsonr(tt[p1], tt[p2])[0]
                out['pearsonr_n_p']=pearsonr(tt[p1], tt[p2])[1]
                pass
            else:
                out['n_n'] = n_n 

            tt=test[test.monotonicity.isin(['d'])]
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
pear=pd.read_csv('/Users/gms/Desktop/pr_500_sq_diff.csv')

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


ipython.magic('R -i u')

#pmod = ro.r("pmod <- lm('moi ~ diff_comp_mse+factor(mono)+diff_comp+nterms', data=u)")

#pmod = ro.r("pmod <- lm('moi_score_diff ~ diff_comp_mse_sq+diff_comp+factor(mono)+nterms', data = u)")

pmod = ro.r("pmod <- lm('moi_score_diff ~ diff_comp_mse+diff_comp+diff_comp_sq+factor(mono)+nterms', data = u)")


print(base.summary(pmod))

vif = ro.r('car::vif(pmod)')

# ---------> corr

cols=['nterms', 'diff_comp', 'diff_comp_sq', 'diff_comp_mse', 'diff_comp_mse_sq', 'moi', 'moi_score_diff']

# Correlations ================> https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
cols=['moi', 'moi_score_range', 'nterms', 'comp_f1_gain_mse', 'comp_p_gain_mse', 'comp_r_gain_mse']
u=df.loc[df.group=='Disorders']
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
df=pd.read_csv('~/Desktop/top_100.csv')
# Monotonicity bar plots: get df from monotonicity summary
semgroups = sorted(list(set(df.group.tolist())))
rows = [['monotonic p', 'non p', 'monotonic r', 'non r', 'monotonic f1', 'non f1'], 
        ['increase p', 'decrease p', 'increase r', 'decrease r', 'increase f1', 'decrease f1']]  # columns for each row of plots

clrs = ['red', 'red', 'blue', 'blue', 'green', 'green']
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
            sns.barplot(data=data, ax=ax, palette = clrs)

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
