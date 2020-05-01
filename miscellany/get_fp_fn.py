import pandas as pd

def get_fp_fn(gold, sys, return_type='FN'):
    '''
    params: gold -> df of gold standard annotations
            sys -> df of system generrated annotations
            return_type -> default as 'FN'
   
    returns: df -> df of either FN or FP
    '''
    
    #gold = gold[gold['note_id']=='NCT00042380_criteria'].drop_duplicates()
    #sys = sys[sys['note_id']=='NCT00042380_criteria']
    
    df = pd.DataFrame()
    if return_type == 'FN':
        #iterate through gold as tuples
        for g in gold.itertuples():
            g_begin = int(g.begin)
            g_end = int(g.end)
            g_case = g.note_id
            mMatch = False # flag when there is a match
            for s in sys.itertuples():
                s_begin = int(s.begin)
                s_end = int(s.end)
                s_case = s.note_id
                if (((g_begin <= s_end and g_end >= s_begin) or 
                     (g_end >= s_begin and s_end >= g_begin)) and g_case == s_case):
                        mMatch = True
                        # create a df of TP here if wanted
                        # break once a match is found
                        break
                        
            # no match found, write list of tuple elements to df
            if mMatch == False:
                l = [g.begin,g.end,g.semtype,g.note_id,g.text,'FN']
                fn = pd.DataFrame(columns=['begin','end','semtypes','note_id','text','value'], data=None)
                fn.loc[fn.shape[0]] = l 
                frames = [df,fn]
                #print(fn)
                df = pd.concat(frames)
                
    else:    
        for s in sys.itertuples(index=False):
            s_begin = int(s.begin)
            s_end = int(s.end)
            s_case = s.note_id
            mMatch = False
            for g in gold.itertuples():
                g_begin = int(g.begin)
                g_end = int(g.end)
                g_case = g.note_id
                if (((g_begin <= s_end and g_end >= s_begin) or 
                     (g_end >= s_begin and s_end >= g_begin)) and g_case == s_case):
                        mMatch = True
                        break

            if mMatch == False:
                l = [s.begin,s.end,s.semtypes,s.note_id,s.text,s.type,s.system, 'FP']
                fp = pd.DataFrame(columns=['begin','end','semtypes','note_id','text','type','system','value'], data=None)
                fp.loc[fp.shape[0]] = l 
                frames = [df,fp]
                df = pd.concat(frames)
                #print(fp)
       
    return df

# path/file
gold='~/Desktop/clinical_trial2_reference.csv' 
system='~/development/nlp/nlpie/data/ensembling-u01/output/analytical_clinical_trial2.csv'
analysis_type = 'FP'  # default for method is 'FN'

df = span_to_text(pd.read_csv(gold).drop_duplicates(),pd.read_csv(system).drop_duplicates(), analysis_type)
print(len(df))
pd.to_csv('fp.csv')
