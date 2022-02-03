import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import pymysql
from datetime import datetime
from pathlib import Path
import time

data_folder = Path("/Users/gms/development/nlp/nlpie/data/ensembling-u01/output/")
start = time.time()

def get_systems_set(sql, engine):

    df = pd.read_sql(sql, engine)
    
    return df
    
def get_sem_types(engine):

    # semantic types for filtering: 'Disorders', 'Procedures', 'Anatomy', 'Chemicals & Drugs'
    sql = """SELECT st.tui as tui, abbreviation, clamp_name, ctakes_name, group_name 
            FROM semantic_groups sg join semantic_types st on sg.tui = st.tui 
            where group_name in ('Disorders');"""
      
    return pd.read_sql(sql, engine)

def get_data_file(file_name, st=None):

    if len(st) > 0: 
        df = pd.read_csv(data_folder / file_name, dtype={'file': str})

        if corpus != 'medmentions':
            df = df[['begin', 'end', 'note_id', 'score', 'semtypes', 'system', 'corpus', 'cui']]
        else:
            df = df[['begin', 'end', 'note_id', 'score', 'semtype', 'polarity', 'system', 'cui']]

        print(df.columns)

        mask = [st for st in list(set(st.tui.tolist()))]
        qumls = df.loc[df.system=='quick_umls']
        qumls = qumls[qumls.semtypes.isin(mask)]

        b9 = df.loc[df.system=='biomedicus']
        b9 = b9[b9.semtypes.isin(mask)]

        if 'Anatomy' not in st.group_name.tolist():
            mask = [st.split(',')  for st in list(set(st.clamp_name.tolist()))][0]
            clamp = df.loc[df.system=='clamp']
            clamp = clamp[clamp.semtypes.isin(mask)]
        else:
            clamp = None
        
        mask = [st.split(',')  for st in list(set(st.ctakes_name.tolist()))][0]
        ctakes =  df.loc[df.system=='ctakes']
        ctakes = ctakes[ctakes.semtypes.isin(mask)]

        mask = [st for st in list(set(st.abbreviation.tolist()))]
        mm =  df.loc[df.system=='metamap']
        mm = mm[mm.semtypes.isin(mask)]

    else:
        df = pd.read_csv(data_folder / file_name, dtype={'note_id': str})

        if corpus in ['medmentions']:
            df = df[['begin', 'end', 'note_id', 'score', 'semtype', 'polarity', 'system', 'cui']]
        elif corpus in ['mipacq']:
            df = df[['begin', 'end', 'note_id', 'score', 'semtypes', 'system', 'cui']]
        else:
            df = df[['begin', 'end', 'note_id', 'score', 'semtypes', 'system', 'cui']]

        qumls = df.loc[df.system=='quick_umls']

        b9 = df.loc[df.system=='biomedicus']

        clamp = df.loc[df.system=='clamp']
        
        ctakes =  df.loc[df.system=='ctakes']

        mm =  df.loc[df.system=='metamap']

    return qumls, b9, clamp, ctakes, mm
   
def get_data(engine):

    ### ------> qumls
    
    sql = """
    SELECT q.begin, q.end, null as concept, q.cui, q.similarity as score, q.semtypes as semtype, q.note_id, q.type, q.system, 0 as polarity 
    FROM qumls_system_combos q
    where overlap='score' and best_match=1;
    """
    
    qumls = get_systems_set(sql, engine)
    qumls['semtype'] = qumls.semtype.str.replace('{','').str.replace('}','').str.replace("'",'').str.strip()
    qumls =  qumls.assign(semtype=qumls.semtype.str.split(", ")).explode('semtype')
    
    ### ------> b9

    sql = """
    select u.begin, u.end, null as concept, u.cui, u.confidence as score, u.tui as semtype, u.note_id, u.type, u.system, -1 as polarity 
    from bio_biomedicus_UmlsConcept u left join bio_biomedicus_Negated n
        on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id 
        where n.begin is not null and n.end is not null and n.note_id is not null

    union 

    select u.begin, u.end, null as concept, u.cui, u.confidence as score, u.tui as semtype, u.note_id, u.type, u.system, 1 as polarity 
    from bio_biomedicus_UmlsConcept u left join bio_biomedicus_Negated n
        on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id  
        where n.begin is null and n.end is null and n.note_id is null;

    """

    b9 = get_systems_set(sql, engine)

    ### ------> clamp
    
    sql = """
    select u.begin, u.end, u.concept, u.cui, COALESCE(u.concept_prob, 0) as score, u.semanticTag as semtype, u.note_id, u.type, u.system, -1 as polarity 
    from cla_edu_ClampNameEntityUIMA u left join cla_edu_ClampRelationUIMA r
        on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id 
        where u.assertion = 'absent' and r.begin is not null and r.end is not null and r.note_id is not null

    union 

    select u.begin, u.end, u.concept, u.cui, COALESCE(u.concept_prob, 0) as score, u.semanticTag as semtype, u.note_id, u.type, u.system, 1 as polarity 
    from cla_edu_ClampNameEntityUIMA u left join cla_edu_ClampRelationUIMA r
        on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id 
        where (u.assertion = 'present' or u.assertion is null) and r.begin is null and r.end is null and r.note_id is null;

    """
    
    clamp = get_systems_set(sql, engine)
    
    ### ------> ctakes
    
    sql = """
    select `begin`, `end`, concept, cui, 0 as score, SUBSTRING_INDEX(SUBSTRING_INDEX(`med_mentions`.`cta_org_AnatomicalSiteMention`.`type`, ',',7),'.', -(1))  as semtype,              note_id, 'ctakes_mentions' as `type`,  `system`, polarity  
    from cta_org_AnatomicalSiteMention 

    union distinct 

    select `begin`, `end`, concept, cui, 0 as score, SUBSTRING_INDEX(SUBSTRING_INDEX(`med_mentions`.`cta_org_DiseaseDisorderMention`.`type`, ',',7),'.', -(1))  as semtype, 
    note_id, 'ctakes_mentions' as `type`,  `system`, polarity  
    from cta_org_DiseaseDisorderMention 

    union distinct 

    select `begin`, `end`, concept, cui, 0 as score, SUBSTRING_INDEX(SUBSTRING_INDEX(`med_mentions`.`cta_org_MedicationMention`.`type`, ',',7),'.', -(1))  as semtype, 
    note_id, 'ctakes_mentions' as `type`,  `system`, polarity  
    from cta_org_MedicationMention 

    union distinct 

    select `begin`, `end`,  concept, cui, 0 as score,  SUBSTRING_INDEX(SUBSTRING_INDEX(`med_mentions`.`cta_org_ProcedureMention`.`type`, ',',7),'.', -(1))  as semtype,  
    note_id, 'ctakes_mentions' as `type`,  `system`, polarity  
    from cta_org_ProcedureMention 

    union distinct 
    
    select `begin`, `end`,  concept, cui, 0 as score,  SUBSTRING_INDEX(SUBSTRING_INDEX(`med_mentions`.`cta_org_SignSymptomMention`.`type`, ',',7),'.', -(1))  as semtype,  
    note_id, 'ctakes_mentions' as `type`,  `system`, polarity  
    from cta_org_SignSymptomMention;
   
    """
    
    ctakes = get_systems_set(sql, engine)
    
    ### ------> mm
    
    sql = """
    select c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score) as score, c.semanticTypes as semtype, c.note_id, c.type, c.system, -1 as polarity 
    from met_org_Candidate c left join met_org_Negation n
        on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id 
        where n.begin is not null and n.end is not null and n.note_id is not null

    union distinct

    select c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score) as score, c.semanticTypes as semtype, c.note_id, c.type, c.system, 1 as polarity 
    from met_org_Candidate c left join met_org_Negation n
        on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id 
        where n.begin is null and n.end is null and n.note_id is null;

    """
    
    mm = get_systems_set(sql, engine)
    # explode list like string into multiple rows
    mm =  mm.assign(semtype=mm.semtype.str.split(" ")).explode('semtype')
    
    return qumls, b9, clamp, ctakes, mm


def disambiguate(arg):
    
    arg['length'] = (arg.end - arg.begin).abs()
    
    arg.sort_values(by=['note_id','begin'],inplace=True)
   
    if corpus in ['medmentions']:
        df = arg[['begin', 'end', 'note_id', 'cui', 'semtype', 'score', 'length', 'system', 'polarity']].copy()
    #df = arg[['begin', 'end', 'note_id', 'score', 'length', 'system', 'cui', 'semtypes']].copy()
    #elif corpus in ['mipacq']:
    #    df = arg[['begin', 'end', 'note_id', 'cui', 'semtypes', 'score', 'length', 'system']].copy()
    else:
        df = arg[['begin', 'end', 'note_id', 'score', 'length', 'system', 'semtypes', 'cui']].copy()
    
    df.sort_values(by=['note_id','begin'],inplace=True)
    
    data = []
    out = pd.DataFrame()
    
    for row in df.itertuples():
        # get overlapping intervals: 
        # https://stackoverflow.com/questions/58192068/is-it-possible-to-use-pandas-overlap-in-a-dataframe
        iix = pd.IntervalIndex.from_arrays(df.begin, df.end, closed='neither')
        span_range = pd.Interval(row.begin, row.end)
        
        fx = df[iix.overlaps(span_range)].copy()
        fx.sort_values(by=['note_id','begin'],inplace=True)
        
        fx = fx.drop_duplicates(subset=['length'])

        maxLength = fx['length'].max()
        minLength = fx['length'].min()
        maxScore = abs(float(fx['score'].max()))
        minScore = abs(float(fx['score'].min()))
        
        if maxLength > minLength:
            fx = fx[fx['length'] == maxLength]
        elif maxScore > minScore:
            fx = fx[fx['score'] == maxScore]        
            if len(fx) > 1:
                n = len(fx)
                fx.drop(fx.tail(n-1).index, inplace = True)
        
        data.append(fx)

    out = pd.concat(data, axis=0)

    # randomly reindex to keep random row when dropping duplicates: https://gist.github.com/cadrev/6b91985a1660f26c2742
    out.reset_index(drop=True, inplace=True)
    out = out.reindex(np.random.permutation(out.index))
 
    '''
    # instead: WHILE #overlaps in list using get_overlaps > 1
    run = True 
    while run:
        test=get_overlaps(out)

        run = False

        i = 0
        for t in test:
            print(t)
            if len(t) > 0:
                run = True
                out = disambiguate(out)
                print('len', llen(t))    
            i+=1
            if i%100==0:
                print(i, run)
    '''
    
    return out.drop_duplicates(subset=['begin', 'end', 'note_id'])
    #return out.drop_duplicates(subset=['begin', 'end', 'note_id', 'polarity'])

def get_overlaps(df):
        
    dfs = []

    df = df.sort_values('begin').reset_index(drop=True)
    idx = 0
    while True:
        lower = 'begin'        
        low = df[lower][idx]
        upper = 'end'        
        high = df[upper][idx]
        sub_df = df[(df['begin'] <= high) & (low <= df['begin'])]
        dfs.append(sub_df)
        idx = sub_df.index.max() + 1
        if idx > df.index.max():
            break

        #print('max/min')
        #print(len(dfs))

    return dfs

def assemble_data(cases, qumls, b9, clamp, ctakes, mm, group=None): 
    
    analytical_cui = pd.DataFrame()
    if corpus not in ['medmentions']:
        cols_to_keep=['begin', 'end', 'semtypes']
    else:
        cols_to_keep=['begin', 'end', 'semtype', 'cui']

    i = 0
    for case in cases:
        print(case, i)
        i += 1
    
        ### ------> qumls
        
        test = qumls[qumls['note_id'] == case].copy()
        test = test.sample(frac=1).drop_duplicates(subset=cols_to_keep)
        print('qumls', len(test))

        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            #frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
        
        ### ------>  b9
        test = b9[b9['note_id'] == case].copy()
        test = test.sample(frac=1).drop_duplicates(subset=cols_to_keep)
        print('b9 umls', len(test))

        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            #frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False)  

        ### ------>  clamp
        
        if group != 'Anatomy':
            test = clamp[clamp['note_id'] == case].copy()
            test = test.sample(frac=1).drop_duplicates(subset=cols_to_keep)
            print('clamp', len(test))
            if len(test) > 0:
                frames = [ analytical_cui, disambiguate(test) ]
                #frames = [ analytical_cui, test ]
                analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  ctakes
        test = ctakes[ctakes['note_id'] == case].copy()
        test = test.sample(frac=1).drop_duplicates(subset=cols_to_keep)
        print('ctakes mentions', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test)]
            #frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  mm
        test = mm[mm['note_id'] == case].copy()
        test = test.sample(frac=1).drop_duplicates(subset=cols_to_keep)
        print('mm candidate', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            #frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

    return analytical_cui
    
def main(corpus, semtypes=None):
    if corpus not in ['medmentions']:
        engine = create_engine('mysql+pymysql://gms:nej123@localhost/concepts', pool_pre_ping=True)
    else:
        engine = create_engine('mysql+pymysql://gms:nej123@localhost/med_mentions', pool_pre_ping=True)
    print('BEGIN')
    
    #GENERATE analytical table

    # TODO get cases method
    sql = """SELECT  distinct file as file 
             FROM mipacq_all
          """
    
    file ='analytical_' + corpus + '.csv'
    #file = 'analytical_disambiguated_fairview_1633366674.934254.csv'
    notes = pd.read_sql(sql, engine)
    
    cases = set(notes['file'].tolist())

    #qumls, b9, clamp, ctakes, mm = get_data(engine)
   
    if semtypes:
        analytical_cui = pd.DataFrame()
        st = get_sem_types(engine)
        groups = set(st.group_name.tolist())

        for group in groups:
            print("Group:", group)


            semtypes = st.loc[st.group_name==group]

            qumls, b9, clamp, ctakes, mm = get_data_file(file, semtypes)

            frames = [analytical_cui, assemble_data(cases, qumls, b9, clamp, ctakes, mm, group)]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
            analytical_cui['sem_group']=group
    else:
            qumls, b9, clamp, ctakes, mm = get_data_file(file)
            analytical_cui = assemble_data(cases, qumls, b9, clamp, ctakes, mm)
            
        
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    #analytical_cui = analytical_cui.drop('length', 1)
    
    return analytical_cui
    
    
if __name__ == '__main__':
    #%prun main()
    corpus = 'mipacq'
    semtype = 'Disorders,Sign_Symptom'
    analytical_cui = main(corpus, semtype)
    elapsed = (time.time() - start)
    print('fini!', 'time:', elapsed)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    #analytical_cui.ro_sql("analytical_cui" + timestamp, engine)
    if semtype:
        file = 'analytical_disambiguated_' + corpus + '_' + semtype + '_' + str(timestamp) +'.csv'
    else: 
        file = 'analytical_disambiguated_' + corpus + '_' + str(timestamp) +'.csv'
    analytical_cui.to_csv(data_folder / file)
    
# fill in null conccepts:
'''
data='/mnt/DataResearch/DataStageData/ed_provider_notes/output/'
file='analytical_medmentions_cui_filtered_by_semtype_1597485061.111386.csv'
df = pd.read_csv(data+file)
sg = df

sg.loc[sg.semtype.isin(['T020', 'T190', 'T047', 'T033', 'T184', 
'problem', 
'DiseaseDisorderMention','SignSymptomMention', 
'acab',
'anab'
'dsyn'
'fndg'
'sosy'])].drop_duplicates(subset='cui') 

test=sg.loc[sg.semtype.isin(['T020', 'T190', 'T047', 'T033', 'T184', 'problem', 'DiseaseDisorderMention','SignSymptomMention', 'acab',
'anab'
'dsyn'
'fndg'
'sosy'])].drop_duplicates(subset='cui')[['concept', 'cui','system']]

test=sg.loc[sg.semtype.isin(['T047', 'T033', 'T184', 'problem', 'DiseaseDisorderMention','SignSymptomMention',
'dsyn'
'fndg'
'sosy'])].drop_duplicates(subset='cui')[['concept', 'cui','system']]

test.sort_values('system', ascending=False).drop_duplicates(subset='cui')
t =test.sort_values('system', ascending=False).drop_duplicates(subset='cui')

mrconso=pd.read_sql('select "CUI", "STR", "ISPREF" from public."MRCONSO"', con=engine))
concepts=mrconso.loc[mrconso.ISPREF=='Y'].drop_duplicates(subset='CUI')[['CUI', 'STR']]

u=t.merge(concepts, left_on='cui', right_on='CUI')
u['concept'] = np.where(u.concept.isnull(), u.STR, u.concept)
df.merge(u, left_on='cui', right_on='CUI')

df.merge(u, left_on='cui', right_on='CUI')['concept_y']
'''
