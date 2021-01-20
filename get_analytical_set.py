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

    # semantic types for filtering
    sql = """SELECT st.tui as tui, abbreviation, clamp_name, ctakes_name 
            FROM semantic_groups sg join semantic_types st on sg.tui = st.tui 
            where group_name = 'Disorders';"""
      
    return pd.read_sql(sql, engine)
    
    
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

    union distinct

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
    
    df = arg[['begin', 'end', 'note_id', 'cui', 'concept', 'semtype', 'score', 'length', 'type', 'system', 'polarity']].copy()
    df.sort_values(by=['note_id','begin'],inplace=True)
    
    data = []
    out = pd.DataFrame()
    
    for row in df.itertuples():
        # get overlapping intervals: 
        # https://stackoverflow.com/questions/58192068/is-it-possible-to-use-pandas-overlap-in-a-dataframe
        iix = pd.IntervalIndex.from_arrays(df.begin, df.end, closed='neither')
        span_range = pd.Interval(row.begin, row.end)
        fx = df[iix.overlaps(span_range)].copy()
 
        maxLength = fx['length'].max()
        minLength = fx['length'].min()
        maxScore = abs(float(fx['score'].max()))
        minScore = abs(float(fx['score'].min()))
        
        if maxLength > minLength:
            fx = fx[fx['length'] == maxLength]
        
        elif maxScore > minScore:
            fx = fx[fx['score'] == maxScore]        
            
        data.append(fx)

    out = pd.concat(data, axis=0)
   
    # randomly reindex to keep random row when dropping duplicates: https://gist.github.com/cadrev/6b91985a1660f26c2742
    out.reset_index(inplace=True)
    out = out.reindex(np.random.permutation(out.index))
    
    return out.drop_duplicates(subset=['begin', 'end', 'note_id', 'polarity'])

    
def main():
    engine = create_engine('mysql+pymysql://gms:nej123@localhost/med_mentions', pool_pre_ping=True)
    print('BEGIN')
    
    #GENERATE analytical table
    analytical_cui = pd.DataFrame()

    # TODO get cases method
    sql = """SELECT  distinct file as note_id 
             FROM medmentions_annotations
          """
           
    notes = pd.read_sql(sql, engine)
    
    cases = set(notes['note_id'].tolist())

    qumls, b9, clamp, ctakes, mm = get_data(engine)
    
    i = 0
    for case in cases:
        print(case, i)
        i += 1
    
        ### ------> qumls
        
        test = qumls[qumls['note_id'] == case].copy()
        print('qumls', len(test))

        if len(test) > 0:
            #frames = [ analytical_cui, disambiguate(test) ]
            frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
        
        ### ------>  b9
        test = b9[b9['note_id'] == case].copy()
        print('b9 umls', len(test))

        if len(test) > 0:
            #frames = [ analytical_cui, disambiguate(test) ]
            frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False)  

        ### ------>  clamp
        test = clamp[clamp['note_id'] == case].copy()
        print('clamp', len(test))
        if len(test) > 0:
            #frames = [ analytical_cui, disambiguate(test) ]
            frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  ctakes
        test = ctakes[ctakes['note_id'] == case].copy()
        print('ctakes mentions', len(test))
        if len(test) > 0:
            #frames = [ analytical_cui, disambiguate(test) ]
            frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  mm
        test = mm[mm['note_id'] == case].copy()
        print('mm candidate', len(test))
        if len(test) > 0:
            #frames = [ analytical_cui, disambiguate(test) ]
            frames = [ analytical_cui, test ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
        
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    #analytical_cui = analytical_cui.drop('length', 1)
    
    return analytical_cui
    
    
if __name__ == '__main__':
    #%prun main()
    corpus = 'medmentions'
    analytical_cui = main()
    elapsed = (time.time() - start)
    print('fini!', 'time:', elapsed)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    #analytical_cui.ro_sql("analytical_cui" + timestamp, engine)
    file = 'analytical_' + corpus + '_' + str(timestamp) +'.csv'
    
    analytical_cui.to_csv(data_folder / file)
    
# fill in null conccepts:
'''
data='/mnt/DataResearch/DataStageData/ed_provider_notes/output/'
file='analytical_fairview_cui_filtered_by_semtype_1597485061.111386.csv'
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