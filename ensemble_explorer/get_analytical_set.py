import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
from datetime import datetime
from pathlib import Path
import time


data_folder = Path("/mnt/DataResearch/DataStageData/ed_provider_notes/output/")
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

    st = get_sem_types(engine)
    
    mask = [st for st in list(set(st.tui.tolist()))]

    ### ------> qumls
    
    sql = """
      SELECT q.begin, q.end, null as concept, q.cui, q.similarity as score, q.semtypes as semtype, split_part(q.note_id, '_', 2) as note_id, q.type, q.system, 0 as polarity 
      FROM qumls q inner join
      (select max(date_added) as da from qumls) as md on md.da = q.date_added;
      """
    
    sql = """
    SELECT q.begin, q.end, null as concept, q.cui, q.similarity as score, q.semtypes as semtype, split_part(q.note_id, '_', 2) as note_id, q.type, q.system, 0 as polarity 
    FROM qumls q;
    """
    
    # qumls = get_systems_set(sql, engine)
    # qumls = qumls[qumls.semtype.isin(mask)]
    
    ### ------> b9

    sql = """
    select u.begin, u.end, null as concept, u.cui, u.confidence::float as score, u.tui as semtype, u.note_id, u.type, u.system, -1 as polarity 
    from "bio_biomedicus_UmlsConcept" u left join "bio_biomedicus_Negated" n
        on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= u.date_added and md.da = n.date_added
        where n.begin is not null and n.end is not null and n.note_id is not null

    union distinct

    select u.begin, u.end, null as concept, u.cui, u.confidence::float as score, u.tui as semtype, u.note_id, u.type, u.system, 1 as polarity 
    from "bio_biomedicus_UmlsConcept" u left join "bio_biomedicus_Negated" n
        on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= u.date_added 
        where n.begin is null and n.end is null and n.note_id is null;

    """
    
    # sql = """
    # select u.begin, u.end, null as concept, u.cui, u.confidence::float as score, u.tui as semtype, u.note_id, u.type, u.system, -1 as polarity 
    # from "bio_biomedicus_UmlsConcept" u left join "bio_biomedicus_Negated" n
        # on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id 
        # where n.begin is not null and n.end is not null and n.note_id is not null

    # union distinct

    # select u.begin, u.end, null as concept, u.cui, u.confidence::float as score, u.tui as semtype, u.note_id, u.type, u.system, 1 as polarity 
    # from "bio_biomedicus_UmlsConcept" u left join "bio_biomedicus_Negated" n
        # on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id  
        # where n.begin is null and n.end is null and n.note_id is null;

    # """

    b9 = get_systems_set(sql, engine)
    b9 = b9[b9.semtype.isin(mask)]

    print('b9!')

    ### ------> clamp
    
    mask = [st.split(',')  for st in list(set(st.clamp_name.tolist()))][0]

    sql = """
    select u.begin, u.end, u.concept, u.cui, u.concept_prob::float as score, "u"."semanticTag" as semtype, u.note_id, u.type, u.system, -1 as polarity 
    from "cla_edu_ClampNameEntityUIMA" u left join "cla_edu_ClampRelationUIMA" r
        on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= u.date_added and md.da = r.date_added
        where u.assertion = 'absent' and r.begin is not null and r.end is not null and r.note_id is not null
        

    union distinct

    select u.begin, u.end, u.concept, u.cui, u.concept_prob::float as score, "u"."semanticTag" as semtype, u.note_id, u.type, u.system, 1 as polarity 
    from "cla_edu_ClampNameEntityUIMA" u left join "cla_edu_ClampRelationUIMA" r
        on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= u.date_added
        where (u.assertion = 'present' or u.assertion is null) and r.begin is null and r.end is null and r.note_id is null;

    """
    
    # sql = """
    # select u.begin, u.end, u.concept, u.cui, u.concept_prob::float as score, "u"."semanticTag" as semtype, u.note_id, u.type, u.system, -1 as polarity 
    # from "cla_edu_ClampNameEntityUIMA" u left join "cla_edu_ClampRelationUIMA" r
        # on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id 
        # where u.assertion = 'absent' and r.begin is not null and r.end is not null and r.note_id is not null
        

    # union 

    # select u.begin, u.end, u.concept, u.cui, u.concept_prob::float as score, "u"."semanticTag" as semtype, u.note_id, u.type, u.system, 1 as polarity 
    # from "cla_edu_ClampNameEntityUIMA" u left join "cla_edu_ClampRelationUIMA" r
        # on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id 
        # where (u.assertion = 'present' or u.assertion is null) and r.begin is null and r.end is null and r.note_id is null;

    # """
    
    clamp = get_systems_set(sql, engine)
    clamp = clamp[clamp.semtype.isin(mask)]
    
    print('clamp!')
    ### ------> ctakes
    
    mask = [st.split(',')  for st in list(set(st.ctakes_name.tolist()))][0]

    # sql = """
    # select  "begin", "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity  
    # from "cta_org_AnatomicalSiteMention" inner join
        # (select max(date_added) as da from "cta_org_AnatomicalSiteMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_DiseaseDisorderMention" inner join
        # (select max(date_added) as da from "cta_org_DiseaseDisorderMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_MedicationMention" inner join
        # (select max(date_added) as da from "cta_org_MedicationMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_ProcedureMention" inner join
        # (select max(date_added) as da from "cta_org_ProcedureMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_SignSymptomMention" inner join
        # (select max(date_added) as da from "cta_org_SignSymptomMention") as md on md.da = date_added ;

    # """
    
    
    
    # sql = """
    # select  "begin", "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity  
    # from "cta_org_AnatomicalSiteMention"  

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_DiseaseDisorderMention" 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_MedicationMention" 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_ProcedureMention" 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_SignSymptomMention"  ;

    # """
    
    # sql = """
    # select  "begin", "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity  
    # from "cta_org_AnatomicalSiteMention" inner join
        # (select max(date_added) as da from "cta_org_AnatomicalSiteMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_DiseaseDisorderMention" inner join
        # (select max(date_added) as da from "cta_org_DiseaseDisorderMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_MedicationMention" inner join
        # (select max(date_added) as da from "cta_org_MedicationMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_ProcedureMention" inner join
        # (select max(date_added) as da from "cta_org_ProcedureMention") as md on md.da = date_added 

    # union distinct 

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_SignSymptomMention" inner join
        # (select max(date_added) as da from "cta_org_SignSymptomMention") as md on md.da = date_added ;

    # """
    
    
    sql = """
     

    select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    from "cta_org_DiseaseDisorderMention" inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= date_added 

    union distinct 


    select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    from "cta_org_SignSymptomMention" inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= date_added ;

    """
    # sql = """
   
    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_DiseaseDisorderMention"  

    # union

    # select begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    # from "cta_org_SignSymptomMention"  ;

    # """
    
    ctakes = get_systems_set(sql, engine)
    
    print('ctakes!')
    ### ------> mm
    
    mask = [st for st in list(set(st.abbreviation.tolist()))]

    sql = """
    select c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score::int) as score, "c"."semanticTypes" as semtype, c.note_id, c.type, c.system, -1 as polarity 
    from "met_org_Candidate" c left join "met_org_Negation" n
        on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= c.date_added and md.da = n.date_added
        where n.begin is not null and n.end is not null and n.note_id is not null

    union distinct

    select c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score::int) as score, "c"."semanticTypes" as semtype, c.note_id, c.type, c.system, 1 as polarity 
    from "met_org_Candidate" c left join "met_org_Negation" n
        on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id inner join
        (select max(date_added) as da from ed_provider_notes where cohort = 1) as md on md.da <= c.date_added
        where n.begin is null and n.end is null and n.note_id is null;

    """
    
    # sql = """
    # select c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score::int) as score, "c"."semanticTypes" as semtype, c.note_id, c.type, c.system, -1 as polarity 
    # from "met_org_Candidate" c left join "met_org_Negation" n
        # on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id 
        # where n.begin is not null and n.end is not null and n.note_id is not null

    # union distinct

    # select c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score::int) as score, "c"."semanticTypes" as semtype, c.note_id, c.type, c.system, 1 as polarity 
    # from "met_org_Candidate" c left join "met_org_Negation" n
        # on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id 
        # where n.begin is null and n.end is null and n.note_id is null;

    # """
    
    mm = get_systems_set(sql, engine)
    # explode list like string into multiple rows
    mm =  mm.assign(semtype=mm.semtype.str.split(" ")).explode('semtype')
    mm = mm[mm.semtype.isin(mask)]
    
    print('mm!')
    #return qumls, b9, clamp, ctakes, mm
    return b9, clamp, ctakes, mm


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

def update_opt_out(engine):
    
    patients = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database__4_Oct_2020.dta")

    notes=pd.read_csv("/mnt/DataResearch/DataStageData/CV_PATIENT_ED_PROVIDER_NOTES.txt", dtype=str, engine='python', sep="~\|~")
    
    df = patients.merge(notes, on="MDM_LINK_ID")

    opt_out = list(set(df.loc[df.research_op_out==1]['MDM_LINK_ID'].tolist()))
    
    import sqlalchemy as sa
    from sqlalchemy.sql import text

    sql = text("""
                UPDATE public.ed_provider_notes 
                SET opt_out=1 
                WHERE "MDM_LINK_ID" in :opt_out 
               """)
               
    conn = engine.connect()
    conn.execute(sql, opt_out=tuple(opt_out))
    
def main():
    engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19')
    print('BEGIN')
    
    #GENERATE analytical table
    analytical_cui = pd.DataFrame()

    # TODO get cases method
    # sql = """SELECT  distinct note_id
            # FROM public."bio_biomedicus_UmlsConcept" inner join
            # (select max(date_added) as da from public."bio_biomedicus_UmlsConcept") as md on md.da = date_added;"""
            
    # sql = """select  distinct note_id
            # from "bio_biomedicus_UmlsConcept" b9 inner join "ed_provider_notes" ed
			# on b9.note_id = ed."NOTE_ID"
			# where  "NOTE_STATUS" != 'Incomplete';"""
            
    sql = """select distinct "NOTE_ID" 
              from ed_provider_notes inner join
              (select max(date_added) as da from ed_provider_notes) as md
              on md.da = date_added
              where cohort = 1 and opt_out is null;
              
          """
    notes = pd.read_sql(sql, engine)
    
    # patients = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database__3_Oct_2020.dta")
    
    # df = notes.merge(patients, on = "MDM_LINK_ID")
    
    # df = df.loc[df.research_op_out.isnull()]
    
    cases = set(notes['NOTE_ID'].tolist())
    
    #__, b9, clamp, ctakes, mm = get_data(engine)
    b9, clamp, ctakes, mm = get_data(engine)
    
    i = 0
    for case in cases:
        print(case, i)
        i += 1
    
        ### ------> qumls
        '''
        test = df[df['note_id'] == case].copy()
        print('qumls', len(test))

        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
        
        #print(analytical_cui[0:1], analytical_cui.columns)
        '''
        
        ### ------>  b9
        test = b9[b9['note_id'] == case].copy()
        print('b9 umls', len(test))

        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False)  

        ### ------>  clamp
        test = clamp[clamp['note_id'] == case].copy()
        print('clamp', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  ctakes
        test = ctakes[ctakes['note_id'] == case].copy()
        print('ctakes mentions', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  mm
        test = mm[mm['note_id'] == case].copy()
        print('mm candidate', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
        
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    analytical_cui = analytical_cui.drop('length', 1)
    
    return analytical_cui
    
    
if __name__ == '__main__':
    #%prun main()
    analytical_cui = main()
    elapsed = (time.time() - start)
    print('fini!', 'time:', elapsed)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    #analytical_cui.ro_sql("analytical_cui" + timestamp, engine)
    file = 'analytical_fairview_cui_filtered_by_semtype_' + str(timestamp) +'.csv'
    
    analytical_cui.to_csv(data_folder / file)
    
# fill in null concepts:
'''
data='/mnt/DataResearch/DataStageData/ed_provider_notes/output/'
file='analytical_fairview_cui_filtered_by_semtype_1603120549.778849.csv'
df = pd.read_csv(data+file)
sg = df

# sg.loc[sg.semtype.isin(['T020', 'T190', 'T047', 'T033', 'T184', 
# 'problem', 
# 'DiseaseDisorderMention','SignSymptomMention', 
# 'acab',
# 'anab'
# 'dsyn'
# 'fndg'
# 'sosy'])].drop_duplicates(subset='cui') 

# test=sg.loc[sg.semtype.isin(['T020', 'T190', 'T047', 'T033', 'T184', 'problem', 'DiseaseDisorderMention','SignSymptomMention', 'acab',
# 'anab'
# 'dsyn'
# 'fndg'
# 'sosy'])].drop_duplicates(subset='cui')[['concept', 'cui','system']]

test=sg.loc[sg.semtype.isin(['T047', 'T033', 'T184', 'T046',
'DiseaseDisorderMention','SignSymptomMention',
'problem', 
'dsyn'
'fndg'
'sosy',
'patf'])].drop_duplicates(subset='cui')[['concept', 'cui','system']]

test.sort_values('system', ascending=False).drop_duplicates(subset='cui')
t =test.sort_values('system', ascending=False).drop_duplicates(subset='cui')

mrconso=pd.read_sql('select "CUI", "STR", "ISPREF" from public."MRCONSO"', con=engine)
concepts=mrconso.loc[mrconso.ISPREF=='Y'].drop_duplicates(subset='CUI')[['CUI', 'STR']]

u=t.merge(concepts, left_on='cui', right_on='CUI')
u['concept'] = np.where(u.concept.isnull(), u.STR, u.concept)
df.merge(u, left_on='cui', right_on='CUI')

df.merge(u, left_on='cui', right_on='CUI')['concept_y']
'''