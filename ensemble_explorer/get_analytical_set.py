import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
from datetime import datetime
from pathlib import Path
import time, io

data_folder = Path("/mnt/DataResearch/DataStageData/ed_provider_notes/output/")
start = time.time()

# https://towardsdatascience.com/optimizing-pandas-read-sql-for-postgres-f31cd7f707ab
def read_sql_inmem_uncompressed(query, db_engine):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
       query=query, head="HEADER"
    )
    conn = db_engine.raw_connection()
    cur = conn.cursor() 
    store = io.StringIO()
    cur.copy_expert(copy_sql, store)
    store.seek(0)
    df = pd.read_csv(store)
    return df
    
def get_systems_set(sql, engine):

    df = pd.read_sql(sql, engine)
    
    return df
    
def get_sem_types(engine):

    # semantic types for filtering
    sql = """SELECT st.tui as tui, abbreviation, clamp_name, ctakes_name 
            FROM semantic_groups sg join semantic_types st on sg.tui = st.tui 
            where group_name = 'Disorders';"""
      
    return pd.read_sql(sql, engine)
    
    
def get_data(engine, cohort):

    st = get_sem_types(engine)
    
    mask = [st for st in list(set(st.tui.tolist()))]
    
    ### ------> b9
    sql = """
    select distinct u.begin, u.end, null as concept, u.cui, u.confidence::float as score, u.tui as semtype, u.note_id, u.type, u.system, -1 as polarity 
    from (select "NOTE_ID"  from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID" limit 4000) as md join
        "bio_biomedicus_UmlsConcept" u on md."NOTE_ID" = u.note_id join
        "bio_biomedicus_Negated" n on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id
	where n.begin is not null and n.end is not null and n.note_id is not null
    
    union distinct
    
    select distinct u.begin, u.end, null as concept, u.cui, u.confidence::float as score, u.tui as semtype, u.note_id, u.type, u.system, 1 as polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join
       "bio_biomedicus_UmlsConcept" u on md."NOTE_ID" = u.note_id left join
       "bio_biomedicus_Negated" n on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id 
    where n.begin is null and n.end is null and n.note_id is null
    """

    #b9 = pd.read_sql(sql, params={"notes":tuple(notes)}, con=engine)
    b9 = read_sql_inmem_uncompressed(sql, engine)
    b9 = b9[b9.semtype.isin(mask)]

    print('b9!')

    ### ------> clamp
    
    mask = [st.split(',')  for st in list(set(st.clamp_name.tolist()))][0]
    
    sql = """
    select distinct u.begin, u.end, u.concept, u.cui, u.concept_prob::float as score, "u"."semanticTag" as semtype, u.note_id, u.type, u.system, -1 as polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join 
        "cla_edu_ClampNameEntityUIMA" u on md."NOTE_ID" = u.note_id join 
        "cla_edu_ClampRelationUIMA" r on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id 
	where u.assertion = 'absent' and r.begin is not null and r.end is not null and r.note_id is not null
     
    union distinct

    select distinct u.begin, u.end, u.concept, u.cui, u.concept_prob::float as score, "u"."semanticTag" as semtype, u.note_id, u.type, u.system, 1 as polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join 
        "cla_edu_ClampNameEntityUIMA" u on md."NOTE_ID" = u.note_id left join 
        "cla_edu_ClampRelationUIMA" r on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id 
	where (u.assertion = 'present' or u.assertion is null) and r.begin is null and r.end is null and r.note_id is null
    """
    
    clamp = read_sql_inmem_uncompressed(sql, engine)
    clamp = clamp[clamp.semtype.isin(mask)]
    
    print('clamp!')
    ### ------> ctakes
    
    mask = [st.split(',')  for st in list(set(st.ctakes_name.tolist()))][0]

    sql = """
    
    select distinct begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join 
        "cta_org_DiseaseDisorderMention" u on md."NOTE_ID" = u.note_id 

	union distinct 

    select distinct begin, "end", concept, cui, null as score, split_part(type, '.', 7) as semtype, note_id, 'ctakes_mentions' as type,  system, polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join 
        "cta_org_SignSymptomMention" u on md."NOTE_ID" = u.note_id
    """
    
    ctakes = read_sql_inmem_uncompressed(sql, engine)
    
    print('ctakes!')
    ### ------> mm
    
    mask = [st for st in list(set(st.abbreviation.tolist()))]

    sql = """
    select distinct c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score::int) as score, "c"."semanticTypes" as semtype, c.note_id, c.type, c.system, -1 as polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join  
        "met_org_Candidate" c on md."NOTE_ID" = c.note_id join 
        "met_org_Negation" n on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id 
	where n.begin is not null and n.end is not null and n.note_id is not null

    union distinct

    select distinct c.begin, c.end, c.preferred as concept, c.cui, ABS(c.score::int) as score, "c"."semanticTypes" as semtype, c.note_id, c.type, c.system, 1 as polarity 
    from (select "NOTE_ID" from ed_provider_notes where date_added >= '2020-11-14' and cohort = 1 order by "NOTE_ID"   limit 4000) as md join  
        "met_org_Candidate" c on md."NOTE_ID" = c.note_id left join 
        "met_org_Negation" n on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id 
    where n.begin is null and n.end is null and n.note_id is null
    """
    mm = read_sql_inmem_uncompressed(sql, engine)
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
   
    del out['length']
    # randomly reindex to keep random row when dropping duplicates: https://gist.github.com/cadrev/6b91985a1660f26c2742
    out.reset_index(inplace=True)
    out = out.reindex(np.random.permutation(out.index))
    
    return out.drop_duplicates(subset=['begin', 'end', 'note_id', 'polarity'])

def update_opt_out(engine):
    
    patients = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database_27_Nov_2020.dta")

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
    
    # set which cohort to extract
    #cohort = 2 
    
    # set opt out status
    update_opt_out(engine)
    
    #GENERATE analytical table
    analytical_cui = pd.DataFrame()

    # TODO get cases method
    
    sql = """
            select distinct "NOTE_ID" 
              from ed_provider_notes
              where date_added>='2020-11-14' 
              order by "NOTE_ID";
         """
    notes = pd.read_sql(sql, engine)
    
    cases = set(notes['NOTE_ID'].tolist())
    
    b9, clamp, ctakes, mm = get_data(engine, cohort)
    
    i = 0
    for case in cases:
        print(case, i)
        i += 1
    
        ### ------>  b9
        test = b9[b9['note_id'] == int(case)].copy()
        print('b9 umls', len(test))

        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False)  

        ### ------>  clamp
        test = clamp[clamp['note_id'] == int(case)].copy()
        print('clamp', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  ctakes
        test = ctakes[ctakes['note_id'] == int(case)].copy()
        print('ctakes mentions', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 

        ### ------>  mm
        test = mm[mm['note_id'] == int(case)].copy()
        print('mm candidate', len(test))
        if len(test) > 0:
            frames = [ analytical_cui, disambiguate(test) ]
            analytical_cui = pd.concat(frames, ignore_index=True, sort=False) 
        
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    
    return analytical_cui
    
    
if __name__ == '__main__':
    #%prun main()
    analytical_cui = main()
    elapsed = (time.time() - start)
    print('fini!', 'time:', elapsed)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    #analytical_cui.ro_sql("analytical_cui" + timestamp, engine)
    file = 'analytical_fairview_cui_filtered_by_semtype_test_' + str(timestamp) +'.csv'
    
    analytical_cui.to_csv(data_folder / file)