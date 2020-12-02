import pandas as pd
from pathlib import Path
import time, os

from sqlalchemy.engine import create_engine
from datetime import datetime

now = datetime.now()
dt = now.strftime('%Y%m%d')

engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19')
data_directory = '/mnt/DataResearch/DataStageData/ed_provider_notes/output/' 

# ---> Create disorders set for downstream usage !!!!!!!!

# TODO: find way to snarf up and concat all files
set_a = pd.read_csv(data_directory + 'ensembled_data/ensemble_1602080938.27128.csv')
set_b = pd.read_csv(data_directory + 'ensembled_data/ensemble_1603122042.536138.csv')
set_c = pd.read_csv(data_directory + 'ensembled_data/ensemble_1603552742.547878.csv')
set_d = pd.read_csv(data_directory + 'ensembled_data/ensemble_1604158456.683375.csv')
set_e = pd.read_csv(data_directory + 'ensembled_data/ensemble_1604766378.92923.csv')
set_f = pd.read_csv(data_directory + 'ensembled_data/ensemble_1606082178.965203.csv')

df = pd.concat([set_a, set_b, set_c, set_d, set_e, set_f])

# cohort 2
 
set_a = pd.read_csv(data_directory + 'cohort_2/24aug2020/ensemble_1605924301.195449.csv')
set_b = pd.read_csv(data_directory + 'cohort_2/24aug2020/ensemble_1605924356.595452.csv')
set_c = pd.read_csv(data_directory + 'cohort_2/24aug2020/ensemble_1605925199.585139.csv')
set_d = pd.read_csv(data_directory + 'cohort_2/24aug2020/ensemble_1605926133.778776.csv')

df = pd.concat([set_a, set_b, set_c, set_d, df])

set_a = pd.read_csv(data_directory + 'cohort_2/13nov2020/ensemble_1606061052.073672.csv')
set_b = pd.read_csv(data_directory + 'cohort_2/13nov2020/ensemble_1606061105.626346.csv')
set_c = pd.read_csv(data_directory + 'cohort_2/13nov2020/ensemble_1606061155.929239.csv')

df = pd.concat([set_a, set_b, set_c, df])

sql = """

SELECT "MDM_LINK_ID", "CONTACT_DATE", "NOTE_ID"::int, "NOTE_STATUS"
	FROM public.ed_provider_notes
	WHERE "NOTE_STATUS" not in ('Incomplete', 'Shared') 
    and (cohort = 1 or (cohort = 2 and pui = 1))

"""

notes = pd.read_sql(sql, engine)

notes = df.merge(notes, how='inner', left_on='case', right_on= 'NOTE_ID')

# Final_Clean_QI_Database_23_Nov_2020 PUI_VER2_Database_22_Nov_2020
db = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database_23_Nov_2020.dta")

# tested elsewhere
db = db[db['Covid_Positive_Date'].notna()]
#db = db[db['Covid_Result_Time'].notna()]

db['covid_positive_date']= pd.to_datetime(db.Covid_Positive_Date-(24*3600000)*3653, unit='ms')
#db['covid_result_time']= pd.to_datetime(db.Covid_Result_Time, unit='ms') - pd.Timedelta((24*36000)*3653, unit='ms') 

notes = notes.merge(db[['covid_positive_date', 'MDM_LINK_ID']], on="MDM_LINK_ID")
#notes = notes.merge(db[['covid_result_time','mdm_link_id']], left_on="MDM_LINK_ID", right_on="mdm_link_id")

notes['contact_date'] = pd.to_datetime(notes['CONTACT_DATE'])
notes['date_diff']=(notes.covid_positive_date - notes.contact_date).astype('timedelta64[D]').abs()

#notes['date_diff']=(notes.covid_result_time - notes.contact_date).astype('timedelta64[D]')

# criteria as per CJT TODO: new criterion
#notes = notes.loc[(notes.date_diff>=-14) & (notes.date_diff<=0)]

# get min date_diff
# get min date_diff
test = notes.loc[notes.groupby('MDM_LINK_ID').date_diff.idxmin()]
test = test[['MDM_LINK_ID', 'date_diff']]
out = test.merge(notes, on=['MDM_LINK_ID', 'date_diff'])
# --->

#out = notes

out = out[[ 'cui', 'polarity', 'MDM_LINK_ID', 'CONTACT_DATE', 'NOTE_ID', 'NOTE_STATUS']] #.sort_values(by=['MDM_LINK_ID', 'CONTACT_DATE', 'NOTE_ID'])

out = out.drop_duplicates(['cui', 'NOTE_ID', 'polarity'])

sql = """
SELECT "CUI" as cui, "STR" as concept
	FROM public."MRCONSO"
    where "ISPREF" = 'Y';
"""
cuis = pd.read_sql(sql, engine)
cuis = cuis.drop_duplicates(subset='cui')

disorders = out.merge(cuis, how='inner',on='cui').drop_duplicates(['cui','polarity','NOTE_ID'])#.sort_values(by=['MDM_LINK_ID', 'CONTACT_DATE', 'NOTE_ID', 'concept'])

#print(disorders)
now = datetime.now()
timestamp = datetime.timestamp(now)
file = 'patient_disorders_'+str(timestamp)+'.csv'
disorders.to_csv(data_directory + file, index=False)

# -----> write to STATA
timestr = time.strftime("%Y%m%d-%H%M%S")

df = pd.read_csv('/mnt/DataResearch/DataStageData/ed_provider_notes/output/' + file)
data_folder = Path("/mnt/DataResearch/DataStageData/ed_provider_notes/output/")

pos = df[df.polarity == 1].copy()
neg = df[df.polarity == -1].copy()

pos = pos.drop_duplicates()
neg = neg.drop_duplicates()

both = pos.merge(neg, how = 'left', on=['cui', 'MDM_LINK_ID'], indicator=True)
both = both[both._merge == 'both']

both.drop(['polarity_y', 'CONTACT_DATE_y', 'NOTE_ID_y', 'NOTE_STATUS_y', 'concept_y'], axis=1, inplace=True)

both = both.rename(columns={'polarity_x': 'polarity', 'CONTACT_DATE_x': 'CONTACT_DATE', 'NOTE_ID_x': 'NOTE_ID', 
'NOTE_STATUS_x': 'NOTE_STATUS', 'concept_x': 'concept'})

# @note_id or MDM_LINK_ID level?
p_only = pos.merge(neg, how = 'left', on=['cui', 'MDM_LINK_ID'], indicator=True)
p_only = p_only[p_only._merge == 'left_only']

p_only.drop(['polarity_y', 'CONTACT_DATE_y', 'NOTE_ID_y', 'NOTE_STATUS_y', 'concept_y'], axis=1, inplace=True)

p_only = p_only.rename(columns={'polarity_x': 'polarity', 'CONTACT_DATE_x': 'CONTACT_DATE', 'NOTE_ID_x': 'NOTE_ID', 
'NOTE_STATUS_x': 'NOTE_STATUS', 'concept_x': 'concept'})
  
n_only = neg.merge(pos, how = 'left', on=['cui', 'MDM_LINK_ID'], indicator=True)
n_only = n_only[n_only._merge == 'left_only']

n_only.drop(['polarity_y', 'CONTACT_DATE_y', 'NOTE_ID_y', 'NOTE_STATUS_y', 'concept_y'], axis=1, inplace=True)

n_only = n_only.rename(columns={'polarity_x': 'polarity', 'CONTACT_DATE_x': 'CONTACT_DATE', 'NOTE_ID_x': 'NOTE_ID', 
'NOTE_STATUS_x': 'NOTE_STATUS', 'concept_x': 'concept'})

# if in both, then positive overides negative
both['polarity'] = 1

combined = pd.concat([n_only, p_only, both])

patients = list(set(combined.MDM_LINK_ID.tolist()))
cuis = list(set(combined.cui.tolist()))

test = dict()
for p in patients:
    test[p] = {}
    df1 = combined[combined.MDM_LINK_ID == p].copy()
    for row in df1.itertuples():
        if row.cui in cuis:
            test[p][row.cui] = row.polarity
             
test1 = dict()
for p in patients:
    test1[p] = {}
    df1 = combined[combined.MDM_LINK_ID == p].copy()
    for row in df1.itertuples():
        if row.cui in cuis:
            test1[p][row.cui] = row.concept.lower()

# for c in cuis:
    # for p in patients:
        # if c in test[p]:
            # print(p,test[p][c])

# empty df?                 
#out = pd.DataFrame(columns=cuis)

        
        
# this might do it with a proper transformation
# https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
# https://pandas.pydata.org/docs/user_guide/reshaping.html
a = pd.concat({
             k: pd.DataFrame.from_dict(v, 'index') for k, v in test.items()
         },
         axis=0).unstack()
    
#  https://stackoverflow.com/questions/44023770/pandas-get-rid-of-multiindex       
a.columns = a.columns.droplevel(0)
a.to_stata(str(data_folder) + '/test_cui.dta',version=117)

'''
b = pd.concat({
             k: pd.DataFrame.from_dict(v, 'index') for k, v in test1.items()
         },
         axis=0).unstack()
    
b.columns = b.columns.droplevel(0) 
b.to_stata(str(data_folder) + '/pat_cui_with_description' + str(timestr) +'.dta',version=117)
'''

# neg

test = dict()
for p in patients:
     test[p] = {}
     df1 = n_only[n_only.MDM_LINK_ID == p].copy()
     for row in df1.itertuples():
         if row.cui in cuis:
             test[p][row.cui] = row.polarity  

'''    
b = pd.concat({
      k: pd.DataFrame.from_dict(v, 'index') for k, v in test.items()
        },
    axis=0).unstack()
    
b.columns = b.columns.droplevel(0) 
b.to_stata(str(data_folder) + '/pat_cui_negative_mention_' + str(timestr) +'.dta',version=117)
'''

# pos

test = dict()
for p in patients:
     test[p] = {}
     df1 = p_only[p_only.MDM_LINK_ID == p].copy()
     for row in df1.itertuples():
         if row.cui in cuis:
             test[p][row.cui] = row.polarity  


'''   
b = pd.concat({
     k: pd.DataFrame.from_dict(v, 'index') for k, v in test.items()
     },
    axis=0).unstack()
    
b.columns = b.columns.droplevel(0) 
b.to_stata(str(data_folder) + '/pat_cui_positive_mention' + str(timestr) +'.dta',version=117)
'''

#len(b[b.apply(lambda row: ~row.astype(str).str.contains('diarrhea').any(), axis=1)])


# Get set for creation of features


patients = list(set(combined.MDM_LINK_ID.tolist()))
cuis = list(set(combined.cui.tolist()))

# get all positive and negative mentions
test = dict()
for p in patients:
    test[p] = {}
    df1 = combined[combined.MDM_LINK_ID == p].copy()
    for row in df1.itertuples():
        if row.cui in cuis:
            test[p][row.cui] = row.polarity

# add in no mention
for p in patients:
    for cui in cuis:
        if cui in test[p]:
            pass
        else:
            #print('no cui!')
            test[p][cui] = 0
            
pids = []
frames = []

# wide format

b = pd.concat({
                k: pd.DataFrame.from_dict(v, 'index') for k, v in test.items()
              }, 
              axis=0).unstack()
b.columns = b.columns.get_level_values(1)

# sql = """
# select "MDM_LINK_ID", max("CONTACT_DATE") as encounter_date 
# from ed_provider_notes 
# where cohort = 1
# group by "MDM_LINK_ID"
# """

# max_encounter = pd.read_sql(sql, engine)
# max_encounter['encounter_date'] = pd.to_datetime(max_encounter['encounter_date'])
# max_encounter = max_encounter.set_index(["MDM_LINK_ID"])
out['encounter_date'] = pd.to_datetime(out['CONTACT_DATE'])
out = out[['encounter_date', 'MDM_LINK_ID']]
id_ = out.drop_duplicates(subset='MDM_LINK_ID')
id_ = id_.set_index(["MDM_LINK_ID"])
b=b.join(id_)

fname='nlp_umls_concept_features_'+str(dt)                
b.to_csv('/mnt/DataResearch/DataStageData/ed_provider_notes/output/'+fname+'.csv', index=True)
b.to_stata('/mnt/DataResearch/DataStageData/ed_provider_notes/output/nlp_umls_concept_features_'+str(dt)+'.dta',version=117)

dta_path = '/mnt/DataResearch/DataStageData/ed_provider_notes/output/'
dta_fname = dta_path + fname + '.dta'
!cp $dta_fname '/mnt/DataResearch/DataStageData/nlp_umls_concept_features.dta'
os.chdir('/mnt/DataResearch/DataStageData/ed_provider_notes/output/')
os.rename(dta_fname, '/mnt/DataResearch/DataStageData/archive/ed_provider_notes/umls_features/' + fname + '.dta')
os.rename(fname+'.csv', '/mnt/DataResearch/DataStageData/archive/ed_provider_notes/umls_features/' + fname+'.csv')

"""
# long format              
for pid, d in test.items():
    pids.append(pid)
    frames.append(pd.DataFrame.from_dict(d, orient='index'))


out = pd.concat(frames, keys=pids)

out = out.reset_index().rename(columns={'level_0': 'MDM_LINK_ID', 'level_1': 'cui', 0: 'polarity'})

out.to_csv('/mnt/DataResearch/DataStageData/ed_provider_notes/output/test_for_features.csv', index=False)
out.to_stata('/mnt/DataResearch/DataStageData/ed_provider_notes/output/test_for_features.dta',version=117)
"""