#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sqlalchemy.engine import create_engine
from pathlib import Path
from datetime import datetime

engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19')
data_directory = '/mnt/DataResearch/DataStageData/ed_provider_notes/output/' 

# ---> Create disorders set for downstream usage !!!!!!!!

df = pd.read_csv(data_directory + 'ensemble_1602080938.27128.csv')

sql = """

SELECT "MDM_LINK_ID", "CONTACT_DATE", "NOTE_ID"::int, "NOTE_STATUS"
	FROM public.ed_provider_notes
	WHERE "NOTE_STATUS" not in ('Incomplete', 'Shared') 
    and cohort = 1 and opt_out is NULL

"""

notes = pd.read_sql(sql, engine)

out = df.merge(notes, how='inner', left_on='case', right_on= 'NOTE_ID')


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


print(disorders)
now = datetime.now()
timestamp = datetime.timestamp(now)
disorders.to_csv(data_directory + 'patient_disorders_'+str(timestamp)+'.csv', index=False)



disorders.columns


#------------------> not needed


# ---> generate cui frequencies:

cui =  pd.read_csv(data_directory + 'analytical_fairview_cui_filtered_by_semtype_1597485061.111386.csv')
cui = cui[['cui','concept']]
cui=cui.drop_duplicates()

df = pd.read_csv(data_directory + 'test_new_no_dedup.csv')
test = df.merge(cui, how="left", on='cui')

test[['concept','cui']].groupby(['concept']).size().to_csv(data_directory+'concept_frequencies_new.csv')

# generate concept list for CUIs
df = pd.read_csv(data_directory + 'analytical_fairview_cui.csv')

df = df[['cui','concept']].dropna()
df = df.applymap(lambda x: x.strip() if type(x)==str else x)

#df.drop_duplicates()
#df.to_sql("umls_cui_concept", engine)

test = pd.read_csv(data_directory + 'test_dedup.csv'):

test_concepts = df.merge(test, how = 'inner', on='cui')

test_concepts['pat_id'] = test_concepts['case'].str.split(pat = '_', n=1, expand=True)
test_concepts

test_concepts = test_concepts[['cui', 'concept', 'pat_id']].drop_duplicates(['cui', 'concept', 'pat_id'])
test_concepts.to_csv('/mnt/DataResearch/DataStageData/note_test/output/patient_signs_and_symptoms.csv')

test_concepts.groupby('concept').count()

file = 'CV_PATIENT_ED_PROVIDER_NOTES_TESTING.txt'
data_directory = '/mnt/DataResearch/DataStageData/note_test/' 


df = pd.read_csv(data_directory + file, dtype=str, engine='python', sep="~\|~")


len(df)
df.columns
print(df['NOTE_TEXT'][0:1])


df.groupby('UPD_AUT_LOCAL_DTTM').count()


data_folder = Path("/mnt/DataResearch/DataStageData/")
file = 'CV_PATIENT_ED_PROVIDER_NOTES.txt'


df = pd.read_csv(data_folder / file, dtype=str, engine='python', sep="~\|~")


df.columns

df = df[['SOURCE_SYSTEM', 'PAT_ID', 'MDM_LINK_ID', 'PAT_ENC_CSN_ID',
       'CONTACT_DATE', 'ENC_TYPE', 'NOTE_ID', 'NOTE_TYPE', 'NOTE_STATUS',
       'PROV_NAME', 'PROV_TYPE', 'UPD_AUT_LOCAL_DTTM']]



df.groupby('PAT_ID').count()



df.to_sql('ed_provider_notes', engine)



data_directory = '/mnt/DataResearch/DataStageData/ed_provider_notes/output/'



# merge ensembled output and metadata
df1 = pd.read_csv(data_directory + 'ensemble_set_1.csv', dtype={'case':int})
df2 = pd.read_csv(data_directory + 'ensemble_1592874399.601703.csv', dtype={'case':int})
frames = [df1, df2]
df = pd.concat(frames)

# processs new notes
# load new files to detemine which have changed

data_folder = Path("/mnt/DataResearch/DataStageData/")
file = 'CV_PATIENT_ED_PROVIDER_NOTES.txt'
df = pd.read_csv(data_folder / file, dtype=str, engine='python', sep="~\|~")
len(df)



# # new cases:
# sql = """
# SELECT tmp."PAT_ID",  tmp."NOTE_ID", tmp. "NOTE_STATUS"
# 	FROM public.ed_provider_16jun2020 as tmp left join public.ed_provider_notes ed 
# 	on tmp."PAT_ID" = ed."PAT_ID" and tmp."NOTE_ID" = ed."NOTE_ID" and tmp."NOTE_STATUS" = ed."NOTE_STATUS"
# 	where ed."PAT_ID" is null and ed."NOTE_ID" is null and ed."NOTE_STATUS" is null
# 	and tmp."NOTE_STATUS" != 'Incomplete';
# """



#compare to what has been processed
sql = """
SELECT "PAT_ID",  "NOTE_ID",  "NOTE_STATUS"
	FROM public.ed_provider_notes 
"""
notes = pd.read_sql(sql, engine)
len(notes)


new_data = df.merge(notes, how="left", on=["PAT_ID",  "NOTE_ID",  "NOTE_STATUS"], indicator=True)


len(new_data[new_data['_merge']=='left_only'])


new_data[new_data['_merge']=='left_only'].groupby('NOTE_STATUS').count()


new_data = new_data[(new_data['_merge']=='left_only') & (new_data["NOTE_STATUS"].isin(['Signed','Addendum']))]


# add new rows to ed_provider_notes
new_data.columns
new_data[['SOURCE_SYSTEM', 'PAT_ID', 'MDM_LINK_ID', 'PAT_ENC_CSN_ID',
       'CONTACT_DATE', 'ENC_TYPE', 'NOTE_ID', 'NOTE_TYPE', 'NOTE_STATUS',
       'PROV_NAME', 'PROV_TYPE', 'UPD_AUT_LOCAL_DTTM']].to_sql('ed_provider_notes', engine, index=False, if_exists='append')

len(new_data)


from datetime import datetime
now = datetime.now()
timestamp = datetime.timestamp(now)


# archive data
#!zip /mnt/DataResearch/DataStageData/ed_provider_notes/initial_set.zip /mnt/DataResearch/DataStageData/ed_provider_notes/data_in/*.txt 
#!ls -lah /mnt/DataResearch/DataStageData/ed_provider_notes/initial_set.zip

get_ipython().system('unzip -l /mnt/DataResearch/DataStageData/ed_provider_notes/{file}')
#!unzip -l /mnt/DataResearch/DataStageData/ed_provider_notes/processed_notes.zip
#!rm /mnt/DataResearch/DataStageData/ed_provider_notes/data_in/*.txt


files = new_data[(new_data['_merge']=='left_only') & (new_data["NOTE_STATUS"].isin(['Signed','Addendum']))]
files
files = files.apply(lambda x: x.str.replace('[^\x00-\x7F]',''))

pat_ids = set(files['PAT_ID'].to_list())
files = files.sort_values(by=['PAT_ID', 'NOTE_ID', 'CONTACT_DATE'])

# write new notes to txt file for processing
from pathlib import Path
data_out = Path('/mnt/DataResearch/DataStageData/ed_provider_notes/data_in')


import re
for p in pat_ids:
    #print(p)
    note_ids = files['NOTE_ID'][files['PAT_ID']==p].copy()  
    for n in set(note_ids.tolist()):
        #print(n)
        fname = p + '_' + n + '.txt'
        print(fname)    
        #lines = df['LINE'][df['NOTE_ID']==n].copy()
        #/lines = lines.sort_values()  
        f = open(data_out / fname, "a")  # append mode
        #for l in lines.tolist():
            #print(l)
        #txt = df['NOTE_TEXT'][(df['PAT_ID']==p) & (df['NOTE_ID']==n) & (df['LINE']==l)].copy()
        txt = files['NOTE_TEXT'][(files['PAT_ID']==p) & (files['NOTE_ID']==n)].copy()
            #fn.write()
            #print(txt.values[0]) 
            #print(re.sub(' +', ' ', txt.values[0]))
        f.write(re.sub(' +', ' ', str(txt.values[0])))
            #f.write(txt.values[0])
        f.close()



qdata = '/mnt/DataResearch/DataStageData/ed_provider_notes/'
q = pd.read_csv(qdata + 'qumls_out.csv')
q.to_sql('qumls', engine, if_exists='append', index=False)

# --> read MRCONSO

columns=['cui','lat','ts','lui','stt','sui','ispref','aui','saui','scui','sdui','sab','tty','code','str','srl','supress','cvf']
fname = 'MRCONSO.RRF'
data_folder = Path("/mnt/DataResearch/DataStageData/ed_provider_notes/UMLS/")
test = pd.read_csv(data_folder / fname, dtype=str, sep='|',names=columns, index_col=False)
test = test[test.lat == 'ENG']

df = test[['cui', 'lat', 'str','ispref']].copy()

t = cui.loc[pd.isnull(cui.concept)]
# case for null concept in patient disorder set: TODO regenerate disorder pattient set with these
t[['cui']].merge(test.loc[test.ispref=='Y'], how='inner', on='cui')

# check mapping 
u = cui.loc[~pd.isnull(cui.concept)]
u[['cui','concept']].merge(test.loc[test.ispref=='Y'], how='inner', left_on='concept', right_on='str')[['cui_x', 'cui_y', 'concept']]









