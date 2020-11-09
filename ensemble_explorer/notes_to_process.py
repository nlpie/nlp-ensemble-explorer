#!/home/gsilver1/anaconda3/bin/python
# coding: utf-8

"""
Workflow:
1. Create notes: 
/mnt/DataResearch/DataStageData/gsilver1/development/ensemble-explorerensemble_explorer/
notes_to_process.py

2. Run through pipeline:
/mnt/DataResearch/DataStageData/gsilver1/development/nlp-adapt-kube/argo-k8s/cpm
uiima-engines-wf.yaml

3. Parse annotated objects -> postgresql:
/mnt/DataResearch/DataStageData/gsilver1/development/ensemble-explorerensemble_explorer/
parse_uima.py

4. Extract features from postgresql:
/mnt/DataResearch/DataStageData/gsilver1/development/ensemble-explorerensemble_explorer/
get_analytical_set.py

5. Create extraction set using ensemble:
/mnt/DataResearch/DataStageData/gsilver1/development/ensemble-explorerensemble_explorer/
extract_ensemble.py

6. Create dta set for import intto database:
/mnt/DataResearch/DataStageData/gsilver1/development/ensemble-explorerensemble_explorer/
analytical_cui_to_feature_set.py


"""
"""
Summary of files processed:

1	"2020-08-28"	4610 -> done
2	"2020-08-28"	85084 -> done
1	"2020-09-23"	679 -> done
1	"2020-09-24"	96 -> done
2	"2020-09-24"	17350
1   "2020-09-30"    223 -> done
2   "2020-09-30"    5154
1   "2020-10-03"    61 -> done
2   "2020-10-03"    1389  
1   "2020-10-17"    696
2   "2020-10-17"    10173 
1   "2020-10-23"    4800
2   "2020-10-23"    382


note: total above off by n-files from physical files for cohort 1

"""

import pandas as pd
import re
import numpy as np
import glob
from sqlalchemy.engine import create_engine
from pathlib import Path
from datetime import datetime

engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19') 
now=datetime.now().date().strftime('%Y%m%d')
#now="20201016"
data_folder = "/mnt/DataResearch/DataStageData/archive/source/"
data_out = Path('/mnt/DataResearch/DataStageData/ed_provider_notes/in_process/data_in')

# Processs new notes

# delete manifest
!rm '/mnt/DataResearch/DataStageData/ed_provider_notes/in_process/nlp_manifest.txt'
 
# 1. get notes for both cohorts
for fname in glob.glob(data_folder + "CV_PATIENT_ED_PROVIDER_NOTES_ALL*.txt"):
    if now in fname:
        #print(fname)
        notes = pd.read_csv(fname, dtype=str, engine='python', sep="~\|~")
        
# 2. get patients to determine which cohort they belong to
for fname in glob.glob(data_folder + "CV_PATIENTS_ALL*.txt"):
    if now in fname:
        print(fname)
        patients = pd.read_csv(fname, dtype=str, engine='python', sep="~\|~")

# 3. set cohort
df = patients[['COHORT_1', 'COHORT_2', 'MDM_LINK_ID', "PAT_ID",  "NOTE_ID",  "NOTE_STATUS"]].merge(notes, how='inner', on='MDM_LINK_ID').drop_duplicates()
df['cohort'] = np.where(df['COHORT_1'] == "1", 1, 2)

# 4. compare to what has been processed
sql = """
SELECT "PAT_ID",  "NOTE_ID",  "NOTE_STATUS"
	FROM public.ed_provider_notes  
"""
existing = pd.read_sql(sql, engine)

new_data = df.merge(existing, how="left", on=["PAT_ID",  "NOTE_ID",  "NOTE_STATUS"], indicator=True)
new_data = new_data[(new_data['_merge']=='left_only') & (new_data["NOTE_STATUS"].isin(['Signed','Addendum']))]

# 5. add new rows to ed_provider_notes table
new_data[['SOURCE_SYSTEM', 'PAT_ID', 'MDM_LINK_ID', 'PAT_ENC_CSN_ID',
       'CONTACT_DATE', 'ENC_TYPE', 'NOTE_ID', 'NOTE_TYPE', 'NOTE_STATUS',
       'PROV_NAME', 'PROV_TYPE', 'UPD_AUT_LOCAL_DTTM', 'cohort']].to_sql('ed_provider_notes', engine, index=False, if_exists='append')

# 6. set archive folder with date
arch_folder = data_folder + 'archive/ed_provider_notes/processed_' + str(now)  
arch_file = arch_folder + '/data_in.zip'

# make archive directory and move zip file
!mkdir -p $arch_folder
!zip -r $arch_file /mnt/DataResearch/DataStageData/ed_provider_notes/data_in/ 

# clean up data_in directory
data_in = data_folder + '/ed_provider_notes/in_process/data_in/'
#get_ipython().system('unzip -l /mnt/DataResearch/DataStageData/ed_provider_notes/{file}')
#!rm /mnt/DataResearch/DataStageData/ed_provider_notes/data_in/*.txt
!for i in $data_in*.txt; do rm "$i"; done

# get new file set
columns = ['COHORT_1', 'COHORT_2', 'MDM_LINK_ID', 'SOURCE_SYSTEM', 'PAT_ID',
           'PAT_ENC_CSN_ID', 'CONTACT_DATE', 'ENC_TYPE', 'NOTE_ID', 'NOTE_TYPE',
           'NOTE_STATUS', 'PROV_NAME', 'PROV_TYPE', 'UPD_AUT_LOCAL_DTTM', 'NOTE_TEXT']

files = new_data[(new_data.cohort==1) & (new_data['_merge']=='left_only') & (new_data["NOTE_STATUS"].isin(['Signed','Addendum']))]
files = files[columns]

files = files.apply(lambda x: x.str.replace('[^\x00-\x7F]',''))

pat_ids = set(files['PAT_ID'].to_list())
files = files.sort_values(by=['PAT_ID', 'NOTE_ID', 'CONTACT_DATE'])

# 7. write new notes to txt file for processing
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
        
# 8. clean up system annotation zip files
!mv /mnt/DataResearch/DataStageData/ed_provider_notes/*.zip $arch_folder

folders = ['biomedicus_out', 'clamp_out', 'ctakes_out', 'metamap']
for folder in folders:
    !rm -rf '/mnt/DataResearch/DataStageData/ed_provider_notes/'$folder










