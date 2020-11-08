import pandas as pd
from sqlalchemy.engine import create_engine
import matplotlib.pyplot as plt
engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19')

# https://blog.softhints.com/pandas-count-percentage-value-column/
def get_count(df):
   counts = df.value_counts(dropna=False)
   percent = df.value_counts(normalize=True, dropna=False)
   
   return counts, percent
   
# get encounter data

contact_date="2020-09-01"
sql = """
select a."CONTACT_DATE", a."MDM_LINK_ID", "PAT_ID", "NOTE_ID"
from ed_provider_notes a join 
(select max("CONTACT_DATE") as "CONTACT_DATE", "MDM_LINK_ID"
    from ed_provider_notes  
    where opt_out is NULL and cohort=1
    group by  "MDM_LINK_ID") b on a."CONTACT_DATE" <= b."CONTACT_DATE" 
	and a."MDM_LINK_ID" = b."MDM_LINK_ID"
where opt_out is NULL and cohort=1 and a."CONTACT_DATE" < %(contact_date)s
"""
pat = pd.read_sql(sql, params={"contact_date":contact_date}, con=engine)

# gazetteer matches 

file_g = 'gazetteer_out_31oct2021.txt'
df = pd.read_json("/mnt/DataResearch/DataStageData/ed_provider_notes/methods_paper/analysis/misc/" + file_g, orient='columns', lines=True)
#df = pd.read_json("Y:\DataStageData\ed_provider_notes\methods_paper\analysis\misc\gazetteer_test.out", orient='columns', lines=True)

df['pat_id'], df['note_id'] = df['note'].str.split('_', 1).str
df['note_id'] = df['note_id'].str.replace('.txt','')
df = df.drop_duplicates(subset=["symptom", "pat_id"])

db = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database__3_Nov_2020.dta")

# tested elsewhere
#db = db[db['Covid_Positive_Date'].notna()]

# look only at those who had 1 ED-visit (TODO: handle revisits, as per M. Usher) 
#db = db.loc[db.ED_Visit_Total==1]
db['covid_positive_date']= pd.to_datetime(db.Covid_Positive_Date-(24*3600000)*3653, unit='ms')

cutoff = db[['MDM_LINK_ID', 'covid_positive_date']]

new = pat.merge(df, left_on=["PAT_ID", "NOTE_ID"], right_on=["pat_id", "note_id"])

ttt = new.merge(cutoff, on='MDM_LINK_ID')
ttt['contact_date'] = pd.to_datetime(ttt['CONTACT_DATE'])
ttt['date_diff']=(ttt.covid_positive_date - ttt.contact_date).astype('timedelta64[D]')
# encounter date -5 days to + 21 day wrt test date
ttt = ttt.loc[((ttt.date_diff>=-5)&(ttt.date_diff<=21))|(ttt.date_diff.isnull())]

new = ttt

symptoms = set(new.symptom.tolist())
patients = set(new.MDM_LINK_ID.tolist())
new = new.drop_duplicates(subset=["symptom", "MDM_LINK_ID"])

test = dict()
for p in patients:
    test[p] = {}
    df1 = new[new.MDM_LINK_ID == p].copy()
    for row in df1.itertuples():
        if row.symptom in symptoms:
            test[p][row.symptom] = 1
            
b = pd.concat({
                k: pd.DataFrame.from_dict(v, 'index') for k, v in test.items()
              }, 
              axis=0).unstack()

b.columns = b.columns.get_level_values(1)            
b = b.fillna(0)
b = b.reset_index()
b = b.rename(columns={"index": "MDM_LINK_ID"})

'''
db = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database__3_Nov_2020.dta")

# look only at those who had 1 ED-visit (TODO: handle revisits, as per M. Usher) 
#db = db.loc[db.ED_Visit_Total==1]
db['covid_positive_date']= pd.to_datetime(db.Covid_Positive_Date-(24*3600000)*3653, unit='ms')
'''

comorb=[p for p in db.columns if 'COMORB' in p]
home_meds=[p for p in db.columns if '3mo_prior' in p]

# labs at time of ED visit
d0_labs=[p for p in db.columns if 'D0' in p]
#d1_labs=[p for p in db.columns if 'D1' in p]
labs = d0_labs # + d1_labs
vitals=[p for p in db.columns if '24' in p] + ["BMI"]

# TARGET: died at home, in hospital, time to death
death = ["true_died", "Mortality", "dead", "days_if_do_cox_PH"]

db = pd.get_dummies(db, columns=["race"], prefix="race")
db = pd.get_dummies(db, columns=["age_cat"], prefix="age_cat")
race = [p for p in db.columns if 'race_' in p] 
age_cat = [p for p in db.columns if 'age_cat_' in p]

cols_to_keep = ["MDM_LINK_ID", "male"] 

for l in ['UA_Nitrite_D0', 'UA_Ketones_D0', 'TEG_MA_D0', 'TEG_R_D0']:
    labs.remove(l)
    
for m in ['Home_antivirals_3mo_prior']:
    home_meds.remove(m)

dates=['covid_positive_date']   
cols_to_keep += comorb + home_meds + labs + vitals + race + death + age_cat + dates
db = db[cols_to_keep]

test = db.merge(b, on="MDM_LINK_ID")
test = test[cols_to_keep]
#no imputation: assume alive till dead!
test['true_died'] = test['true_died'].fillna(0)

# test = test.drop(['UA_Nitrite_D0', 'UA_Ketones_D0', 'TEG_MA_D0', 'TEG_R_D0'], axis=1)
    
data = db.merge(b, on="MDM_LINK_ID")

# feature engineering:
data['CKD1+']=np.where((data.COMORB_any_CKD == 1)&(data.COMORB_stage1_CKD==1), 1, 0)
data['CKD2+']=np.where((data.COMORB_any_CKD == 1)&(data.COMORB_stage2_CKD==1), 1, 0)
data['CKD3+']=np.where((data.COMORB_any_CKD == 1)&(data.COMORB_stage3_CKD==1), 1, 0)
data['CKD4+']=np.where((data.COMORB_any_CKD == 1)&(data.COMORB_stage4_CKD==1), 1, 0)
data['CKD5+']=np.where((data.COMORB_any_CKD == 1)&(data.COMORB_stage5_CKD==1), 1, 0)

# ---> summary:
print("\n HOME MEDS:")
print("===================================================")
for c in sorted(home_meds):
    counts, percent = get_count(data[c])
    print(c, ':')
    print("-----------------------------------")
    print(pd.DataFrame({'counts': counts, 'per': percent}))
    print("-----------------------------------")
    print("\n")
   
   
print("DEMOGRAPHICS:")
print("===================================================")
for c in ["male"] + race + age_cat:
    counts, percent = get_count(data[c])
    print(c, ':')
    print("-----------------------------------")
    print(pd.DataFrame({'counts': counts, 'per': percent}))
    print("-----------------------------------")
    print("\n")
   
print("COMORBIDITIES:")
print("===================================================")
for c in sorted(comorb):
    counts, percent = get_count(data[c])
    print(c, ':')
    print("-----------------------------------")
    print(pd.DataFrame({'counts': counts, 'per': percent}))
    print("-----------------------------------")
    print("\n")
   
print("DEAD:")
print("===================================================")
for c in ["Mortality", "true_died"]: 
    counts, percent = get_count(data[c])
    if c == "Mortality":
        c="In hospital"
    else:
        c="At home"
    print(c, ':')
    print("-----------------------------------")
    print(pd.DataFrame({'counts': counts, 'per': percent}))
    print("-----------------------------------")
    print("\n")
   
print("MISSING LABS:")
print("===================================================")
# missing labs 
for c in sorted(labs):
    print(c, ':') 
    print("-----------------------------------")
    #if data[c].notnull().sum()/len(data) >= 0.05:
    print('n total:', len(data)) 
    print('n missing:', data[c].isna().sum()) 
    print('percent missing:', data[c].isna().sum()/len(data))
    print('n present:', data[c].notnull().sum())
    print('prevalance:', data[c].notnull().sum()/len(data))
    print("-----------------------------------")
    print("\n")

# missing labs 
'''
for c in sorted(d1_labs):
    print(c, ':') 
    print("-----------------------------------")
    if data[c].notnull().sum()/len(data) >= 0.05:
        print('n total:', len(data)) 
        print('n missing:', data[c].isna().sum()) 
        print('percent missing:', data[c].isna().sum()/len(data))
        print('n present:', data[c].notnull().sum())
        print('prevalance:', data[c].notnull().sum()/len(data))
        print("-----------------------------------")
        print("\n")
'''
        
print("MISSING VITALS:")
print("===================================================")
# missing vitals
for c in sorted(vitals):
    print(c, ':') 
    print("-----------------------------------")
    print('n total:', len(data)) 
    print('n missing:', data[c].isna().sum()) 
    print('percent missing:', data[c].isna().sum()/len(data))
    print('n present:', data[c].notnull().sum())
    print('prevalance:', data[c].notnull().sum()/len(data))
    print("-----------------------------------")
    print("\n")
