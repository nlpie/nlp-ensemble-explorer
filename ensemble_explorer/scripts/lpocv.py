import pandas as pd
import re
import numpy as np
import glob
from sqlalchemy.engine import create_engine
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import LeavePOut
import random as rnd

"""
NB: 84 opt out
Unaccounted our db: 11
Unaccounted FV: 2
Total: 3107
"""
engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19')

# this is already done in nlp_features_tests.py
# date for training, development

# test data set

contact_date="2020-10-01"
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
existing = pd.read_sql(sql, params={"contact_date":contact_date}, con=engine)

# get patient metadata
db = pd.read_stata("/mnt/DataResearch/DataStageData/ForChris/Final_Clean_QI_Database_28_Nov_2020_with_cdc_umls_symptoms.DTA")
#cols_to_keep = ["race", "age_", "MDM_LINK_ID"]

cdc=[p for p in db.columns if 'cdc_' in p]

cols_to_keep = ['Inpatient', 'MDM_LINK_ID', 'true_died', 'age_cat', 'race']
db = db[cols_to_keep]

# features imputed in R w/ indicator
features=pd.read_csv("/mnt/DataResearch/DataStageData/ed_provider_notes/methods_paper/t.impute_1dec2020.csv")
features = features.merge(db, on="MDM_LINK_ID")

# TODO: merge with NLP features

# get list of comorbidities
# instead, match on Elixhauser?
comorb=[p for p in features.columns if 'COMORB' in p]
features['comorb']=features[comorb].values.tolist()

df=features.loc[features.Inpatient==1] #[cols_to_keep]
df['dependent_var']=df['true_died'].values.tolist()

# use for case/control pairwise match
df['match_on']=df[['male', 'age_cat', 'race']].values.tolist()

df = df.drop_duplicates(subset='MDM_LINK_ID')
df = df[['MDM_LINK_ID', 'race', 'age_cat', 'male', 'dependent_var', 'match_on', 'comorb', 'true_died']]
df = df.drop_duplicates(subset='MDM_LINK_ID')

independent=[] # list of dictionaries of independent variable  
target=[] # list of dictionaries for dependent variable
v = dict() # dictionary of features
# get list of dictionaries for independent and dependent variables
for t in df.to_dict(orient="records"):
    v[t['MDM_LINK_ID']] = {}
    print(t['MDM_LINK_ID'])
    v[t['MDM_LINK_ID']] = t
    independent.append(t)
    target.append(t['true_died'])

 
# X=np.array(independent)
# y=np.array(target)

#lpo = LeavePOut(2)
# get set of dead and alive
dead=list()
alive=list()

# generate test dead and alive sets of patients using lpo cv
# for train_index, test_index in lpo.split(X):
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    # #print("TRAIN:", train_index, len(train_index), "TEST:", test_index)
    # if list(X_test)[0]['true_died'] == 1.:
        # dead.append(list(X_test)[0]['MDM_LINK_ID'])
    # else:
        # alive.append(list(X_test)[0]['MDM_LINK_ID'])

for i in independent:
    if i['true_died'] == 1.:
        dead.append(i['MDM_LINK_ID'])
    else:
        alive.append(i['MDM_LINK_ID'])
        
print("done with getting dead/alive") 

# get random matching pair for training, etc.

# TODO: Add NLP features
already_matched=[] # for control/alive only use once
i=0
# iterate through randomly shuffled list of deaths

master_dead = dead.copy()
master_alive = alive.copy()
pair = {}
for d in list(set(rnd.sample(dead, len(dead)))):
    unpaired = True
    
    while unpaired:
        # random pick of live patient 
        a = rnd.choice(list(set(alive)))
        if v[d]['match_on']==v[a]['match_on'] and a not in already_matched:
            print(i, len(v), 'match!', d, v[d]['match_on'], a, v[a]['match_on'])
            already_matched.append(a)
            
            #print(len(v))
            unpaired = False
        
            pair[d]=a
            if i == 0:
                alive = list(filter((a).__ne__, alive))
                dead = list(filter((d).__ne__, dead))
                del v[d]
                del v[a]
            
    i += 1
            

#get features
ed_ = [p for p in features.columns if '48hrs' in p]
comorb=[p for p in features.columns if 'COMORB' in p]
home_meds=[p for p in features.columns if '3mo_prior' in p]
cdc=[p for p in features.columns if 'cdc_' in p]
race = [p for p in features.columns if 'race_' in p] 
demogrpahics = ['age_raw', 'male'] 
others = ['MDM_LINK_ID', 'true_died']
dummy = [p for p in features.columns if 'dummy' in p] 



# features selected via HPC
fs = ["COMORB_CAD",
"COMORB_Hx_VTE",
"COMORB_Hypocoag_state",
"COMORB_Any_HeartFail",
"COMORB_COPD",
"COMORB_ILD",
"COMORB_Xeno_Heart_valve",
"COMORB_Pulm_HTN",
"COMORB_Afib_flutter",
"COMORB_VTach_VFib",
"COMORB_Prior_Card_arrest",
"COMORB_CerebroVascDz",
"COMORB_Lung_Cancer",
"Home_BB_3mo_prior",
"Home_Rivaroxaban_3mo_prior",
"Home_Leflunomide_3mo_prior",
"Home_Inhal_Ipratropium_3mo_prior",
"Home_Glimepiride_3mo_prior",
"Home_Ezetimibe_3mo_prior",
"Home_Etodolac_3mo_prior",
"Home_Enoxaparin_3mo_prior",
"Home_Oral_Steroids_3mo_prior",
"Home_Clopidogrel_3mo_prior",
"Home_Azithromycin_3mo_prior",
"Home_loop_diuretic_3mo_prior",
"Home_Zinc_3mo_prior",
"Home_Multi_vitamin_3mo_prior",
"Home_Menaquinone_3mo_prior",
"Home_Memantine_3mo_prior",
"Home_Donepezil_3mo_prior",
"Home_Ascorbic_Acid_3mo_prior",
"Home_Statin_3mo_prior",
"age_raw",
"ED_IL6_48hrs",
"ED_IL1B_48hrs",
"ED_DDIMER_48hrs",
"ED_FERRITIN_48hrs",
"ED_K_48hrs",
"ED_TBILI_48hrs",
"ED_ABLUMIN_48hrs",
"ED_LACTATE_48hrs",
"ED_MCV_48hrs",
"ED_RDW_48hrs",
"ED_BUN_48hrs",
"ED_RR_max_48hrs",
"ED_SBP_max_48hrs",
"ED_SpO2_min_48hrs",
"ED_SBP_min_48hrs",
"ED_DBP_min_48hrs",
"CKD1.",
"ED_TRPONIN_48hrs.dummy",
"ED_INR_48hrs.dummy",
"ED_PROCAL_48hrs.dummy",
"ED_MG_48hrs.dummy",
"ED_BNP_48hrs.dummy",
"ED_LACTATE_48hrs.dummy",
"ED_PHOS_48hrs.dummy",
"ED_Xa_48hrs.dummy",
"ED_CORTISOL_48hrs.dummy"]

fs1 = ["COMORB_Hypocoag_state", 
"COMORB_Any_HeartFail", 
"COMORB_COPD", 
"COMORB_ILD",
"COMORB_Xeno_Heart_valve",
"COMORB_Pulm_HTN",
"COMORB_Prior_Card_arrest",
"COMORB_CerebroVascDz",
"Home_Leflunomide_3mo_prior",
"Home_Inhal_Ipratropium_3mo_prior",
"Home_Glimepiride_3mo_prior",
"Home_Etodolac_3mo_prior",
"Home_Aspirin_3mo_prior",
"Home_Menaquinone_3mo_prior",
"Home_Memantine_3mo_prior",
"age_raw",
"ED_IL6_48hrs",
"ED_DDIMER_48hrs",
"ED_K_48hrs",
"ED_ABLUMIN_48hrs",
"ED_PLT_48hrs",
"ED_LACTATE_48hrs",
"ED_CA_48hrs",
"ED_MCV_48hrs",
"ED_RDW_48hrs",
"ED_BUN_48hrs",
"ED_RR_max_48hrs",
"ED_SBP_max_48hrs",
"ED_SpO2_min_48hrs",
"ED_DBP_min_48hrs"]

# add in missing indicator variables
for f in fs:
    if 'dummy' not in f:
        if f + '.dummy' in fs:
            pass
        else:
            if f+'.dummy' in features:
                print(f)
                fs.append(f+'.dummy')

# ------------> TEST logistic regression

"""
ed_ = [p for p in features.columns if '48hrs' in p]
comorb=[p for p in features.columns if 'COMORB' in p]
home_meds=[p for p in features.columns if '3mo_prior' in p]
cdc=[p for p in features.columns if 'cdc_' in p]
race = [p for p in features.columns if 'race_' in p] 
demogrpahics = ['age_raw', 'male'] 
others = ['MDM_LINK_ID', 'true_died']
dummy = [p for p in features.columns if 'dummy' in p] 
"""

cols_to_keep = fs + cdc + others
#cols_to_keep = ed_ + comorb + home_meds + cdc + race + demogrpahics + dummy + others


f = features.loc[features.Inpatient==1][cols_to_keep]


# get empty
# test_X.columns[test_X.isna().any()].tolist()
# drop = ['ED_C3_48hrs', 'ED_UA_NITRITE_48hrs']
drop = []

train_X = f.loc[f.MDM_LINK_ID.isin(alive+dead)]
train_X =  train_X.drop(['true_died', 'MDM_LINK_ID'] + drop, axis = 1)
train_y = f.loc[f.MDM_LINK_ID.isin(alive+dead)]['true_died']
test_X = f.loc[~f.MDM_LINK_ID.isin(alive+dead)]
test_X =  test_X.drop(['true_died', 'MDM_LINK_ID'] + drop, axis = 1)
test_y = f.loc[~f.MDM_LINK_ID.isin(alive+dead)]['true_died']

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

scaler = StandardScaler()

scaled = scaler.fit_transform(train_X.values)
scaled_test = scaler.fit_transform(test_X.values)

from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
counter = Counter(train_y)
weights = {0:1.0, 1:7}

model = LogisticRegression(solver='lbfgs', class_weight='balanced')
model.fit(scaled, train_y)


lr_probs = model.predict_proba(scaled_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(scaled_test)
# calculate precision and recall for each threshold
lr_precision, lr_recall, _ = precision_recall_curve(test_y, lr_probs)
# calculate scores
lr_f1, lr_auc = f1_score(test_y, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(test_y[test_y==1]) / len(test_y)

AUC=[]
for k, v in pair.items():
    #print(k,v)
    
    test_X = f.loc[f.MDM_LINK_ID.isin([k, v])]
    test_X =  test_X.drop(['true_died', 'MDM_LINK_ID'] + drop, axis = 1)
    test_y = f.loc[f.MDM_LINK_ID.isin([k, v])]['true_died']
    
    #scaled = scaler.fit_transform(train_X.values)
    scaled_test = scaler.fit_transform(test_X.values)
    lr_probs = model.predict_proba(scaled_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(scaled_test)
    # calculate precision and recall for each threshold
    lr_precision, lr_recall, _ = precision_recall_curve(test_y, lr_probs)
    # calculate scores
    lr_f1, lr_auc = f1_score(test_y, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    #print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    AUC.append(lr_auc)
    no_skill = len(test_y[test_y==1]) / len(test_y)
