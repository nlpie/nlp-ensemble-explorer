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
existing = pd.read_sql(sql, params={"contact_date":contact_date}, con=engine)

# get patient metadata
db = pd.read_stata("/mnt/DataResearch/DataStageData/analytical_tables/Final_Clean_QI_Database_31_Oct_2020.DTA")
cols_to_keep = ["race", "age_cat", "MDM_LINK_ID"]
db = db[cols_to_keep]

# features imputed in R w/ indicator
features=pd.read_csv("/mnt/DataResearch/DataStageData/ed_provider_notes/methods_paper/analysis/imputed_features_with_indicator.csv")
features = features.merge(db, on="MDM_LINK_ID")

# TODO: merge with NLP features

# get list of comorbidities
# instead, match on Elixhauser?
comorb=[p for p in features.columns if 'COMORB' in p]
features['comorb']=features[comorb].values.tolist()

df=features #[cols_to_keep]
df['dependent_var']=df['true_died'].values.tolist()

# use for case/control pairwise match
df['match_on']=df[['male', 'age_cat', 'race']].values.tolist()

df = df.drop_duplicates(subset='MDM_LINK_ID')
df = df[['MDM_LINK_ID', 'race', 'age_cat', 'male', 'dependent_var', 'match_on', 'comorb', 'true_died']]
df = df.drop_duplicates(subset='MDM_LINK_ID')

independent=[] # list of dictionaries of indeependent variable  
target=[] # list of dictionaries for dependent variable
v = dict() # dictionary of features
# get list of dictionaries for independent and dependent variables
for t in df.to_dict(orient="records"):
    v[t['MDM_LINK_ID']] = {}
    print(t['MDM_LINK_ID'])
    v[t['MDM_LINK_ID']] = t
    independent.append(t)
    target.append(t['true_died'])

 
X=np.array(independent)
y=np.array(target)

lpo = LeavePOut(2)
# get set of dead and alive
dead=list()
alive=list()

# generate test dead and alive sets of patients using lpo cv
for train_index, test_index in lpo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if list(X_test)[0]['true_died'] == 1.:
        dead.append(list(X_test)[0]['MDM_LINK_ID'])
    else:
        alive.append(list(X_test)[0]['MDM_LINK_ID'])

print("done with getting dead/alive") 
# get random matching pair for training, etc.
# TODO: Add NLP features
already_matched=[] # for control/alive only use once
i=0
for d in list(set(dead)):
     unpaired = True
     
     while unpaired:
        a = rnd.choice(list(set(alive)))
        if v[d]['match_on']==v[a]['match_on'] and a not in already_matched:
            print(i, 'match!', d, v[d]['match_on'], a, v[a]['match_on'])
            already_matched.append(a)
            unpaired = False
     i += 1
            


    


