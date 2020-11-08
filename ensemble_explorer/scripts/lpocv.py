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

# kosher for research
#patients=patients.loc[patients.research_op_out.isna()]

# output from nlp_features_tests.py
features=pd.read_csv("/mnt/DataResearch/DataStageData/ed_provider_notes/methods_paper/analysis/discrete_features_01sep2020.csv")

features = features.merge(db, on="MDM_LINK_ID")

#cols_to_keep=['male', 'true_died', 'age_cat', 'MDM_LINK_ID', 'race'] 
# get list of comorbidities


# instead, maatch on Elixhauser?
features['comorb']=features[comorb].values.tolist()

#cols_to_keep = cols_to_keep + ['Vent', 'ICU', 'Inpatient', 'ED_Revisit', 'comorb'] 

p=features #[cols_to_keep]
# impute values
values = {'male': 0, 'race': 'Declined'}
df=p.fillna(value=values)

#df=p.merge(existing,how="right", on='MDM_LINK_ID', indicator=True)
#df = df.loc[df._merge=="both"]

df['dependent_var']=df['true_died'].values.tolist()
#df['match_on']=df[['race', 'male', 'age_cat']].values.tolist()

# use for case/control pair match
df['match_on']=df[['male', 'age_cat', 'race']].values.tolist()

test = df.drop_duplicates(subset='MDM_LINK_ID')
testing=test[['MDM_LINK_ID', 'race', 'age_cat', 'male', 'dependent_var', 'match_on', 'comorb', 'true_died']]

testing = testing.drop_duplicates(subset='MDM_LINK_ID')
l=[]
target=[]
v = dict()
# get list of dictionaries for independent and dependent variables
for t in testing.to_dict(orient="records"):
    v[t['MDM_LINK_ID']] = {}
    print(t['MDM_LINK_ID'])
    v[t['MDM_LINK_ID']] = t
    l.append(t)
    target.append(t['true_died'])

  
X=np.array(l)
y=np.array(target)

lpo = LeavePOut(2)
# print(lpo)
# for train_index, test_index in lpo.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    
    # if list(X_test)[0]['true_died'] != list(X_test)[1]['true_died']:
        # if list(X_test)[0]['match_on'] == list(X_test)[1]['match_on']:
            # print (list(X_test)[0]['PAT_ID'], list(X_test)[1]['PAT_ID'])

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

    # if list(X_test)[0]['true_died'] != list(X_test)[1]['true_died']:
        # if list(X_test)[0]['match_on'] == list(X_test)[1]['match_on']:
            # pass #print (list(X_test)[0]['PAT_ID'], list(X_test)[1]['PAT_ID'])

print("done with getting dead/alive") 
# get random matching pair
already_matched=[]
i=0
for d in list(set(dead)):
     unpaired = True
     
     while unpaired:
        a = rnd.choice(list(set(alive)))
        if v[d]['match_on']==v[a]['match_on'] and a not in already_matched:
            print(i, 'match!', v[d]['match_on'], v[a]['match_on'])
            already_matched.append(a)
            unpaired = False
     i += 1
            


    


