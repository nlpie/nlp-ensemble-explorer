from config import output_folder, engine
from sqlalchemy.types import Integer, String, Float
import pandas as pd
import os
os.chdir(output_folder)
df = pd.read_csv('mipacq_ann_4_ben.csv')
def convert_cui(x):
    try:
        return int(x[1:])
    except ValueError:
        return None
df['cui'] = df['value'].apply(convert_cui)
df['begin'] = df['start']
df['corpus'] = 'mipacq'
df['system'] = 'reference'
df['type'] = None
df['string'] = df['text']
df['semantic_type'] = None
df = df.drop(['text', 'mentionSlot', 'start', 'class', 'value', 'classtype'], axis=1)
df.to_sql('spacy', if_exists='append', con=engine, index=False,
    dtype={
    'cui': Integer(),
    'begin': Integer(),
    'end': Integer(),
    'corpus': String(),
    'system': String(),
    'type': String(),
    'score': Float(),
    'semantic_type': String(),
    'string': String(),
    })
