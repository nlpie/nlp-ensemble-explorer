import os
import spacy
import pandas as pd
import time
from scispacy.umls_linking import UmlsEntityLinker
linker = UmlsEntityLinker(resolve_abbreviations=True)
from sqlalchemy.types import Integer, String, Float
from config import engine, input_folder



database_type = 'mysql+pymysql' # We use mysql+pymql as default
corpus = 'mipacq'
files = os.listdir(input_folder)
for model in ("en_core_sci_sm","en_core_sci_md","en_core_sci_lg",
              "en_ner_craft_md", "en_ner_jnlpba_md","en_ner_bc5cdr_md",
              "en_ner_bionlp13cg_md",):
    nlp = spacy.load(model)
    nlp.add_pipe(linker)
    os.chdir(input_folder)
    start_time = time.time()
    for i, file in enumerate(files):
        with open(file) as f:
            print(file)
            doc = nlp(f.read().replace('^', ' '))
            df = pd.DataFrame(
                [
                    [ent.end_char, ent.start_char, ent.text,
                    int(ent._.umls_ents[0][0][1:]), ent._.umls_ents[0][1]]
                    if len(ent._.umls_ents) > 0
                    else [ent.end_char, ent.start_char, ent.text,None, None]
                    for ent in doc.ents
                ],
                columns=['end', 'begin', 'string', 'cui', 'score']
            )
            df['file'] = file[:-4]
            df['corpus'] = 'mipacq'
            df['system'] = model
            df['type'] = None
            df['semantic_type'] = None
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

    time_per_file = (time.time() - start_time) / i
    print(f'time per file: {time_per_file}')



    #grouping by end, start, and file converts these columns into a multiindex,
    #which makes splitting by filename faster
