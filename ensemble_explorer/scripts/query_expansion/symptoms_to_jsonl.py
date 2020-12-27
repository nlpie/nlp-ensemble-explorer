import json
import jsonlines
import pandas as pd
     
file_name = 'master_symptoms_lexicon.txt' # was master_symptom_list.txt

with open("/Users/gms/Documents/manuscripts/fairview_covid/" + file_name) as fp:
    Lines = fp.readlines()
    out = []
    print(Lines)
    with jsonlines.open("/Users/gms/Documents/manuscripts/fairview_covid/master_symptoms_gazetteer.txt", 'w') as writer:
        for line in Lines:
            print([line.strip()])
            line = line.strip()
            # write jsonl
            symptom = line.strip()
            s={"text": symptom}
            writer.write(s)

writer.close()

