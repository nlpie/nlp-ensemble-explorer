import json
import jsonlines
import pandas as pd
     
#file_out = jsonlines.open("/Users/gms/Documents/manuscripts/fairview_covid/symptoms_jsonl.txt", 'a')

umls_symptoms = pd.DataFrame()

file_name = 'validate_symptoms.txt' # was master_symptom_list.txt
with open("/Users/gms/Desktop/" + file_name) as fp:
    Lines = fp.readlines()
    out = []
    print(Lines)
#    with jsonlines.open("/Users/gms/Documents/manuscripts/fairview_covid/symptoms_jsonl.txt", 'a') as writer:
#        for line in Lines:
#            print([line.strip()])
#            line = line.strip()
#            # write jsonl
#            symptom = line.strip()
#            s={"text": symptom}
#            writer.write(s)
#
#writer.close()

    for line in Lines:
        print([line.strip()])
        line = line.strip()

        # get mm annotation
        t = !echo $line | /Users/gms/development/nlp/engines/metamap/public_mm/bin/metamap18 --silent --JSONn -z -g --conj -i -u -y -r 900 -J fndg,sosy,dsyn,patf
        #print(t[1])

        d = json.loads(t[1])

        mapping = d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings']
        if len(mapping) > 0:
            cui=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateCUI']
            preferred=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidatePreferred']
            semtypes=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['SemTypes']
            score=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateScore']
            matched=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateMatched']

            out = {'cui': cui, 'preferred':  preferred, 'semtypes':  semtypes, 'score': score, 'matched': matched, 'symptom': line}
        else:
            out = {'cui': 'no match', 'symptom': line}
   
        frames = [umls_symptoms,  pd.DataFrame(out,  index=[0])]
        umls_symptoms = pd.concat(frames)

# get other sem types
umls_symptoms_other = pd.DataFrame()
for o in umls_symptoms.loc[umls_symptoms.cui=="no match"].symptom.tolist():
    print(o)
    t = !echo $o | /Users/gms/development/nlp/engines/metamap/public_mm/bin/metamap18 --silent --JSONn -z -g --conj -i -u -y -r 900
    d = json.loads(t[1])

    mapping = d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings']
    if len(mapping) > 0:
        cui=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateCUI']
        preferred=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidatePreferred']
        semtypes=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['SemTypes']
        score=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateScore']
        matched=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateMatched']
    
        out = {'cui': cui, 'preferred':  preferred, 'semtypes':  semtypes, 'score': score, 'matched': matched, 'symptom': o}
    else:
        out = {'cui': 'no match', 'symptom': o}
    
    frames = [umls_symptoms_other,  pd.DataFrame(out,  index=[0])]
    umls_symptoms_other = pd.concat(frames)
        
# get term from SList
'''
d = json.loads(t[1])
d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['PhraseText']
d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]

# dict_keys(['CandidateScore', 'CandidateCUI', 'CandidateMatched', 'CandidatePreferred', 'MatchedWords', 'SemTypes', 'MatchMaps', 'IsHead', 'IsOverMatch', 'Sources', 'ConceptPIs', 'Status', 'Negated'])

d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateCUI']
d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidatePreferred']
d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['SemTypes']
d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateScore']
'''
