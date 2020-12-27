import gensim
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer
import json


def get_terms():
    with open("/Users/gms/Documents/manuscripts/fairview_covid/symptoms.txt") as fp:
        
        Lines = fp.readlines()
        out = []

        out = []
        for line in Lines:
            line = line.strip()
            line = line.replace(' ', '_')
            line = line.replace('/', '_').lower()
            out.append(line)
        
        return out 


def get_preferred():

    df = pd.read_csv("/Users/gms/Desktop/symptoms_to_process.csv") # "Documents/manuscripts/fairview_covid/final_symptoms_list.csv")
    
    df['preferred'] = df['preferred'].str.replace(', CTCAE 3.0', '')
    df['preferred'] = df['preferred'].str.replace(', CTCAE', '')

    df['preferred'] = df['preferred'].str.replace(r" \(.*\)","")
    df['preferred'] = df['preferred'].str.replace(', ','')
    df['preferred'] = df['preferred'].str.replace('3.0', '')
    df['preferred'] = df['preferred'].str.replace(' -','')
    df['preferred'] = df['preferred'].str.replace(' ', '_')
    df['preferred'] = df['preferred'].str.replace('__', '')
    
    df['preferred'] = df['preferred'].str.replace('-','_')

    df['preferred'] = df['preferred'].str.lower()

    df['cui_concept'] = list(zip(df.cui, df.preferred, df.symptom))

    terms = df['cui_concept'].tolist()
    no_match = None
    
    return terms, no_match


def get_concept(term):

        t = !echo $term | /Users/gms/development/nlp/engines/metamap/public_mm/bin/metamap18 --silent --JSONn -g --conj -i -z -u -r 900 -J fndg,sosy,dsyn,patf

        d = json.loads(t[1])

        mapping = d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings']
        if len(mapping) > 0:
            cui=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateCUI']
            preferred=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidatePreferred']
            semtypes=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['SemTypes']
            score=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateScore']
            matched=d['AllDocuments'][0]['Document']['Utterances'][0]['Phrases'][0]['Mappings'][0]['MappingCandidates'][0]['CandidateMatched']

            out = {'mappedCui': cui, 'mappedConcept':  preferred, 'mappedSemanticTypes':  semtypes, 'mmScore': score, 'mappedMatchedOn': matched, 'lookupSymptom': term}

        return out


def semantic_phrases():
    """
    method for generating semantic lists 
    NB: need to run remotely, due to w2p model containing PHI
    """

    file = '/Volumes/Data2/PakhomovS-U01/word2vec/fairview-vectors-c5_2010-2014-phrase1.bin'
    phrase = KeyedVectors.load_word2vec_format(datapath(file), binary=True)
    writer = pd.ExcelWriter('/Users/gms/Desktop/semantic_similarity_covid_symptoms.xlsx')
    
    def get_sys_similarities(terms):
        """
        get semantically similar terms
        """
        
        out = pd.DataFrame()
        for term in terms:

            d = {}
            print(term)
            if term[1] in phrase.vocab: #and ("abdominal_bloating" in term[1] or "ageusia" in term[1]):

                test = phrase.most_similar(positive=[term[1]], topn = 100)

                # get symptom as top element 
                d['semSimLookupTerm'] = term[1]
                d['cosineDistance'] = 1
                d['UmlsSymptomConcept'] = term[1]
                d['UmlsSymtomCui'] = term[0]
                d['symptom'] = term[2]
                d['mappedConcept'] = term[1]
                d['mappedCui'] = term[0]

                frames = [out,  pd.DataFrame(d,  index=[0])]
                out = pd.concat(frames)
                
                # rule: 
                # if there are tuples with cosine distance >= .75
                if len([t for t in test if t[1] >= 0.75]) > 0:
                    test = [t for t in test if t[1] >= 0.75]
                    
                    for t in test:
                        d = get_concept(t[0].replace('_',' '))
                        if not d:
                            pass
                        else:
                            d['semSimLookupTerm'] = t[0]
                            d['cosineDistance'] = t[1]
                            d['UmlsSymptomConcept'] = term[1]
                            d['UmlsSymtomCui'] = term[0]
                            d['symptom'] = term[2]

                            frames = [out,  pd.DataFrame(d,  index=[0])]
                            out = pd.concat(frames)

                else: # get top 2 similarity terms w/ mapped concepts
                    i = 0
                    while i < 2:
                        for t in test:
                            if i == 2:
                                break

                            else:
                                d = get_concept(t[0].replace('_',' '))
                                if not d:
                                    pass
                                else:
                                    d['semSimLookupTerm'] = t[0]
                                    d['cosineDistance'] = t[1]
                                    d['UmlsSymptomConcept'] = term[1]
                                    d['UmlsSymtomCui'] = term[0]
                                    d['symptom'] = term[2]

                                    frames = [out,  pd.DataFrame(d,  index=[0])]
                                    out = pd.concat(frames)

                                    i +=1
            
            elif term[1] not in phrase.vocab:
                d['semSimLookupTerm'] = 'No similararity terms'
                d['UmlsSymptomConcept'] = term[1]
                d['UmlsSymtomCui'] = term[0]
                d['symptom'] = term[2]
                d['mappedConcept'] = term[1]
                d['mappedCui'] = term[0]

                frames = [out,  pd.DataFrame(d,  index=[0])]
                out = pd.concat(frames)

        return out
    
    def get_sys_similarities_nomatch(no_match):
        """
        get semantically similar terms
        """
        
        out = pd.DataFrame()
        for term in no_match:

            d = {}
            print(term)
            if term in phrase.vocab: #and ("abdominal_bloating" in term[1] or "ageusia" in term[1]):

                test = phrase.most_similar(positive=[term], topn = 100)
                
                # rule: 
                # if there are tuples with cosine distance >= .75
                if len([t for t in test if t[1] >= 0.75]) > 0:
                    test = [t for t in test if t[1] >= 0.75]
                    
                    for t in test:
                        d = get_concept(t[0].replace('_',' '))
                        if not d:
                            pass
                        else:
                            d['semSimLookupTerm'] = t[0]
                            d['cosineDistance'] = t[1]
                            d['symptom'] = term

                            frames = [out,  pd.DataFrame(d,  index=[0])]
                            out = pd.concat(frames)

                else: # get top 2 similarity terms w/ mapped concepts
                    i = 0
                    while i < 2:
                        for t in test:
                            if i == 2:
                                break

                            else:
                                d = get_concept(t[0].replace('_',' '))
                                if not d:
                                    pass
                                else:
                                    d['semSimLookupTerm'] = t[0]
                                    d['cosineDistance'] = t[1]
                                    d['symptom'] = term

                                    frames = [out,  pd.DataFrame(d,  index=[0])]
                                    out = pd.concat(frames)

                                    i +=1

        return out 

    
    terms, no_match = get_preferred()

    # parent symptom has UMLS mapping
    mapped_terms = get_sys_similarities(terms)

    # case where parent symptom has no UMLS mapping
    #unmapped_terms = get_sys_similarities_nomatch(no_match)
    
    unmapped_terms = None

    return mapped_terms, unmapped_terms

mapped_terms, unmapped_terms = semantic_phrases()
