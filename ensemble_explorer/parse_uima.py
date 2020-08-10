# annotation class for UIMA systems
class AnnotationSystems(object):
    """
    CAS XMI Annotations of interest
    """
    
    def __init__(self):
        """ 
        annotation base types
        """
        
        self.biomedicus_dir = "biomedicus_out/"
        self.biomedicus_types = ["biomedicus.v2.UmlsConcept",
                                 "biomedicus.v2.Negated"]#,
                                 #"biomedicus.v2.Acronym",
                                 #"biomedicus.v2.TemporalPhrase"]
        
        
        self.clamp_dir = "clamp_out/"
        #self.clamp_dir = "Annotated_XMI/"
        self.clamp_types = ["edu.uth.clamp.nlp.typesystem.ClampNameEntityUIMA",
                            "edu.uth.clamp.nlp.typesystem.ClampRelationUIMA"]
        
        
        self.ctakes_dir = "ctakes_out/"
        self.ctakes_types = ["org.apache.ctakes.typesystem.type.textsem.DiseaseDisorderMention",
                             "org.apache.ctakes.typesystem.type.textsem.MedicationMention",
                             "org.apache.ctakes.typesystem.type.textsem.ProcedureMention",
                             "org.apache.ctakes.typesystem.type.textsem.SignSymptomMention",
                             "org.apache.ctakes.typesystem.type.textsem.AnatomicalSiteMention"]#,
                             #"org.apache.ctakes.typesystem.type.textsem.DateAnnotation",
                             #"org.apache.ctakes.typesystem.type.textsem.MeasurementAnnotation"]
       
        self.metamap_dir = "metamap_out/"
        self.metamap_types = ["org.metamap.uima.ts.Candidate",
                              #"org.metamap.uima.ts.CuiConcept",
                              "org.metamap.uima.ts.Negation"]
                
       
    def get_system_type(self, system):
        """
        return system types
        """
        
        if system == "biomedicus":
            view = "Analysis"
            
        else:
            view = "_InitialView"

        if system == 'biomedicus':
            types = self.biomedicus_types
            output = self.biomedicus_dir

        elif system == 'clamp':
            types = self.clamp_types
            output = self.clamp_dir

        elif system == 'ctakes':
            types = self.ctakes_types
            output = self.ctakes_dir

        elif system == 'metamap':
            types = self.metamap_types
            output = self.metamap_dir
            
        return types, view, output
    
annSys = AnnotationSystems()

# extract attributes from cas Annotation object
def get_attribs(v):
    attribs = []
    for sentence in v:
        #print(sentence)
        for s in sentence.__dir__():
            if '__' not in s:
                if s not in attribs:
                    #print(s)
                    attribs.append(s)
                else:
                    break

    return attribs


def init_cassis(system, typesystem):
   
    #tic=timeit.default_timer() 

    print(system)

    # types for metamap

    if system == 'metamap':
        t = typesystem.create_type(name='org.apache.uima.examples.SourceDocumentInformation', supertypeName='uima.tcas.Annotation')
        typesystem.add_feature(t, name='uri', rangeTypeName='uima.cas.String')
        typesystem.add_feature(t, name="offsetInSource", rangeTypeName="uima.cas.Integer")
        typesystem.add_feature(t, name="documentSize", rangeTypeName="uima.cas.Integer")
        typesystem.add_feature(t, name="lastSegment", rangeTypeName="uima.cas.Integer")

    # features for ctakes

    if system == 'ctakes':
        t = typesystem.get_type('org.apache.ctakes.typesystem.type.structured.Metadata')
        typesystem.add_feature(t, name='patientIdentifier', rangeTypeName='uima.cas.String')
        
        t = typesystem.get_type('org.apache.ctakes.typesystem.type.textsem.DiseaseDisorderMention')
        typesystem.add_feature(t, name='id', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='ontologyConceptArr', rangeTypeName='uima.cas.FSArray', elementType='org.apache.ctakes.typesystem.type.refsem.UmlsConcept')
        typesystem.add_feature(t, name='subject', rangeTypeName='uima.cas.String')
        #typesystem.add_feature(t, name='typeID', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='discoveryTechnique', rangeTypeName='uima.cas.Integer')
        #typesystem.add_feature(t, name='confidence', rangeTypeName='uima.cas.Double')
        typesystem.add_feature(t, name='polarity', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='uncertainty', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='conditional', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='generic', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='historyOf', rangeTypeName='uima.cas.Integer')

        t = typesystem.get_type('org.apache.ctakes.typesystem.type.textsem.SignSymptomMention')
        typesystem.add_feature(t, name='id', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='ontologyConceptArr', rangeTypeName='uima.cas.FSArray', elementType='org.apache.ctakes.typesystem.type.refsem.UmlsConcept')
        typesystem.add_feature(t, name='subject', rangeTypeName='uima.cas.String')
        #typesystem.add_feature(t, name='typeID', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='discoveryTechnique', rangeTypeName='uima.cas.Integer')
        #typesystem.add_feature(t, name='confidence', rangeTypeName='uima.cas.Double')
        typesystem.add_feature(t, name='polarity', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='uncertainty', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='conditional', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='generic', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='historyOf', rangeTypeName='uima.cas.Integer')

        t = typesystem.get_type('org.apache.ctakes.typesystem.type.textsem.MedicationMention')
        typesystem.add_feature(t, name='id', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='ontologyConceptArr', rangeTypeName='uima.cas.FSArray', elementType='org.apache.ctakes.typesystem.type.refsem.UmlsConcept')
        typesystem.add_feature(t, name='subject', rangeTypeName='uima.cas.String')
        #typesystem.add_feature(t, name='typeID', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='discoveryTechnique', rangeTypeName='uima.cas.Integer')
        #typesystem.add_feature(t, name='confidence', rangeTypeName='uima.cas.Double')
        typesystem.add_feature(t, name='polarity', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='uncertainty', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='conditional', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='generic', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='historyOf', rangeTypeName='uima.cas.Integer')

        t = typesystem.get_type('org.apache.ctakes.typesystem.type.textsem.ProcedureMention')
        typesystem.add_feature(t, name='id', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='subject', rangeTypeName='uima.cas.String')
        #typesystem.add_feature(t, name='typeID', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='discoveryTechnique', rangeTypeName='uima.cas.Integer')
        #typesystem.add_feature(t, name='confidence', rangeTypeName='uima.cas.Double')
        typesystem.add_feature(t, name='polarity', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='uncertainty', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='conditional', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='generic', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='historyOf', rangeTypeName='uima.cas.Integer')

        t = typesystem.get_type('org.apache.ctakes.typesystem.type.textsem.AnatomicalSiteMention')
        typesystem.add_feature(t, name='id', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='ontologyConceptArr', rangeTypeName='uima.cas.FSArray', elementType='org.apache.ctakes.typesystem.type.refsem.UmlsConcept')
        typesystem.add_feature(t, name='subject', rangeTypeName='uima.cas.String')
        #typesystem.add_feature(t, name='typeID', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='discoveryTechnique', rangeTypeName='uima.cas.Integer')
        #typesystem.add_feature(t, name='confidence', rangeTypeName='uima.cas.Double')
        typesystem.add_feature(t, name='polarity', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='uncertainty', rangeTypeName='uima.cas.Integer')
        typesystem.add_feature(t, name='conditional', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='generic', rangeTypeName='uima.cas.Boolean')
        typesystem.add_feature(t, name='historyOf', rangeTypeName='uima.cas.Integer')

# sofa -> db
def write_sofa(u):
    d = {}
    d["note_id"] = str(u)
    d["sofa"] = view.sofa_string
    d["corpus"] = corpus

    # does it exist?
    if engine.dialect.has_table(engine, "sofas"):
        sql = text("SELECT * FROM test.sofas WHERE note_id = :e1")
        resp = engine.execute(sql, e1=u).fetchall()
    else:
        resp = []

    if len(resp) == 0:            
        #pd.DataFrame(d, index=[0]).to_sql("sofas", engine, if_exists="append")  
        pd.DataFrame(d, index=[0]).to_sql("sofas", engine, if_exists="append")  
        

def get_dict(keys, sentence, d):
    for i in range(len(keys)):
        key = keys[i]
        val = sentence.__getattribute__(keys[i])
        d[key] = val
    return d

def add_keys(df, system, t, u, corpus, fname):

    pat_id = u.split('_')[0]
    note_id = u.split('_')[1]
    
    df["system"] = system
    df["type"] = t
    df["note_id"] = note_id
    df["pat_id"] = pat_id
    df["corpus"] = corpus
    df["filename"] = fname
    
    return df

def append_to_df(df, d):
    import pandas as pd
    frames = [ df, pd.DataFrame(d, index=[0]) ]
    df = pd.concat(frames, ignore_index=True)
    
    return df

#%%time
# parse system annotations
def main():
    import os, glob
    import pandas as pd
    import json
    from sqlalchemy.engine import create_engine
    from sqlalchemy.sql import text
    from cassis import load_typesystem, load_cas_from_xmi

    # connection string
    engine = create_engine('postgresql+psycopg2://gsilver1:nej123@d0pconcourse001/covid-19')
    systems = ["clamp", "metamap", "ctakes", "biomedicus"]
    
    corpora = ["fairview"]
    parse_to_sql = True
    
    if parse_to_sql:
        
        for corpus in corpora:
            print("CORPUS:", corpus)

            for system in systems:

                print("SYSTEM:", system)

                types, view_, output = annSys.get_system_type(system)
                
                dir_test = '/mnt/DataResearch/gsilver1/development/ensemble-explorer/system_annotations/typesystems/' + system + '/'

                with open(dir_test + 'TypeSystem.xml', 'rb') as f:
                    typesystem = load_typesystem(f)
                
                init_cassis(system, typesystem)
                
                # parse directory
                directory_to_parse = '/mnt/DataResearch/DataStageData/ed_provider_notes/' +  system + '_out/' 
               
                print(directory_to_parse)

                for fname in glob.glob(directory_to_parse + '/*.xmi'):
                #for fname in glob.glob(directory_to_parse + '/527982345.txt.xmi'):

                    file = os.path.basename(fname)
                    u = file.split('.')[0]
                    
                    print(u)

                    # load cas
                    with open(directory_to_parse + file, 'rb') as f:
                        cas = load_cas_from_xmi(f, typesystem=typesystem)

                    # load view
                    view = cas.get_view(view_)
                    
                    # write sofa object here:
                    #write_sofa(u)

                    for t in types:
                        print("TYPE:", t)
                       
                        attribs = get_attribs(view.select(t))
                        x = t.split('.')
                        table_name = system[0:3] + '_' + x[0] + '_' + x[len(x)-1]

                        # Annotation object -> dataframe
                        def get_df(v, attribs):
                            df = pd.DataFrame()
                            d = dict() 

                            # only parse if type exists in file
                            if view.select(t):
                                for sentence in view.select(t):
                                    
                                    dd = dict()
                                    if t == 'org.metamap.uima.ts.Negation':
                                        keys = ['cuiConcepts', 'ncSpans', 'negTrigger', 'negType']
                                        d = get_dict(keys, sentence, d)

                                        if len(d['cuiConcepts']) == len(d['ncSpans']):
                                             for x in range(len(d['cuiConcepts'])):
                                                dd['cui'] = d['cuiConcepts'][x].negExCui 
                                                dd['concept'] = d['cuiConcepts'][x].negExConcept 
                                                dd['begin'] = d['ncSpans'][x].begin 
                                                dd['end'] = d['ncSpans'][x].end
                                                dd['negTrigger'] = d['negTrigger']
                                                dd['negType'] = d['negType']
                                                dd['cuiConcepts'] = ' '.join(map(str,d['cuiConcepts']))
                                                dd['ncSpans'] = ' '.join(map(str,d['ncSpans']))
                                            
                                                df = append_to_df(df, dd)
                                                df = add_keys(df, system, t, u, corpus, fname)
                                            
                                        elif len(d['cuiConcepts']) > len(d['ncSpans']) and len(d['ncSpans']) == 1:
                                            for x in range(len(d['cuiConcepts'])):
                                                dd['cui'] = d['cuiConcepts'][x].negExCui 
                                                dd['concept'] = d['cuiConcepts'][x].negExConcept
                                                dd['begin'] = d['ncSpans'][0].begin
                                                dd['end'] = d['ncSpans'][0].end
                                                dd['negTrigger'] = d['negTrigger']
                                                dd['negType'] = d['negType']
                                                dd['cuiConcepts'] = ' '.join(map(str,d['cuiConcepts']))
                                                dd['ncSpans'] = ' '.join(map(str,d['ncSpans']))
                                                
                                                df = append_to_df(df, dd)
                                                df = add_keys(df, system, t, u, corpus, fname)
                                            
                                                
                                        elif len(d['cuiConcepts']) < len(d['ncSpans']) and len(d['cuiConcepts']) == 1:
                                            for x in range(len(d['ncSpans'])):
                                                dd['cui'] = d['cuiConcepts'][0].negExCui 
                                                dd['concept'] = d['cuiConcepts'][0].negExConcept
                                                dd['begin'] = d['ncSpans'][x].begin
                                                dd['end'] = d['ncSpans'][x].end
                                                dd['negTrigger'] = d['negTrigger']
                                                dd['negType'] = d['negType']
                                                dd['cuiConcepts'] = ' '.join(map(str,d['cuiConcepts']))
                                                dd['ncSpans'] = ' '.join(map(str,d['ncSpans']))
                                                
                                                df = append_to_df(df, dd)
                                                df = add_keys(df, system, t, u, corpus, fname)
                                                
                                        else:
                                            print('Error: MetaMap Negation', d)
                                    
                                    elif t == 'edu.uth.clamp.nlp.typesystem.ClampRelationUIMA':
                                        keys = ['semanticTag', 'entTo', 'entFrom']
                                        d = get_dict(keys, sentence, d)
                                        
                                        if d['semanticTag'] == 'NEG_Of':
                                            #print(d['entTo'], '\n', d['entFrom'])
                                            attribute =  json.loads(d['entFrom'].attribute)
                                            if 'umlsCuiDesc' in attribute:
                                                dd['cui'] = d['entFrom'].cui.split()[0].replace(',','')  
                                                dd['concept'] = attribute['umlsCuiDesc']
                                                dd['begin'] = d['entFrom'].begin
                                                dd['end'] = d['entFrom'].end
                                                dd['semanticTag'] = d['semanticTag']
                                                dd['entTo'] = str(d['entTo'])
                                                dd['entFrom'] = str(d['entFrom'])
                                                
                                                df = append_to_df(df, dd)
                                                df = add_keys(df, system, t, u, corpus, fname)
                                        
                                    elif t == 'biomedicus.v2.Negated':
                                        keys = ['begin', 'end']
                                        d = get_dict(keys, sentence, d)
                                        df = append_to_df(df, d)
                                        df = add_keys(df, system, t, u, corpus, fname)
                                        
                                    elif t == 'biomedicus.v2.UmlsConcept':
                                        keys = ['begin', 'end', 'cui', 'tui', 'confidence']
                                        d = get_dict(keys, sentence, d)
                                        
                                        df = append_to_df(df, d)
                                        df = add_keys(df, system, t, u, corpus, fname)
                                    
                                    # no labvalue -> no UMLS info
                                    elif t == 'edu.uth.clamp.nlp.typesystem.ClampNameEntityUIMA':
                                        keys = ['assertion', 'cui', 'begin', 'end', 'attribute', 'semanticTag']
                                        d = get_dict(keys, sentence, d)
                                        
                                        if 'attribute' in d and d['attribute']:
                                            attribute =  json.loads(d['attribute'])
                                            #if d['assertion'] == 'present':
                                            if 'umlsCuiDesc' in attribute:
                                                dd['concept'] = attribute['umlsCuiDesc']
                                                dd['sentence_prob'] = attribute['sentence_prob'] 
                                                dd['concept_prob'] = attribute['concept_prob']
                                                dd['assertion'] = d['assertion']
                                                dd['cui'] = d['cui'].split()[0].replace(',','') 
                                                dd['begin'] = d['begin']
                                                dd['end'] = d['end']
                                                dd['semanticTag'] = d['semanticTag']
                                                dd['attribute'] = str(d['attribute'])

                                                df = append_to_df(df, dd)
                                                df = add_keys(df, system, t, u, corpus, fname)
                                        else:
                                            if d['semanticTag'] == 'drug':
                                                if d['cui'] and 'C' in d['cui']:
                                                    dd['cui'] = str(d['cui'].split()[0]).replace(',','') 
                                                    dd['begin'] = d['begin']
                                                    dd['end'] = d['end']
                                                    dd['semanticTag'] = d['semanticTag']

                                                    df = append_to_df(df, dd)
                                                    df = add_keys(df, system, t, u, corpus, fname)
                                            
                                                    
                                    elif 'Mention' in t:
                                        keys = ['begin', 'conditional', 'confidence', 'end', 'historyOf', 'ontologyConceptArr', 'polarity', 
                                                'uncertainty']
                                        d = get_dict(keys, sentence, d)
                                        
                                        for x in range(len(d['ontologyConceptArr'])):
                                                dd['cui'] = d['ontologyConceptArr'][x].cui
                                                dd['tui'] = d['ontologyConceptArr'][x].tui
                                                dd['concept'] = d['ontologyConceptArr'][x].preferredText
                                                dd['ontologyConceptArr'] = ' '.join(map(str,d['ontologyConceptArr']))
                                                dd['begin'] = d['begin']
                                                dd['end'] = d['end']
                                                dd['conditional'] = d['conditional']
                                                dd['confidence'] = d['confidence']
                                                dd['historyOf'] = d['historyOf']
                                                dd['polarity'] = d['polarity']
                                                dd['uncertainty'] = d['uncertainty']                    
                                                                    
                                                df = append_to_df(df, dd)
                                                df = add_keys(df, system, t, u, corpus, fname)
                                                
                                    elif t == 'org.metamap.uima.ts.Candidate':
                                        keys =  ['begin', 'concept', 'cui', 'end',
                                                 'preferred', 'score', 'semanticTypes', 'matchedwords']
                                        d = get_dict(keys, sentence, d)
                                        d['semanticTypes'] = " ".join(d['semanticTypes'])
                                        d['matchedwords'] = " ".join(d['matchedwords'])
                                        
                                        df = append_to_df(df, d)
                                        df = add_keys(df, system, t, u, corpus, fname)
                            
                            return df
                            
                        annotations = get_df(view.select(t), attribs)
                        
                        # write to database
                        annotations.to_sql(table_name, engine, if_exists="append") 
                        
    # write out annotations for non-cui tables
    else:
        
        sys_ann_other = pd.DataFrame()
        for system in systems:
                
            types, view_, output = annSys.get_system_type(system)
            print("SYSTEM:", system)
           
            for t in types:

                x = t.split('.')
                table_name = system[0:3] + '_' + x[0] + '_' + x[len(x)-1]

                sql = "SELECT * FROM test." + table_nam 
                df = pd.read_sql(sql, engine)

                cols_to_keep = ['begin', 'end', 'type', 'system', 'note_id', 'corpus', 'filename']
                #print(system, t, table_name, list(df[cols_to_keep].columns.values))
                
                frames = [ sys_ann_other, df[cols_to_keep] ]
                sys_ann_other = pd.concat(frames, ignore_index=True)
        
        print(sys_ann_other.drop_duplicates())
        sys_ann_other.drop_duplicates().to_csv('/Users/gms/development/nlp/nlpie/data/amia-2019/output/analytical_' + corpus + '.csv')


if __name__ == '__main__':
    %prun main()
    print('done!')
    pass
    #main()