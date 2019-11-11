

class Annotations(object):
     
    def __init__(self):
            
        """ 
        annotation base types
        """   

        self.types = {
                        'biomedicus': ["biomedicus.v2.UmlsConcept",
                                        "biomedicus.v2.Negated",
                                        "biomedicus.v2.Acronym"],

                        'clamp': ["edu.uth.clamp.nlp.typesystem.ClampNameEntityUIMA",
                                    "org.apache.ctakes.typesystem.type.syntax.ConllDependencyNode",
                                    "edu.uth.clamp.nlp.typesystem.ClampRelationUIMA"],    

                        'ctakes': ["org.apache.ctakes.typesystem.type.textsem.DiseaseDisorderMention",
                                    "org.apache.ctakes.typesystem.type.textsem.MedicationMention",
                                    "org.apache.ctakes.typesystem.type.textsem.ProcedureMention",
                                    "org.apache.ctakes.typesystem.type.refsem.UmlsConcept",
                                    "org.apache.ctakes.typesystem.type.textsem.SignSymptomMention",
                                    "org.apache.ctakes.typesystem.type.textsem.AnatomicalSiteMention"],

                        'metamap': ["org.metamap.uima.ts.Candiate",
                                    "org.metamap.uima.ts.CuiConcept",
                                    "org.metamap.uima.ts.Negation"],

                        'quick_umls': ['concept_cosine_length_false',
                                        'concept_cosine_length_true',
                                        'concept_cosine_score_false',
                                        'concept_cosine_score_true',
                                        'concept_dice_length_false',
                                        'concept_dice_length_true',
                                        'concept_dice_score_false',
                                        'concept_dice_score_true',
                                        'concept_jaccard_length_false',
                                        'concept_jaccard_length_true',
                                        'concept_jaccard_score_False',
                                        'concept_jaccard_score_true']
                        }
     
    def get_system_type(self):

        """
        return system types
        """

        types = self.types
            
        return types


