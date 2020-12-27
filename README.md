# Tools for ensembling:


Code for parsing and evaluating system performance of extracted UMLS concepts and CUIs from relevant corpora

*TODO: add other annotation tasks (e.g., abbreviation disambiguation)*

## Parsing manually annotated corpora:

   - reference_annotations/mipacq_reference.ipynb: MiPACQ clinical notes
   - reference_annotations/i2b2_reference.ipynb: i2b2 2010 challenge set of clinical notes
   - reference_annotations/fairview_reference.ipynb: fvr01 clinical notes

## Parsing system annotated corpora:

   - system_annotations/system_annotations.ipynb

1. Parse UIMA based CAS objects
2. Create analytical sets based on corpus with general format

TODO: clean up and consolidate with system_master.ipynb

## Performance evaluation of single system and all combinatoric permutations of Boolean merges (unions and intersections)

   - nlp_ensemble_explorer.ipynb

## Data

1. System annotated data have been consolidated by corpora into files named: analytical_CORPUSNAME.csv

General format of system data:

   - id; pk
   - cui: assigned cui (preferred; when available)
   - span: begin and end
   - note_id: corpus assigned case_id/mrn
   - corpus: identifes corpus by name
   - system: NLP system
   - type: for quick_umls this is the similarity metric; for UIMA systems, this is the assigned annotation type
   - score: MetaMap scoring threshold; similarity score for QuickUmls; probability for CLAMP; n/a for cTAKES 
   - semantic_type: culled from Figure 1 in paper, based on system

2. Refence/gold standard data are parsed into aa MySQL database

General format of reference/gold standard data:

   - file: case_id
   - text: annotated text from span
   - type (or class for mipacq): concept
   - span: start and end 
   - value: cui (when available)
   - classType (mipacq only): string if no modifier; boolean if negation; degree if (?)
   - semantic_type: culled from Figure 1 in paper, based on corpus

### To use data in nlp_ensemble_explorer.ipynb:

Extract ensembling.sql.zip and import into MySQL. 

Configure `engine` variable with database name and user credentials and data directory in first cell. Reference tables are then defined in the `AnalysisConfig` class, according to corpus.

Place `analytical_cui_CORPUSNAME_concepts.csv` files in desired data directory and change `data_dir` attribute accordingly in first cell.

## Requirements:

Anaconda python version 3.7.x should contain most libraries need for this.

Some special libraries include:

   - cassis (needed to parse UIMA CAS objects into JSON for use in python; (if needed, install from the dkpro github instance)
   - pymsql
   - pythonds (for use of parse tree data structures and methods)
   
## Desiderata

### Set operation notation

We use the operator `|` to represent a Boolean OR operation (or union: ∪) and `&` to represent a Boolean AND oepration (or intersection: ∩)

### Optimizations

Initial optimizations to NLP-Ensemble-Explorer, including use of memozie caching, sped up processing time very significantly, it could still be optimized further. For example, run time for all corpora and all semantic aggregations was over 5 hours. 

### Logical equivalence

For the current release version of NLP-Ensemble-Explorer, we were not able to account for all cases of logical equivalence. While we we able to account for simple cases like (A∪B) = (B∪A), we were not able to exclude cases like (((A∩B)∪C)∪(D∩E)) = (((D∩E)∪C)∪(A∩B)), which could potential decrease total run time. To account for these cases in various analyses, we used the Pandas “drop_duplicates” method.

### Semantic group usage 
Choice of Annotation Type for grouping cTAKES concepts was a convenience over use of the available TUI, especially since it mapped directly to the groupings defined for all corpora. An examination of TUIs associated with specific Annotation Type aggregation in cTAKES confirms that there is a slight advantage of the use of this over use of TUIs, since a number of TUIs mapped by cTAKES to the annotation types DiseaseDisordermention and SignSymptomMention were classified within the UMLS in semantic groups not covered by any of the corpora (specifically,  Activities & Behaviors, Phenomena and Physiology), but were properly classified by cTAKES.

### System UMLS lookup

BioMedICUS uses a tiered scoring technique for matching UMLS concepts to phrases by first performing direct dictionary phrase matches, second by lower-cased dictionary phrase matches, and lastly using a discontinuous bag of SPECIALIST normalized terms matches. 

cTAKES matches UMLS concepts to phrases, by each phrase’s lexical and non-lexical permutations and variations against concepts in a dictionary and a list of maintained terms.[1]

CLAMP matches UMLS concepts to phrases using the BM25 algorithm for UMLS lookup to find candidates concepts from the UMLS and then apply RankSVM to rank those candidates, from which the top ranked concept is selected.

MetaMap uses a shallow parser to generate candidate phrases then, for each candidate phrase, many lexical variations are generated; finally, each phrase is then assigned a score based on its distance to concepts in the UMLS.[2] For this study, we did not use word sense disambiguation.

Lastly, QuickUMLS generates and validates all possible sequences for each token in the document, then using an indexing algorithm to determine if a string in the UMLS is similar to a candidate set of tokens, it returns the matching set with a similarity measure based on the given threshold.  Larger values of α increase precision but decrease recall; the opposite holds true for smaller values of α.[3]


References:

1 Savova GK, Masanz JJ, Ogren PV, et al. Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES): architecture, component evaluation and applications. J Am Med Inform Assoc JAMIA 2010;17 5:507–13.

2 Aronson A, Lang F-M. An Overview of MetaMap: Historical Perspective and Recent Advances. J Am Med Inform Assoc JAMIA 2010;17:229–36. doi:10.1136/jamia.2009.002733

3 Soldaini L, Goharian N. QuickUMLS: a fast, unsupervised approach for medical concept extraction. Proc Med Inf Retr MedIR Workshop SIGIR 2016;:4.




