# Tools for ensembling:

Code for extracting UMLS concepts and CUIs from relevant corpora

## Parsing manually annotated corpora:

-> mipacq_reference.ipynb: MiPACQ clinical notes
-> i2b2_reference.ipynb: i2b2 2010 challenge set of clinical notes
-> fairview_reference.ipynb: fv017 clinical notes

TODO: fv017 to be processed through NLP-ADAPT

## Parsing system annotated corpora:

-> system_annotations.ipynb:

1. Parse UIMA based CAS objects
2. Create analytical sets based on corpus with general format

TODO: clean up and consolidate with system_master.ipynb

## Performance evaluation of single system and all combinatoric permutations of Boolean merges (unions and intersections)

-> nlp_ensemble_explorer.ipynb

TODO:

## Data

1. System annotated data have been consolidated by corpora into files named: analytical_cui_CORPUSNAME_concepts.csv

General format of system data:

id; pk
cui: assigned cui (preferred) 
span: begin and end
note_id: corpus assigned case_id/mrn
system: NLP system
similarity: similarity score based on given algorithm (used for quick_umls)
corpus: not used
type: for quick_umls this is the similarity metric; for UIMA systems, this is the assigned type
polarity: -1 or 1 for negation; null if not used
score: MetaMap scoring threshold 
qscore: ????

2. Refence/gold standard data are parsed into aa MySQL database

General format of reference/gold standard data:

file: case_id
text: annotated text from span
type (or class for mipacq): concept
span: start and end 
value: cui (when available)
classType (mipacq only): string if no modifier; boolean if negation; degree if (?)

### To use data in nlp_ensemble_explorer.ipynb:

Extract ensembling.sql.zip and import into MySQL. Configure `engine` variable in `get_metric_data` method with database name and user credentials. Reference tables are then defined in the `AnalysisConfig` class, according to corpus.

Place `analytical_cui_CORPUSNAME_concepts.csv` files in desired directory and change `data_dir` attribute accordingly in the `AnalysisConfig` class.

Within `data_dir` location create a direcory called `single_system_out` (for running single system performance evaluation)


## Requirements:

Anaconda python version 3.7.x should contain most libraries need for this.

Some special libraries include:
cassis (needed to parse UIMA CAS objects into JSON for use in python; (if needed, install from the dkpro github instance)
pymsql
shelve (for persistant storage of i-o labeled vectors)
pythonds (for use of binary parse tree data structures and methods)

*TODO: add otherannotation tasks (e.g., abbreviation disambiguation)*


