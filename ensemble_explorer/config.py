from sqlalchemy.engine import create_engine
import pymysql

input_folder = '/Users/jacobsolinsky/programming/serguei/ensemble-explorer/data/input'
output_folder = '/Users/jacobsolinsky/programming/serguei/ensemble-explorer/data/output'

database_type = 'mysql+pymysql' # We use mysql+pymql as default
database_username = 'gms'
database_password = 'nej123'
database_url = 'localhost' # HINT: use localhost if you're running database on your local machine
database_name = 'concepts' # Enter database name

engine_request = str(database_type)+'://'+database_username+':'+database_password+"@"+database_url+'/'+database_name
engine = create_engine(engine_request, pool_pre_ping=True, pool_size=20, max_overflow=30)
all_systems = ("en_core_sci_sm","en_core_sci_md","en_core_sci_lg",
              "en_ner_craft_md", "en_ner_jnlpba_md","en_ner_bc5cdr_md",
              "en_ner_bionlp13cg_md",)
