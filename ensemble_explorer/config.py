from sqlalchemy.engine import create_engine
import pymysql
from pathlib import Path

database_type = 'mysql+pymysql' # We use mysql+pymql as default
database_username = 'gms'
database_password = 'nej123' 
database_url = 'localhost' # HINT: use localhost if you're running database on your local machine
database_name = 'test' # Enter database name

engine_request = str(database_type)+'://'+database_username+':'+database_password+"@"+database_url+'/'+database_name
engine = create_engine(engine_request, pool_pre_ping=True, pool_size=20, max_overflow=30)

data_dir =  '/Users/gms/development/nlp/nlpie/data/amicus-u01/output/'

single_sys_dir = Path(data_dir + "single_system_out")
single_sys_dir.mkdir(parents=True, exist_ok=True)
dir_out = Path(data_dir + 'single_system_out/')

systems = ['biomedicus', 'clamp', 'metamap'] 

corpus = 'casi' 

system_annotations = 'analytical_cui_' + corpus + '_concepts.csv'

reference_annotations = corpus + '_all'
