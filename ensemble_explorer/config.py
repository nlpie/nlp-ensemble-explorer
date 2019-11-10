from sqlalchemy.engine import create_engine
import pymysql
from pathlib import Path

engine = create_engine('mysql+pymysql://gms:nej123@localhost/test', pool_pre_ping=True, pool_size=20, max_overflow=30)

data_dir =  '/Users/gms/development/nlp/nlpie/data/amicus-u01/output/'

single_sys_dir = Path(data_dir + "single_system_out")
single_sys_dir.mkdir(parents=True, exist_ok=True)
dir_out = Path(data_dir + 'single_system_out/')

systems = ['biomedicus', 'clamp', 'metamap'] 

corpus = 'casi' 

system_annotations = 'analytical_cui_' + corpus + '_concepts.csv'

reference_annotations = corpus + '_all'
