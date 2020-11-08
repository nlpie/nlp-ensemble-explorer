import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.environ["R_HOME"] = r"C:\Program Files\R\R-3.6.3"
os.environ["PATH"] =  r"C:\Program Files\R\R-3.6.3\bin\x64" + ";" + os.environ["PATH"]


ipython = get_ipython()
ipython.magic('load_ext rpy2.ipython')

# --->
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
pandas2ri.activate()
import rpy2.robjects.lib.ggplot2 as ggplot2
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('load_ext rpy2.ipython')

# packages used
grdevices = importr('grDevices')
R = ro.r
# --->

"""
t <- read.csv('C:\\Users\\gsilver1\\Desktop\\test_with_impute.csv')
vitals <- c('weightel24',
 'HR_max24h',
 'SpO2_max24h',
 'RR_max24h',
 'SBP_max24h',
 'DBP_max24h',
 'Temp_max24h',
 'HR_min24h',
 'SpO2_min24h',
 'RR_min24h',
 'SBP_min24h',
 'DBP_min24h',
 'Temp_min24h',
 'ynel24_Absent',
 'ynel24_Present',
 'BMI')

test=t[,vitals]

labs=t[grep("D0", names(t))]

dem = c('male', 
 'age_cat', 
 'race_White',
 'race_Black',
 'race_Asian',
 'race_Hispanic',
 'race_Declined',
 'race_Other')
 
test = t[, dem]

comorb = t[grep("COMORB", names(t))]


# https://stackoverflow.com/questions/35372365/how-do-i-generate-a-histogram-for-each-column-of-my-table/35373419
ggplot(gather(comorb), aes(value)) +
geom_histogram(bins = 10) + 
facet_wrap(~key, scales = 'free_x')

meds = t[grep("3mo", names(t))]
ggplot(pivot_longer(meds,cols=everything(), names_to="key", values_to="value"), aes(x=value)) +
geom_histogram(bins = 10) + 
facet_wrap(~key, scales = 'free_x')

--->
source("C:\\Users\\gsilver1\\Downloads\\utils_fs.R")
drops <- c("true_died","Mortality", "days_if_do_cox_PH", "MDM_LINK_ID", 'x')
test = t[ , !(names(t) %in% drops)]
target = t[grep("days_if_do_cox_PH", names(t))]
run.fs(test, target, 'HPC', 0.05)

"""

dir_ = Path("~\Desktop")
test = pd.read_csv(dir_ / "test_for_model.csv")

comorb=[p for p in test.columns if 'COMORB' in p]
home_meds=[p for p in test.columns if '3mo_prior' in p]
d0_labs=[p for p in test.columns if 'D0' in p]
#d1_labs=[p for p in db.columns if 'D1' in p]
labs = d0_labs # + d1_labs
vitals=[p for p in test.columns if '24' in p] + ["BMI"]
# TARGET: died at home, in hospital, time to death
death = ["true_died", "Mortality", "days_if_do_cox_PH"]
race = [p for p in test.columns if 'race_' in p]
other=['age_cat', 'male']


