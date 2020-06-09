import numpy as np
from config import engine, input_folder, output_folder, all_systems
import click
import os
import pandas as pd




def compare_system_and_reference(system):
    system_sql = f'select * from spacy where `system` = "{system}";'
    reference_sql = f'select * from spacy where `system` = "reference";'
    system_df = pd.read_sql(system_sql, con=engine)

    reference_df = pd.read_sql(reference_sql, con=engine).set_index(['file', 'corpus'])
    print(system_df)
    print(reference_df)
    def apply_func(x, file, corpus):
        referand = reference_df.loc[file, corpus]
        return confused(label_vector(x, file), label_vector(referand, file))
    result = pd.DataFrame([
        apply_func(x, file, corpus)
        for (file, corpus), x in system_df.groupby(['file', 'corpus'])
        if file in reference_df.index.get_level_values('file')
        ])

    print(result)
    os.chdir(output_folder)
    result.to_csv(f'{system}_comparison.csv')


    mean_precision, sd_precision, mean_recall, sd_recall, mean_f1, sd_f1 =(
        result['precision'].mean(), result['precision'].std(),
        result['recall'].mean(), result['recall'].std(),
        2/(1/(result['recall']) + 1/(result['precision'])).mean(),
        2/(1/(result['recall']) + 1/(result['precision'])).std(),
        )
    with open(f'{system}_summary.csv', 'w+') as f:
        f.write(f'''
        mean precision: {mean_precision}
        sd precision:{sd_precision}
        mean recall:{mean_recall}
        sd recall:{sd_recall}
        mean f1:{mean_f1}
        sd f1:{sd_f1}

        ''')

def label_vector(df, file):
    os.chdir(input_folder)
    with open(file + '.txt') as f:
        #get length of file
        length = len(f.read())
    vector = np.zeros(length)
    if analysis_type == 'entity':
        for i, row in df.iterrows():
            #1 for label, 0 for no label
            vector[row['begin']:row['end']] = 1
    elif analysis_type == 'full':
        for i, row in df.iterrows():
            #cui number for label, 0 otherwise
            vector[int(row['begin']):int(row['end'])] = row['cui']
    return vector

def confused(sys1, ann1):
    retval =  {
    # True Negative (TN): we predict a label of 1 (positive), and the true label is 1.
        'TP':np.sum(np.logical_and(sys1 != 0, ann1 == sys1 )),
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        'TN': np.sum(np.logical_and(sys1 == 0, ann1 == 0)),

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        'FP': np.sum(np.logical_and(sys1 != 0, ann1 == 0)),
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        'FN': np.sum(np.logical_and(sys1 == 0, ann1 != 0)),
    }
    retval['precision'] = retval['TP'] / (retval['TP'] + retval['FP'])
    retval['recall'] = retval['TP'] / (retval['TP'] + retval['FN'])
    retval['f1'] = 2 / ((1 / retval['precision']) + (1 / retval['recall']))
    return retval

if __name__ == '__main__':
    @click.command()
    @click.option('--system', help='system of analysis to compare')
    @click.option('--analysistype', default='entity', help='preform entity or full analysis?')
    def analyze(system, analysistype):
        global analysis_type
        analysis_type = analysistype
        if system == 'all':
            for s in all_systems:
                compare_system_and_reference(s)
        else:
            compare_system_and_reference(system)
    analyze()
