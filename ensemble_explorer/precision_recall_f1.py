import numpy as np
from config import engine, input_folder, output_folder, all_systems
import click
import os
import pandas as pd
from scipy.stats import norm





def compare_system_and_reference(system):
    system_sql = f'select * from spacy where `system` = "{system}";'
    reference_sql = f'select * from spacy where `system` = "reference";'
    system_df = pd.read_sql(system_sql, con=engine)

    reference_df = pd.read_sql(reference_sql, con=engine).set_index(['file', 'corpus'])
    def apply_func(x, file, corpus):
        referand = reference_df.loc[file, corpus]
        return confused(label_vector(x, file), label_vector(referand, file))
    result = pd.DataFrame([
        apply_func(x, file, corpus)
        for (file, corpus), x in system_df.groupby(['file', 'corpus'])
        if file in reference_df.index.get_level_values('file')
        ])

    os.chdir(output_folder)
    result.to_csv(f'{system}_comparison.csv')


    tp, fn, fp =result['TP'].sum(), result['FN'].sum(), result['FP'].sum(),
    recall_obs = tp + fn
    precision_obs = tp + fp

    [r, dr, r_upper_bound, r_lower_bound] = normal_approximation_binomial_confidence_interval(tp, recall_obs)
    [p, dp, p_upper_bound, p_lower_bound] = normal_approximation_binomial_confidence_interval(tp, precision_obs)
    [f, df, f_upper_bound, f_lower_bound] = f1_score_confidence_interval(r, p, dr, dp)

    return {
    'system': system,
    'precision':p,
    'upper_precision':p_upper_bound,
    'lower_precision':p_lower_bound,
    'recall':r,
    'upper_recall':r_upper_bound,
    'lower_recall':r_lower_bound,
    'f1':f,
    'upper_f1':f_upper_bound,
    'lower_f1':f_lower_bound,
    }


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


def normal_approximation_binomial_confidence_interval(s, n, confidence_level=.95):
	'''Computes the binomial confidence interval of the probability of a success s,
	based on the sample of n observations. The normal approximation is used,
	appropriate when n is equal to or greater than 30 observations.
	The confidence level is between 0 and 1, with default 0.95.
	Returns [p_estimate, interval_range, lower_bound, upper_bound].
	For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book.'''

	p_estimate = (1.0 * s) / n

	interval_range = norm.interval(confidence_level)[1] * np.sqrt( (p_estimate * (1-p_estimate))/n )

	return p_estimate, interval_range, p_estimate - interval_range, p_estimate + interval_range


def f1_score_confidence_interval(r, p, dr, dp):
	'''Computes the confidence interval for the F1-score measure of classification performance
	based on the values of recall (r), precision (p), and their respective confidence
	interval ranges, or absolute uncertainty, about the recall (dr) and the precision (dp).
	Disclaimer: I derived the formula myself based on f(r,p) = 2rp / (r+p).
	Nobody has revised my computation. Feedback appreciated!'''

	f1_score = (2.0 * r * p) / (r + p)

	left_side = np.abs( (2.0 * r * p) / (r + p) )

	right_side = np.sqrt( np.power(dr/r, 2.0) + np.power(dp/p, 2.0) + ((np.power(dr, 2.0)+np.power(dp, 2.0)) / np.power(r + p, 2.0)) )

	interval_range = left_side * right_side

	return f1_score, interval_range, f1_score - interval_range, f1_score + interval_range


@click.command()
@click.option('--system', help='system of analysis to compare')
@click.option('--analysistype', default='entity', help='preform entity or full analysis?')
def analyze(system, analysistype):
    global analysis_type
    analysis_type = analysistype
    if system == 'all':
        systems = all_systems
    else:
        systems = [system]
    os.chdir(output_folder)
    pd.DataFrame([
        compare_system_and_reference(s) for s in systems
    ]).to_csv('output_summarized.csv')


if __name__ == '__main__':

    analyze()
