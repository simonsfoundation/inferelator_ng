"""
Workflow class that splits the prior into a gold standard and new prior
"""

import random
import pandas as pd
import numpy as np
from . import workflow
from . import results_processor

class PriorGoldStandardSplitWorkflowBase(workflow.WorkflowBase):
    split_ratio = 0.5

    def set_gold_standard_and_priors(self):
        """
        Ignore the gold standard file. Instead, create a gold standard
        from a 50/50 split of the prior. Half of original prior becomes the new prior, 
        the other half becomes the gold standard
        """
        self.priors_data = self.input_dataframe(self.priors_file)
        prior = pd.melt(self.priors_data.reset_index(), id_vars='index')
        prior_edges = prior.index[prior.value != 0]
        np.random.seed(self.random_seed)
        keep = np.random.choice(prior_edges, int(len(prior_edges)*self.split_ratio), replace=False)
        prior_subsample = prior.copy(deep=True)
        gs_subsample = prior.copy(deep=True)
        prior_subsample.loc[prior_edges[~prior_edges.isin(keep)], 'value'] = 0
        gs_subsample.loc[prior_edges[prior_edges.isin(keep)], 'value'] = 0
        prior_subsample = pd.pivot_table(prior_subsample, index='index', columns='variable', values='value', fill_value=0)
        gs_subsample = pd.pivot_table(gs_subsample, index='index', columns='variable', values='value', fill_value=0)
        self.priors_data = prior_subsample
        self.gold_standard = gs_subsample

class ResultsProcessorForGoldStandardSplit(results_processor.ResultsProcessor):

    def calculate_precision_recall(self, combined_confidences, gold_standard, priors_data):
        # this code only runs for a positive gold standard, so explicitly transform it using the absolute value: 
        gold_standard = np.abs(gold_standard)
        # filter gold standard
        gold_standard_nozero = gold_standard.loc[(gold_standard!=0).any(axis=1), (gold_standard!=0).any(axis=0)]
        intersect_index = combined_confidences.index.intersection(gold_standard_nozero.index)
        intersect_cols = combined_confidences.columns.intersection(gold_standard_nozero.columns)
        gold_standard_filtered = gold_standard_nozero.loc[intersect_index, intersect_cols]
        priors_data_filtered = priors_data.loc[intersect_index, intersect_cols]
        combined_confidences_filtered = combined_confidences.loc[intersect_index, intersect_cols]
        # removing correctly predicted interactions that were removed from GS because GS was split:
        combined_confidences_filtered = combined_confidences_filtered*(1-priors_data_filtered.abs())
        # rank from highest to lowest confidence
        
        sorted_candidates = np.argsort(combined_confidences_filtered.values, axis = None)[::-1]
        gs_values = gold_standard_filtered.values.flatten()[sorted_candidates]
        #the following mimicks the R function ChristophsPR
        precision = np.cumsum(gs_values).astype(float) / np.cumsum([1] * len(gs_values))
        recall = np.cumsum(gs_values).astype(float) / sum(gs_values)
        precision = np.insert(precision,0,precision[0])
        recall = np.insert(recall,0,0)
        return (recall, precision)
