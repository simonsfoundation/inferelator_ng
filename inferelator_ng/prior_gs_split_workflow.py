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

    def create_filtered_gold_standard_and_confidences(self, combined_confidences, gold_standard, priors, filter_index, filter_cols):
        # this code only runs for a positive gold standard, so explicitly transform it using the absolute value: 
        gold_standard = np.abs(gold_standard)
        gold_standard_filtered = gold_standard.loc[filter_index, filter_cols]
        priors_data_filtered = priors.loc[filter_index, filter_cols]
        combined_confidences_filtered = combined_confidences.loc[filter_index, filter_cols]
        # removing correctly predicted interactions that were removed from GS because GS was split:
        combined_confidences_filtered = combined_confidences_filtered*(1-priors_data_filtered.abs())
        # rank from highest to lowest confidence
        return(combined_confidences_filtered, gold_standard_filtered)

