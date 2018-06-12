import numpy as np
import os
from . import workflow
import design_response_translation #added python design_response
from tfa import TFA
from results_processor import ResultsProcessor
import mi_R
import bbsr_python
import datetime
from kvsstcp.kvsclient import KVSClient
import pandas as pd
from . import utils
from bbsr_tfa_workflow import BBSR_TFA_Workflow
from prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase
from prior_gs_split_workflow import ResultsProcessorForGoldStandardSplit


# Connect to the key value store service (its location is found via an
# environment variable that is set when this is started vid kvsstcp.py
# --execcmd).
kvs = KVSClient()
# Find out which process we are (assumes running under SLURM).
rank = int(os.environ['SLURM_PROCID'])


class InfereCLaDR_Workflow(BBSR_TFA_Workflow, PriorGoldStandardSplitWorkflowBase):
    expr_clust_files = ["clusters/expr_clust1.tsv", "clusters/expr_clust2.tsv", "clusters/expr_clust3.tsv", "clusters/expr_clust4.tsv"]
    meta_clust_files = ["clusters/meta_clust1.tsv", "clusters/meta_clust2.tsv", "clusters/meta_clust3.tsv", "clusters/meta_clust4.tsv"]
    seeds = range(42,62)
    taus = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250]

    def optimize_taus(self):
        run_objs = []
        ## loop over condition clusters
        for clust in range(len(self.expr_clust_files)):
            print("Optimizing tau for expression cluster {} out of {}".format(clust+1,len(self.expr_clust_files)))
            self.expression_matrix_file = self.expr_clust_files[clust]
            self.meta_data_file = self.meta_clust_files[clust]
            ## loop over splits
            for seed in self.seeds:
                print("Resampling the prior for seed {} out of {}".format(seed,max(self.seeds)))
                self.random_seed = seed
                ## loop over taus
                for tau in self.taus:
                    # set self.tau to current value
                    # set self.prior, expression, etc. 
                    print("Running BBSR for tau = {} ({} out of {})".format(tau, np.concatenate(np.where(tau==np.asarray(self.taus)))+1, len(self.taus)))
                    self.tau = tau
                    self.run()
                    run_result = Results_of_a_Run()
                    (run_result.clust, run_result.seed, run_result.tau) = (clust, seed, tau)
                    (run_result.betas, run_result.rescaled_betas) = (self.betas, self.rescaled_betas)
                    (run_result.priors_data, run_result.gold_standard) = (self.priors_data, self.gold_standard)
                    run_result.evaluate_tau_run()
                    import pdb; pdb.set_trace()
                    run_objs.append(run_result)

                self.results_by_clust_seed_tau = run_objs

    ## with a vector of taus, build custom logic for a final design and response
    # loop over bootstraps:
    def run_with_full_prior(self):
        #self.run(X, Y, etc...)
        raise NotImplementedError # implement later

    #define your own emit_results:
    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        print "skipping emit results"
        #raise NotImplementedError # implement later
        
class Results_of_a_Run(object):
    
    #define precision-recall evaluations for every tau
    def evaluate_tau_run(self):
        self.results_processor = ResultsProcessorForGoldStandardSplit(self.betas, self.rescaled_betas)
        self.combined_confidences = self.results_processor.compute_combined_confidences()
        (self.recall, self.precision) = self.results_processor.calculate_precision_recall(self.combined_confidences, self.gold_standard, self.priors_data)
        self.aupr = self.results_processor.calculate_aupr(self.recall, self.precision)


