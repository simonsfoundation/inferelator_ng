import numpy as np
import os
from . import workflow
import design_response_translation #added python design_response
from tfa import TFA
from results_processor import ResultsProcessor
import mi_R
import bbsr_python
import datetime
#from kvsstcp.kvsclient import KVSClient
import pandas as pd
import xarray as xr
from . import utils
from bbsr_tfa_workflow import BBSR_TFA_Workflow
from prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase
from prior_gs_split_workflow import ResultsProcessorForGoldStandardSplit
import matplotlib.pyplot as plt


# Connect to the key value store service (its location is found via an
# environment variable that is set when this is started vid kvsstcp.py
# --execcmd).
#kvs = KVSClient()
# Find out which process we are (assumes running under SLURM).
#rank = int(os.environ['SLURM_PROCID'])


class InfereCLaDR_Workflow(BBSR_TFA_Workflow, PriorGoldStandardSplitWorkflowBase):
    expr_clust_files = ["clusters/expr_clust1.tsv", "clusters/expr_clust2.tsv", "clusters/expr_clust3.tsv", "clusters/expr_clust4.tsv"]
    gene_clust_files = ["clusters/genes_clust1.tsv", "clusters/genes_clust2.tsv", "clusters/genes_clust3.tsv", "clusters/genes_clust4.tsv", "clusters/genes_clust5.tsv"]
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
                    self.set_gene_clusters()
                    run_result.gene_clust_index = self.gene_clust_index
                    run_result.evaluate_tau_run()
                    run_objs.append(run_result)

        self.results_by_clust_seed_tau = run_objs
        self.emit_results_by_cluster()

    def set_gene_clusters(self):
        gene_clust_index=[]
        for gene_clust_file in self.gene_clust_files:
            file_path = self.input_file(gene_clust_file)
            genes_index = self.read_gene_clust_names(file_path)
            gene_clust_index.append(genes_index)
        self.gene_clust_index = gene_clust_index

    def read_gene_clust_names(self, file_like):
        "Read gene names from one-column tsv file.  Return list of names."
        exp = pd.read_csv(file_like, sep="\t", header=None)
        assert exp.shape[1] == 1, "gene cluster file should have one column "
        return list(exp[0])
    
    ## with a vector of taus, build custom logic for a final design and response
    # loop over bootstraps:
    def run_with_full_prior(self):
        #self.optimize_taus(X, Y, etc...)
        raise NotImplementedError # implement later

    def emit_results_by_cluster(self):
        self.create_auprs_xarray()
        self.calculate_predicted_half_lives()
        self.output_dir = os.path.join(self.input_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.output_dir)
        self.predicted_half_lives.to_pandas().to_csv(os.path.join(self.output_dir, 'predicted_half-lives.tsv'), sep = '\t')
        self.plot_aupr_vs_tau()

    def create_auprs_xarray(self):
        #Create a 4D DataArray with predicted AUPR as a function of 1) condition cluster, 2) gene cluster, 3) random seed, and 4) tau:
        cond_clust_coords = ["cond_clust" + str(s+1) for s in range(len(self.expr_clust_files))]
        gene_clust_coords = ["gene_clust" + str(s+1) for s in range(len(self.gene_clust_files))]
        #Empty 4D array of zeros:
        auprs_xarray = xr.DataArray(np.zeros((len(self.expr_clust_files), len(self.gene_clust_files), len(self.seeds), len(self.taus))), dims=['cond_clusts', 'gene_clusts', 'rand_seeds', 'taus'], coords=[cond_clust_coords, gene_clust_coords, self.seeds, self.taus])
        #Fill in the AUPR values:
        for run_result in self.results_by_clust_seed_tau:
            auprs_xarray[run_result.clust,:,:,:].loc[:,run_result.seed,run_result.tau] =  run_result.recall_precision_by_gene_clust[2]

        self.auprs_xarray = auprs_xarray

    def calculate_predicted_half_lives(self):
        max_taus_by_GC_CC_seed=self.auprs_xarray.coords['taus'][self.auprs_xarray.argmax('taus')[:,:,:]]*np.log(2)
        self.predicted_half_lives = max_taus_by_GC_CC_seed.median('rand_seeds')

    def plot_aupr_vs_tau(self):
        fig, axes = plt.subplots(nrows=len(self.auprs_xarray.coords['gene_clusts']), ncols=len(self.auprs_xarray.coords['cond_clusts']), figsize=(10,8))
        #plt.setp(axes.flat, xlabel="RNA half-life (minutes)", ylabel="AUPR")
        ylim_min = self.auprs_xarray.min()
        ylim_max = self.auprs_xarray.max()
        half_lives = self.auprs_xarray.coords['taus']*np.log(2)

        for gene_idx in range(len(axes)):
            axes[gene_idx, 0].set_ylabel("AUPR", fontsize=8)
            for cond_idx in range(len(axes[0])):
                axes[-1, cond_idx].set_xlabel("RNA half-life (minutes)", fontsize=8)
                axes[gene_idx, cond_idx].set_ylim([ylim_min,ylim_max])
                axes[gene_idx, cond_idx].tick_params(labelsize=6)
                for seed in self.auprs_xarray.coords['rand_seeds']:
                    auprs_profile = self.auprs_xarray[cond_idx,gene_idx,:,:].loc[seed,:]
                    axes[gene_idx, cond_idx].plot(half_lives, auprs_profile)
                    axes[gene_idx, cond_idx].plot(half_lives[np.argmax(auprs_profile)], np.max(auprs_profile), 'o')

        fig.tight_layout()
        fig.subplots_adjust(left=0.15, top=0.95)
        plt.savefig(os.path.join(self.output_dir, 'auprs_panel.pdf'))
        plt.close()
        

    #define your own emit_results:
    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        print "skipping emit results"
        #raise NotImplementedError # implement later
        
class Results_of_a_Run(object):
    
    #define precision-recall evaluations for every tau
    def evaluate_tau_run(self):
        self.results_processor = ResultsProcessorForGoldStandardSplit(self.betas, self.rescaled_betas)
        self.combined_confidences = self.results_processor.compute_combined_confidences()
        (filter_index, filter_cols) = self.results_processor.get_nonempty_rows_cols(self.combined_confidences, self.gold_standard)
        (combined_confidences_filtered, gold_standard_filtered) = self.results_processor.create_filtered_gold_standard_and_confidences(self.combined_confidences, self.gold_standard, self.priors_data, filter_index, filter_cols)
        (self.recall, self.precision) = self.results_processor.calculate_precision_recall(combined_confidences_filtered, gold_standard_filtered)
        self.aupr = self.results_processor.calculate_aupr(self.recall, self.precision)

        recall_by_gene_clust = []
        precision_by_gene_clust = []
        aupr_by_gene_clust = []
        for this_gene_clust_index in self.gene_clust_index:
            genes_index_filtered = set(this_gene_clust_index).intersection(filter_index)
            (combined_confidences_gclust, gold_standard_gclust) = self.results_processor.create_filtered_gold_standard_and_confidences(self.combined_confidences, self.gold_standard, self.priors_data, genes_index_filtered, filter_cols)
            (recall, precision) = self.results_processor.calculate_precision_recall(combined_confidences_gclust, gold_standard_gclust)
            aupr = self.results_processor.calculate_aupr(recall, precision)

            recall_by_gene_clust.append(recall)
            precision_by_gene_clust.append(precision)
            aupr_by_gene_clust.append(aupr)
        
        self.recall_precision_by_gene_clust = [recall_by_gene_clust, precision_by_gene_clust, aupr_by_gene_clust]



