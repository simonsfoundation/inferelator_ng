"""
Run BSubtilis Network Inference with TFA BBSR.
"""

import numpy as np
import os
from workflow import WorkflowBase
import design_response_R
from tfa import TFA
from results_processor import ResultsProcessor
import mi_R
import bbsr_R
import datetime
from prior import Prior
from . import utils


class BBSR_TFA_Chromatin_Prior_Workflow(WorkflowBase):

    motifs_file = 'motifs.bed'
    annotation_file = 'tss.bed'
    target_genes_file = 'target_genes.tsv'
    prior_mode = 'closest'
    prior_max_distance = float('Inf')
    prior_ignore_downstream = True
    output_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def run(self):
        """
        Execute workflow, after all configuration.
        """
        np.random.seed(self.random_seed)

        self.mi_clr_driver = mi_R.MIDriver()
        self.regression_driver = bbsr_R.BBSR_driver()
        self.design_response_driver = design_response_R.DRDriver()

        self.get_data()
        self.build_prior()
        self.compute_common_data()
        self.compute_activity()

        betas = []
        rescaled_betas = []

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            print('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps))
            X = self.activity.ix[:, bootstrap]
            Y = self.response.ix[:, bootstrap]
            print('Calculating MI, Background MI, and CLR Matrix')
            (self.clr_matrix, self.mi_matrix) = self.mi_clr_driver.run(X, Y)
            print('Calculating betas using BBSR')
            current_betas, current_rescaled_betas = self.regression_driver.run(X, Y, self.clr_matrix, self.priors_data)
            betas.append(current_betas)
            rescaled_betas.append(current_rescaled_betas)
        self.emit_results(betas, rescaled_betas, self.gold_standard, self.priors_data)


    def get_data(self):
        """
        Read data files in to data structures.
        """
        self.expression_matrix = self.input_dataframe(self.expression_matrix_file)
        tf_file = self.input_file(self.tf_names_file)

        self.tf_names = utils.read_tf_names(tf_file)
        target_genes_file = self.input_file(self.target_genes_file)
        self.target_genes = utils.read_tf_names(target_genes_file)

        # Read metadata, creating a default non-time series metadata file if none is provided
        self.meta_data = self.input_dataframe(self.meta_data_file, has_index=False, strict=False)
        if self.meta_data is None:
            self.meta_data = self.create_default_meta_data(self.expression_matrix)
        self.gold_standard = self.input_dataframe(self.gold_standard_file)


    def build_prior(self):
        print('Generating Prior Matrix from input Motifs and Annotation ... ')
        prior_generator = Prior(self.input_file(self.motifs_file), self.input_file(self.annotation_file), self.target_genes, self.tf_names,
                                mode = self.prior_mode, max_distance = self.prior_max_distance, ignore_downstream = self.prior_ignore_downstream)
        self.priors_data = prior_generator.make_prior()

    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        print('Computing Transcription Factor Activity ... ')
        TFA_calculator = TFA(self.priors_data, self.design, self.half_tau_response)
        self.activity = TFA_calculator.compute_transcription_factor_activity()

    def emit_results(self, betas, rescaled_betas, gold_standard, priors):
        """
        Output result report(s) for workflow run.
        """
        output_dir = os.path.join(self.input_dir, self.output_dir)
        os.makedirs(output_dir)
        self.results_processor = ResultsProcessor(betas, rescaled_betas)
        self.results_processor.summarize_network(output_dir, gold_standard, priors)
