from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow

workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/dmel'
workflow.num_bootstraps = 100
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 1
workflow.priors_file = 'prior_with_tfs_filtered_by_one_percent_expression.tsv'
workflow.run() 

