from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow

workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/dmel'
workflow.num_bootstraps = 20
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.random_seed = 1
workflow.run() 

