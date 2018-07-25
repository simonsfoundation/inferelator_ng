from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
#from inferelator_ng.prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase
from inferelator_ng.inferecladr_workflow import InfereCLaDR_Workflow


workflow = InfereCLaDR_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/yeast'
#workflow.expr_clust_files = ["clusters/expr_clust1.tsv", "clusters/expr_clust2.tsv", "clusters/expr_clust3.tsv", "clusters/expr_clust4.tsv"]
workflow.expr_clust_files = ["clusters/expr_clust2.tsv", "clusters/expr_clust3.tsv"]
#workflow.meta_clust_files = ["clusters/meta_clust1.tsv", "clusters/meta_clust2.tsv", "clusters/meta_clust3.tsv", "clusters/meta_clust4.tsv"]
workflow.meta_clust_files = ["clusters/meta_clust2.tsv", "clusters/meta_clust3.tsv"]
workflow.num_bootstraps = 1
workflow.delTmax = 6000
workflow.delTmin = 1
#workflow.seeds = range(42,62)
workflow.seeds = range(42,44)
#workflow.taus = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250]
workflow.taus = [0, 5]
workflow.reduce_searchspace=True
#workflow.optimize_taus()
workflow.run()
