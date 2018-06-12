from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
from inferelator_ng.prior_gs_split_workflow import PriorGoldStandardSplitWorkflowBase
from inferelator_ng.inferecladr_workflow import InfereCLaDR_Workflow

class InfereCLaDR_Workflow_with_Prior_GS_split(InfereCLaDR_Workflow, PriorGoldStandardSplitWorkflowBase):
    """ 
        The class BBSR_TFA_Workflow_with_Prior_GS_split is a case of multiple inheritance,
        as it inherits both from BBSR_TFA_Workflow and PriorGoldStandardSplitWorkflowBase      
    """

workflow = InfereCLaDR_Workflow_with_Prior_GS_split()
# Common configuration parameters
workflow.input_dir = 'data/yeast'
#workflow.expression_matrix_file = "clusters/expr_clust1.tsv"
#workflow.meta_data_file = "clusters/meta_clust1.tsv"
workflow.num_bootstraps = 1
workflow.delTmax = 6000
workflow.delTmin = 0
#workflow.tau = 45
#workflow.run() 
workflow.reduce_searchspace=True
workflow.optimize_taus()
