import arparse
from inferelator_ng.multitask_sparse_blocksparse_workflow import MTL_SBS_Workflow

workflow = MTL_SBS_Workflow()

parser = argparse.ArgumentParser(description = 'run MTL Bsubtilis reps with ipcluster -- ipyparallel')
parser.add_argument('-cid','--cluster_id', help = 'cluster id for ipcluster', required = False, default = '')
parser.add_argument('-idx','--index', help = 'replicate index', required = False, default = 1)
args = vars(parser.parse_args())

idx = args['index']
cluster_id = args['cluster_id']

workflow = MTL_SBS_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/bsubtilis_MTL'
workflow.expression_filelist = ["expression_py79.tsv", "expression_bsb1.tsv"]
workflow.meta_data_filelist = ["meta_data_py79.tsv", "meta_data_bsb1.tsv"]
workflow.tf_names_file = 'tf_names_MTL.tsv'
workflow.delTmax = 60
workflow.delTmin = 15
workflow.tau = 15
workflow.num_bootstraps = 20
workflow.cluster_id = cluster_id
workflow.priors_filelist = [''.join(['prior_gs_splits/prior_', str(idx), '.tsv'])]*2
workflow.gold_standard_filelist = ['gold_standard.tsv', 'gold_standard.tsv']
workflow.outdir = ''.join(['MTL_SBSebic_output/replicate_', str(idx)])
workflow.run()

