from workflow_class import RecDMTA
import sys

sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/models/")
from RF_class import RF_model
import pandas as pd

# Commonly Changed Variables
start_iter = int(sys.argv[3])
total_iters = int(sys.argv[4])
n_cmpds = int(sys.argv[1])
selection_method = sys.argv[2]
run_name = f"{sys.argv[5]}_{n_cmpds}_{selection_method}_{sys.argv[7]}"
sel_size=float(sys.argv[6])


# Less Commonly Changed Variables
docking_column = "Affinity(kcal/mol)"
id_prefix = "PMG-"
max_confs = 100

# Pathing
full_data_fpath = (
    "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/"
)
full_data_fprefix = "PMG_rdkit_*.csv.gz"
desc_fpath = "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/"
desc_fprefix = "PMG_rdkit_desc_*.csv"
docking_dir = "/users/yhb18174/Recreating_DMTA/docking/PyMolGen/"
docking_file_dir = "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/docking/"
results_dir = "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/"
init_model_dir = "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/init_RF_model/it0/"
chosen_mol_file = results_dir + run_name + "/chosen_mol.csv"
docking_score_files = "PMG_docking_*.csv"
held_out_test_feats = (
    "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_desc.csv"
)
held_out_test_targs = (
    "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_docked.csv"
)
remove_unreliable_molecules = False
n_molecules_to_remove = 5

# Running the Workflow
run = RecDMTA(
    full_data_fpath=full_data_fpath,
    full_data_fprefix=full_data_fprefix,
    desc_fpath=desc_fpath,
    desc_fprefix=desc_fprefix,
    start_iter=start_iter,
    total_iters=total_iters,
    n_cmpds=n_cmpds,
    docking_dir=docking_dir,
    docking_file_dir=docking_file_dir,
    results_dir=results_dir,
    init_model_dir=init_model_dir,
    chosen_mol_file=chosen_mol_file,
    selection_method=selection_method,
    docking_score_files=docking_score_files,
    run_name=run_name,
    docking_column=docking_column,
    max_confs=max_confs,
    rm_unrel_tr_data=remove_unreliable_molecules,
    n_to_remove=n_molecules_to_remove,
    sel_size=sel_size
)

run.RunIterations(
    held_out_test_feats=held_out_test_feats, held_out_test_targs=held_out_test_targs
)
