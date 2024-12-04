# %%
import sys
from pathlib import Path
import pandas as pd

PROJ_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, PROJ_DIR / "scripts/run/")
from average_all import AverageAll

sys.path.insert(0, PROJ_DIR / "scripts/misc/")
from misc_functions import (
    get_sel_mols_between_iters,
    molid_to_smiles,
    molid_ls_to_smiles,
)

avg = AverageAll(results_dir=str(PROJ_DIR) + '/results/rdkit_desc/finished_results/10_mol_sel/')

# %%
avg._average_experiment(exp_suffix='10_mu', n_iters=150)
# avg._average_experiment(exp_suffix='10_mp', n_iters=150)
# avg._average_experiment(exp_suffix='10_mpo', n_iters=150)
# avg._average_experiment(exp_suffix='10_rmp', n_iters=150)
# avg._average_experiment(exp_suffix='10_rmpo', n_iters=150)
# avg._average_experiment(exp_suffix='10_r', n_iters=150)


# %%
avg = AverageAll(results_dir=str(PROJ_DIR) + '/results/rdkit_desc/finished_results/50_mol_sel/')

avg._average_experiment(exp_suffix='50_mu', n_iters=30)
avg._average_experiment(exp_suffix='50_mp', n_iters=30)
avg._average_experiment(exp_suffix='50_mpo', n_iters=30)
avg._average_experiment(exp_suffix='50_rmp', n_iters=30)
avg._average_experiment(exp_suffix='50_rmpo', n_iters=30)
avg._average_experiment(exp_suffix='50_r', n_iters=30)
# %%
