# %%
import sys
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/models")
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/run")

from RF_class import PredictNewTestSet
import pandas as pd

path = "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/"
ft = path + 'PMG_held_out_desc_top.csv'
tg = path + 'PMG_held_out_targ_top.csv'
fl = path + 'PMG_rdkit_full_top.csv'

# %%

PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=["20241002_10_mp", "20241002_10_rmpo", "20241011_10_rmp",  "20241015_10_rmp", 
"20241002_10_mpo", "20241011_10_mp", "20241011_10_rmpo", "20241015_10_rmpo",
"20241002_10_mu", "20241011_10_mpo", "20241015_10_mp",   
"20241002_10_r", "20241011_10_mu", "20241015_10_mpo",
"20241002_10_rmp", "20241011_10_r", "20241015_10_r"
],
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/10_mol_sel/'
)

# %%

PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=['20241011_50_mp', '20241015_50_mp', '20241023_50_mp',
'20241011_50_mpo', '20241015_50_mpo', '20241023_50_mpo', 
'20241011_50_mu', '20241015_50_mu', '20241023_50_mu',
'20241011_50_r', '20241015_50_r', '20241023_50_r', 
'20241011_50_rmp', '20241015_50_rmp', '20241023_50_rmp', 
'20241011_50_rmpo', '20241015_50_rmpo', '20241023_50_rmpo'  

],
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/50_mol_sel/'
)
# %%
