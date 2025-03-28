# %%
import sys
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/models")
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/run")

from RF_class import PredictNewTestSet
import pandas as pd

path = "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/"
# ft = path + 'PMG_held_out_desc_top.csv'
# tg = path + 'PMG_held_out_targ_top.csv'
# fl = path + 'PMG_rdkit_full_top.csv'

tg = path + 'PMG_held_out_targ_trimmed.csv'
ft = path + "PMG_held_out_desc_trimmed.csv"
fl = path +  "PMG_rdkit_full.csv"

# %%

PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'held_out_test',
    experiment_ls=[
"20240910_10_rmp"


],
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/10_sel/'
)

# %%
