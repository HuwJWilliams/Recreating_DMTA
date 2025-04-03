# %%
import sys
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/models")
sys.path.insert(0, "/users/yhb18174/Recreating_DMTA/scripts/run")

from RF_class import PredictNewTestSet
import pandas as pd
from glob import glob
from pathlib import Path

path = "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/"
tft = path + 'PMG_held_out_desc_top.csv'
ttg = path + 'PMG_held_out_targ_top.csv'
tfl = path + 'PMG_rdkit_full.csv'

tg = path + 'PMG_held_out_targ_trimmed.csv'
ft = path + "PMG_held_out_desc_trimmed.csv"
fl = path +  "PMG_rdkit_full.csv"

# %%
experiments = glob('/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/diff_pool/202*')
all_experiments = [str(Path(path).name) for path in experiments]


PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/diff_pool/'
)

PredictNewTestSet(
    feats=tft,
    targs=ttg,
    full_data=tfl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/diff_pool/'
)

# %%
experiments = glob('/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/mp_mu_hybrid/202*')
all_experiments = [str(Path(path).name) for path in experiments]
PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/mp_mu_hybrid/'
)

# %%
PredictNewTestSet(
    feats=tft,
    targs=ttg,
    full_data=tfl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/mp_mu_hybrid/'
)

# %%
experiments = glob('/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/rmp_rmu_hybrid/202*')
all_experiments = [str(Path(path).name) for path in experiments]
PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/rmp_rmu_hybrid/'
)

# %%
PredictNewTestSet(
    feats=tft,
    targs=ttg,
    full_data=tfl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=all_experiments,
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/rmp_rmu_hybrid/'
)

