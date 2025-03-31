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

# tg = path + 'PMG_held_out_targ_trimmed.csv'
# ft = path + "PMG_held_out_desc_trimmed.csv"
# fl = path +  "PMG_rdkit_full.csv"

# %%

PredictNewTestSet(
    feats=ft,
    targs=tg,
    full_data=fl,
    test_set_name = 'trimmed_held_out',
    experiment_ls=[
        "20241024_10_scramb_mp" ,  
        "20241024_10_scramb_r"  ,   
        "20241024_10_scramb_mpo"  ,
        "20241024_10_scramb_rmp"   ,
        "20241024_10_scramb_mu"   ,
        "20241024_10_scramb_rmpo" , 
        "20241105_50_scramb_mu"   ,
        "20241105_50_scramb_rmpo",
        "20241105_50_scramb_mp"  , 
        "20241105_50_scramb_r",
        "20241105_50_scramb_mpo"  ,
        "20241105_50_scramb_rmp",

],
    results_dir = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/scrambled/'
)

# %%
