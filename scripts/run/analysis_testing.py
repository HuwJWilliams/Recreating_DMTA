# %%
import sys
from pathlib import Path
import pandas as pd

PROJ_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, PROJ_DIR / "scripts/run/")
from analysis_class import Analysis

sys.path.insert(0, PROJ_DIR / "scripts/misc/")
from misc_functions import (
    get_sel_mols_between_iters,
    molid_to_smiles,
    molid_ls_to_smiles,
)
results_dir=str(PROJ_DIR) + '/results/rdkit_desc/'
an = Analysis(results_dir=results_dir)

# %%
an.Plot_Perf(
    experiments=[
        "average_50_mp",
        "average_50_rmp",
        "average_50_rmpo",
        "average_50_mpo",
        "average_50_r",
        "average_50_mu",
    ],
    plot_int = True,
    plot_fname='int_plot',
)

# %%
an.Plot_Perf(
    experiments=[
        "average_10_mp",
        "average_10_rmp",
        "average_10_rmpo",
        "average_10_mpo",
        "average_10_r",
        "average_10_mu",
        "average_50_mp",
        "average_50_rmp",
        "average_50_rmpo",
        "average_50_mpo",
        "average_50_r",
        "average_50_mu",
    ],
    plot_ho = False,
    plot_int= True,
    plot_chembl_int= False,
    plot_tr_ho= False,
    plot_fname='all_avg_int_plot_cod_r',
    set_ylims=True,
    r_type = 'r2',
    # rmse_ylim=(1.2, 0),
    # sdep_ylim=(-0.2, 0.5),
    # r2_ylim=(-0, 1),
    bias_ylim= (-0.6, 0.2),
)
# %%
an.Plot_Perf(
    experiments=[
        "average_50_mp",
        "average_50_rmp",
        "average_50_rmpo",
        "average_50_mpo",
        "average_50_r",
        "average_50_mu",
    ],
    plot_tr_ho = True,
    set_ylims=False,
    plot_fname='tr_ho_plot',
    r_type = 'r2'
)
# %%

an.Plot_Perf(
    experiments=[
        "average_10_r",
        "average_10_rmp",
        "average_10_rmpo",
        "average_10_mp",
        "average_10_mpo",
        "average_10_mu",
    ],
    plot_chembl_int= True,
    plot_fname='chembl_plot'
)

# %%

# chembl_feats = '/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/rdkit/ChEMBL_rdkit_desc_1.csv.gz'
# ho_feats = '/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_desc.csv'
# prediction = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/PMG_rdkit_desc*'

# an.PCA_Plot(train=chembl_feats,
#             validation=ho_feats,
#             prediction=prediction,
#             source_ls=['ChEMBL', 'Held_Out', 'PyMolGen'])

# %%
# an.Prediction_Development("20240910_10_",
#                           iter_ls=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
#                           plot_filename="10_mp_it_preds",
#                           save_plot=False,
#                           underlay_it0=True)


# %%
# molids_a = get_sel_mols_between_iters(experiment_dir='/users/yhb18174/Recreating_DMTA/results/rdkit_desc/20240910_10_mp',
#                            start_iter = 1,
#                            end_iter= 1)
# molids_b = get_sel_mols_between_iters(experiment_dir='/users/yhb18174/Recreating_DMTA/results/rdkit_desc/20240910_10_mp',
#                            start_iter = 2,
#                            end_iter= 2)

# smiles_a = molid_ls_to_smiles(molids_a, 'PMG-', "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")
# smiles_b = molid_ls_to_smiles(molids_b, 'PMG-', "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")

# %%
# an.Tanimoto_Heat_Maps(smiles_a, smiles_b, molids_a, molids_b)

# %%
# an.Avg_Tanimoto_Avg_Across_Iters(experiments=['20240910_10_mu',
#                                               '20240910_10_mp',
#                                               '20240910_10_r',
#                                               '20240910_10_rmp',
#                                               '20240910_10_rmpo',
#                                               '20240910_10_mpo',
#                                               '20240916_50_mu',
#                                               '20240916_50_mp',
#                                               '20240916_50_r',
#                                               '20240916_50_rmp',
#                                               '20240916_50_rmpo',
#                                               '20240916_50_mpo'])

# an.MP_Avg_Tanimoto_Across_Iters(experiments=['20240910_10_mu',
#                                               '20240910_10_mp',
#                                               '20240910_10_r',
#                                               '20240910_10_rmp',
#                                               '20240910_10_rmpo',
#                                               '20240910_10_mpo',
#                                               '20240916_50_mu',
#                                               '20240916_50_mp',
#                                               '20240916_50_r',
#                                               '20240916_50_rmp',
#                                               '20240916_50_rmpo',
#                                               '20240916_50_mpo'])

# %%
# an.MP_Top_Preds_Analysis(
#     experiments=[
#         "20240910_10_mu",
#         "20240910_10_mp",
#         "20240910_10_r",
#         "20240910_10_rmp",
#         "20240910_10_rmpo",
#         "20240910_10_mpo",
#         "20240916_50_mu",
#         "20240916_50_mp",
#         "20240916_50_r",
#         "20240916_50_rmp",
#         "20240916_50_rmpo",
#         "20240916_50_mpo",
#     ],
#     ascending=False,
#     filename="Avg_Bottom_Preds_Plot",
# )

# an.MP_Top_Preds_Analysis(
#     experiments=[
#         "20240910_10_mu",
#         "20240910_10_mp",
#         "20240910_10_r",
#         "20240910_10_rmp",
#         "20240910_10_rmpo",
#         "20240910_10_mpo",
#         "20240916_50_mu",
#         "20240916_50_mp",
#         "20240916_50_r",
#         "20240916_50_rmp",
#         "20240916_50_rmpo",
#         "20240916_50_mpo",
#     ],
#     ascending=True,
#     filename="Avg_Top_Preds_Plot",
# )


# %%
