# %%
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJ_DIR = Path(__file__).parent.parent.parent

sys.path.insert(0, PROJ_DIR / "scripts/run/")
from analysis_class import Analysis

sys.path.insert(0, PROJ_DIR / "scripts/misc/")
from misc_functions import (
    get_sel_mols_between_iters,
    molid_to_smiles,
    molid_ls_to_smiles,
    get_descs_for_molid,
    create_gif,
    )
results_dir=str(PROJ_DIR) + '/results/rdkit_desc/finished_results/10_mol_sel/'
an = Analysis(results_dir=results_dir, 
              held_out_stat_json="trimmed_held_out_test/trimmed_held_out_stats.json",
              docking_column='Experimental_pIC50'
              )
# %%
# an.Plot_Perf(
#     experiments=[
#         "average_10_mp",
#         "average_10_rmp",
#         "average_10_rmpo",
#         "average_10_mpo",
#         "average_10_r",
#         "average_10_mu",
#         "average_50_mp",
#         "average_50_rmp",
#         "average_50_rmpo",
#         "average_50_mpo",
#         "average_50_r",
#         "average_50_mu",
#     ],
#     plot_ho = False,
#     plot_int= True,
#     plot_chembl_int= False,
#     plot_fname='all_avg_int_plot_pear_r',
#     set_ylims=True,
#     r_type = 'pearson_r',
#     rmse_ylim=(0, 1),
#     sdep_ylim=(0, 1),
#     r2_ylim=(0, 1),
#     bias_ylim= (-0.1, 0.1),
# )

# %%

# chembl_feats = '/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/rdkit/ChEMBL_rdkit_desc_1.csv.gz'
# ho_feats = '/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_desc.csv'
# prediction = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/PMG_rdkit_desc*'

# an.PCA_Plot(train=chembl_feats,
#             validation=ho_feats,
#             prediction=prediction,
#             source_ls=['ChEMBL', 
#                        'Held_Out', 
#                        'PyMolGen'],
#             n_components=5,
#             plot_scatter=True,
#             plot_area=True,
#             save_extra_data=True,
#             plot_loadings=True,
#             kdep_sample_size=0.5,
#             kdep_sample_ls=['PyMolGen'])

# %%
# experiment_ls = ["20241002_10_mp", "20241002_10_mpo", '20241002_10_mu', '20241002_10_r', '20241002_10_rmp', '20241002_10_rmpo']
# mol_sel_ls = ['mp', 'mpo', 'mu', 'r', 'rmp', 'rmpo']
# for exp, sel in zip(experiment_ls, mol_sel_ls):
#     an.Prediction_Development(exp,
#                             prediction_fpath = "/trimmed_held_out_test/trimmed_held_out_preds.csv",
#                             true_path= "/users/yhb18174/Recreating_DMTA/datasets/held_out_data/PMG_held_out_targ_top.csv",
#                             iter_ls=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
#                             plot_filename=f"10_{sel}_it_preds_top",
#                             save_plot=True,
#                             underlay_it0=True)


# %%
# molids_a = get_sel_mols_between_iters(experiment_dir='/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/10_mol_sel/20241002_10_mp',
#                            start_iter = 1,
#                            end_iter= 150)
# molids_b = get_sel_mols_between_iters(experiment_dir='/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/10_mol_sel/20241002_10_mpo',
#                            start_iter = 1,
#                            end_iter= 150)

# smiles_a = molid_ls_to_smiles(molids_a, 'PMG-', "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")
# smiles_b = molid_ls_to_smiles(molids_b, 'PMG-', "/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")

# %%
# an.Tanimoto_Heat_Maps(smiles_a, smiles_b, molids_a, molids_b,
#                       plot_fname='mp_mpo_tanimoto_heatmap')

# %%
# an.MP_Avg_Tanimoto_Across_Iters(experiments=['20241002_10_mu',
#                                               '20241002_10_mp',
#                                               '20241002_10_r',
#                                               '20241002_10_rmp',
#                                               '20241002_10_rmpo',
#                                               '20241002_10_mpo',
#                                               '20241002_50_mu',
#                                               '20241002_50_mp',
#                                               '20241002_50_r',
#                                               '20241002_50_rmp',
#                                               '20241002_50_rmpo',
#                                               '20241002_50_mpo'])

# %%
# an.MP_Top_Preds_Analysis(
#     experiments=[
#         "20241002_10_mp",
#         "20241002_10_mpo",
#         "20241002_10_mu",
#         "20241002_10_r",
#         "20241002_10_rmp",
#         "20241002_10_rmpo",
#         "20241011_50_mp",
#         "20241011_50_mpo",
#         "20241011_50_mu",
#         "20241011_50_r",
#         "20241011_50_rmp",
#         "20241011_50_rmpo",
#     ],
#     ascending=False,
#     filename="Avg_Bottom_Preds_Plot",
# )

# %%
# an.Plot_MPO_Potency_Correlation(
    # full_data_fpath=f"{PROJ_DIR}/datasets/held_out_data/PMG_rdkit_full.csv",
    # preds_df_fpath=f"{PROJ_DIR}/datasets/held_out_data/PMG_held_out_targ_trimmed.csv",
    # save_fname='MPO_Aff_corr_ho_test',
    # save_plot=True
    # )
# %%
# an.Draw_Chosen_Mols(experiment = "20241002_10_mpo", iter_ls = [1, 25, 50, 75, 100, 125, 150],
#                     save_img = True, img_fname='mpo_mols')
# %%

#RMPO
# an.PCA_Plot_Across_Iters('20241002_10_rmpo', 
#                          iter_ls=[0, 10, 20, 30, 40, 50, 60, 70],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="rmpo_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# an.PCA_Plot_Across_Iters('20241002_10_rmpo', 
#                          iter_ls=[80, 90, 100, 110, 120, 130, 140, 150],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="rmpo_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )
# print('RMPO completed...')


# an.PCA_Plot_Across_Iters('20241002_10_mpo', 
#                          iter_ls=[0, 10, 20, 30, 40, 50, 60, 70],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="mpo_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# an.PCA_Plot_Across_Iters('20241002_10_mpo', 
#                          iter_ls=[80, 90, 100, 110, 120, 130, 140, 150],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="mpo_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# print('MPO completed...')

# an.PCA_Plot_Across_Iters('20241002_10_rmp', 
#                          iter_ls=[0, 10, 20, 30, 40, 50, 60, 70],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="rmp_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# an.PCA_Plot_Across_Iters('20241002_10_rmp', 
#                          iter_ls=[80, 90, 100, 110, 120, 130, 140, 150],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="rmp_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# print('RMP completed...')

# an.PCA_Plot_Across_Iters('20241002_10_mp', 
#                          iter_ls=[0, 10, 20, 30, 40, 50, 60, 70],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="mp_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# an.PCA_Plot_Across_Iters('20241002_10_mp', 
#                          iter_ls=[80, 90, 100, 110, 120, 130, 140, 150],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="mp_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )
# print('MP completed...')

# an.PCA_Plot_Across_Iters('20241002_10_mu', 
#                          iter_ls=[0, 10, 20, 30, 40, 50, 60, 70],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="mu_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# an.PCA_Plot_Across_Iters('20241002_10_mu', 
#                          iter_ls=[80, 90, 100, 110, 120, 130, 140, 150],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="mu_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# print('MU completed...')

# an.PCA_Plot_Across_Iters('20241002_10_r', 
#                          iter_ls=[0, 10, 20, 30, 40, 50, 60, 70],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="r_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# an.PCA_Plot_Across_Iters('20241002_10_r', 
#                          iter_ls=[80, 90, 100, 110, 120, 130, 140, 150],
#                          save_plot=True,
#                          n_rows=4,
#                          plot_fname="r_PCA_across_iters",
#                          n_components=5,
#                          indi_plot_suffix='RM_outliers',
#                          remove_outliers=True,
#                          use_multiprocessing=True
# )

# print('R completed...')


# %%

# data = pd.DataFrame()
an = Analysis(results_dir=results_dir, 
              held_out_stat_json="trimmed_held_out_test/trimmed_held_out_stats.json",
              docking_column='Experimental_pIC50'
              )
mpo = an.Scaffold_Analysis(experiment='20241002_10_mpo',
                     iter=0,
                     search_in_top=0,
                     plot_fname='chembl_test',
                     show_value='Experimental_pIC50',
                     save_data=False,
                     use_external_data=True,
                     external_data_path='/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/20_random_raw_ChEMBL_data.csv',
                     num_reoccurring_scaff=0,
                     num_total_scaff_shown_graph=20)

an = Analysis(results_dir=results_dir, 
              held_out_stat_json="trimmed_held_out_test/trimmed_held_out_stats.json",
              docking_column='Affinity(kcal/mol)'
              )
mpo = an.Scaffold_Analysis(experiment='20241002_10_mpo',
                     iter=0,
                     search_in_top=0,
                     plot_fname='chembl_test',
                     show_value='Affinity(kcal/mol)',
                     save_data=False,
                     use_external_data=True,
                     external_data_path='/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/dock/20_random_docked_ChEMBL_data.csv',
                     num_reoccurring_scaff=0,
                     num_total_scaff_shown_graph=20)
# %%
# mp = an.Scaffold_Analysis(experiment='20241002_10_mp',
#                      iter=150,
#                      search_in_top=10000,
#                      plot_fname='mp_scaff_ds_ranked',
#                      show_value='pred_Affinity(kcal/mol)',
#                      save_data=False)
# mp['Ref'] = 'MP'

# %%

# r = an.Scaffold_Analysis(experiment='20241002_10_r',
#                      iter=150,
#                      search_in_top=10000,
#                      plot_fname='r_scaff_ds_ranked',
#                      show_value='pred_Affinity(kcal/mol)',
#                      save_data=False)
# r['Ref'] = 'R'

# rmpo = an.Scaffold_Analysis(experiment='20241002_10_rmpo',
#                      iter=150,
#                      search_in_top=10000,
#                      plot_fname='rmpo_scaff_ds_ranked',
#                      show_value='pred_Affinity(kcal/mol)',
#                      save_data=False)
# rmpo['Ref'] = 'RMPO'

# rmp = an.Scaffold_Analysis(experiment='20241002_10_rmp',
#                      iter=150,
#                      search_in_top=10000,
#                      plot_fname='rmp_scaff_ds_ranked',
#                      show_value='pred_Affinity(kcal/mol)',
#                      save_data=False)
# rmp['Ref'] = 'RMP'

# mu = an.Scaffold_Analysis(experiment='20241002_10_mu',
#                      iter=150,
#                      search_in_top=10000,
#                      plot_fname='mu_scaff_ds_ranked',
#                      show_value='pred_Affinity(kcal/mol)',
#                      save_data=False)
# mu['Ref'] = 'MU'

# data = pd.concat([mpo, mp, r, rmpo, rmp, mu], axis=0)
# data.reset_index(drop=True, inplace=True)

# data.to_csv(f"{PROJ_DIR}/results/rdkit_desc/finished_results/10_mol_sel/scaff_data.csv", index='Ref')

# %%
# an.Loadings_Plot(loadings=f"{PROJ_DIR}/scripts/run/pca_loadings.csv")
# %%
# from rdkit import Chem

# scaff_data = pd.read_csv(f"{PROJ_DIR}/results/rdkit_desc/finished_results/10_mol_sel/scaff_data.csv", index_col='Unnamed: 0')
# murcko_smiles = scaff_data['Murcko_Scaffold'].tolist()
# murcko_mols = [Chem.MolFromSmiles(smi) for smi in murcko_smiles]

# %%
# path = '/users/yhb18174/Recreating_DMTA/results/rdkit_desc/plots'
# create_gif(
#     image_ls= [
#         path + '/PCA_20241002_10_mp_iter_0_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_10_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_20_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_30_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_40_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_50_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_60_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_70_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_80_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_90_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_100_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_110_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_120_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_130_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_140_RM_outliers.png',
#         path + '/PCA_20241002_10_mp_iter_150_RM_outliers.png',

#     ],
#     save_path = path,
#     save_fname='PCA_mp_development',
# )

# create_gif(
#     image_ls= [
#         path + '/PCA_20241002_10_mu_iter_0_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_10_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_20_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_30_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_40_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_50_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_60_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_70_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_80_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_90_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_100_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_110_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_120_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_130_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_140_RM_outliers.png',
#         path + '/PCA_20241002_10_mu_iter_150_RM_outliers.png',


#     ],
#     save_path = path,
#     save_fname='PCA_mu_development'
# )

# create_gif(
#     image_ls= [
#         path + '/PCA_20241002_10_r_iter_0_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_10_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_20_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_30_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_40_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_50_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_60_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_70_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_80_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_90_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_100_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_110_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_120_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_130_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_140_RM_outliers.png',
#         path + '/PCA_20241002_10_r_iter_150_RM_outliers.png',

#     ],
#     save_path = path,
#     save_fname='PCA_r_development'
# )

# create_gif(
#     image_ls= [
#         path + '/PCA_20241002_10_rmp_iter_0_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_10_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_20_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_30_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_40_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_50_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_60_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_70_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_80_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_90_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_100_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_110_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_120_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_130_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_140_RM_outliers.png',
#         path + '/PCA_20241002_10_rmp_iter_150_RM_outliers.png',
#     ],
#     save_path = path,
#     save_fname='PCA_rmp_development'
# )

# create_gif(
#     image_ls= [
#         path + '/PCA_20241002_10_rmpo_iter_0_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_10_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_20_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_30_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_40_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_50_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_60_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_70_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_80_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_90_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_100_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_110_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_120_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_130_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_140_RM_outliers.png',
#         path + '/PCA_20241002_10_rmpo_iter_150_RM_outliers.png',
#     ],
#     save_path = path,
#     save_fname='PCA_rmpo_development'
# )

# create_gif(
#     image_ls= [
#         path + '/PCA_20241002_10_mpo_iter_0_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_10_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_20_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_30_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_40_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_50_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_60_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_70_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_80_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_90_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_100_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_110_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_120_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_130_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_140_RM_outliers.png',
#         path + '/PCA_20241002_10_mpo_iter_150_RM_outliers.png',
#     ],
#     save_path = path,
#     save_fname='PCA_mpo_development'
# )
# %%

# an.Dock_Top_Pred(experiment='20241002_10_mp',
#                  iter=150)
# %%
# an.Plot_Top_Pred_Docked(experiment_ls=['20241002_10_mp', '20241002_10_mpo',
#                                        '20241002_10_rmp', '20241002_10_rmpo',
#                                        '20241002_10_mu', '20241002_10_r'],
#                         iter=150,
#                         save_plot=True,
#                         save_structures=True,
#                         search_in_top=50,
#                         plot_name="Top_50_pred_docked_boxplot")

# %%
# df = pd.read_csv(f"{PROJ_DIR}/results/rdkit_desc/plots/20241002_10__it150_structs.csv", index_col='ID')
# df.sort_values(by='Affinity(kcal/mol)', ascending=True)
# %%
an.UniqueFragCount(experiment_ls=['20241002_10_mp'], max_iter=3)
# %%