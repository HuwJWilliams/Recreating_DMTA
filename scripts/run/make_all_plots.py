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
results_dir=str(PROJ_DIR) + '/results/rdkit_desc/'
ho_an = Analysis(results_dir=results_dir, 
            #   held_out_stat_json="trimmed_held_out_test/trimmed_held_out_stats.json",
             held_out_stat_json="held_out_test/held_out_stats.json",

              docking_column='Affinity(kcal/mol)'
              )

trho_an=Analysis(results_dir=results_dir, 
             held_out_stat_json="trimmed_held_out_test/trimmed_held_out_stats.json",
            # held_out_stat_json="held_out_test/held_out_stats.json",

              docking_column='Affinity(kcal/mol)'
              )

# %%
plot_ls_ls = [
[
        "average_50_rmp",
        "average_50_mp",
        "average_50_rmpo",
        "average_50_mpo",
        "average_50_r",
        "average_50_mu",
        "average_50_rmu",
        "average_10_rmp",
        "average_10_mp",
        "average_10_rmpo",
        "average_10_mpo",
        "average_10_r",
        "average_10_mu",
        "average_50_rmu",
],
[
        "average_50_mp_mu_2:8",
        "average_50_mp_mu_5:5",
        "average_50_mp_mu_8:2", 
        "average_50_mu",
        "average_50_mp",
],
[
        "average_50_rmp_rmu_2:8",
        "average_50_rmp_rmu_5:5",
        "average_50_rmp_rmu_8:2",
        "average_50_rmp",
        "average_50_rmu",
],
[
        "average_50_mp",
        "average_50_rmp_0025",
        "average_50_rmp_005",
        "average_50_rmp",
        "average_50_rmp_025",
        "average_50_rmp_05",
        "average_50_r",
]]
plot_ls_ref = ["sing", "mp_mu", "rmp_rmu", "diff_pool"]

#%%
for ref, plot_ls in zip(plot_ls_ref, plot_ls_ls):
    break
    ho_an.Plot_Perf(
        experiments=plot_ls,
        plot_ho =True,
        plot_int= False,
        plot_chembl_int= False,
        plot_fname=f'{ref}_ho_perf_plot',
        set_ylims=True,
        r_type = 'pearson_r',
        rmse_ylim=(0.25, 1.25),
        sdep_ylim=(0.2, 0.6),
        r2_ylim=(-0.2, 1),
        bias_ylim= (-1.2, 0.1),
        # r2_ylim=(-0.2, 0.8), #scrambled
        # bias_ylim= (-0.5, 0.5), #scrambled
        yticks=4,
        custom_xticks=[0, 500, 1000, 1500],
        tick_fontsize=20,
        label_fontsize=24,
        title_fontsize=20,
        legend_fontsize=20,
        save_plot=True,
        )

    ho_an.Plot_Perf(
        experiments=plot_ls,
        plot_ho =False,
        plot_int= True,
        plot_chembl_int= False,
        plot_fname=f'{ref}_int_perf_plot',
        set_ylims=True,
        r_type = 'pearson_r',
        rmse_ylim=(0.25, 1.25),
        sdep_ylim=(0.2, 0.6),
        r2_ylim=(-0.2, 1),
        bias_ylim= (-1.2, 0.1),
        yticks=4,
        custom_xticks=[0, 500, 1000, 1500],
        tick_fontsize=20,
        label_fontsize=24,
        title_fontsize=20,
        legend_fontsize=20,
        save_plot=True,
        )

    trho_an.Plot_Perf(
        experiments=plot_ls,
        plot_ho =True,
        plot_int= False,
        plot_chembl_int= False,
        plot_fname=f'{ref}_trho_perf_plot',
        set_ylims=True,
        r_type = 'pearson_r',
        rmse_ylim=(0.25, 1.25),
        sdep_ylim=(0.2, 0.6),
        r2_ylim=(-0.2, 1),
        bias_ylim= (-1.2, 0.1),
        yticks=4,
        custom_xticks=[0, 500, 1000, 1500],
        tick_fontsize=20,
        label_fontsize=24,
        title_fontsize=20,
        legend_fontsize=20,
        save_plot=True,
        )

# %%
    
results_dir=str(PROJ_DIR) + '/results/rdkit_desc/complete_archive/50_sel/'
an = Analysis(results_dir=results_dir, 
              held_out_stat_json="held_out_test/held_out_stats.json",
              docking_column='Affinity(kcal/mol)'
              )

# %%
# an.UniqueFragCountGrouped(suffix_ls=['_mp',
#                                         "_mpo",
#                                         "_mu",
#                                         "_rmp",
#                                         "_rmpo",
#                                         "_rmu",
#                                         "_r"], 
#                                         max_iter=30,
#                                      save_plot=True)
# %%
# an.UncertaintyChecker(experiment_ls=['average_50_mu'],
#                                      iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
#                                      save_plot=True,
#                                      plot_name='uncertainty_checker_50_mu')
# an.UncertaintyChecker(experiment_ls=['average_50_mp'],
#                                      iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
#                                      save_plot=True,
#                                      plot_name='uncertainty_checker_50_mp')
# an.UncertaintyChecker(experiment_ls=['average_50_rmp'],
#                                      iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
#                                      save_plot=True,
#                                      plot_name='uncertainty_checker_50_rmp')
# %%

an.PlotFeatureImportanceAndRidgelines(experiment='average_50_mp',
                               iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                               )

# %%
an.PlotFeatureImportanceAndRidgelines(experiment='average_50_rmp',
                               iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                               )
# %%
an.PlotFeatureImportanceAndRidgelines(experiment='average_50_mu',
                               iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                               )

an.PlotFeatureImportanceAndRidgelines(experiment='average_50_mp_mu_2:8',
                               iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                               )

an.PlotFeatureImportanceAndRidgelines(experiment='average_50_mp_mu_5:5',
                               iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                               )

an.PlotFeatureImportanceAndRidgelines(experiment='average_50_mp_mu_8:2',
                               iter_ls=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                               )

# %%