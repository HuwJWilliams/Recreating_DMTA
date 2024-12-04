import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import json
from glob import glob
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import subprocess
from functools import partial
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import math
from PIL import Image as PILImage
from io import BytesIO
from scipy.spatial import ConvexHull
import random as rand
import time

from rdkit.DataStructs import FingerprintSimilarity
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

import colorcet as cc

FILE_DIR = Path(__file__).parent
PROJ_DIR = Path(__file__).parent.parent.parent

# Misc
sys.path.insert(0, str(PROJ_DIR) + "/scripts/misc/")
from misc_functions import (
    molid2batchno,
    count_number_iters,
    count_conformations,
    get_chembl_molid_smi,
    get_sel_mols_between_iters,
    molid_ls_to_smiles,
    get_descs_for_molid,
    fig2img,
    get_top,
    WaitForJobs
)

# Docking
sys.path.insert(0, str(PROJ_DIR) + "/scripts/docking/")
from docking_fns import (
    WaitForDocking,
    Run_GNINA,
    GetUndocked
)

# Dataset
sys.path.insert(0, str(PROJ_DIR) + "/scripts/dataset/")
from dataset_functions import Dataset_Accessor


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

def CalcStats(
    df: pd.DataFrame,
    true_col: str = "Affinity(kcal/mol)",
    pred_col: str = "pred_Affinity(kcal/mol)",
):

    # Extract True and Predicted Values from DF
    true = df[true_col].values
    pred = df[pred_col].values

    # Pearsons r^2
    pearson_r, _ = pearsonr(true, pred)
    pearson_r2 = pearson_r**2

    # Coefficient of Determination (R^2)
    cod = r2_score(true, pred)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(true, pred))

    # Bias (Systematic Error / Mean Error)
    bias = np.mean(true - pred)

    # SDEP (Standard Deviation of the Error of Prediction / Random Error)
    sdep = np.sqrt(np.mean((true - pred - bias) ** 2))

    return (true, pred, pearson_r2, cod, rmse, bias, sdep)


class Analysis:
    def __init__(
        self,
        results_dir = str(PROJ_DIR) + '/results/rdkit_desc/',
        rdkit_or_mordred: str = "rdkit",
        held_out_stat_json: str = "/rerun_held_out_test/rerun_held_out_stats.json",
        docking_column: str = "Affinity(kcal/mol)",
    ):
        """
        Description
        -----------
        Class to carry out the Recreating_DMTA workflow analysis

        Parameters
        ----------
        rdkit_or_mordred (str)      Value to set the working results directory, set as 'rdkit' or 'mordred'
        held_out_stats_json (str)   Name of the json file containing the performance stats on the held out
                                    test set
        docking_column (str)        Name of column which the docking scores are saved under

        Returns
        -------
        Initialised analysis class
        """
        global PROJ_DIR

        self.rdkit_or_mordred = rdkit_or_mordred.lower()

        self.results_dir = results_dir
        self.results_10_dir = self.results_dir + '/finished_results/10_mol_sel/'
        self.results_50_dir = self.results_dir + '/finished_results/50_mol_sel/'

        self.held_out_stat_json = held_out_stat_json

        self.docking_column = docking_column

    def _get_stats(
        self,
        experiment_dirs: list = [],
        perf_stats_json: str = "performance_stats.json",
    ):
        """
        Description
        -----------
        Function to get the performance statistics for all experiments on both the internal and held out tests

        Parameters
        ----------
        experiment_dirs (list)      List of experiment names e.g., [20240910_10_r, 20241012_10_r]

        Returns
        -------
        Dictionary containing all available iteration statistics on internal and held out tests for each given expetiment
        """

        all_stats = {}

        # Looping through all provided experiments
        for exp in experiment_dirs:
            
            if "_50_" in exp:
                step = 50
                results_dir = self.results_50_dir
            else:
                step = 10
                results_dir = self.results_10_dir

            # Initialising empty lists
            rmse = []
            r2 = []
            bias = []
            sdep = []
            pearson = []

            no_mols_ls = []

            # Defining the working directory
            working_dir = results_dir + '/' + exp

            # For each iteration obtain and save the statistics data
            # If this doesnt work change back to (0, cnt_n_iters())
            for n in range(0, count_number_iters(working_dir)):
                no_mols_ls.append(n * step)

                stats_path = f"{working_dir}/it{n}/{perf_stats_json}"

                try:
                    with open(stats_path, "r") as perf_stats:
                        data = json.load(perf_stats)

                    rmse.append(round(float(data.get("RMSE", 0)), 3))
                    r2.append(round(float(data.get('r2', 0)), 3))
                    bias.append(round(float(data.get("Bias", 0)), 3))
                    sdep.append(round(float(data.get("SDEP", 0)), 3))
                    pearson.append(round(float(data.get('pearson_r', 0)), 3))


                except Exception as e:
                    print(e)

            # Format the statistics data
            all_stats[exp] = {
                "n_mols": no_mols_ls,
                "rmse": rmse,
                "r2": r2,
                "bias": bias,
                "sdep": sdep,
                "pearson_r": pearson
            }

        return all_stats

    def Plot_Perf(
        self,
        experiments: list,
        save_plot: bool = True,
        results_dir: str = f"{PROJ_DIR}/results/rdkit_desc/",
        plot_fname: str = "Perf_Plot",
        plot_int: bool = False,
        plot_ho: bool = False,
        plot_chembl_int: bool = False,
        set_ylims: bool = True,
        r2_ylim: tuple = (-1, 1),
        bias_ylim: tuple = (-0.5, 0.5),
        rmse_ylim: tuple = (0, 1),
        sdep_ylim: tuple = (0, 1),
        r_type: str = 'r2'
    ):

        # Load performance stats for the selected datasets
        all_int_stats = (
            self._get_stats(
                experiment_dirs=experiments, perf_stats_json="performance_stats.json",
            )
            if plot_int
            else None
        )

        all_ho_stats = (
            self._get_stats(
                experiment_dirs=experiments,
                perf_stats_json=self.held_out_stat_json,
            )
            if plot_ho
            else None
        )

        all_chembl_stats = (
            self._get_stats(
                experiment_dirs=experiments,
                perf_stats_json="chembl_performance_stats.json",
            )
            if plot_chembl_int
            else None
        )

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        colours = sns.color_palette(cc.glasbey, n_colors=20)
        method_color_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }

        linestyles = {"_10_": "-", "_50_": "--"}

        # Determine the maximum length of metrics from any of the datasets to pad missing data
        max_length = 0
        for stats in [all_int_stats, all_ho_stats, all_chembl_stats]:
            if stats:
                max_length = max(
                    max_length, max(len(s["rmse"]) for s in stats.values())
                )
        
        colour_ls = []
        for exp in experiments:
            step = 50 if "_50_" in exp else 10
            name = exp.split("_")[-1]
            exp_name = f"_{name}" 
            method = next((m for m in method_color_map.keys() if exp.endswith(m)), None)
            colour = method_color_map.get(method, "black")
            colour_ls.append(colour_ls)
            linestyle = linestyles.get("_50_" if "_50_" in exp else "_10_", "-")

            print(f"Experiment: {exp}, Color: {colour}, Line Style: {linestyle}")

            def plot_metric(ax, stats, metric, linestyle, color, label=None):
                padded_data = np.pad(
                    stats[exp][metric],
                    (0, max_length - len(stats[exp][metric])),
                    constant_values=np.nan,
                )
                sns.lineplot(
                    x=list(range(0, max_length * step, step)),
                    y=padded_data,
                    legend=False,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                    ax=ax,
                )

            if plot_int:
                plot_metric(
                    ax[0, 0],
                    all_int_stats,
                    "rmse",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_int_stats,
                    r_type,
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_int_stats,
                    "bias",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_int_stats,
                    "sdep",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )

            if plot_ho:
                plot_metric(
                    ax[0, 0],
                    all_ho_stats,
                    "rmse",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_ho_stats,
                    r_type,
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_ho_stats,
                    "bias",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_ho_stats,
                    "sdep",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )

            if plot_chembl_int:
                plot_metric(
                    ax[0, 0],
                    all_chembl_stats,
                    "rmse",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 0],
                    all_chembl_stats,
                    r_type,
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[1, 1],
                    all_chembl_stats,
                    "bias",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )
                plot_metric(
                    ax[0, 1],
                    all_chembl_stats,
                    "sdep",
                    linestyle=linestyle,
                    color=colour,
                    label=exp_name,
                )

            ax[0, 0].set_title("RMSE")
            ax[0, 0].set_ylabel("RMSE")

            ax[1, 0].set_title(r_type)
            ax[1, 0].set_ylabel(r_type)

            ax[1, 1].set_title("Bias")
            ax[1, 1].set_ylabel("Bias")

            ax[0, 1].set_title("SDEP")
            ax[0, 1].set_ylabel("SDEP")

            if set_ylims:
                ax[0, 0].set_ylim(rmse_ylim[0], rmse_ylim[1])
                ax[1, 0].set_ylim(r2_ylim[0], r2_ylim[1])
                ax[1, 1].set_ylim(bias_ylim[0], bias_ylim[1])
                ax[0, 1].set_ylim(sdep_ylim[0], sdep_ylim[1])

            for a in ax.flat:
                a.set_xlabel("Molecule Count")

        lines = [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        leg1 = fig.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            ncol=1,
            borderaxespad=0.0,
        )


        exp_names = [e.split("_")[-1] for e in experiments]
        labels = []
        for e in exp_names:
            name = f"_{e}"
            if name not in labels:
                labels.append(name)

        print(labels)
        handles = []
        colour_ls = []
        for label in labels:
            colour = method_color_map[label]
            colour_ls.append(colour)
            if colour:
                handle = Line2D([0], [0], color=colour, lw=2)
                handles.append(handle)
            else:
                print(f"Warning: No color found for label {label}")

        leg2 = fig.legend(
            handles,
            [label.lstrip('_') for label in labels],
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )

        fig.add_artist(leg1)
        fig.add_artist(leg2)

        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_plot:
            plt.savefig(results_dir + "/plots/" + plot_fname + ".png", dpi=600)

        plt.show()

    def PCA_Plot(
        self,
        train: str,
        prediction: str,
        source_ls: list,
        validation: str=None,
        n_components: int = 5,
        loadings_filename: str = "pca_loadings",
        pca_df_filename: str = "pca_components",
        kdep_sample_size: float = 0.33,
        contamination: float = 0.00001,
        plot_fname: str = "PCA_Plot",
        save_plot: bool = True,
        save_extra_data: bool=False,
        save_fpath: str=f"{PROJ_DIR}/results/rdkit_desc/plots/",
        plot_area: bool=False,
        plot_scatter: bool=True,
        random_seed: int=None,
        plot_loadings: bool=False,
        plot_title: str='PCA Plot',
        remove_outliers: bool=True, 
    ):
        """
        Description
        -----------
        Function to do a PCA analysis on the training, validation and prediction molecule sets

        Parameters
        ----------
        train (str)                 File pathway to the features used to train ML models
        validation (str)            File pathway to the features used in a validation/held out test set
        prediction (str)            File pathway to the features used to make predictions
                                    (this requires a general pathway with * to replace batch numbers
                                    e.g., "/path/to/desc/PMG_rdkit_desc_*" )
        source_ls (list)            List of the datasets comparing. (e.g., ChEMBL, Held Out, PyMolGen)
        n_components (int)          Number of principal components to create and plot
        loadings_filename (str)     Name to save the loadings DataFrame under
        pca_df_filename (str)       Name to save the PC DataFrame under
        plot_fname (str)         Name to save the PCA plots under
        kdep_sample_size (float)    Set to decimal for the size of the sample to do the
                                    Kernel Density Estimate Plot from (0.33 = 33 %)
        contamination (float)       Fracion of outlying molecules to remove from the PCA data

        Returns
        -------
        Plot of a n_components x n_components PCA scatter plot.
        """

        if random_seed is None:
            random_seed = rand.randint(0, 2**31)

        # Reading in the training data. Setting the name of the data to colour plot by
        if isinstance(train, str):
            train_df = pd.read_csv(train, index_col="ID")
        else:
            train_df = train
        used_cols = train_df.columns
        train_df["Source"] = source_ls[0]

        # Reading in the validation data. Setting the name of the data to colour plot by
        if validation is not None:
            if isinstance(train, str):
                validation_df = pd.read_csv(validation, index_col="ID")
            else:
                validation_df = validation

            validation_df = validation_df[used_cols]
            validation_df["Source"] = source_ls[1]
        else:
            validation_df = pd.DataFrame()

        # Reading in the prediction data. Setting the name of the data to colour plot by
        prediction_df = pd.DataFrame()
        files = glob(prediction)
        for file in files:
            df = pd.read_csv(file, index_col="ID")
            df = df[used_cols]
            prediction_df = pd.concat([prediction_df, df], axis=0)
        prediction_df["Source"] = source_ls[-1]

        # Making a dictionary for the data and sorting by data length
        # This means that the larger data sets are at the bottom and dont cover smaller ones
        df_dict = {
            "train": train_df,
            "validation": validation_df,
            "prediction": prediction_df,
        }
        sorted_dfs = sorted(df_dict.items(), key=lambda x: len(x[1]), reverse=True)
        combined_df = pd.concat([df for _, df in sorted_dfs], axis=0)

        # Scaling the data between 0 and 1
        scaler = StandardScaler()
        scaled_combined_df = combined_df.copy().dropna()
        scaled_combined_df[used_cols] = scaler.fit_transform(combined_df[used_cols])

        # Doing the PCA on the data
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_combined_df[used_cols])
        explained_variance = pca.explained_variance_ratio_ * 100

        # Isolating the loadings for each principal components and labelling the associated features
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings, columns=[f"PC{i+1}" for i in range(n_components)], index=used_cols
        )
        
        abs_loadings_df = loadings_df.abs()
        abs_loadings_df.rename_axis("Features", inplace=True)
        
        loadings_df.rename_axis("Features", inplace=True)

        if save_extra_data:
            loadings_df.to_csv(
                f"{PROJ_DIR}/scripts/run/{loadings_filename}.csv", index_label="Features"
            )
            abs_loadings_df.to_csv(
                f"{PROJ_DIR}/scripts/run/{loadings_filename}_abs.csv", index_label="Features"
            )
        

        # Creating a DataFrame for the principal component results. Saves to .csv
        pca_df = pd.DataFrame(
            principal_components,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=combined_df.index,
        )
        pca_df["Source"] = combined_df["Source"].values
        if save_extra_data:
            pca_df.to_csv(f"{PROJ_DIR}/scripts/run/{pca_df_filename}.csv.gz", index_label="ID", compression='gzip')

        # Removing outlying molecules from the PCA data
        def remove_outliers(df, columns, n_neighbors=20, contamination=contamination):
            """
            Description
            -----------
            Function to remove outlying molecules from a DataFrame

            Parameters
            ----------
            df (pd.DataFrame)       DataFrame from which you wish to remove the outlying molecules
            columns (list)          Columns you want to consider when defining outliers
            contamination (float)   Fraction of outlying points you wish to remove from the dataset

            Returns
            -------
            New DataFrame with outlying molecules removed

            """
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination
            )
            outlier_labels = lof.fit_predict(df[columns])
            return df[outlier_labels == 1]

        if remove_outliers:
            pca_df = remove_outliers(pca_df, [f"PC{i+1}" for i in range(n_components)])

        # Initialise PCA subplots
        fig, axs = plt.subplots(
            nrows=n_components, ncols=n_components, figsize=(15, 15)
        )

        # Filling in the subplots
        for i in range(n_components):
            for j in range(n_components):

                # If not on the diagonal, make a scatter plot for the PCA overlap
                if i != j:
                    
                    if plot_scatter:
                        sns.scatterplot(
                                x=f"PC{j+1}",
                                y=f"PC{i+1}",
                                hue="Source",
                                data=pca_df,
                                ax=axs[i, j], 
                                legend=False,
                                edgecolor="none",
                                palette="dark",
                            )
                    
                    dark_colours = sns.color_palette('dark')
                    for idx, source in enumerate(pca_df['Source'].unique()):

                        source_data = pca_df[pca_df['Source'] == source]
                        area_colour = dark_colours[idx]

                                            
                        if plot_area:
                            # Calculate convex hull of points
                            points = source_data[[f"PC{j+1}", f"PC{i+1}"]].values
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]

                            # Close the polygon by appending the first point
                            hull_points = np.vstack((hull_points, hull_points[0]))
                            
                            # Plot the area
                            axs[i, j].fill(
                                hull_points[:, 0],
                                hull_points[:, 1],
                                alpha=0.2,  # Transparency for the filled area
                                color=area_colour,
                                edgecolor = area_colour,
                                linewidth=2,
                                label=f"{source} area"
                            )


                    axs[i, j].set_xlabel(f"PC{j+1}")
                    axs[i, j].set_ylabel(f"PC{i+1}")

                    # Remove x and y plots from every PCA plot apart from the left and bottom most plots
                    if i != n_components - 1:
                        axs[i, j].set_xlabel("")
                        axs[i, j].set_xticks([])

                    if j != 0:
                        axs[i, j].set_ylabel("")
                        axs[i, j].set_yticks([])

                # If on the diagonal, make the Kernel Density Estimate Plots for each Principal Component
                else:
                    # Because this is slow, you can take a sample of the principal component data rather than using the full data
                    src1_data = pca_df[pca_df["Source"] == source_ls[0]]
                    subset_src1_data = src1_data.sample(
                        n=int(len(src1_data) * kdep_sample_size),
                        random_state=random_seed
                    )

                    src2_data = pca_df[pca_df["Source"] == source_ls[1]]
                    subset_src2_data = src2_data.sample(
                        n=int(len(src2_data) * kdep_sample_size),
                        random_state=random_seed

                    )
                    try:
                        src3_data = pca_df[pca_df["Source"] == source_ls[2]]
                        subset_src3_data = src3_data.sample(
                            n=int(len(src3_data) * kdep_sample_size),
                            random_state=random_seed

                        )
                    except IndexError:
                        subset_src3_data = pd.DataFrame()        

                    sampled_pca_df = pd.concat(
                        [subset_src1_data, subset_src2_data, subset_src3_data], axis=0
                    )

                    # Making the Kernel Density Estimate Plot
                    sns.kdeplot(
                        x=f"PC{i+1}",
                        hue="Source",
                        data=sampled_pca_df,
                        common_norm=False,
                        fill=True,
                        ax=axs[i, i],
                        legend=False,
                        palette="dark",
                    )

                    axs[i, i].set_xlabel("")
                    axs[i, i].set_ylabel("Density Estimate")

                # Adjusting labels and titles, including the variance for each principal component
                if i == n_components - 1:
                    axs[i, j].set_xlabel(
                        f"PC{j+1} ({explained_variance[j]:.2f}% Variance)"
                    )
                if j == 0:
                    axs[i, j].set_ylabel(
                        f"PC{i+1} ({explained_variance[i]:.2f}% Variance)"
                    )

        # Define handles and labels for the legend
        fig = plt.gcf()  # Get current figure
        ax = axs[0, 0]  # Use first subplot as reference

        # Create custom legend handles
        custom_handles = []
        custom_labels = []
        dark_colours = sns.color_palette('dark')

        for idx, source in enumerate(pca_df['Source'].unique()):
            # Scatter point handle
            scatter_handle = plt.scatter([], [], 
                color=dark_colours[idx], 
                edgecolor='black', 
                linewidth=0.5, 
                marker='o', 
                label=source
            )
            
            # Area patch handle
            area_handle = plt.Rectangle((0,0), 1, 1, 
                color=dark_colours[idx], 
                alpha=0.2, 
                label=f"{source} area"
            )
            
            custom_handles.extend([scatter_handle, area_handle])
            custom_labels.extend([source, f"{source} area"])

        # Place legend
        fig.legend(
            custom_handles, 
            custom_labels,
            loc="center left", 
            bbox_to_anchor=(0.75, 0.95), 
            ncol=1,
            borderaxespad=0.0
        )

        fig.suptitle(plot_title, fontsize=16, y=0.98)

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        if save_plot:
            plt.savefig(
                save_fpath  + plot_fname + ".png",
                dpi=600,
                bbox_inches='tight'
            )

        if plot_loadings:
            loadings_df['Max'] = loadings_df.max(axis=1)
            loadings_df = loadings[loadings['Max'] > 0.3]
            loadings_df.drop(columns=['Max'])

            fig, ax = plt.subplots(n_components, 1, figsize=(25,25), sharex=True)

            for n in range(1, n_components + 1):
                sns.barplot(x=loadings_df.index, y=loadings[f'PC{n}'], ax=ax[n-1])
                ax[n-1].set_ylabel(f"PC{n} Loadings")

            ax[n-1].set_xticklabels(range(1, len(loadings_df) + 1), rotation=90)

            # Create a legend mapping numbers to feature names
            legend_labels = [f"{i+1}: {feature}" for i, feature in enumerate(loadings_df.index)]
            fig.legend(legend_labels, loc="center right", title="Feature Legend", fontsize=10)

            # Add a shared xlabel
            fig.supxlabel("Features (Mapped to Index Numbers)")
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(right=0.85)  # Leave space for the legend
            plt.savefig(
                save_fpath + plot_fname + '_loadings.png',
                dpi=600,
                )


        return fig

    def Conformer_Analysis(
        self,
        docking_dir: str = f"{PROJ_DIR}/docking/PyMolGen/",
        sample_size: int = 15,
        conf_gen_plot: bool = True,
        score_convergence: bool = True,
    ):
        """
        Description
        -----------
        Function to look into the conformer generation process. Plots the number of conformers generated for a sample of molecules
        along with the number of docked conformers. Can also plot the highest docking score by iteration to see how far through the
        conformer search are we finding the best docking scores.

        Parameters
        ---------
        docking_dir (str)           Directory to find all of the docking .tar.gz files in
        sample_size (int)           Number of molecules to consider at any given time
        conf_gen_plot (bool)        Flag to plot the number of conformers made (and docked) for each molecule
        score_convergence (bool)    Flag to plot the convergence of highest docking scores by iteration

        Returns
        -------
        None
        """

        # Initiailising empty lists to be used
        n_confs_ls = []
        n_docked_ls = []
        molid_ls = []
        scores_ls = []
        pose_ls = []

        # Obtaining all of the molid files available
        tar_files = glob(docking_dir + "PMG*.tar.gz")

        # Taking a random sample of molecules from all available
        for file in random.sample(tar_files, sample_size):
            file = Path(file)

            # Obtaining just the Molecule ID (.stem removes the .gz suffix)
            molid = file.stem[:-4]

            # Make temporary directory to investigate data in
            output_dir = PROJ_DIR / "docking" / "PyMolGen" / f"extracted_{molid}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # tar command to unzip and untar the molecule docking dataa
            command = ["tar", "-xzf", str(file), "-C", str(output_dir)]
            try:
                subprocess.run(command, check=True)

                # Unzip the .csv.gz file
                try:
                    # Trying to unzip the all_scores file, if fails continues onto next molecule ID
                    gz_file = output_dir / molid / f"{molid}_all_scores.csv.gz"
                    docked_confs_df = pd.read_csv(gz_file)

                    # Updating lists with necessary data
                    scores_ls.append(docked_confs_df[self.docking_column].tolist())
                    n_docked_confs = len(docked_confs_df)
                    n_docked_ls.append(n_docked_confs / 9)
                    pose_ls.append(docked_confs_df.index)

                    # Counting the number of conformations in the all_confs .sdf file
                    n_total_confs = count_conformations(
                        f"{output_dir}/{molid}/all_confs_{molid}_pH74.sdf"
                    )
                    n_confs_ls.append(n_total_confs)
                except:
                    continue

                # Remove the extracted directory
                rm_command = ["rm", "-r", str(output_dir)]
                subprocess.run(rm_command, check=True)

                # Adding the molecule ID to the list if successful
                molid_ls.append(molid)
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract {file}. Error: {e}")

        if conf_gen_plot:

            # Creating a pd.DataFrame with all of the necessary data to make the
            # number of conformers per molecule ID plot

            conf_df = pd.DataFrame(
                {
                    "n_confs_made": n_confs_ls,
                    "molids": molid_ls,
                    "n_confs_docked": n_docked_ls,
                }
            )

            conf_df.index = molid_ls

            # Making the scatter plots
            sns.scatterplot(data=conf_df, x="molids", y="n_confs_made")
            sns.scatterplot(data=conf_df, x="molids", y="n_confs_docked")

            # Formatting the scatter plots
            plt.title("Conformer Generation Analysis")
            plt.xticks(rotation=90)
            plt.ylabel("Number of conformers made")
            plt.xlabel("Molecule ID")
            plt.tight_layout()
            plt.savefig("/users/yhb18174/Recreating_DMTA/scripts/run/conf_gen_plot.png")
            plt.show()

        if score_convergence:

            # Initialising an empty list for all normalised scores
            all_norm_score_lists = []

            for ds_ls in scores_ls:

                # Finding the best scores after each iteration
                best_score = 0
                best_score_ls = []
                for score in ds_ls:
                    if score <= best_score:
                        best_score = score
                    best_score_ls.append(best_score)

                # Normalising the scores between 0 and 1
                min_score = min(best_score_ls)
                max_score = max(best_score_ls)
                if max_score == min_score:
                    normalised_scores = [0.5] * len(best_score_ls)
                else:
                    normalised_scores = [
                        (score - min_score) / (max_score - min_score)
                        for score in best_score_ls
                    ]

                # Updating the normalised scores list
                all_norm_score_lists.append(normalised_scores)

            # Plot the best score lists
            plt.figure()
            for best_score_ls, molid in zip(all_norm_score_lists, molid_ls):
                plt.plot(best_score_ls, label=molid, alpha=0.5)

            # Formatting the plots
            plt.xlabel("Pose Number")
            plt.ylabel("Best Score")
            plt.title("Best Scores Over Time")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")

            plt.tight_layout()
            plt.savefig(
                "/users/yhb18174/Recreating_DMTA/scripts/run/conf_conv_plot.png"
            )
            plt.show()

        return

    def Prediction_Development(
        self,
        experiment: str,
        iter_ls: list,
        n_plots: int=16,
        prediction_fpath: str = "/held_out_test/held_out_test_preds.csv",
        true_path: str = f"{PROJ_DIR}/datasets/held_out_data/PMG_held_out_targ_trimmed.csv",
        dot_size: int = 3,
        save_plot: bool=True,
        plot_filename: str = "preds_dev_plot.png",
        tl_box_position: tuple = (0.35, 0.95),
        br_box_position: tuple = (0.95, 0.05),
        underlay_it0: bool=False,
        regression_line_colour: str = 'gold',
        x_equals_y_line_colour: str = 'red',
        it0_dot_colour: str='purple',
        it_dot_colour: str='teal'
    ):
        """
        Description
        -----------
        Function to look at the true vs predicted values over the iterations for a given test set.
        It will take an even distribution of iteration data from n number of iterations and plot them in a grid.

        Parameters
        ----------
        experiment (str)        Name of the experiment (e.g., 20240910_10_r)
        iter_ls (list)          Iterations to plot recommended for 10 and 50 molecule selections are as follows:
                                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
                                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        n_plots (int)           Number of plots to make/iterations to consider. Needs to be a square number (e.g., 4, 9, 16, etc.)
        prediction_fpath (str)  Path to the predicted values for each iteration, considering the pathing is the same.
                                (Only need to specify the dirs after the iteration dir e.g.,
                                DO -> '/held_out_test/preds.csv
                                NOT -> '/path/to/results/it{n}/held_out_test/preds.csv)
        true_path (str)         Pathway to true data .csv file
        dot_size (int)          Size of the dots on the scatter plot
        plot_filename (str)     Name to save the plot under

        Returns
        -------
        None
        """

        step = 50 if "_50_" in experiment else 10

        # Defining the results directory
        working_dir = str(self.results_dir) + "/" + experiment

        # Counting the number of iterations within the working directory
        n_its = count_number_iters(working_dir)

        n_y_plots = int(np.sqrt(n_plots))
        n_x_plots = n_y_plots

        # Reading in the true values
        true_scores = pd.read_csv(true_path, index_col="ID")[self.docking_column]

        # Obtaining the iteration numbers which will be considered.
        # Evenly picks n_plots number of iterations out of the total data
        #its_to_plot = np.round(np.linspace(1, n_its, n_plots)).astype(int).tolist()
    
        its_to_plot = iter_ls

        # Initialising the subplots
        fig, ax = plt.subplots(nrows=n_x_plots, ncols=n_y_plots, figsize=(12, 12))

        it0_preds_df = pd.read_csv(working_dir + f"/it0/{prediction_fpath}", index_col='ID')
        it0_df = pd.DataFrame()
        it0_df[self.docking_column] = true_scores
        it0_df[f'pred_{self.docking_column}'] = it0_preds_df[f"pred_{self.docking_column}"].tolist()

        # Saving the prediciton dataframes
        df_list = []
        for it in its_to_plot:
            it_dir = working_dir + f"/it{it}"
            preds = it_dir + prediction_fpath

            pred_df = pd.read_csv(preds, index_col="ID")
            pred_df[self.docking_column] = true_scores
            df_list.append(pred_df)

        # Plotting the results
        for i, (df, iter) in enumerate(zip(df_list, its_to_plot)):
            row = i // n_x_plots
            col = i % n_y_plots

            if underlay_it0:
                sns.scatterplot(
                    data=it0_df,
                    x=self.docking_column,
                    y=f'pred_{self.docking_column}',
                    ax=ax[row,col],
                    s=dot_size,
                    color=it0_dot_colour
                )

            sns.scatterplot(
                data=df,
                x=self.docking_column,
                y=f"pred_{self.docking_column}",
                ax=ax[row, col],
                s=dot_size,
                color=it_dot_colour
            )

            # Add line of best fit using regplot
            sns.regplot(
                data=df,
                x=self.docking_column,
                y=f"pred_{self.docking_column}",
                ax=ax[row, col],
                scatter=False,
                line_kws={"linestyle": "-", "color": regression_line_colour},
            )

            # Calculate the slope of the line of best fit (regression line)
            slope, intercept = np.polyfit(
                df[self.docking_column], df[f"pred_{self.docking_column}"], 1
            )

            # Plot y=x line for reference
            min_val = min(
                df[self.docking_column].min(), df[f"pred_{self.docking_column}"].min()
            )
            max_val = max(
                df[self.docking_column].max(), df[f"pred_{self.docking_column}"].max()
            )
            ax[row, col].plot(
                [min_val, max_val], [min_val, max_val], color=x_equals_y_line_colour, linestyle="-"
            )

            # Calculate angle between the regression line and x=y
            # angle_rad = np.arctan(np.abs((slope - 1) / (1 + slope * 1)))
            # angle_deg = round(np.degrees(angle_rad), 2)

            # Calculate stats
            true, pred, pearson_r2, cod, rmse, bias, sdep = CalcStats(df)
            #print(f"Iteration: {iter}")
            # print(
            #     f"Stats\npr:{pearson_r2}\nr2:{cod}\nrmse:{rmse}\nbias:{bias}\nsdep:{sdep}"
            # )

            avg_pred = np.mean(pred)

            ax[row, col].set_title(f"Iteration {iter} ({iter * step} mols)", fontsize=12)

            # Add text box with metrics
            br_textstr = (
                f"Avg Pred: {avg_pred:.2f}\n"
                f"$R^2_{{cod}}$: {cod:.2f}\n"
                f"$R^2_{{pear}}$: {pearson_r2:.2f}\n"
                # f"Angle: {angle_deg}"
                f"Grad: {round(slope, 2)}"
            )

            tl_textstr = (
                f"RMSE: {rmse:.2f}\n" f"$Bias$: {bias:.2f}\n" f"$sdep$: {sdep:.2f}"
            )

            ax[row, col].text(
                br_box_position[0],
                br_box_position[1],
                br_textstr,
                transform=ax[row, col].transAxes,
                fontsize=7,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

            ax[row, col].text(
                tl_box_position[0],
                tl_box_position[1],
                tl_textstr,
                transform=ax[row, col].transAxes,
                fontsize=7,
                verticalalignment="top",
                horizontalalignment="right",
            )

            # Set axis labels only for bottom and left most plots
            if row == np.sqrt(n_plots) - 1:  # Bottom row
                ax[row, col].set_xlabel(self.docking_column)
            else:
                ax[row, col].set_xlabel("")

            if col == 0:  # Left-most column
                ax[row, col].set_ylabel(f"pred_{self.docking_column}")
            else:
                ax[row, col].set_ylabel("")
        
        legend_elements = [
            Line2D([0], [0], color='red', linestyle=':', label='x=y'),
            Line2D([0], [0], color='gold', linestyle=':', label='Line of\nBest Fit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='teal', markersize=10, label='Working it\n preds'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=it0_dot_colour, markersize=10, label='it0 preds') if underlay_it0 else None,
        ]

        legend_elements = [elem for elem in legend_elements if elem is not None]

        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.94), ncol=1)

        plt.tight_layout(rect=[0, 0, 0.9, 0.96], pad=1)
        plt.suptitle(f"Prediction Development for {experiment}", fontsize=16, fontweight='bold')

        fig.set_size_inches(12, 10)

        if save_plot:
            plt.savefig(working_dir + "/" + plot_filename, dpi=600)

        plt.show()

        return

    def _pairwise_similarity(self, fngpts_x: list, fngpts_y: list):
        """
        Description
        -----------
        Function to calculate the Tanimoto Similarity matrix between two lists of SMILES strings

        Parameters
        ----------
        fngpts_x (list)     List of molecular fingerprints
        fngpts_y (list)     List of molecular fingerprints

        Returns
        -------
        Similarity matrix for fingerprints x and y
        """

        n_fngpts_x = len(fngpts_x)
        n_fngpts_y = len(fngpts_y)

        similarities = np.zeros((n_fngpts_x, n_fngpts_y))

        for i, fp_x in enumerate(fngpts_x):
            for j, fp_y in enumerate(fngpts_y):
                similarities[i, j] = FingerprintSimilarity(fp_x, fp_y)

        return similarities

    def Tanimoto_Heat_Maps(
        self,
        smiles_a: list,
        smiles_b: list,
        molids_a: list,
        molids_b: list,
        save_plots: bool = False,
        save_path: str = f"{PROJ_DIR}/results/rdkit_desc/plots/",
        plot_fname: str="tanimoto_heatmap"
    ):
        """
        Description
        -----------
        Function which takes 2 lists of smiles as inputs and plots the Tanimoto Similarities.
        This analyses both within and across the two lists giving a comprehensive look into the 
        structural similarities

        Parameters
        ----------
        smiles_a (list)         list of SMILES strings
        smiles_b (list)         list of SMILES strings
        molids_a (list)         list of molecule IDs for labelling axes
        molids_b (list)         list of molecule IDs for labelling axes
        save_plots (bool)       flag to save the Tanimoto Similarity heat plots

        Returns
        -------
        Figure containing 3 heat plots:
            1- Tanimoto Similarity between SMILES in smiles_a
            2- Tanimoto Similarity between SMILES in smiles_b
            3- Tanimoto Similarity between SMILES in smiles_a and smiles_b
        """
        mols_a = [Chem.MolFromSmiles(smi) for smi in smiles_a]
        mols_b = [Chem.MolFromSmiles(smi) for smi in smiles_b]

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        fngpts_a = [rdkit_gen.GetFingerprint(mol) for mol in mols_a]

        fngpts_b = [rdkit_gen.GetFingerprint(mol) for mol in mols_b]

        sim_a = self._pairwise_similarity(fngpts_x=fngpts_a, fngpts_y=fngpts_a)

        sim_b = self._pairwise_similarity(fngpts_x=fngpts_b, fngpts_y=fngpts_b)

        sim_ab = self._pairwise_similarity(fngpts_x=fngpts_a, fngpts_y=fngpts_b)

        def heatmap(sim, x_labels, y_labels, ax):
            plot = sns.heatmap(
                sim,
                annot=True,
                annot_kws={"fontsize": 10},
                cmap="crest",
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax,
                cbar=False,
            )

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

        heatmap(sim=sim_a, x_labels=molids_a, y_labels=molids_a, ax=axes[0])
        axes[0].set_title("Heatmap Smiles A")

        heatmap(sim=sim_b, x_labels=molids_b, y_labels=molids_b, ax=axes[1])
        axes[1].set_title("Heatmap Smiles B")

        heatmap(sim=sim_ab, x_labels=molids_a, y_labels=molids_b, ax=axes[2])
        axes[2].set_title("Heatmap Smiles A vs Smiles B")

        cbar = fig.colorbar(
            axes[0].collections[0],
            ax=axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
        )
        cbar.set_label("Tanimoto Similarity")

        if save_plots:
            plt.savefig(save_path + plot_fname + ".png", dpi=(600))

        plt.show()

    def Avg_Tanimoto_Avg_Across_Iters(
        self,
        experiments: list,
        smiles_df: str = str(PROJ_DIR)
        + "/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv",
        prefix: str = "PMG-",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Tanimoto_Plot",
    ):
        """
        Dictionary
        ----------
        Function to calculate the average pairwise Tanimoto Similarity of the added training molecules
        for each experiment provided and plot them.

        Parameters
        ----------
        experiments (list)          List of experiment names (name of directories results are in)
        smiles_df (str)             Generic pathway to the .csv file containing all of the SMILES
                                    data (uses glob, e.g., /path/to/file/smiles_df_* )
        results_dir (str)           Pathway to results directory where the experiment directories are held
        save_plot (bool)            Flag to save generated plots
        save_path (str)             Pathway to directory you want to save the plots in
        filename (str)              Name of the file to save plots as
        """

        plt.figure(figsize=(10, 6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        tan_sim_dict = {}

        for i, exp in tqdm(
            enumerate(experiments), desc="Processing Experiments", unit="exp"
        ):
            experiment_dir = results_dir + exp
            n_iters = count_number_iters(experiment_dir)

            all_mols = pd.DataFrame(columns=["ID", "SMILES", "Mol", "Fingerprints"])

            if "__50__" in exp:
                step = 50
            else:
                step = 10

            avg_tanimoto_sim_ls = []
            iter_ls = []
            n_mols_chosen = []

            for iter in range(0, n_iters + 1):
                temp_df = pd.DataFrame()
                start_iter = iter
                end_iter = iter + 1

                molids = get_sel_mols_between_iters(
                    experiment_dir=experiment_dir,
                    start_iter=start_iter,
                    end_iter=end_iter,
                )
                temp_df["ID"] = molids

                smiles = molid_ls_to_smiles(
                    molids=molids, prefix=prefix, data_fpath=smiles_df
                )
                temp_df["SMILES"] = smiles

                mols = [Chem.MolFromSmiles(smi) for smi in smiles]
                temp_df["Mols"] = mols

                added_fngpts = [rdkit_gen.GetFingerprint(mol) for mol in mols]
                temp_df["Fingerprints"] = added_fngpts

                all_mols = pd.concat([all_mols, temp_df], ignore_index=True)
                iter_ls.append(end_iter)
                n_mols_chosen.append(end_iter * step)

                sim = self._pairwise_similarity(
                    fngpts_x=all_mols["Fingerprints"], fngpts_y=all_mols["Fingerprints"]
                )

                avg_sim = round(np.mean(sim), 4)
                avg_tanimoto_sim_ls.append(avg_sim)
                tan_sim_dict[exp] = avg_tanimoto_sim_ls

            sns.lineplot(
                x=n_mols_chosen, y=avg_tanimoto_sim_ls, label=exp, color=colours[i]
            )

        plt.xlabel("Iteration")
        plt.ylabel("Average Tanimoro Similarity")
        plt.ylim(0, 1)
        plt.title("Average Tanimoto Similarity of Chosen Mols")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600)

        plt.show()

        return tan_sim_dict

    def Top_Preds_Analysis(
        self,
        experiments: list,
        preds_fname: str = "all_preds*",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        preds_column: str = "pred_Affinity(kcal/mol)",
        n_mols: int = 1000,
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Top_Preds_Plot",
        sort_by_descending: bool = True,
    ):

        ascending = False if sort_by_descending else True

        # plt.figure(figsize=(10,6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)
        linestyles = {"_10_": "-", "_50_": "--"}
        method_color_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }

        for exp in tqdm(experiments, desc="Processing Experiments", unit="exp"):
            avg_top_preds = []
            n_mols_chosen = []
            n_iters = count_number_iters(results_dir + exp)

            step = 50 if "_50_" in exp else 10
            linestyle = linestyles["_50_"] if "_50_" in exp else linestyles["_10_"]
            method = next((m for m in method_color_map.keys() if exp.endswith(m)), None)
            colour = method_color_map.get(method, "black")

            chosen_mols = pd.read_csv(
                results_dir + exp + "/chosen_mol.csv", index_col="ID"
            ).index

            for iter in range(0, n_iters + 1):
                working_dir = Path(results_dir + exp + f"/it{iter}/")
                print(working_dir)
                preds_files = glob(str(working_dir) + "/" + preds_fname)
                top_preds_ls = []

                for file in preds_files:
                    preds = pd.read_csv(file)
                    top_preds = get_top(preds, n_mols, preds_column, ascending)
                    top_preds = top_preds[~top_preds["ID"].isin(chosen_mols)]
                    top_preds_ls.extend(top_preds[preds_column].tolist())

                top_preds_df = pd.DataFrame(columns=[preds_column], data=top_preds_ls)
                abs_top_preds = get_top(top_preds_df, n_mols, preds_column, ascending)

                avg_top_preds.append(round(np.mean(abs_top_preds[preds_column]), 4))
                n_mols_chosen.append(iter * step)

            sns.lineplot(
                x=n_mols_chosen,
                y=avg_top_preds,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )

        plt.xlabel("Number of Molecules")
        plt.ylabel(f"Average {preds_column}")
        plt.title(f"Average {preds_column} of top {n_mols} molecules")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5), ncol=1)
        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600, bbox_inches="tight")

        plt.show()

    def _calc_avg_tanimoto_exp(self, exp, results_dir, smiles_df, prefix):
        experiment_dir = results_dir + exp
        n_iters = count_number_iters(experiment_dir)

        step = 50 if "_50_" in exp else 10

        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)

        all_mols = pd.DataFrame(columns=["ID", "SMILES", "Mol", "Fingerprints"])
        n_mols_chosen = []
        avg_tanimoto_sim_ls = []

        for iter in range(0, n_iters + 1):
            temp_df = pd.DataFrame()
            start_iter = iter
            end_iter = iter + 1

            molids = get_sel_mols_between_iters(
                experiment_dir=experiment_dir, start_iter=start_iter, end_iter=end_iter
            )

            temp_df["ID"] = molids

            temp_df["SMILES"] = molid_ls_to_smiles(
                molids=molids, prefix=prefix, data_fpath=smiles_df
            )
            temp_df["Mols"] = [Chem.MolFromSmiles(smi) for smi in temp_df["SMILES"]]

            temp_df["Fingerprints"] = [
                rdkit_gen.GetFingerprint(mol) for mol in temp_df["Mols"]
            ]

            all_mols = pd.concat([all_mols, temp_df], ignore_index=True)
            n_mols_chosen.append(end_iter * step)

            sim = self._pairwise_similarity(
                fngpts_x=all_mols["Fingerprints"], fngpts_y=all_mols["Fingerprints"]
            )

            avg_sim = round(np.mean(sim), 4)
            avg_tanimoto_sim_ls.append(avg_sim)

        return exp, avg_tanimoto_sim_ls, n_mols_chosen, step

    def MP_Avg_Tanimoto_Across_Iters(
        self,
        experiments: list,
        smiles_df: str = str(PROJ_DIR)
        + "/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv",
        prefix: str = "PMG-",
        results_dir: str = f"{str(PROJ_DIR)}/results/rdkit_desc/",
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Tanimoto_Plot",
    ):
        """
        Dictionary
        ----------
        Function to calculate the average pairwise Tanimoto Similarity (with Multiprocessing) of the added training molecules
        for each experiment provided and plot them.

        Parameters
        ----------
        experiments (list)          List of experiment names (name of directories results are in)
        smiles_df (str)             Generic pathway to the .csv file containing all of the SMILES
                                    data (uses glob, e.g., /path/to/file/smiles_df_* )
        results_dir (str)           Pathway to results directory where the experiment directories are held
        save_plot (bool)            Flag to save generated plots
        save_path (str)             Pathway to directory you want to save the plots in
        filename (str)              Name of the file to save plots as
        """

        plt.figure(figsize=(10, 6))
        colours = sns.color_palette(cc.glasbey, n_colors=12)

        method_color_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }

        with Pool() as pool:
            results = pool.starmap(
                self._calc_avg_tanimoto_exp,
                [(exp, results_dir, smiles_df, prefix) for exp in experiments],
            )

        for exp, avg_tanimoto_sim_ls, n_mols_chosen, step in results:
            linestyle = "-" if step == 10 else "--"
            method = next((m for m in method_color_map.keys() if exp.endswith(m)), None)
            colour = method_color_map.get(method, "black")

            sns.lineplot(
                x=n_mols_chosen,
                y=avg_tanimoto_sim_ls,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )

        plt.xlabel("Iteration")
        plt.ylabel("Average Tanimoro Similarity")
        plt.ylim(0, 1)
        plt.title("Average Tanimoto Similarity of Chosen Mols")

        labels = pd.Series([e[12:] for e in experiments])
        handles = [
            Line2D([0], [0], color=colours, lw=1)
            for colours in method_color_map.values()
        ]

        plt.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.75, 0.5),
            ncol=1,
            borderaxespad=0.0,
        )

        lines = [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        plt.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(0.75, 0.75),
            ncol=1,
            borderaxespad=0.0,
        )

        plt.tight_layout()

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600)

        plt.show()

        tan_sim_dict = {exp: avg_sim for exp, avg_sim, _, _ in results}

        return tan_sim_dict

    def _process_top_preds_exp(
        self,
        exp: str,
        preds_fname: str,
        preds_column: str,
        n_mols: int,
        ascending: bool,
        linestyles: dict,
        method_colour_map: dict,
    ):

        avg_top_preds = []
        n_mols_chosen = []

        results_10_dir = f'{PROJ_DIR}/results/rdkit_desc/finished_results/10_mol_sel/'
        results_50_dir = f'{PROJ_DIR}/results/rdkit_desc/finished_results/50_mol_sel/'

        results_dir = results_10_dir if "_10_" in exp else results_50_dir

        n_iters = count_number_iters(results_dir + exp)

        step = 50 if "_50" in exp else 10
        linestyle = linestyles["_50_"] if "_50_" in exp else linestyles["_10_"]
        method = next((m for m in method_colour_map.keys() if exp.endswith(m)), None)
        colour = method_colour_map.get(method, "black")
        chosen_mols = pd.read_csv(
            results_dir + exp + "/chosen_mol.csv", index_col="ID"
        ).index

        for iter in range(0, n_iters + 1):
            working_dir = Path(results_dir + exp + f"/it{iter}/")
            preds_files = glob(str(working_dir) + "/" + preds_fname)
            top_preds_ls = []

            for file in preds_files:
                preds = pd.read_csv(file)
                top_preds = get_top(preds, n_mols, preds_column, ascending)
                top_preds = top_preds[~top_preds["ID"].isin(chosen_mols)]
                top_preds_ls.extend(top_preds[preds_column].tolist())

            top_preds_df = pd.DataFrame(columns=[preds_column], data=top_preds_ls)
            abs_top_preds = get_top(top_preds_df, n_mols, preds_column, ascending)
            avg_top_preds.append(round(np.mean(abs_top_preds[preds_column]), 4))
            n_mols_chosen.append(iter * step)

        return exp, n_mols_chosen, avg_top_preds, colour, linestyle

    def MP_Top_Preds_Analysis(
        self,
        experiments: list,
        preds_fname: str = "all_preds*",
        preds_column: str = "pred_Affinity(kcal/mol)",
        n_mols: int = 1000,
        save_plot: bool = True,
        save_path: str = f"{str(PROJ_DIR)}/results/rdkit_desc/plots/",
        filename: str = "Avg_Top_Preds_Plot",
        ascending: bool = False,
    ):

        colours = sns.color_palette(cc.glasbey, n_colors=20)
        linestyles = {"_10_": "-", "_50_": "--"}

        method_colour_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }
        process_func = partial(
            self._process_top_preds_exp,
            preds_fname=preds_fname,
            preds_column=preds_column,
            n_mols=n_mols,
            ascending=ascending,
            linestyles=linestyles,
            method_colour_map=method_colour_map,
        )

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, experiments),
                    total=len(experiments),
                    desc="Processing Experiments",
                )
            )

        fig, ax = plt.subplots()

        print(type(results))

        for exp, n_mols_chosen, avg_top_preds, colour, linestyle in results:
            sns.lineplot(
                x=n_mols_chosen,
                y=avg_top_preds,
                label=exp,
                color=colour,
                linestyle=linestyle,
            )


        ax.set_xlabel("Number of Molecules")
        ax.set_ylabel(f"Average {preds_column}")
        description = "top" if not ascending else "bottom"
        ax.set_title(f"Average {preds_column} of {description} {n_mols} mols")


        lines = [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
        line_labels = ["50 Molecules", "10 Molecules"]

        leg1 = ax.legend(
            lines,
            line_labels,
            loc="upper left",
            bbox_to_anchor=(1.05, 0.75),
            ncol=1,
            borderaxespad=0.0,
        )

        exp_names = [e.split("_")[-1] for e in experiments]
        labels = []
        for e in exp_names:
            name = f"_{e}"
            if name not in labels:
                labels.append(name)

        print(labels)
        handles = []
        colour_ls = []
        for label in labels:
            colour = method_colour_map[label]
            colour_ls.append(colour)
            if colour:
                handle = Line2D([0], [0], color=colour, lw=2)
                handles.append(handle)
            else:
                print(f"Warning: No color found for label {label}")

        leg2 = ax.legend(
            handles,
            [label.lstrip('_') for label in labels],
            loc="center left",
            bbox_to_anchor=(1.05, 0.40),
            ncol=1,
            borderaxespad=0.0,
        )

        fig.add_artist(leg1)
        fig.add_artist(leg2)

        if save_plot:
            plt.savefig(save_path + filename + ".png", dpi=600, bbox_inches="tight")
        plt.show()

    def Plot_MPO_Potency_Correlation(self,
                                     full_data_fpath: str,
                                     preds_df_fpath: str,
                                     save_plot: bool=False,
                                     save_fname: str='mpo_aff_correlation',
                                     save_fpath: str=f"{PROJ_DIR}/results/rdkit_desc/plots/",
                                     dpi: int=600):
        
        """
        Description
        -----------
        Function to plot the correlation between MPO and Docking Scores

        Parameters
        ----------
        full_data_fpath (str)       Path to the full data which includes PFI and oe_logp
        preds_df_fpath (str)        Path to the Docking Scores or Predictions
        save_plot (bool)            Flag to save plot
        save_fname (str)            Name to save plot under
        save_fpath (str)            Path to save plot to
        dpi (int)                   Image quality

        Returns
        -------
        None
        
        """
        
        desc_df = pd.read_csv(
            full_data_fpath, index_col="ID", usecols=['ID', 'PFI', 'oe_logp']
        )

        preds_df = pd.read_csv(
            preds_df_fpath, index_col='ID'
        )

        mpo_df = pd.DataFrame()
        mpo_df.index = preds_df.index
        mpo_df['MPO'] = [
            -score * (1 / (1 + math.exp(PFI - 8)))
            for score, PFI in zip(preds_df[self.docking_column], desc_df["PFI"])
        ]
        mpo_df[self.docking_column] = preds_df[self.docking_column]

        sns.scatterplot(x= 'MPO', y= self.docking_column, data=mpo_df)
        plt.title(f"MPO & {self.docking_column} Correlation")
        plt.xlabel('MPO')
        plt.ylabel(self.docking_column)

        # Add line of best fit using regplot
        sns.regplot(
            data=mpo_df,
            x="MPO",
            y=self.docking_column,
            scatter=False,
            line_kws={"linestyle": "-", "color": 'gold'},
        )

        slope, intercept = np.polyfit(mpo_df['MPO'], mpo_df[self.docking_column], 1)

        plt.text(
            x=mpo_df['MPO'].max() * 0.75,  # Position on x-axis
            y=mpo_df[self.docking_column].max(),  # Position on y-axis
            s=f"$y = {slope:.2f}x + {intercept:.2f}$",
            fontsize=12,
            color='black',
        )

        r2 = r2_score(mpo_df['MPO'], mpo_df[self.docking_column])
        r_pearson, p_pearson = pearsonr(mpo_df['MPO'], mpo_df[self.docking_column])

        plt.text(
            x=mpo_df['MPO'].max() * 0.75,
            y=mpo_df[self.docking_column].max() * 1.05,
            s = f"$pearson r = {r_pearson: .2f}$",
            fontsize=12,
            color='black',
        )

        if save_plot:
            plt.savefig(save_fpath + save_fname + '.png', dpi=dpi)

        plt.show()
        
        return
    
    def Draw_Chosen_Mols(self,
                         experiment: str,
                         iter_ls: list,
                         save_img: bool=False,
                         img_fpath: str=f"{PROJ_DIR}/results/rdkit_desc/plots/",
                         img_fname: str=None):
        
        """
        Description
        -----------
        Function to take chosen mols at specified iterations, draw their structure 
        and save as a .png file.

        Parameters
        ----------
        experiment (str)        name of experiment e.g., 202410002_10_mp
        iter_ls (list)          List of iterations from which to obtain the chosen molecules
                                from
        save_img (bool)         Flag to save image of drawn molecules
        img_fpath (str)         Path to save image to
        img_fname (str)         Name to save image under

        Returns
        -------
        Image of drawn chosen molecules

        """
        
        chosen_mol_df = pd.read_csv(self.results_dir + experiment + '/chosen_mol.csv', index_col='ID')
        trimmed_df = chosen_mol_df[chosen_mol_df['Iteration'].isin(iter_ls)]

        trimmed_molids = trimmed_df.index
        iteration_ls = trimmed_df['Iteration'].tolist()
        iteration_legend = [f"Iter {n}" for n in iteration_ls]

        smiles_ls = molid_ls_to_smiles(trimmed_molids, 'PMG-', f"{PROJ_DIR}/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*.csv")
        mol_ls = [Chem.MolFromSmiles(smiles) for smiles in smiles_ls]

        img = Chem.Draw.MolsToGridImage(mols=mol_ls, 
                                        molsPerRow=5, 
                                        subImgSize=(200,200), 
                                        legends=iteration_legend,
                                        )          
        
        if save_img:
            img_pil = PILImage.open(BytesIO(img.data))
            img_pil.save(f'{img_fpath}{img_fname}.png')

        return img
    
    def _process_iteration(self, args):
        experiment, pred_desc, n_components, indi_plot_suffix, remove_outliers, iter = args
        print(f"Processing Iteration {iter}")
        training_data = f"{self.results_dir}/{experiment}/it{iter}/training_data/training_targets.csv.gz"
        molid_ls = pd.read_csv(training_data, index_col='ID').index
        desc_df = get_descs_for_molid(molid_ls)

        pca_plot = self.PCA_Plot(
                    train = desc_df,
                    prediction=pred_desc,
                    source_ls=[
                        'Train',
                        'PyMolGen'
                    ],
                    n_components=n_components,
                    save_plot=True,
                    plot_area=True,
                    plot_scatter=True,
                    plot_fname=f'PCA_{experiment}_iter_{iter}_{indi_plot_suffix}',
                    plot_title=f"{experiment} Iteration {iter}",
                    remove_outliers=remove_outliers
                    )
        plt_img = fig2img(pca_plot)

        return (iter, plt_img)
            
    def PCA_Plot_Across_Iters(self,
                              experiment: str,
                              iter_ls: list,
                              n_components: int=2,
                              n_rows: int=2,
                              save_plot: bool=False,
                              indi_plot_suffix: str=None,
                              plot_fname: str="PCA_Across_Iters",
                              save_fpath: str=f"{PROJ_DIR}/results/rdkit_desc/plots",
                              plot_in_one_fig: bool=False,
                              remove_outliers:bool=True,
                              use_multiprocessing: bool=False
                              ):
        
        """
        Description
        -----------
        Function to plot a grid image of PCA plots across iterations, showing the development
        of the overlap between training data and full data

        Parameters
        ----------
        experiment (str)        Name of the experiment e.g., 20241002_10_mp
        iter_ls (list)          List of iterations to plot
        n_components (int)      Number of principal components to plot
        n_rows (int)            Number of rows and columns to make a NxN grid of PCA plots
        save_plot (bool)        Flag to save plot
        save_fpath (str)        Path to save plot to
        plot_fname (str)        Name to save plot as

        
        Returns
        -------
        None
        """

        prediction_descs = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/rdkit/PMG_rdkit_desc*'

        process_args = [(
            experiment, 
            prediction_descs, 
            n_components,
            indi_plot_suffix,
            remove_outliers,
            it,
        ) for it in iter_ls]

        if use_multiprocessing:
            with Pool() as pool:
                results = pool.map(self._process_iteration, process_args)
        
        else:
            results = []
            for args in process_args:
                res = self._process_iteration(args)
                results.append(res)

        if plot_in_one_fig:
            fig, outer_ax = plt.subplots(n_rows, n_rows, figsize=(20, 25), gridspec_kw={'wspace':0, 'hspace':0.2})
            results_dict = dict(results)

            for idx, it in enumerate(iter_ls):
                row = idx // n_rows
                col = idx % n_rows
                outer_ax[row, col].imshow(results_dict[it])
                outer_ax[row, col].axis('off')
                outer_ax[row, col].set_title(f'Iteration {it}')
            
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout(pad=0)

            if save_plot:
                plt.savefig(save_fpath + f'/{plot_fname}.png', dpi=300, bbox_inches='tight')

        return
    
    def Scaffold_Analysis(
            self,
            experiment: str,
            iter: int,
            preds_fname: str = 'all_preds_*',
            path_to_preds: str = None,
            num_reoccurring_scaff: int = 2,
            num_total_scaff_shown_graph: int = 10,
            num_total_scaff_shown_img: int = 9,
            search_in_top: int=10000,
            save_data: bool=True,
            ascending: bool= True,
            full_data: str = f"{PROJ_DIR}/datasets/PyMolGen/desc/rdkit/full_data/PMG_rdkit_*",
            show_value: str = 'pred_Affinity(kcal/mol)',
            plot_fname: str='scaff_plot',
            plot_spath: str=f'{PROJ_DIR}/results/rdkit_desc/plots/'
    ):
        """
        Description
        -----------
        Function to check for reoccurring Murcko scaffolds in the self.sorted_results file and plot them in a boxplot and barplot. Also outputs an image of the top 6 scaffolds.
        Boxplot shows the range and average value of the scaffolds
        Barplot shows the number of molecules in the sorted results that have that scaffold.bool


        Parameters
        ----------
        value (str):                        Which value the plot is based off of (e.g., MPO or pIC50)
        csv_filename (str):                 Name of the .csv file to filter molecules.
        num_reoccurring_scaff (int):        Include Murcko scaffold if >= number of molecules have it
        num_total_scaff_shown_graph (int):  How many scaffolds (bars and boxes) to show on the plots
        num_total_scaff_shown_img (int):    How many scaffolds shown in the image
        save_data (bool)                    Flad to save returned plot and image

        Returns
        -------
        A dual plot of a boxplot and barplot showing the statistics of the given scaffolds.
        &
        An image showing the structure of the top scaffolds
        A DataFrame containing the top scaffolds based on average value. Gives max and min values for given value as
        well as the SMILES string which corresponds to that value.
        """
    
        if path_to_preds is None:
            path_to_preds = self.results_dir + '/' + experiment + f'/it{iter}/'

        if '*' in preds_fname or '?' in preds_fname:
            flist = glob(path_to_preds + preds_fname)
            df_list = [pd.read_csv(file, index_col='ID') for file in flist]
            top_df = get_top(
                df=df_list, 
                n=search_in_top, 
                column=f'{self.docking_column}', 
                ascending=ascending
                )

        else: 
            top_df = get_top(
                df=pd.read_csv(path_to_preds + preds_fname, index_col='ID'),
                n=search_in_top, 
                column=f'{self.docking_column}', 
                ascending=ascending
                )

        print(f"Obtained top {search_in_top} predicted molecules")

            
        if '*' in full_data or '?' in full_data:
            flist = glob(full_data)
            smi_df = pd.DataFrame()
            smi_df = pd.concat(
                [pd.read_csv(file, index_col='ID', usecols=['ID', 'SMILES']) for file in flist],
                ignore_index=True
            )
         
        else:
            smi_df=pd.read_csv(full_data, index_col='ID', usecols=['ID', 'SMILES'])

        top_smi_df = smi_df.loc[top_df.index]

        top_smi_df['Murcko_Scaffold'] = [MurckoScaffold.MurckoScaffoldSmiles(smi) for smi in top_smi_df['SMILES']]
        print('Generated Murcko Scaffolds')

        top_smi_df[f'{self.docking_column}'] = top_df[f'{self.docking_column}']
        top_smi_df['MPO'] = top_df['MPO']
        top_smi_df['PFI'] = top_df['PFI']

    
        # Obtaining first the unique scaffolds in the data, then the reoccurring ones
        unique_scaffolds = top_smi_df['Murcko_Scaffold'].unique().tolist()

        print(
            f"Total number of molecules in data:\n{len(top_smi_df)}\n"
            f"Total number of unique scaffolds:\n{len(unique_scaffolds)}\n"
        )

        reocurring_scaffolds = [
            scaffold 
            for scaffold in unique_scaffolds
            if top_smi_df['Murcko_Scaffold'].value_counts()[scaffold] > num_reoccurring_scaff
        ]

        # Making dictionary which holds the scaffolds as a key which contains all of the molecules MPOs or pIC50s with
        # the given scaffold key
        scaff_value_dict = {
            s: top_smi_df[top_smi_df["Murcko_Scaffold"] == s][
                ["MPO", self.docking_column, "PFI"]
            ].values.tolist()
            for s in reocurring_scaffolds
        }

        # Taking the previous dictionary and obtaining an average value (MPO or pIC50) for each scaffold
        avg_key = {
            key: (
                np.mean([item[0] for item in values]),
                np.mean([item[1] for item in values]),
                np.mean([item[2] for item in values]),
            )
            for key, values in scaff_value_dict.items()
        }

        data = [(key, val[0], val[1], val[2]) for key, val in avg_key.items()]
        self.avg_scaff_value_df = pd.DataFrame(
            data,
            columns=[
                "Murcko_Scaffold",
                "Average_MPO",
                f"Average_{self.docking_column}",
                "Average_PFI",
            ],
        )

        if show_value == "MPO":
            ascending = False
        if show_value == "pIC50_pred":
            ascending = True

        # Making the DataFrame containing the top n scaffolds and their average value (MPO or pIC50) and labelling them a number so they can be identified

        # Sort by average value
        self.avg_scaff_value_df = self.avg_scaff_value_df.sort_values(
            by=f"Average_{show_value}", ascending=ascending
        )

        # Add Scaffold_Number
        self.avg_scaff_value_df["Scaffold_Number"] = range(
            1, len(self.avg_scaff_value_df) + 1
        )

        # Add Scaffold_Count
        self.avg_scaff_value_df["Scaffold_Count"] = self.avg_scaff_value_df[
            "Murcko_Scaffold"
        ].map(lambda x: len(scaff_value_dict.get(x, [])))

        # Add 'sorted by' column
        self.avg_scaff_value_df["Sorted By"] = show_value

        # Head and set index
        self.avg_scaff_value_df = self.avg_scaff_value_df.head(
            num_total_scaff_shown_graph
        ).set_index("Scaffold_Number")


        for val in ["MPO", self.docking_column]:
            self.avg_scaff_value_df[f'Min_{val}'] = self.avg_scaff_value_df[
                "Murcko_Scaffold"
            ].map(lambda x: min(scaff_value_dict.get(x, [])))

            self.avg_scaff_value_df[f'Max_{val}'] = self.avg_scaff_value_df[
                "Murcko_Scaffold"
            ].map(lambda x: max(scaff_value_dict.get(x, [])))


        if show_value == 'MPO':
            top_3 = (
                top_smi_df.groupby("Murcko_Scaffold")
                .apply(lambda x: x.nlargest(3, show_value))
                .reset_index(drop=True)
            )

        else:
            top_3 = (
                top_smi_df.groupby("Murcko_Scaffold")
                .apply(lambda x: x.nsmallest(3, show_value))
                .reset_index(drop=True)
            ) 

        # Merge avg_scaff_value_df with the top_3 DataFrame based on 'Scaffold_SMILES'
        merged_df = pd.merge(
            self.avg_scaff_value_df,
            top_3[["Murcko_Scaffold", "SMILES", show_value]],
            on="Murcko_Scaffold",
            how='left',
            suffixes=("", "_top3")
        )

        # Group the merged DataFrame by 'Murcko_Scaffold' and aggregate the 'SMILES' and values into lists
        grouped_df = (
            merged_df.groupby("Murcko_Scaffold")
            .agg({"SMILES": list, show_value: list})
            .reset_index()
        )

        # Round all floats in the value list to 2 decimal places
        grouped_df[show_value] = grouped_df[show_value].apply(
            lambda x: [round(elem, 2) for elem in x]
        )

        # Rename columns
        grouped_df.rename(
            columns={"SMILES": f"Top_3_{show_value}_SMILES", show_value: f"Top_3_{show_value}_Values"},
            inplace=True,
        )

      # Merge the aggregated lists back to avg_scaff_value_df
        self.avg_scaff_value_df = pd.merge(
            self.avg_scaff_value_df,
            grouped_df[
                ["Murcko_Scaffold", f"Top_3_{show_value}_SMILES", f"Top_3_{show_value}_Values"]
            ],
            on="Murcko_Scaffold",
            how="left",
        )

        # Rounding all floats to 2 decimal places
        self.avg_scaff_value_df = self.avg_scaff_value_df.round(2)
        smiles_order = self.avg_scaff_value_df["Murcko_Scaffold"].tolist()

        value_dict = {}

        if show_value == 'MPO':
            n = 0
        else:
            n = 1

        for key, values_list in scaff_value_dict.items():
            value_dict[key] = [values[n] for values in values_list]

        scaff_value_df = pd.DataFrame.from_dict(value_dict, orient="index").transpose()
        scaff_value_df = scaff_value_df[smiles_order]

        palette = sns.color_palette("Set2", len(self.avg_scaff_value_df))

        # Creating the subplots
        plot, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Box plot
        sns.boxplot(
            data=scaff_value_df,
            showmeans=True,
            ax=ax1,
            meanprops={
                "marker": "x",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": "5",
            },
            palette=palette
        )
        ax1.set_ylabel(show_value)
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Bar plot
        sns.barplot(
            x=self.avg_scaff_value_df['Murcko_Scaffold'],
            y="Scaffold_Count",
            data=self.avg_scaff_value_df,
            ax=ax2,
            palette=palette
        )

        ax2.set_ylabel(show_value)
        ax2.set_xlabel("Scaffold Tag")
        ax2.set_xticklabels(
            range(1, len(self.avg_scaff_value_df.index) + 1), rotation=90
        )

        plt.tight_layout()

        # Showing the top scaffolds, number defined in the function
        top_scaffolds = scaff_value_df.iloc[:, :num_total_scaff_shown_img].columns
        molecule_list = [Chem.MolFromSmiles(smiles) for smiles in top_scaffolds]
        scaffolds = Draw.MolsToGridImage(
            molecule_list, molsPerRow=3, subImgSize=(300, 300), returnPNG=False
        )

        if save_data:
            plt.savefig(
                f"{plot_spath}{plot_fname}.png", dpi=360
            )
            scaffolds.save(
                f"{plot_spath}{plot_fname}_structures.png"
            )

        return self.avg_scaff_value_df

    def _submit_docking_wrapper(self, 
                                args, 
                                docking_score_files = f"{PROJ_DIR}/datasets/PyMolGen/docking/PMG_docking_*.csv",
                                docking_column:str='Affinity(kcal/mol)',
                                docking_dir:str=f"{PROJ_DIR}/docking/PyMolGen/",
                                receptor_path:str=f"{PROJ_DIR}/scripts/docking/receptors/4bw1_5_conserved_HOH.pdbqt",
                                max_confs:int=100
        ):
        
        batch_no, idxs_in_batch = args

        docking_score_batch_file = docking_score_files.replace("*", str(batch_no))

        da = Dataset_Accessor(
            original_path=docking_score_batch_file,
            temp_suffix=".dock",
            wait_time=30,
        )

        # Obtain exclusive access to the docking file
        docking_file = da.get_exclusive_access()
        if docking_file is None:
            print(f"Failed to access file:\n{docking_score_batch_file}")
            print(f"Redocking of IDs:\n{idxs_in_batch} required")

        dock_df = pd.read_csv(docking_file, index_col=0)

        # Isolating the molecule ids which have not already been docked or in the process of being docked
        for_docking = GetUndocked(
            dock_df=dock_df,
            idxs_in_batch=idxs_in_batch,
            scores_col=docking_column,
        )

        if for_docking.empty:
            print(f"No molecules to dock in batch {batch_no}...")
            da.release_file()
            return None, None, docking_score_batch_file, [], idxs_in_batch

        # Change docking value for each molecule being docked as 'PD' (pending)
        da.edit_df(
            column_to_edit=docking_column,
            idxs_to_edit=for_docking.index,
            vals_to_enter=["PD" for idx in for_docking.index],
        )

        # Releases exclusive access on file so parallel runs can access it
        da.release_file()

        print(
            "** Docking compounds: " + ", ".join(for_docking.index.tolist()),
            end="\r",
        )

        molid_ls = []
        smi_ls = []

        for molid, smi in for_docking["SMILES"].items():
            molid_ls.append(molid)
            smi_ls.append(smi)

        # Initialising the docker
        docker = Run_GNINA(
            docking_dir=docking_dir,
            molid_ls=molid_ls,
            smi_ls=smi_ls,
            receptor_path=receptor_path,
            max_confs=max_confs,
        )

        # Creating sdfs with numerous conformers and adjusting for pH 7.4
        docker.ProcessMols(use_multiprocessing=True)

        # Docking the molecules and saving scores in for_docking
        job_ids = docker.SubmitJobs(run_hrs=0, run_mins=20, use_multiprocessing=True)

        return docker, job_ids, docking_score_batch_file, molid_ls, idxs_in_batch

    def _docking_score_retrieval(
        self,
        dock_scores_ls: list,
        docking_batch_file: list,
        mols_to_edit_ls: list,
        idxs_in_batch: list,
        docking_column:str='Affinity(kcal/mol)'
    ):

        da = Dataset_Accessor(
            original_path=docking_batch_file,
            temp_suffix=".dock",
            wait_time=30,
        )

        if mols_to_edit_ls:
            da.get_exclusive_access()

            da.edit_df(
                column_to_edit=docking_column,
                idxs_to_edit=mols_to_edit_ls,
                vals_to_enter=dock_scores_ls,
            )

            da.release_file()

            WaitForDocking(
                docking_batch_file,
                idxs_in_batch=idxs_in_batch,
                scores_col=docking_column,
                check_interval=60,
            )

        file_accessed = False
        while not file_accessed:
            try:
                batch_dock_df = pd.read_csv(docking_batch_file, index_col=0)
                file_accessed = True
            except FileNotFoundError as e:
                print("Waiting for file to be accessable again...")
                time.sleep(30)

        batch_dock_df = batch_dock_df.loc[idxs_in_batch]

        return batch_dock_df
     
    def Dock_Top_Pred(self,
                      experiment:str,
                      iter:int,
                      preds_fname: str='all_preds_*',
                      ascending:bool=True,
                      search_in_top:int=20,
                      prefix: str='PMG-',
                      data_fpath: str=f'{PROJ_DIR}/datasets/PyMolGen/docking/PMG_docking_*.csv',
                      chunksize:int=100000,
                      docking_column:str='Affinity(kcal/mol)',
                      preds_column:str='pred_Affinity(kcal/mol)',
    ):
        
    
        path_to_preds = f"{self.results_dir}/{experiment}/it{iter}/"

        if path_to_preds is None:
            path_to_preds = self.results_dir + '/' + experiment + f'/it{iter}/'

        if '*' in preds_fname or '?' in preds_fname:
            flist = glob(path_to_preds + preds_fname)
            df_list = [pd.read_csv(file) for file in flist]
            top_df = get_top(
                df=df_list, 
                n=search_in_top, 
                column=f'{preds_column}', 
                ascending=ascending
                )
            
        else: 
            top_df = get_top(
                df=pd.read_csv(path_to_preds + preds_fname),
                n=search_in_top, 
                column=f'{preds_column}', 
                ascending=ascending
                )

        print(f"Obtained top {search_in_top} predicted molecules")

        molid_ls = top_df['ID'].tolist()

        smi_ls = molid_ls_to_smiles(
                molids = molid_ls,
                prefix = prefix,
                data_fpath = data_fpath
            )

        df_select = pd.DataFrame()
        df_select['ID'] = molid_ls
        df_select["batch_no"] = [
            molid2batchno(
                molid=molid, prefix=prefix, dataset_file=data_fpath, chunksize=chunksize
            )
            for molid in molid_ls
        ]
        df_select['SMILES'] = smi_ls

        sdw_args = [
            (batch_no, idxs_in_batch)
            for batch_no, idxs_in_batch in (
                df_select.reset_index()
                .groupby("batch_no")['ID']
                .apply(list)
                .items()
            )
        ]

        # Getting all job ids
        all_job_id_ls = []
        initialised_dockers = []
        all_docking_score_batch_files = []
        all_molid_ls = []
        all_idxs_in_batch = []
        all_dock_scores_ls = []

        for args in sdw_args:
            docker, job_ids, ds_batch_file, mols_to_edit_ls, idx_ls = (
                self._submit_docking_wrapper(args)
            )
        
            if docker is not None:
                initialised_dockers.append(docker)
                all_job_id_ls.extend(job_ids)
                all_docking_score_batch_files.append(ds_batch_file)
                all_molid_ls.append(mols_to_edit_ls)
                all_idxs_in_batch.append(idx_ls)
        
            else:
                docked_df = pd.read_csv(ds_batch_file, index_col="ID")
                all_dock_scores_ls.append(docked_df[docking_column].loc[idx_ls])
                all_idxs_in_batch.append(idx_ls)
                all_molid_ls.append(mols_to_edit_ls)
                all_docking_score_batch_files.append(ds_batch_file)

        if all_job_id_ls:
            WaitForJobs(all_job_id_ls)

        for docker in initialised_dockers:
            molids, top_cnn_scores, top_aff_scores = docker.MakeCsv()
            all_dock_scores_ls.append(top_aff_scores)
            docker.CompressFiles()

        dsr_args = [
            (docking_scores_ls, docking_score_batch_file, molids, idxs_in_batch)
            for docking_scores_ls, docking_score_batch_file, molids, idxs_in_batch in zip(
                all_dock_scores_ls,
                all_docking_score_batch_files,
                all_molid_ls,
                all_idxs_in_batch,
            )
        ]

        df = pd.DataFrame()
        df['ID'] = molid_ls
        df['SMILES'] = smi_ls
        df['Batch_No'] = df_select['batch_no']
        df[preds_column] = df.merge(top_df[['ID', preds_column]], on='ID', how='left')[preds_column]
        df[docking_column] = 'NaN'


        with Pool() as pool:
            results = pool.starmap(self._docking_score_retrieval, dsr_args)

        fin_dock_df = pd.concat(results, axis=0)
        # Reset index and keep the old index as a column
        fin_dock_df.reset_index(drop=False, inplace=True)   

        # Make sure 'ID' exists in fin_dock_df after resetting index
        if 'ID' not in fin_dock_df.columns:
            fin_dock_df['ID'] = fin_dock_df['index']  # If the 'ID' column was not found, use the 'index' column

        # Now iterate over the index of fin_dock_df to update docking scores in df
        for id in fin_dock_df['ID']:  # Use the 'ID' column from fin_dock_df to access values
            df.loc[df['ID'] == id, docking_column] = fin_dock_df.loc[fin_dock_df['ID'] == id, docking_column].values[0]

        # Set the 'ID' column as the index in df
        df.set_index('ID', inplace=True)

        return df
    
    def Plot_Top_Pred_Docked(self,
                             experiment_ls:list,
                             iter:int,
                             preds_fname: str='all_preds_*',
                             ascending:bool=True,
                             search_in_top:int=20,
                             prefix: str='PMG-',
                             data_fpath: str=f'{PROJ_DIR}/datasets/PyMolGen/docking/PMG_docking_*.csv',
                             chunksize:int=100000,
                             docking_column:str='Affinity(kcal/mol)',
                             preds_column:str='pred_Affinity(kcal/mol)',
                             save_plot:bool=False,
                             plot_name:str='Top_pred_docked_boxplot',
                             save_structures: bool=True):
                
        """
        Description
        -----------
        Function to plot the docking scores of the top predicted molecules for specified experiments
        
        Parameters
        ----------
        experiment_ls
        iter
        preds_fname
        ascending
        search_in_top
        prefix
        data_fpath
        chunksize
        docking_column
        preds_column

        Returns
        -------
        
        """

        colours = sns.color_palette(cc.glasbey, n_colors=20)
        method_colour_map = {
            "_mp": colours[0],
            "_mu": colours[1],
            "_r": colours[2],
            "_rmp": colours[3],
            "_rmpo": colours[4],
            "_mpo": colours[5],
        }


        df_list = []

        for exp in experiment_ls:
            print(f"Processing {exp}")
            exp_docking_df = self.Dock_Top_Pred(
                            experiment=exp,
                            iter=iter,
                            preds_fname=preds_fname,
                            ascending=ascending,
                            search_in_top=search_in_top,
                            prefix=prefix,
                            data_fpath=data_fpath,
                            chunksize=chunksize,
                            docking_column=docking_column,
                            preds_column=preds_column
                            )
            exp_docking_df['Experiment'] = exp
            df_list.append(exp_docking_df)

        
        full_df = pd.concat(df_list, axis=0)
        experiment_colors = [
            next((color for key, color in method_colour_map.items() if exp.endswith(key)), "gray")
            for exp in experiment_ls
        ]      

        fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
        
        # Convert docking_column to numeric
        full_df[docking_column] = pd.to_numeric(full_df[docking_column], errors='coerce')

        # Convert preds_column to numeric
        full_df[preds_column] = pd.to_numeric(full_df[preds_column], errors='coerce')

        sns.boxplot(y=full_df[docking_column], x=full_df['Experiment'], ax=ax[0], palette=experiment_colors, legend=False, hue=None)
        sns.boxplot(y=full_df[preds_column], x=full_df['Experiment'], ax=ax[1], palette=experiment_colors, legend=False, hue=None)

        # Set y-ticks for both axes
        min_value = min(full_df[docking_column].min(), full_df[preds_column].min())
        max_value = max(full_df[docking_column].max(), full_df[preds_column].max())

        min_value = np.floor(min_value)
        max_value = np.ceil(max_value)

        y_ticks = np.arange(min_value, max_value, 0.5)

        ax[0].set_yticks(y_ticks)
        ax[1].set_yticks(y_ticks)

        ax[0].set_ylim([min_value, max_value])
        ax[1].set_ylim([min_value, max_value])

        exp_names = [e.split("_")[-1] for e in experiment_ls]
        tick_positions = range(len(exp_names))
        ax[1].set_xticks(tick_positions)  # Set tick positions
        ax[1].set_xticklabels(exp_names, rotation=30, fontsize=14)
        ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=14)
        ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=14)
        ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=14)

        plt.tight_layout()

        if save_plot:
            plt.savefig(f"{PROJ_DIR}/results/rdkit_desc/plots/{plot_name}.png")

        plt.show()

        if save_structures:
            for df, exp in zip(df_list, experiment_ls):
                mol_ls = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
                drawn_mols = Draw.MolsToGridImage(mols=mol_ls, 
                                        molsPerRow=5, 
                                        subImgSize=(200,200),
                                        legends=df['ID'].astype(str).tolist())
                drawn_mols.save(f"{PROJ_DIR}/results/rdkit_desc/plots/{exp}_it{iter}_structs.png")