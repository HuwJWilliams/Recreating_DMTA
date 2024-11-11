import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import joblib
from joblib import Parallel, delayed
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
import sys
import pickle as pk
from pathlib import Path

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings('ignore', category=DataConversionWarning)

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, f"{str(PROJ_DIR)}/scripts/misc/")
from misc_functions import count_number_iters


def PredictNewTestSet(
                     feats: str,
                     targs: str,
                     full_data: str,
                     test_set_name: str,
                     experiment_ls: list,
                     results_dir: str=f"{str(PROJ_DIR)}/results/rdkit_desc/",
                     docking_column: str = 'Affinity(kcal/mol)'):
    """
    Description
    -----------
    Test all trained models on a new test set

    Parameters
    ----------
    feats (str)             Path to features .csv to predict, make sure ID in file
    targs (str)             Path to targets .csv to predict, make sure ID in file
    experiment_ls (list)    List of experiments to do predictions with

    Returns
    -------
    """

    rf_class = RF_model(docking_column=docking_column)

    feat_df = pd.read_csv(feats, index_col='ID')
    targ_df = pd.read_csv(targs, index_col='ID')
    targ_df = targ_df.values.ravel() if isinstance(targ_df, pd.DataFrame) else targ_df

    for exp in experiment_ls:
        print(f"Running {exp}")
        n_iters = count_number_iters(results_dir + exp)
        for it in range(0, n_iters + 1):
            print(f"it{it}")
            it_dir = results_dir + exp + f'/it{it}/'
            model_fpath = it_dir + 'final_model.pkl'
            model = joblib.load(model_fpath)

            
            preds_dir = Path(it_dir + f"{test_set_name}_test/")
            preds_dir.mkdir(parents=True, exist_ok=True)

            bias, sdep, mse, rmse, r2, r_pearson, p_pearson, true, pred = rf_class._calculate_performance(
                feature_test = feat_df,
                target_test = targ_df,
                best_rf = model
                )
            

            performance_dict = {
                "Bias": round(
                    float(bias), 4
                ),
                "SDEP": round(
                    float(sdep), 4
                ),
                "MSE": round(
                    float(mse), 4
                ),
                "RMSE": round(
                    float(rmse), 4
                ),
                "r2": round(
                    float(r2), 4
                ),
                "Pearson_r": round(
                    float(r_pearson), 4
                ),
                "Pearson_p": round(
                    float(p_pearson), 4
                ),
            }

            with open(f"{preds_dir}/{test_set_name}_stats.json", "w") as file:
                json.dump(performance_dict, file, indent=4)

            pred_df = pd.DataFrame()
            pred_df.index = feat_df.index
            pred_df[f"pred_{docking_column}"] = pred
            pred_df.to_csv(f"{preds_dir}/{test_set_name}_preds.csv", index_label='ID')
    
    return


class RF_model:
    def __init__(self, docking_column: str):
        """
        Description
        -----------
        Initialising the ML models class
        """
        self.docking_column = docking_column

    def _set_inner_cv(self, cv_type: str = "kfold", n_splits: int = 5):
        """
        Description
        -----------
        Setting up the Cross Validation for the inner loop. (Add to this as needed)

        Parameters
        ----------
        cv_type (str)       Name of Cross-Validation type. Current compatible CVs:
                            'kfold'
        n_splits (int)      Number splits to perform

        Returns
        -------
        The object for inner CV.

        """
        rng = rand.randint(0, 2**31)

        if cv_type == "kfold":
            self.inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=rng)

        return self.inner_cv, rng

    def _calculate_performance(
        self, feature_test: pd.DataFrame, target_test: pd.DataFrame, best_rf: object
    ):
        """
        Description
        -----------
        Function to calculate the performance metrics used to verify models

        Parameters
        ----------
        feature_test (pd.DataFrame)      pd.DataFrame of feature values (x) from the test set
        target_test (pd.DataFrame)       pd.DataFrame of targets (y) from the test set
        best_rf (object)                 RF model from the current resample

        Returns
        -------
        Series of performance metrics-
                                        1. Bias
                                        2. Standard Error of Potential
                                        3. Mean Squared Error
                                        4. Root Mean Squared Error (computed from SDEP and Bias)
                                        5. Pearson R coefficient
                                        6. Spearman R coefficient
                                        7. r2 score

        """
        # Get predictions from the best model in each resample
        predictions = best_rf.predict(feature_test)

        # Calculate Errors
        true = target_test.astype(float)
        pred = predictions.astype(float)
        errors = true - pred

        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = (np.mean((true - pred - (np.mean(true - pred))) ** 2)) ** 0.5
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        r_pearson, p_pearson = pearsonr(true, pred)

        return bias, sdep, mse, rmse, r2, r_pearson, p_pearson, true, pred

    def _get_inner_cv_info(self, hp_search_object, rng: int, feats, targs):
        all_results = {}
        cv = hp_search_object.cv_results_
        param_ls = cv['params']

        # Recreating CV splits
        n_splits = len([col for col in cv.keys() if 'split' in col and 'test_score' in col])
        kf = KFold(n_splits, shuffle=True, random_state=rng)
        cv_splits = list(kf.split(feats))

        # Initialising the prediciton DataFrames
        all_kf_preds = pd.DataFrame()
        all_kf_preds.index = feats.index

        # Iterating over each parameter set
        for param_idx, params in enumerate(param_ls):
            param_key = f'params_{param_idx}'
            all_results[param_key] = {
                'parameters': params,
                'folds': {},
                'all_fold_preds': None
            }

            # Fitting a new model with given set of hyperparameters
            estimator = clone(hp_search_object.estimator).set_params(**params)

            for train_idx, test_idx in cv_splits:
                kf_feat_tr = feats.iloc[train_idx] if hasattr(feats, 'iloc') else feats[train_idx]
                kf_feat_te = feats.iloc[test_idx] if hasattr(feats, 'iloc') else feats[test_idx]
                kf_targ_tr = targs.iloc[train_idx] if hasattr(targs, 'iloc') else targs[train_idx]

                estimator.fit(kf_feat_tr, kf_targ_tr)

                test_preds = estimator.predict(kf_feat_te)
                
                fold_df = pd.DataFrame()
                fold_df['ID'] = kf_feat_te.index
                fold_df['preds'] = test_preds
                fold_df.set_index('ID')

                all_fold_preds = all_results[param_key]['all_fold_preds']
                all_fold_preds = pd.concat([all_fold_preds, fold_df], ignore_index=True)
                all_results[param_key]['all_fold_preds'] =all_fold_preds

            pred_ls =all_results[param_key]['all_fold_preds']['preds'].tolist()
            all_kf_preds[f'Param_set_{param_idx}'] = pred_ls
            
        return all_kf_preds
        
    def _get_reliability_score(self,
                               all_kfold_preds,
                               targs):
        
        stats = pd.DataFrame(index=all_kfold_preds.index)

        stats['Mean_Pred'] = all_kfold_preds.mean(axis=1)
        stats['Docking_Score'] = targs[self.docking_column]
        stats['Stdev_Pred'] = all_kfold_preds.std(axis=1)
        stats['Prediction_Error'] = stats['Mean_Pred'].astype(float) - targs[self.docking_column].astype(float)
        stats['Absolute_Prediction_Error'] = stats['Prediction_Error'].abs()

        max_error = np.max(stats['Absolute_Prediction_Error'])

        stats['Normalised_Error'] = stats['Absolute_Prediction_Error']/max_error if max_error > 0 else stats['Absolute_Prediction_Error']
        stats['Reliability_Score'] = 1 - (stats['Normalised_Error']* stats['Stdev_Pred'])

        return stats.round(3)

    def _fit_model_and_evaluate(
        self,
        n: int,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        test_size: float,
        save_interval_models: bool,
        save_path: str,
        hyper_params: dict,
        get_full_cv_data: bool=False,
        get_reliability_score: bool=False,

    ):
        """
        Description
        -----------
        Function to carry out single resample and evaluate the performance of the predictions

        Parameters
        ----------
        n (int)                      Resample number
        features (pd.DataFrame)      Training features used to make predictions
        targets (pd.DataFrame)       Training targets used to evaluate training
        test_size (float)            Decimal of test set size (0.3 = 70/30 train/test split)
        save_interval_models (bool)  Flag to save the best rf model from each resample
        save_path (str)              Pathway to save interval models to

        Returns
        -------
        1: best parameters from the hyperparameters search
        2: Performance metrics from the best RF from the given resample
        3: Feature importances from each RF
        """

        # Setting a random seed value
        rng = rand.randint(0, 2**31)

        resample_number = n + 1

        # Doing the train/test split
        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features, targets, test_size=test_size, random_state=rng
        )

        chembl_row_indices = [
            i for i, id in enumerate(tar_te.index) if id.startswith("CHEMBL")
        ]

        # Convert DataFrames to NumPy arrays if necessary
        tar_tr = tar_tr.values.ravel() if isinstance(tar_tr, pd.DataFrame) else tar_tr
        tar_te = tar_te.values.ravel() if isinstance(tar_te, pd.DataFrame) else tar_te

        # Initialize the model and inner cv and pipeline it to prevent data leakage
        rf = Pipeline([("rf", RandomForestRegressor())])

        self.inner_cv_, kfold_rng = self._set_inner_cv(cv_type=self.inner_cv_type, n_splits=self.n_splits)

        # Setting the search type for hyper parameter optimisation
        if self.search_type == "grid":
            search = GridSearchCV(
                estimator=rf,
                param_grid=hyper_params,
                cv=self.inner_cv,
                scoring=self.scoring,
            )
        else:
            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=hyper_params,
                n_iter=self.n_resamples,
                cv=self.inner_cv,
                scoring=self.scoring,
                random_state=rng,
            )
        
        # Training the model
        search.fit(feat_tr, tar_tr)

        # Obtaining full cv data
        if get_full_cv_data:
            print("Obtaining full Cross-Validation Data")
            all_kfold_preds = self._get_inner_cv_info(
                hp_search_object=search,
                feats=features,
                targs=targets,
                rng=kfold_rng)
        else:
            all_kfold_preds = None
        
        if get_reliability_score and get_full_cv_data is not None:
            reliability_scores = self._get_reliability_score(all_kfold_preds=all_kfold_preds,
                                                             targs=targets,)
        else: 
            reliability_scores = None


        # Obtaining the best model in
        best_pipeline = search.best_estimator_
        best_rf = best_pipeline.named_steps["rf"]

        # Calculating the performance of each resample
        performance = self._calculate_performance(
            target_test=tar_te, feature_test=feat_te, best_rf=best_rf
        )

        ChEMBL_perf = self._calculate_performance(
            target_test=tar_te[chembl_row_indices],
            feature_test=feat_te.iloc[chembl_row_indices],
            best_rf=best_rf,
        )


        # Isolating the true and predicted values used in performance calculations
        # so analysis can be done on them
        true_vals_ls = performance[-2]
        pred_vals_ls = performance[-1]
        performance = performance[:-2]

        # Calculate cross-validation scores for the best estimator
        cross_val_scores = cross_val_score(
            search.best_estimator_,
            feat_tr,
            tar_tr,
            cv=self.inner_cv,
            scoring=self.scoring,
        )

        # Saving the model at each resample
        if save_interval_models:
            joblib.dump(best_rf, f"{save_path}{n}.pkl")

        return (
            search.best_params_,
            performance,
            ChEMBL_perf,
            best_rf.feature_importances_,
            resample_number,
            true_vals_ls,
            pred_vals_ls,
            cross_val_scores,
            all_kfold_preds,
            reliability_scores
        )

    def Train_Regressor(
        self,
        search_type: str,
        scoring: str = "neg_mean_squared_error",
        n_resamples: int = 2,
        inner_cv_type: str = "kfold",
        n_splits: int = 5,
        test_size: float = 0.3,
        hyper_params: dict = None,
        features: pd.DataFrame = None,
        targets: pd.DataFrame = None,
        save_interval_models: bool = False,
        save_path: str = None,
        save_final_model: bool = False,
        plot_feat_importance: bool = False,
        batch_size: int = 2,
        get_full_cv_data: bool=False,
        get_reliability_score: bool=False,
    ):
        """
        Description
        -----------
        Function to train the RF Regressor model.

        Parameters
        ----------
        search_type (str)               Type of hyperparameter search:
                                        'grid' = grid search, exhaustive and more computationally expensive
                                        'random' = random search, non-exhaustive and less computationally expensive
        scoring (str)                   Loss function to map the hyperparameter optimisation to
        n_resamples (int)               Number of Outer Loop resamples
        inner_cv_type (str)             Setting the inner Cross-Validation method
        n_splits (int)                  Number of splits in the inner Cross-Validation
        test_size (float)               Decimal fort he train/test split. 0.3 = 70/30
        hyper_params (dict)             Dictionary of hyperparameters to optimise on
        features (pd.DataFrame)         Features to train the model on
        targets (pd.DataFrame)          Targets to train the model on
        save_interval_models (bool)     Flag to save best individual models from each resample
        save_path (str)                 Path to save individual models to
        save_final_model (bool)         Flag to save the final model after all resampling
        plot_feat_importance (bool)     Flag to plot the feature importance generated by RF model

        Returns
        -------
        1: Final Random Forect model in pickle format
        2: Best hyper parameters for the final model
        3: Dictionary of performance metrics
        4: Dataframe of feature importances
        """

        # Setting the training parameters
        self.inner_cv_type = inner_cv_type
        self.n_splits = n_splits
        self.search_type = search_type
        self.scoring = scoring
        self.n_resamples = n_resamples

        # Dropping indexes which failed to dock
        targets = targets[targets[self.docking_column] != "False"]
        features = features.loc[targets.index]

        def process_batch(batch_indices: list):
            """
            Description
            -----------
            Wrapper function for processing a batch of resamples.
            Calls the _fit_model_and_evaluate() function for each resample index provided

            Parameters
            ----------
            batch_indices (list of int)     List of indices representing the current batch of
                                            resamples to process. Each index corresponds to
                                            a specific resample

            Returns
            -------
            List of tuples
                Each element is a tuple containing information from each resample:
                - Best parameters from hyperparameter search (dict)
                - Performance metrics from the best RFR (tuple)
                - Performance metrics on just the ChEMBL data from the best RFR (dict)
                - Feature importances from the best RFR (array)
                - Resample number (int)
                - True target values used to create the performance metrics (array)
                - Predicted target values used to create the performance metrics (array)
            """

            results_batch = []
            for n in batch_indices:
                result = self._fit_model_and_evaluate(
                    n,
                    features,
                    targets,
                    test_size,
                    save_interval_models,
                    save_path,
                    hyper_params,
                    get_full_cv_data,
                    get_reliability_score,
                )
                results_batch.append(result)
            return results_batch

        # Calculating the batch size for the number of resamples to submit
        n_batches = (n_resamples + batch_size - 1) // batch_size
        batches = [
            range(i * batch_size, min((i + 1) * batch_size, n_resamples))
            for i in range(n_batches)
        ]

        # Multiprocessing the to process eatch batch
        results_batches = Parallel(n_jobs=40)(
            delayed(process_batch)(batch) for batch in batches
        )

        # Flattening the results into a single list
        results = [result for batch in results_batches for result in batch]

        # Obtaining each value from result's list of lists
        (
            best_params_ls,
            self.performance_list,
            self.ChEMBL_perf_list,
            feat_importance_ls,
            self.resample_number_ls,
            self.true_vals_ls,
            self.pred_vals_ls,
            self.cv_score_ls,
            all_kfold_preds_ls,
            reliability_scores_ls
        ) = zip(*results)

        # Obtaining average reliability scores, if flag to calculate
        # reliability scores True but fails, prints the error message
        if get_full_cv_data and get_reliability_score:
            converted_dfs = []
            for df in reliability_scores_ls:
                df_copy = df.copy()

                for col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                converted_dfs.append(df_copy)

            reliability_df = pd.concat(converted_dfs).groupby(level=0).mean()
        else:
            reliability_df = pd.DataFrame()

        # Putting the best parameters into a dictionary and forcing float type onto them to
        # remove potential issues
        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self.best_params_df.mode().iloc[0].to_dict()

        for key, value in best_params.items():
            if key != "rf__max_features":
                best_params[key] = int(value)

        # Calculating average performance metrics across all training resamples
        self.performance_dict = {
            "Bias": round(
                float(np.mean([perf[0] for perf in self.performance_list])), 4
            ),
            "SDEP": round(
                float(np.mean([perf[1] for perf in self.performance_list])), 4
            ),
            "MSE": round(
                float(np.mean([perf[2] for perf in self.performance_list])), 4
            ),
            "RMSE": round(
                float(np.mean([perf[3] for perf in self.performance_list])), 4
            ),
            "r2": round(float(np.mean([perf[4] for perf in self.performance_list])), 4),
            "Pearson_r": round(
                float(np.mean([perf[5] for perf in self.performance_list])), 4
            ),
            "Pearson_p": round(
                float(np.mean([perf[6] for perf in self.performance_list])), 4
            ),
        }

        self.ChEMBL_perf_dict = {
            "Bias": round(
                float(np.mean([perf[0] for perf in self.ChEMBL_perf_list])), 4
            ),
            "SDEP": round(
                float(np.mean([perf[1] for perf in self.ChEMBL_perf_list])), 4
            ),
            "MSE": round(
                float(np.mean([perf[2] for perf in self.ChEMBL_perf_list])), 4
            ),
            "RMSE": round(
                float(np.mean([perf[3] for perf in self.ChEMBL_perf_list])), 4
            ),
            "r2": round(float(np.mean([perf[4] for perf in self.ChEMBL_perf_list])), 4),
            "Pearson_r": round(
                float(np.mean([perf[5] for perf in self.ChEMBL_perf_list])), 4
            ),
            "Pearson_p": round(
                float(np.mean([perf[6] for perf in self.ChEMBL_perf_list])), 4
            ),
        }

        # Calculating average feature importances across all training resamples
        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": features.columns.tolist(), "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)

        # Plotting the feature importances in a bar plot
        if plot_feat_importance:
            print("Plotting feature importance")
            self._plot_feature_importance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

        # Removing the 'rf__' prefix on the best determined hyper parameters
        # ('rf__' needed for the pipelining of the rf model in _fit_model_and_evaluate function)
        cleaned_best_params = {
            key.split("__")[1]: value for key, value in best_params.items()
        }

        # Training final model on best hyper parameters and on whole data
        self.final_rf = RandomForestRegressor(**cleaned_best_params)
        self.final_rf.fit(features, targets.to_numpy())

        # Saving model, performance, best hyper parameters, training features and targets
        if save_final_model:
            print(f"Saving final model to:\n{save_path}/final_model.pkl")
            joblib.dump(self.final_rf, f"{save_path}/final_model.pkl")

            with open(f"{save_path}/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file, indent=4)

            with open(f"{save_path}/chembl_performance_stats.json", "w") as file:
                json.dump(self.ChEMBL_perf_dict, file, indent=4)

            with open(f"{save_path}/best_params.json", "w") as file:
                json.dump(best_params, file, indent=4)

            features.to_csv(
                f"{save_path}/training_data/training_features.csv.gz",
                index_label="ID",
                compression="gzip",
            )
            targets.to_csv(
                f"{save_path}/training_data/training_targets.csv.gz",
                index_label="ID",
                compression="gzip",
            )


        print("Performance on total training data")
        print(self.performance_dict)
        print("Performance on ChEMBL training data")
        print(self.ChEMBL_perf_dict)

        return (
            self.final_rf,
            best_params,
            self.performance_dict,
            self.ChEMBL_perf_dict,
            feat_importance_df,
            self.true_vals_ls,
            self.pred_vals_ls,
            self.cv_score_ls,
            reliability_df,
            )

    def AnalyseModel(
        self,
        plot_data_leakage: bool = True,
        plot_resample_preds: bool = False,
        resample_n: int = None,
        show_cv_scores: bool = True,
        check_preds_on_val: bool = True,
    ):

        if plot_data_leakage:
            mse_list = [perf[2] for perf in self.performance_list]

            # Prepare data for seaborn
            data = pd.DataFrame(
                {"Resample Number": self.resample_number_ls, "MSE": mse_list}
            )

            # Create the scatter plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=data, x="Resample Number", y="MSE", marker="o", color="blue"
            )

            plt.title("Data Leakage Plot")
            plt.xlabel("Resample Number")
            plt.ylabel("MSE")
            plt.grid(True)

            plt.show()

        if plot_resample_preds:
            true = self.true_vals_ls[resample_n]
            pred = self.pred_vals_ls[resample_n]

            data = pd.DataFrame({"True_Val": true, "Pred_Val": pred})

            plt.figure(figsize=(8, 6))

            sns.regplot(
                x="True_Val",
                y="Pred_Val",
                data=data,
                scatter_kws={"s": 50},
                line_kws={"color": "red", "linestyle": "--"},
            )

            plt.axis("equal")

            # Determine axis limits
            min_val = min(np.min(true), np.min(pred))
            max_val = max(np.max(true), np.max(pred))
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.plot([min_val, max_val], [min_val, max_val], "g--", label="x=y")

            plt.legend()
            plt.xlabel("True Value")
            plt.ylabel("Predicted Value")
            plt.title("True vs Predicted with Trendline and x=y line")
            plt.show()

        if show_cv_scores:
            print(
                f"Mean Cross Validation Score:\n{round(np.mean(self.cv_score_ls), 4)}"
            )
            print(
                f"Std Deviation of Cross Valisation Score:\n{round(np.std(self.cv_score_ls), 4)}"
            )

        if check_preds_on_val:
            return

    def _plot_feature_importance(
        self,
        feat_importance_df: pd.DataFrame = None,
        top_n_feats: int = 20,
        save_data: bool = False,
        save_path: str = None,
        filename: str = None,
        dpi: int = 500,
    ):
        """
        Description
        -----------
        Function to plot feature importance

        Parameters
        ----------
        feature_importance_df (pd.DataFrame)    pd.DataFrame containing feature importances
        top_n_feats (int)                       Number of features shown
        save_data (bool)                        Flag to save plot
        save_path (str)                         Path to save plot to
        filename (str)                          Filename to save plot a
        dpi (int)                               Value for quality of saved plot

        Returns
        -------
        None
        """

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feat_importance_df.head(top_n_feats),
            x="Importance",
            y="Feature",
            palette="viridis",
            dodge=False,
            hue="Feature",
            legend=False,
        )

        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        if save_data:
            plt.savefig(f"{save_path}/{filename}.png", dpi=dpi)
            feat_importance_df.to_csv(f"{save_path}/feature_importance_df.csv")

        return

    def _calc_mpo(self, full_data_fpath, preds_df, preds_col_name):
        """
        Description
        -----------
        Function to get the PFI and LogP scores from the full data files and calculate the MPO values from predicted
        scores

        Parameters
        ----------
        full_data_fpath (str)       String path to the full data file
        preds_df (pd.DataFrame)     Pandas DataFrame of the predictions
        preds_col_name (str)        Column name containing the predictions

        Returns
        -------
        New pandas DataFrame object containing the MPO scores

        """
        df = pd.read_csv(
            full_data_fpath, index_col="ID", usecols=["ID", "PFI", "oe_logp"]
        )
        df[preds_col_name] = preds_df[preds_col_name]
        df["MPO"] = [
            -score * 1 / (1 + math.exp(PFI - 8))
            for score, PFI in zip(preds_df[preds_col_name], df["PFI"])
        ]

        return df

    def Predict(
        self,
        feats: pd.DataFrame,
        save_preds: bool,
        preds_save_path: str = None,
        preds_filename: str = None,
        final_rf: str = None,
        pred_col_name: str = "pred_Affinity(kcal/mol)",
        calc_mpo: bool = True,
        full_data_fpath: str = None,
    ):
        """
        Descripton
        ----------
        Function to take make predictions using the input RF model

        Parameters
        ----------
        feats (pd.DataFrame)        DataFrame object containing all of the features used for predictions
        save_preds (bool)           Flag to save the predictions
        preds_save_path (str)       Path to save the predictions to
        preds_filename (str)        Name to save the .csv.gz prediction dfs to
        final_rf (str)              Path to the RF pickle file used to make predictions
        pred_col_name (str)         Name of the column in filename to save predictions to

        Returns
        -------
        pd.DataFrame object containing all of the predictions
        """

        if final_rf is not None:
            rf_model = joblib.load(final_rf)
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame()
        preds_df[pred_col_name] = rf_model.predict(feats)
        preds_df.index = feats.index

        all_tree_preds = np.stack(
            [tree.predict(feats) for tree in rf_model.estimators_]
        )

        if calc_mpo:
            preds_df = self._calc_mpo(
                full_data_fpath, preds_df=preds_df, preds_col_name=pred_col_name
            )

        preds_df["Uncertainty"] = np.std(all_tree_preds, axis=0)

        if save_preds:
            preds_df.to_csv(
                f"{preds_save_path}/{preds_filename}.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        return preds_df
