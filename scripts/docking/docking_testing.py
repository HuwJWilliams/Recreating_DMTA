# %%

import pandas as pd
import sys
from rdkit import Chem
import re
from pathlib import Path
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool
import os
import random as rand
import textwrap
from glob import glob
import numpy as np
from docking_fns import CalcMPO

# Import Openeye Modules
from openeye import oechem

# Muting GPU warning
oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
from openeye import oequacpac, oeomega

PROJ_DIR = Path(__file__).parent.parent.parent

# %%
def CleanFiles(fpath:str=f"{str(PROJ_DIR)}/datasets/PyMolGen/docking/",
                      fname:str="PMG_docking_*.csv",
                      docking_column:str="Affinity(kcal/mol)",
                      contaminants: list=["PD"],
                      replacement: str="",
                      index_col:str='ID'):
    """
    Description
    -----------
    Function to remove any contaminants from files (e.g., 'PD', 'NaN', False, ...)
    
    Parameters
    ----------
    fpath (str)             Path to docking files
    fname (str)             Docking file file name. Can be either generic (e.g., * or ?) or specific
    docking column (str)    Column which to remove contaminants from
    contaminants (list)     Values to remove
    replacement (str)       Values to replace contaminants with
    index_col (str)         Name of index column

    Returns
    -------
    None
    """

    working_path = fpath + fname
    replace_dict = {value: replacement for value in contaminants}

    if "*" in fname or "?" in fname:
        fpath_ls = glob(working_path)

    else:
        fpath_ls = [working_path]

    for path in fpath_ls:
        working_df = pd.read_csv(path, index_col=index_col)

        contaminant_counts = {contaminant: 0 for contaminant in contaminants}

        for contaminant in contaminants:
            if contaminant == "NaN":
                contaminant_counts["NaN"] = working_df.isna().sum().sum()
            else:
                contaminant_counts[contaminant] = (working_df == contaminant).sum().sum()

        working_df.replace(replace_dict, inplace=True)
        if "NaN" in contaminants or np.nan in contaminants:
            working_df.fillna(replacement, inplace=True)

        working_df.to_csv(path, index_label=index_col)

        print(f"Processed file: {Path(path).name}")

        for contaminant, count in contaminant_counts.items():
            print(f" - Found and replaced {count} instances of '{contaminant}'.\n")
# %%

CleanFiles()
# %%
CalcMPO()
