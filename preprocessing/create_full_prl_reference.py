#%%
import pandas as pd
import os
from pathlib import Path


#%%
# PRL_PROJECT_ROOT = Path(os.getenv("PRL_PROJECT_ROOT"))
PRL_PROJECT_ROOT=Path("/home/srs-9/Projects/prl_project")
resource_dir = PRL_PROJECT_ROOT / "resources"

#%%
prl_updated_df = pd.read_csv(resource_dir / "PRL_spreadsheet-lstai_update_label_reference.csv", index_col="subid")
# prl_master_df = pd.read_csv(resource_dir / "PRL_labels_master_orig.csv", index_col="subid")
clinical_data_df = pd.read_csv(resource_dir / "Clinical_Data_All_updated.csv")
subject_sessions = pd.read_csv(resource_dir / "subject-sessions-updated.csv", index_col="sub")
subject_sessions.index.name = "subid"
# Create the column directly from the 'ID' column

clinical_data_df["subid"] = (
    clinical_data_df["ID"]
    .str.removeprefix("ms")
    .astype(int) # Use 'Int64' (capital I) if you have missing values (NaNs)
)

# If you need it to be the first column:
cols = ["subid"] + [c for c in clinical_data_df.columns if c != "subid"]
clinical_data_df = clinical_data_df[cols]
clinical_data_df.set_index("subid", inplace=True)


#%%
# Select the 4 columns + join everything from prl_master_df
final_df = clinical_data_df[["PRL_possible", "PRL_probable", "PRL_definite", "PRL"]].join(
    prl_updated_df, 
    how="left"
)
final_df = subject_sessions.join(
    final_df, 
    how="left"
)
final_df.loc[:, 'date_mri'].update(subject_sessions['ses'].astype("Int64"))
final_df.drop(columns=['ses'], inplace=True)

print(f"Final length: {len(final_df)}") 
final_df.to_csv(resource_dir / "PRL_labels_master_full.csv")

# %%
