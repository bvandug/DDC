import sqlite3
import pandas as pd
from tabulate import tabulate

# Path to your Optuna DB
db_path = "jax_optuna_ppo.db"

# Connect to the database
conn = sqlite3.connect(db_path)

# Fetch trial metadata and values
trials_query = """
SELECT t.trial_id, t.number AS trial_number, t.state, v.value
FROM trials t
LEFT JOIN trial_values v ON t.trial_id = v.trial_id
ORDER BY t.number ASC;
"""
trials_df = pd.read_sql(trials_query, conn)

# Fetch parameters
params_query = "SELECT trial_id, param_name, param_value FROM trial_params;"
params_df = pd.read_sql(params_query, conn)

# Pivot parameters into wide format
params_wide = params_df.pivot(index="trial_id", columns="param_name", values="param_value").reset_index()

# Merge with trial info
full_df = pd.merge(trials_df, params_wide, on="trial_id", how="left").drop(columns="trial_id")

# Decode activation_fn if needed
activation_map = ["tanh", "relu", "leaky_relu", "elu"]
if "activation_fn" in full_df.columns:
    full_df["activation_fn"] = full_df["activation_fn"].apply(lambda x: activation_map[int(x)] if pd.notna(x) else None)

# Save as CSV
csv_path = "sac_trials_export.csv"
full_df.to_csv(csv_path, index=False)

# # Save as Excel
# excel_path = "sac_trials_export.xlsx"
# full_df.to_excel(excel_path, index=False)

# Print preview
# print(f"✅ Exported to:\n - {csv_path}\n - {excel_path}")
print("\n📋 Preview:")
print(tabulate(full_df.head(5), headers="keys", tablefmt="fancy_grid"))

# Close connection
conn.close()
