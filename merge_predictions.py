import pandas as pd
import numpy as np
import os
import glob
import utils
import sys

PREDICTIONS_PATH=os.path.join(".", "to_merge")
LOG_PATH=LOG_PATH = os.path.join(".", "log_Merged")
OUT_PATH=os.path.join(LOG_PATH, utils.timestamp())

print("\n\n")
if not os.path.exists(PREDICTIONS_PATH):
        print("3D predictions folder: '{}' does not exists".format(PREDICTIONS_PATH))
        os.makedirs(PREDICTIONS_PATH)
        print("Created predictions folder: '{}'".format(PREDICTIONS_PATH))
        print("Please copy the 3D predictions .csv.gz files to folder '{}' and run again".format(PREDICTIONS_PATH))
        sys.exit(-1)

list_of_files = glob.glob(os.path.join(PREDICTIONS_PATH, "*.csv.gz"))
file_names = [path.split("/")[-1] for path in list_of_files]

print("Merging files: {}".format(file_names))

merged_preds = None
n=0
for f in list_of_files:
    preds = pd.read_csv(f, header=0,index_col=0).values
    if merged_preds is None:
        merged_preds = preds
    else:
        merged_preds = merged_preds + preds
    n += 1
        
merged_preds = merged_preds/n
print("Done.")


print("Saving merged 3D predictions and code to '{}'".format(OUT_PATH))

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

utils.generate_submission_3d(merged_preds, os.path.join(OUT_PATH, "submission_merged.csv.gz"))
                
submission_files = [
    "data.py",
    "hourglass2D_model.py",
    "hourglass3D_model.py",
    "linear_model.py",
    "merge_predictions.py",
    "test_hourglass2D.py",
    "test_hourglass3D.py",
    "test_simple_baseline.py",
    "train_hourglass2D.py",
    "train_hourglass3D.py",
    "train_simple_baseline.py",
    "layers.py",
    "setup.py",
    "utils.py",
    "vis.py"
]
utils.create_zip_code_files(os.path.join(OUT_PATH, "code.zip"), submission_files)

print("Done.")

