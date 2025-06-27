"""
Workflow for processing extremely large files containing >40 million rows (cells).

Keep an eye on memory and reduce the "sample" size in Harmony correction if needed.

First make a large file elsewhere with all features combined, features in columns, cells in rows. In code this is called "pca_df_pruned_no_edge_cells_redo_no_outliers_intensity.csv.gz".
Make sure this large file has all the metadata columns needed for downstream analyses.
Do this to remove outliers for combined large file (below). Threshold chosen based on histograms to exclude likely artifacts.

    high = 40
    # remove outliers
    bad_feats_list = ["NUCLEUS INTENSITY-MEAN Alexa 568", "NUCLEUS INTENSITY-MEAN Hoechst 33342"]

    for feature in bad_feats_list:
        pca_df = pca_df[pca_df[feature] < high]

wanted_features.csv is a file listing all features considered for clustering and analysis, excludes metadata such as plate, well, fov.

harmony_modified.py = slightly modified version of Harmony, https://pypi.org/project/harmony-pytorch/

1. 5_harmony_correction_sampling.py: split the big dataframe to sample and non-sample, and correct sample with Harmony. Save non-sample.
2. 6_harmony_nearest_neighbours.py:  now correct the non-sample by applying nearest neighbours averaging based on CORRECTED sample. Combine both to file called "full_corrected_df_100000.csv". You can choose any name.
3. 7_create_giant_umap.py: you create umap coordinates for the big dataframe chunk by chunk by transforming new data with an existing embedding for the sample. you need these coordinates to start clustering. The first time you use DBSCAN to cluster the sample umap coordinates.
4. 8_clustering.py: Now you predict cluster for all umap coordinates in chunks, using knn. As the "model example" for this, you use the DBSCAN clustering from code number 3 create_giant_umap. The dataframe result is: "FINAL_df_coords_included.csv.gz". You can choose any name.
5. 9_subclustering.py: optional if your clusters already look good. You have to know numbers of clusters you want to re-cluster. You can check what different parameters would look like for these. In this code you also remove OUTLIERS between clusters (-1 cluster from DBSCAN). 
"""

import pandas as pd
import sys
from harmony_modified import harmonize

savepath = "/your/save/path/"
pca_df = pd.read_csv(savepath+"pca_df_pruned_no_edge_cells_redo_no_outliers_intensity.csv.gz")
print(pca_df.shape)
print(list(pca_df.columns.values))

#pca_df.drop(["Unnamed: 0","Unnamed: 0.1","Unnamed: 0.2", "Unnamed: 0.3", "Unnamed: 0.4"], axis=1, inplace=True) # if there are unwanted columns

wanted_features = pd.read_csv(savepath+"wanted_features.csv")
wanted_features=wanted_features.drop("Unnamed: 0", axis=1)
wanted_features = wanted_features["0"].tolist()

#  test how large a fragment you can correct by splitting to sample and non-sample
n=35000000
df = pca_df.sample(n)
print("Sample: ")
print(df)
sys.stdout.flush()

# save sample and non-sample
non_sample_df = pca_df[~pca_df.index.isin(df.index)]
non_sample_df.to_csv(savepath+"non_sample_df.csv.gz", compression="gzip")
df.to_csv(savepath+"uncorrected_df_sample_{}.csv.gz".format(str(n)), compression="gzip")

print("Non sample df: ")
print(non_sample_df)
sys.stdout.flush()
del non_sample_df
del pca_df

# harmony correction for large sample
features_corrected = harmonize(df[wanted_features].to_numpy(), df[["plate","well"]], batch_key=["plate","well"])
df.drop(wanted_features, axis=1, inplace=True)
features_corrected = pd.DataFrame(features_corrected, columns=wanted_features)
features_corrected.set_axis(wanted_features, axis='columns', inplace=True)
features_corrected.index = df.index
comb_df = pd.concat([df[["plate", "well", "donor", "fov","id"]], features_corrected], axis=1) # add wanted columns back
print(comb_df)
print("Saving...")
sys.stdout.flush()
comb_df.to_csv(savepath+"corrected_df_sample_{}.csv.gz".format(str(n)), compression="gzip")