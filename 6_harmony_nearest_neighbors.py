"""
Workflow for processing extremely large files containing >40 million rows (cells).

Keep an eye on memory and reduce the "sample" size in Harmony correction if needed.

First make a large file with all features combined. In code this is called "pca_df_pruned_no_edge_cells_redo_no_outliers_intensity.csv.gz".
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

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
import umap
import pandas as pd
from scipy.spatial import KDTree

def create_umaps(pca_df, savepath, size, alpha, name_of_run, wanted_features):
    """
    Creates UMAPs for the previously created, combined and corrected file.
    """
    print("Creating umaps in ", savepath)
    sys.stdout.flush()
    if not os.path.exists(savepath + "umaps"):
        os.makedirs(savepath + "umaps")

    reducer = umap.UMAP(n_components=2, n_neighbors=100, metric="euclidean", min_dist=0.01)
    df_for_umap = pca_df[wanted_features]
    
    print(list(df_for_umap.columns.values)[:10])
    sys.stdout.flush()
    embedding = reducer.fit_transform(df_for_umap) 
    np.savetxt(savepath + "embedding_indices_{}.txt".format(str(name_of_run)), embedding, delimiter="\t")

    umap_df = pd.DataFrame({"x":embedding[:,0],
            "y":embedding[:,1],
            "fov":pca_df["fov"],
            "well":pca_df["well"],
            "donor":pca_df["donor"],
            "plate":pca_df["plate"],
            "Harmony_batch":pca_df["Harmony_batch"]
            })

    print("Creating umaps ")
    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="plate", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["plate"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/plate_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="well", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["well"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(savepath + "umaps/well_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="fov", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["fov"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(savepath + "umaps/fov_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="plate", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["plate"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/plate_umap_{}_2.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="Harmony_batch", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["Harmony_batch"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/Harmony_batch_umap_{}.png".format(str(name_of_run)))
    plt.close()

def main():
    # apply k-nn to average correct the rest of the dataset that didn't fit into memory
    savepath = "/your/save/path/"
    name_of_run = "BCP"

    # create wanted features list
    wanted_features = pd.read_csv(savepath+"wanted_features.csv")
    wanted_features=wanted_features.drop("Unnamed: 0", axis=1)
    wanted_features = wanted_features["0"].tolist()

    alpha=0.2
    size=3

    #pd.set_option('display.max_columns', 20)
    n=35000000
    print("Reading in dataframes: ")
    sys.stdout.flush()

    # uncorrected df
    uncorrected_df_nonsample = pd.read_csv(savepath+"non_sample_df.csv.gz", index_col="Unnamed: 0") # index col depending how you import and save Pandas df
    print("Non-sample")
    print(uncorrected_df_nonsample)
    sys.stdout.flush()

    uncorrected_df_sample = pd.read_csv(savepath+"uncorrected_df_sample_{}.csv.gz".format(str(n)), index_col="Unnamed: 0")
    print("Sample")
    print(uncorrected_df_sample)
    sys.stdout.flush()

    # import all data
    dataA = uncorrected_df_nonsample[wanted_features]
    dataB = uncorrected_df_sample[wanted_features]
    uncorrected_df_nonsample = uncorrected_df_nonsample.drop(wanted_features,axis=1)
    
    print("Shape of uncorrected nonsample after drop: ", uncorrected_df_nonsample.shape)
    print("Shapes of dataA, dataB")
    print(dataA.shape)
    print(dataB.shape)
    sys.stdout.flush()

    # for remaining uncorrected dataset, predict closest points
    kdB = KDTree(dataB.values.sample(1000000))
    print("kdB done")
    sys.stdout.flush()
    del uncorrected_df_sample
    del dataB
    nearest = kdB.query(dataA.values, k=3, workers=-1)[-1]

    """
    # in case of memory issues, do query in chunks:

    nearest = []
    for i, chunk in enumerate(np.array_split(dataA, 1000)):
        print("Processing chunk: ", i, chunk.shape)
        sys.stdout.flush()
        nn = kdB.query(chunk.values, k=3, workers=-1)[-1]
        nn = pd.DataFrame(nn)
        nn.to_csv(savepath+"nn/chunk_{}.csv.gz".format(i), compression="gzip")
        np.savetxt(savepath+"nn_1mil/chunk_{}.csv.gz".format(i), nn)
        nearest.append(nn)
    del kdB

    nearest = np.vstack(nearest)

    """
    print("NN search done: ")
    print(nearest.shape)

    # then index the corrected df for the indices of nearest neighbours, and take a mean of 3 closest
    sys.stdout.flush()
    nearest.to_csv(savepath+"nearest_neighbour_nonsample_1mil.csv.gz", compression="gzip")

    # corrected df
    corrected_df = pd.read_csv(savepath+"corrected_df_sample_{}.csv.gz".format(str(n)), index_col="Unnamed: 0")
    print("Corrected df: ")
    print(corrected_df)
    sys.stdout.flush()
    dataC = corrected_df[wanted_features] # same sample as dataA
    print("Data C shape: ", dataC.shape)
    sys.stdout.flush()

    # take mean of 3 closest points to uncorrected data points
    means = []
    printcounter = 0
    print("Iterating over rows...")
    for row in nearest:
        a = dataC.iloc[row[0]][wanted_features]
        b = dataC.iloc[row[1]][wanted_features]
        c = dataC.iloc[row[2]][wanted_features]
        mean = np.mean([a,b,c], axis=0)
        means.append(mean)
        printcounter += 1
        if (printcounter == 100000):
            print("Progress report: len(nearest)", len(nearest))
            printcounter = 0
            sys.stdout.flush()

    print("Means: ", len(means))
    means = np.vstack(means)
    sys.stdout.flush()
    means = pd.DataFrame(means)
    means.columns = wanted_features
    means.index = dataA.index
    print(means.shape)
    print(means)
    del dataA
    sys.stdout.flush() 

    # add the rest of the columns 
    means.to_csv(savepath+"corrected_nonsample_no_feat_cols_100000.csv.gz", compression="gzip")
    means = pd.concat([uncorrected_df_nonsample, means], axis=1)
    print(means)
    means.to_csv(savepath+"corrected_nonsample_100000.csv.gz", compression="gzip")
    sys.stdout.flush()
    corrected_df = corrected_df.assign(Harmony_batch=1)
    means = means.assign(Harmony_batch=2)

    # concatenate into final dataframe, now all cells are corrected
    final_df = pd.concat([corrected_df, means],axis=0)
    final_df.sort_index(inplace=True)
    print("Final df:")
    print(final_df)
    sys.stdout.flush()
    final_df.to_csv(savepath+"full_corrected_df_100000.csv.gz", compression="gzip") # this is your final, corrected dataframe
    
    print("Creating umaps:")
    umap_sample = final_df.sample(100000)
    create_umaps(pca_df=umap_sample, savepath=savepath, size=size, alpha=alpha, name_of_run=name_of_run, wanted_features=wanted_features)

if __name__ == "__main__":
    main()
