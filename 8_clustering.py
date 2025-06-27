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

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os

def create_umaps(pca_df, embedding, savepath, size, alpha, name_of_run):
    """
    Creates UMAPs for the previously created, combined and corrected file.
    """
    print("Creating umaps in ", savepath)
    sys.stdout.flush()
    if not os.path.exists(savepath + "umaps"):
        os.makedirs(savepath + "umaps")

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
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/well_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="fov", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["fov"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
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
    # after create_giant_umap.py you can use this code to start clustering: you need umap coordinates for ALL data
    savepath = "/your/save/path/"
    name_of_run = "BCP"
    print("Importing data...")
    sys.stdout.flush()
    data = pd.read_csv(savepath+"full_corrected_df_100000.csv.gz", compression="gzip") # or any other name of df
    alpha=0.1
    size=2
    print(data.shape)
    sys.stdout.flush()

    # create wanted features list
    wanted_features = pd.read_csv(savepath+"wanted_features.csv")
    wanted_features=wanted_features.drop("Unnamed: 0", axis=1)
    wanted_features = wanted_features["0"].tolist()

    # these come from create_giant_umap.py
    umap_emb = np.loadtxt(savepath + "embedding_indices_clustering_first_sample.txt")
    labels = np.loadtxt(savepath + "labels_clustering_for_first_sample.txt")
    print("Sample umap emb: ", umap_emb.shape)
    print("Labels: ", len(labels))
    sys.stdout.flush()

    # import non-sample umap coordinates
    non_sample_coords = np.loadtxt(savepath + "coords_looped_clustering.txt")
    print("Non sample coords: ", non_sample_coords.shape)
    sys.stdout.flush()

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # print stats
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # then predict with knn from the DBSCAN clusters & umap coordinates the values for remaining datapoints
    print("Fitting knn..")
    sys.stdout.flush()

    knn_model = KNeighborsClassifier(n_neighbors=3, algorithm="auto")
    knn_model.fit(umap_emb, labels)

    print("Predicting with knn..")
    sys.stdout.flush()

    # here you predict a cluster for all chunks of data, based on UMAP coordinates
    predicted_clusters = []
    for i, chunk in enumerate(np.array_split(non_sample_coords, 1000)):
        print("Processing chunk: ", i)
        sys.stdout.flush()
        predicted_clusters.append(knn_model.predict(chunk)) 

    predicted_clusters = np.concatenate(predicted_clusters).ravel().tolist()
    print("Predicted clusters: ", len(predicted_clusters))
    print(predicted_clusters[:10])
    sys.stdout.flush()

    np.savetxt(savepath + "labels_{}_for_non_sample.txt".format(str(name_of_run)), predicted_clusters, delimiter="\t")

    # import sample
    sample = pd.read_csv(savepath+"first_sample_for_clustering.csv.gz", compression="gzip", index_col="Unnamed: 0")
    print("Sample: ", sample.shape)
    print(sample)
    sys.stdout.flush()

    # import non-sample features
    non_sample = data[~data.index.isin(sample.index)]
    print("Non-sample: ", non_sample.shape)
    print(non_sample)
    sys.stdout.flush()
    del data

    # insert labels
    sample.insert(0, "cluster", labels)
    non_sample.insert(0, "cluster", predicted_clusters)

    sample.insert(0, "x", umap_emb[:,0])
    sample.insert(0, "y", umap_emb[:,1])
    non_sample.insert(0, "x", non_sample_coords[:,0])
    non_sample.insert(0, "y", non_sample_coords[:,1])

    print("Sample: ", sample.shape, len(labels))
    print(sample)
    print("Non-sample: ", non_sample.shape, len(predicted_clusters))
    print(non_sample)
    sys.stdout.flush()

    # create final df
    final_df = pd.concat([sample, non_sample],axis=0)
    final_df.sort_index(inplace=True)
    print(final_df)
    print("Saving final df...")
    sys.stdout.flush()

    final_df.to_csv(savepath+"FINAL_df_coords_included.csv.gz", compression="gzip")

    umap_sample = final_df.sample(1000000)
    print("Creating umaps...")
    sys.stdout.flush()

    fig = plt.figure()
    plt.gca().set_aspect("equal", "datalim")
    sns.scatterplot(x="x", y="y", data=umap_sample, hue="cluster", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_sample["cluster"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/after_knn_cluster_umap_{}.png".format(str(name_of_run)))
    plt.close()

    # then divide by donors and take means if needed
    # if you move to subclustering, this step is optional
    dfs = [x for _, x in final_df.groupby(final_df["cluster"])]

    # for each donor, get mean for each cluster
    donorlist = []
    means = []
    clusters_list = []
    features_list = []
    for i, df in enumerate(dfs):
        print("Processing cluster: ", i)
        for donor in range(1,401):
            d = "donor_" + str(donor)
            donorfile = df[df["donor"]==d]
            for feature in wanted_features:
                values = donorfile[feature]
                mean = np.mean(values)
                means.append(mean)
                donorlist.append(donor)
                clusters_list.append(i)
                features_list.append(feature)

    means_df = pd.DataFrame({"donor":donorlist,
                            "cluster":clusters_list,
                            "feature":features_list,
                            "mean":means})
    print(means_df)
    means_df.to_csv(savepath+"means_of_clusters.csv")

if __name__ == "__main__":
    main()
