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
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
import sys

def main():

    savepath = "/your/save/path/"
    name_of_run = "BCP"
    print("Importing data...")
    sys.stdout.flush()
    alpha=0.3
    size=10
    sys.stdout.flush()

    # create wanted features list
    wanted_features = pd.read_csv(savepath+"wanted_features.csv")
    wanted_features=wanted_features.drop("Unnamed: 0", axis=1)
    wanted_features = wanted_features["0"].tolist()
    
    data = pd.read_csv(savepath+"full_corrected_df_100000.csv.gz", compression="gzip")

    # take a big sample from the corrected file and create umap coordinates for them in create_giant_umap.py
    # take as big a sample as possible
    sample = data.sample(1000000)
    sample.to_csv(savepath+"first_sample_for_clustering.csv.gz", compression="gzip")
    print(sample)
    print("Sample above")
    sys.stdout.flush()
    non_sample = data[~data.index.isin(sample.index)]
    print(non_sample)
    print("Non-sample above")
    sys.stdout.flush()
    del data
    umap_emb = umap.UMAP(n_components=2, n_neighbors=100, metric="euclidean", min_dist=0.01).fit_transform(sample[wanted_features])

    np.savetxt(savepath + "embedding_indices_{}_first_sample.txt".format(str(name_of_run)), umap_emb, delimiter="\t")
    print(umap_emb.shape)
    sys.stdout.flush()

    # use DBSCAN on umap coordinates
    clustering = DBSCAN(eps=0.08, min_samples=35).fit(umap_emb)
    print(clustering)
    print("Unique clusters: ", np.unique(clustering.labels_))
    sys.stdout.flush()
    labels = clustering.labels_
    np.savetxt(savepath + "labels_{}_for_first_sample.txt".format(str(name_of_run)), labels, delimiter="\t")
    
    # plot dbscan
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = umap_emb[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
            alpha=alpha
        )

        xy = umap_emb[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
            alpha=alpha
        )

    plt.tight_layout()
    plt.savefig(savepath + "umaps/dbscan_clusters.png")
    plt.close()

    # now to predict umap coordinates for all data, not just sample
    # due to UMAP memory requirements for extremely large dataset
    sample = pd.read_csv(savepath+"first_sample_for_clustering.csv.gz", compression="gzip", index_col="Unnamed: 0")
    print("Sample: ", sample.shape)
    sys.stdout.flush()

    non_sample = data[~data.index.isin(sample.index)]
    del data
    print("Non-sample: ", non_sample.shape)
    sys.stdout.flush()

    labels_for_sample = np.loadtxt(savepath+"labels_{}_for_first_sample.txt".format(str(name_of_run))).astype(int)

    print("Sample: ")
    print(np.unique(labels_for_sample))

    sample.insert(0, "cluster", labels_for_sample)
    print("Sample: ", sample.shape, len(labels_for_sample))
    
    umap_emb = umap.UMAP(n_components=2, n_neighbors=100, metric="euclidean", min_dist=0.01).fit(sample[wanted_features])

    umap_df = pd.DataFrame({"x":umap_emb.embedding_[:,0],
            "y":umap_emb.embedding_[:,1],
            "cluster":sample["cluster"]
            })

    print(umap_df)
    print("Creating umaps ")
    plt.gca().set_aspect("equal", "datalim")
    sns.scatterplot(x="x", y="y", data=umap_df, hue="cluster", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["cluster"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/LARGE_UMAP_BEFORE_{}.png".format(str(name_of_run)))
    plt.close()
    
    # now apply existing umap to transform chunks of all data to create a very large umap mapping
    predicted_coords = []
    for i, chunk in enumerate(np.array_split(non_sample, 1000)):
        print("Processing chunk: ", i)
        sys.stdout.flush()
        coords = umap_emb.transform(chunk[wanted_features])
        print(coords.shape)
        predicted_coords.append(coords)
        with open(savepath + "coords_looped_{}.txt".format(str(name_of_run)), "a+") as outfile:
            np.savetxt(outfile, coords, delimiter="\t")

    predicted_coords = np.concatenate(predicted_coords)
    
    # save ALL umap coords for clustering
    np.savetxt(savepath + "{}_coords_predicted_umap_non_sample.txt".format(str(name_of_run)), predicted_coords, delimiter="\t")

    # to make umap plot
    predicted_coords = np.loadtxt(savepath + "coords_looped_{}.txt".format(str(name_of_run)))
    print(predicted_coords.shape)
    predicted_coords = pd.DataFrame(predicted_coords)
    emb_sample = predicted_coords.sample(1000000)
    print(emb_sample[:10])
    umap_df = pd.DataFrame({"x":emb_sample[0],
            "y":emb_sample[1]
            })

    print("Creating umaps")
    plt.gca().set_aspect("equal", "datalim")
    sns.scatterplot(x="x", y="y", data=umap_df, s=size, alpha=alpha)
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/LARGE_UMAP_AFTER_{}.png".format(str(name_of_run)))
    plt.close()


if __name__ == "__main__":
    main()
