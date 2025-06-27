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
from sklearn.cluster import DBSCAN
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
    ax = sns.scatterplot(x="x", y="y", data=umap_df, hue="plate", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["plate"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/plate_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot(x="x", y="y", data=umap_df, hue="well", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["well"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/well_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot(x="x", y="y", data=umap_df, hue="fov", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["fov"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/fov_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot(x="x", y="y", data=umap_df, hue="plate", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["plate"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/plate_umap_{}_2.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot(x="x", y="y", data=umap_df, hue="Harmony_batch", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["Harmony_batch"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "umaps/Harmony_batch_umap_{}.png".format(str(name_of_run)))
    plt.close()


def main():
    # now you can do subclustering if any of the clusters look like they should be re-clustered

    savepath = "/your/save/path/"
    name_of_run = "BCP"
    print("Importing data...")

    # create wanted features list
    wanted_features = pd.read_csv(savepath+"wanted_features.csv")
    wanted_features=wanted_features.drop("Unnamed: 0", axis=1)
    wanted_features = wanted_features["0"].tolist()

    sample_size = 1000000
    umap_destination = "final_umaps"
    alpha=0.2
    size=5

    final_df = pd.read_csv(savepath + "FINAL_df_coords_included.csv.gz", compression="gzip")

    print("Final df shape: ", final_df.shape)
    clusters_list = final_df.cluster.unique()
    print("Clusters list: ", clusters_list)

    # colour the clusters with their names so that only wanted cluster has colour, everything else grey
    # for easier viewing of smaller clusters

    print("Now saving UMAP:")
    sys.stdout.flush()
    umap_sample = final_df.sample(sample_size)
    umap_df = umap_sample[["x","y","cluster"]]
    print(umap_df)
    print("Creating umaps ")

    plt.gca().set_aspect("equal", "datalim")
    sns.scatterplot( x="x", y="y", data=umap_df, hue="cluster", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["cluster"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "{}/before_outlier_removal_cluster_umap_{}.png".format(str(umap_destination),str(name_of_run)))
    plt.close()

    for cluster in clusters_list:
        fig = plt.figure()
        points = umap_df[umap_df["cluster"]==cluster]
        other_points = umap_df[umap_df["cluster"]!=cluster]
        plt.scatter(points["x"], points["y"], label=str(cluster), facecolor="blue", s=size+2, alpha=alpha)
        plt.scatter(other_points["x"], other_points["y"], label="other", facecolor="lightgrey", s=size, alpha=alpha)
        plt.tight_layout(h_pad=1)
        plt.savefig(savepath + "{}/cluster_{}_{}_before.png".format(str(umap_destination),str(cluster), str(name_of_run)))
        plt.close()

    for cluster in clusters_list:
        fig = plt.figure()
        points = umap_df[umap_df["cluster"]==cluster]
        other_points = umap_df[umap_df["cluster"]!=cluster]
        plt.scatter(points["x"], points["y"], label=str(cluster), facecolor="blue", s=size+2, alpha=alpha)
        plt.scatter(other_points["x"], other_points["y"], label="other", facecolor="lightgrey", s=size, alpha=alpha)
        plt.tight_layout(h_pad=1)
        plt.savefig(savepath + "{}/cluster_{}_{}_before.png".format(str(umap_destination),str(cluster), str(name_of_run)))
        plt.close()

    print("Final df before outlier removal: ", final_df.shape)

    # remove outliers marked by DBSCAN (-1)
    final_df = final_df[final_df.cluster != -1]
    print("Final df after outlier removal: ", final_df.shape)
    sys.stdout.flush()
    
    clusters_list = final_df.cluster.unique()
    print("Should not contain -1: ", clusters_list)
    sys.stdout.flush()
    print("Dropping extra columns:")

    final_df.drop(["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2"], axis=1, inplace=True) # if there are unwanted columns
    print(final_df)
    print("Starting visualizing...")
    sys.stdout.flush()

    for cluster in clusters_list:
        points = final_df[final_df["cluster"]==cluster]
        print("Datapoints in cluster: ", cluster, " shape: ", points.shape)
        sys.stdout.flush()
    
    print("Now saving UMAP after knn:")
    sys.stdout.flush()
    umap_df = final_df.sample(sample_size)

    print(umap_df)
    print("Creating umaps ")
    
    plt.gca().set_aspect("equal", "datalim")
    sns.scatterplot( x="x", y="y", data=umap_df, hue="cluster", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["cluster"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "{}/after_outlier_removal_cluster_umap_{}.png".format(str(umap_destination),str(name_of_run)))
    plt.close()

    for cluster in clusters_list:
        fig = plt.figure()
        points = umap_df[umap_df["cluster"]==cluster]
        other_points = umap_df[umap_df["cluster"]!=cluster]
        plt.scatter(points["x"], points["y"], label=str(cluster), facecolor="blue", s=size+2, alpha=alpha)
        plt.scatter(other_points["x"], other_points["y"], label="other", facecolor="lightgrey", s=size, alpha=alpha)
        plt.tight_layout(h_pad=1)
        plt.savefig(savepath + "{}/cluster_{}_{}_after.png".format(str(umap_destination),str(cluster), str(name_of_run)))
        plt.close()
    
    """
    # here you can plot how different parameters would look when subclustering
    print("Starting subclustering visualization: ")

    # testing DBSCAN
    print("Moving onto DBSCAN: ")
    eps_list = [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    point_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    for cluster in clusters_list:
        print("Processing cluster: ", cluster)
        points = final_df[final_df["cluster"]==cluster]
        print(points.shape)
        sys.stdout.flush()

        if points.shape[0] <= 100000:
            sample = points
        else:
            sample = points.sample(100000)

        #umap_emb = umap.UMAP(n_components=2, n_neighbors=100, metric="euclidean", min_dist=0.01).fit_transform(sample[wanted_features])        
        umap_emb = sample[["x","y"]]
        for eps in eps_list:
            for point in point_list:
                print("Processing points and eps: ", point, eps)
                # use DBSCAN on umap coordinates
                clustering = DBSCAN(eps=eps, min_samples=point).fit(umap_emb) # eps=0.1, min_samples=30
                print(clustering)
                print("Unique clusters: ", np.unique(clustering.labels_))
                sys.stdout.flush()
                labels = clustering.labels_
                sys.stdout.flush()
                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)

                # print stats
                print("Estimated number of clusters: %d" % n_clusters_)
                print("Estimated number of noise points: %d" % n_noise_)
                sys.stdout.flush()
                
                # plot dbscan

                umap_emb = np.array(umap_emb)
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
                        markersize=10,
                        alpha=0.5)

                    xy = umap_emb[class_member_mask & ~core_samples_mask]
                    plt.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=10,
                        alpha=0.5)

                plt.title(f"Estimated number of clusters: {n_clusters_}")
                plt.tight_layout()
                plt.savefig(savepath + "subclustering/dbscan_clusters_eps{}_points{}_cluster{}.png".format(str(eps), str(point), str(cluster)))
                plt.close()

                umap_df = pd.DataFrame({"x":umap_emb[:,0],
                    "y":umap_emb[:,1],
                    "cluster":labels
                    })
                #print(umap_df)

                #print("Creating umaps ")
                plt.gca().set_aspect("equal", "datalim")
                sns.scatterplot( x="x", y="y", data=umap_df, hue="cluster", s=10, alpha=0.5, palette=sns.color_palette("hls", len(np.unique(umap_df["cluster"]))))
                sns.set(rc={"figure.figsize":(20,20)})
                plt.legend()
                plt.tight_layout()
                plt.savefig(savepath + "subclustering/cluster_umap_eps{}_points{}_cluster{}.png".format(str(eps), str(point), str(cluster)))
                plt.close()
    #sys.exit()
    """
    # subclustering and replacement after finding good parameters for each cluster
    print("Starting cluster replacement...")
    starting_cluster_len = len(np.unique(final_df["cluster"]))
    print("Starting with n clusters: ", starting_cluster_len)
    sys.stdout.flush()
    #####################################################################################################################################
    
    # in BCP, clusters 0, 1 and 2 were considered for subclustering
    to_do_clusters = [0,1,2]
    to_add_number = 0

    for cluster in to_do_clusters:
        if cluster == 0:
            # cluster 0
            eps = 0.1 
            point = 65 
        elif cluster == 1:
            # cluster 1
            eps = 0.09 
            point = 60 
        elif cluster == 2:
            # cluster 2
            eps = 0.11
            point = 45
        print("Processing cluster: ", cluster)
        print(eps, point, cluster)

        points = final_df[final_df["cluster"]==cluster]
        print(points.shape)
        sys.stdout.flush()

        sample = points.sample(100000)
        
        non_sample = points[~points.index.isin(sample.index)]
        print("Sample and non sample: ", sample.shape, non_sample.shape)

        umap_emb = sample[["x","y"]]

        # use DBSCAN on umap coordinates
        clustering = DBSCAN(eps=eps, min_samples=point).fit(umap_emb)
        print("Unique clusters: ", np.unique(clustering.labels_))
        sys.stdout.flush()
        labels = clustering.labels_

        knn_model = KNeighborsClassifier(n_neighbors=3, algorithm="auto")
        knn_model.fit(umap_emb, labels)

        print("Predicting with knn..")
        sys.stdout.flush()
        predicted_clusters = []
        for i, chunk in enumerate(np.array_split(non_sample[["x","y"]], 100)):
            predicted_clusters.append(knn_model.predict(chunk))
        
        predicted_clusters = np.concatenate(predicted_clusters).ravel().tolist()
        print("First 10 sample labels: ", sample["cluster"][:10])
        print("First 10 non-sample labels: ", non_sample["cluster"][:10])
        print("Sample and non sample above BEFORE REPLACEMENT: SHOULD BE ONLY 0, 1 or 2!")
        
        sample["cluster"] = labels
        non_sample["cluster"] = predicted_clusters
        print("First 10 sample labels: ", sample["cluster"][:10])
        print("First 10 non-sample labels: ", non_sample["cluster"][:10])

        print("OLD label names: ", np.unique(labels))
        sys.stdout.flush()

        # create final df
        sample = pd.concat([sample, non_sample],axis=0)
        print(sample)
        print("Sample after reattaching nonsample...")
        sys.stdout.flush()

        # remove new outliers
        outliers_to_remove = sample[sample.cluster == -1]
        print("OUTLIERS TO REMOVE IN CLUSTER: ", outliers_to_remove.shape)
        print("df before removing outliers from cluster: ", cluster, final_df.shape)
        final_df = final_df[~final_df.index.isin(outliers_to_remove.index)]
        print("FINAL df after removing outliers from cluster: ", cluster, final_df.shape)
        sample = sample[sample.cluster != -1]
        print("Sample after outlier removal: ", sample.shape)
        print("Should not contain -1 anymore: ")
        print(np.unique(sample["cluster"]))
        print(np.unique(final_df["cluster"]))
        sys.stdout.flush()

        # rename the new clusters
        umap_labels = labels
        labels = sample["cluster"]
        print("before ", labels[:10])
        labels = [x+starting_cluster_len+to_add_number for x in labels]
        print("after ", labels[:10])
        print("NEW label names: ", np.unique(labels))
        print("len of labels: ", len(labels))
        print(sample.shape)
        to_add_number += len(np.unique(labels)) # for next round
        sys.stdout.flush()

        #replace label names with new ones in original umap:
        sample["cluster"] = labels
        print(np.unique(sample["cluster"]))

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # print stats
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        sys.stdout.flush()

        # plot dbscan
        print("Umap emb ", umap_emb.shape)
        print("Umap labels ", len(umap_labels))
        umap_emb = np.array(umap_emb)
        unique_labels = set(umap_labels)
        core_samples_mask = np.zeros_like(umap_labels, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = umap_labels == k

            xy = umap_emb[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
                alpha=alpha)

            xy = umap_emb[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                 xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
                alpha=alpha)

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.tight_layout()
        plt.savefig(savepath + "{}/FINAL_dbscan_clusters_eps{}_points{}_cluster{}.png".format(str(umap_destination),str(eps), str(point), str(cluster)))
        plt.close()

        print(sample)

        final_df.loc[sample.index] = sample
        print("New clusters for final df after cluster: ", cluster)
        print(final_df.cluster.unique())

    #####################################################################################################################################

    # final UMAP and taking means of donors
    clusters_list = final_df.cluster.unique()
    umap_df = final_df.sample(sample_size)

    for cluster in clusters_list:
        fig = plt.figure()
        points = umap_df[umap_df["cluster"]==cluster]
        other_points = umap_df[umap_df["cluster"]!=cluster]
        plt.scatter(points["x"], points["y"], label=str(cluster), facecolor="blue", s=size+2, alpha=alpha)
        plt.scatter(other_points["x"], other_points["y"], label="other", facecolor="lightgrey", s=size, alpha=alpha)
        plt.tight_layout(h_pad=1)
        plt.savefig(savepath + "{}/cluster_{}_{}_after_subclustering.png".format(str(umap_destination),str(cluster), str(name_of_run)))
        plt.close()

    plt.gca().set_aspect("equal", "datalim")
    sns.scatterplot( x="x", y="y", data=umap_df, hue="cluster", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["cluster"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath + "{}/FINAL_UMAP_AFTER_SUBCLUSTERING_umap_{}.png".format(str(umap_destination),str(name_of_run)))
    plt.close()

    print("SAVING FINAL DF")
    print(final_df)
    print("New clusters: ", clusters_list)
    sys.stdout.flush()

    # then divide by donors and take means if needed
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
    print("Means: ", means_df)

    print("Saving...")
    means_df.to_csv(savepath+"means_of_subclusters.csv")
    final_df.to_csv(savepath+"FINAL_df_outliers_removed_subclustering_done.csv.gz", compression="gzip")


if __name__ == "__main__":
    main()
