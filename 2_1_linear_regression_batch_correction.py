"""
Code to correct features with linear regression and to create UMAPs for corrected features. Processes one feature at a time for all donors, and can be scaled up.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys
from tqdm import tqdm
import umap

def create_uncorrected_features(wanted_features, savepath, donorlist, datapath):
    """
    This function creates a separate file for each feature with all donors combined.
    """
    for feature in wanted_features:
        if os.path.exists(savepath+"uncorrected_barcodes/{}.csv.gz".format(feature)):
            print("File exists for feature ", feature)
        else:
            print("Processing feature ", feature, " for uncorrected barcode files")
            fovlist = []
            well_list = []
            donornames = []
            platelist = []
            valuelist = []
            id_list = []
            for filename in tqdm(donorlist, desc="Processed donor:"):
                    sys.stdout.flush()
                    filename, _ = filename.split(".")
                    file = pd.read_csv(datapath + filename + ".tsv.gz",
                        usecols=[feature, "donor","fov","well","plate","id"])

                    fovlist.extend(list(file["fov"]))
                    well_list.extend(list(file["well"]))
                    donornames.extend(list(file["donor"]))
                    platelist.extend(list(file["plate"]))
                    valuelist.extend(list(file[feature]))
                    id_list.extend(list(file["id"]))
            save_df = pd.DataFrame({"value":valuelist,"donor":donornames, "plate":platelist, "well":well_list, "fov":fovlist, "id":id_list})
            barcodes_for_donors = save_df["donor"].astype(str) + "_" + save_df["plate"].astype(str) + "_" + save_df["well"].astype(str) + "_" + save_df["fov"].astype(str) + "_" + save_df["id"].astype(str)
            save_df["barcode"] = barcodes_for_donors
            print(save_df.shape)
            print(save_df)
            sys.stdout.flush()
            save_df.to_csv(savepath+"uncorrected_barcodes/{}.csv.gz".format(feature))

def extract_outliers(wanted_features, savepath):
    """
    This function goes through all feature values and finds outliers in the data and saves their indices. 
    All features should be same length and share same indices.
    """
    total_len = 0
    for feature in wanted_features:
        print(feature)
        df = pd.read_csv(savepath + "uncorrected_barcodes/{}.csv.gz".format(feature))
        low = np.quantile(df["value"], 0.001)
        high = np.quantile(df["value"], 0.999)
        indices_to_remove = df.index[-(df["value"] >= low) & (df["value"] <= high)].tolist()
        total_len += len(indices_to_remove)
        with open(savepath+"indices_to_remove.csv", "ab") as f:
            np.savetxt(f,indices_to_remove)
    
    # import complete file, remove duplicates and overwrite
    indices_to_remove = np.loadtxt(savepath+"indices_to_remove.csv", dtype=float).astype(int)
    indices_to_remove = pd.DataFrame(indices_to_remove)
    indices_to_remove = indices_to_remove.drop_duplicates()
    indices_to_remove = indices_to_remove[0].tolist()
    np.savetxt(savepath+"indices_to_remove.csv",indices_to_remove)

def create_corrected_features(wanted_features, savepath, indices_to_remove, donorlist, plates_codes, wells_codes, n, feature_location, name_of_run):
    """
    This function creates corrected feature values and saves the files.
    """
    for i, feature in enumerate(wanted_features):
        if not os.path.exists(savepath+feature_location):
            os.makedirs(savepath+feature_location)
            print("Directory created successfully!", savepath+feature_location)
        else:
            print("Directory already exists!")
        
        if os.path.exists(savepath+"{}/{}_corrected.csv.gz".format(feature_location, feature)):
            print("File exists for feature ", feature)
        else:
            print(feature, i, "  out of ", len(wanted_features))
            df = pd.read_csv(savepath + "uncorrected_barcodes/{}.csv.gz".format(feature))
            print("Shape before remcoving outliers: ", df.shape)
            df.drop(indices_to_remove,inplace=True) 
            print("Shape after removing outliers: ", df.shape)
            platelist_subset = []
            feats_subset = []
            donors_subset = []
            fov_subset = []
            well_subset = []
            barcode_subset = []
            id_subset = []
            for donorfile in tqdm(donorlist, desc="Processing donors..."):
                donor, _ = donorfile.split(".")
                y_corrected_subset = df[df["donor"]==donor]
                platelist_subset.extend(y_corrected_subset["plate"].iloc[:n])
                feats_subset.extend(y_corrected_subset["value"].iloc[:n])
                donors_subset.extend(y_corrected_subset["donor"].iloc[:n])
                fov_subset.extend(y_corrected_subset["fov"].iloc[:n])
                well_subset.extend(y_corrected_subset["well"].iloc[:n])
                barcode_subset.extend(y_corrected_subset["barcode"].iloc[:n])
                id_subset.extend(y_corrected_subset["id"].iloc[:n])

            platelist_subset = pd.Series(platelist_subset)
            feats_subset = pd.Series(feats_subset)
            donors_subset = pd.Series(donors_subset)
            fov_subset = pd.Series(fov_subset)
            well_subset = pd.Series(well_subset)
            small_df = pd.DataFrame()
            small_df["donor"] = donors_subset
            small_df["plate"] = platelist_subset
            small_df["well"] = well_subset
            small_df["fov"] = fov_subset
            small_df["barcode"] = barcode_subset
            small_df["id"] = id_subset
            small_df["value"] = feats_subset

            print(small_df)
            print("Small df shape: ", small_df.shape)

            if "platewellfov" in name_of_run:
                print(name_of_run)
                X = small_df[["plate","well","fov"]].copy() # the factors for which to correct
                y = small_df["value"].copy()  # feature values
                X["fov"] = X["fov"].astype("category")
                X["plate"] = X["plate"].map(plates_codes)
                X["well"] = X["well"].map(wells_codes)
                X["plate"] = X["plate"].astype("category")
                X["well"] = X["well"].astype("category")
            elif "_platewell_" in name_of_run:
                print(name_of_run)
                X = small_df[["plate","well"]].copy() 
                y = small_df["value"].copy() 
                X["plate"] = X["plate"].map(plates_codes)
                X["well"] = X["well"].map(wells_codes)
                X["plate"] = X["plate"].astype("category")
                X["well"] = X["well"].astype("category")
            elif "_fov_" in name_of_run:
                print(name_of_run)
                X = small_df[["fov"]].copy() 
                y = small_df["value"].copy() 
                X["fov"] = X["fov"].astype("category")
            elif "_well_" in name_of_run:
                print(name_of_run)
                X = small_df[["well"]].copy() 
                y = small_df["value"].copy() 
                X["well"] = X["well"].map(wells_codes)
                X["well"] = X["well"].astype("category")
            elif "_plate_" in name_of_run:
                print(name_of_run)
                X = small_df[["plate"]].copy() 
                y = small_df["value"].copy()
                X["plate"] = X["plate"].map(plates_codes)
                X["plate"] = X["plate"].astype("category")
            elif "_wellfov_" in name_of_run:
                print(name_of_run)
                X = small_df[["well","fov"]].copy() 
                y = small_df["value"].copy() 
                X["fov"] = X["fov"].astype("category")
                X["well"] = X["well"].map(wells_codes)
                X["well"] = X["well"].astype("category")
            else:
                print("Not found")
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            y_corr = y - y_pred
            save_df = pd.DataFrame({"value":y_corr,"donor":small_df["donor"], "plate":small_df["plate"], "well":small_df["well"], "fov":small_df["fov"], "id":small_df["id"], "barcode":small_df["barcode"]})
            save_df.to_csv(savepath+"{}/{}_corrected.csv.gz".format(feature_location, feature))

def create_combined_df(wanted_features, savepath, feature_location, name_of_run):
    """
    Creating a dataframe of all features for all donors for UMAP creation.
    """
    filepath = savepath+"pca_df_{}.csv.gz".format(str(name_of_run))

    if os.path.isfile(filepath):
        print(f"{filepath} exists.")
        pca_df = pd.read_csv(savepath+"pca_df_{}.csv.gz".format(str(name_of_run))) 
        return pca_df
    else:
        print(f"{filepath} does not exist.")
        dfs = []
        names = []
        metadata = []
        for i in wanted_features:
            if i == "NUCLEUS AREA": # the first feature name, check if different
                print(i)
                y_corrected = pd.read_csv(savepath+"{}/{}_corrected.csv.gz".format(feature_location, i))
                dfs.append(y_corrected["value"])
                metadata = y_corrected[["Unnamed: 0", "plate", "well", "donor", "fov", "barcode"]]
                names.append(i)
            else:
                print(i)
                y_corrected = pd.read_csv(savepath+"{}/{}_corrected.csv.gz".format(feature_location, i))
                dfs.append(y_corrected["value"])
                names.append(i)

        df = pd.concat(dfs,axis=1)
        df.columns=names
        df = pd.concat([metadata, df], axis=1) # should have features and metadata as columns
        df.to_csv(savepath+"pca_df_{}.csv.gz".format(str(name_of_run)), compression="gzip")
        return pca_df

def create_umaps(pca_df, savepath, size, alpha, name_of_run):
    """
    Creates UMAPs for the previously created, combined and corrected file.
    """

    if not os.path.exists(savepath + "umaps"):
        os.makedirs(savepath + "umaps")
    
    if os.path.exists(savepath + "embedding_indices_{}.txt".format(str(name_of_run))):
        embedding = np.loadtxt(savepath + "embedding_indices_{}.txt".format(str(name_of_run)), delimiter="\t")
    else:
        reducer = umap.UMAP(n_components=2, n_neighbors=100, metric="euclidean", min_dist=0.01)
        df_for_umap = pca_df.drop(["Unnamed: 0", "plate", "well", "donor", "fov", "barcode"], axis=1) # check for any other unwanted columns
        embedding = reducer.fit_transform(df_for_umap) 
        np.savetxt(savepath + "embedding_indices_{}.txt".format(str(name_of_run)), embedding, delimiter="\t")

    umap_df = pd.DataFrame({"x":embedding[:,0],
            "y":embedding[:,1],
            "fov":pca_df["fov"],
            "well":pca_df["well"],
            "donor":pca_df["donor"],
            "plate":pca_df["plate"]
            })

    print("Creating umaps ")
    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="plate", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["plate"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.25, fontsize="15", ncol=4, title="Plate")
    plt.savefig(savepath + "umaps/plate_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="well", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["well"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.25, fontsize="15", ncol=4, title="Well")
    plt.savefig(savepath + "umaps/well_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="fov", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["fov"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.25, fontsize="15", ncol=4, title="Fov")
    plt.savefig(savepath + "umaps/fov_umap_{}.png".format(str(name_of_run)))
    plt.close()

    plt.gca().set_aspect("equal", "datalim")
    ax = sns.scatterplot( x="x", y="y", data=umap_df, hue="plate", s=size, alpha=alpha, palette=sns.color_palette("hls", len(np.unique(umap_df["plate"]))))
    sns.set(rc={"figure.figsize":(20,20)})
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.25, fontsize="15", ncol=4, title="Plate")
    plt.savefig(savepath + "umaps/plate_umap_{}_2.png".format(str(name_of_run)))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_of_run", type=str, default="test")
    parser.add_argument("--feature_location", type=str, default="/path/to/corrected/features/")
    parser.add_argument("--savepath", type=str, default="/path/to/savefolder/")
    parser.add_argument("--datapath", type=str, default="/path/to/data/")
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    # import list of donors
    donorlist = pd.read_csv("/path/to/list/of/donors/", header=None)
    donorlist = donorlist[0].to_list()

    # create wanted features list
    if os.path.exists(args.savepath+"wanted_features.csv"):
        wanted_features = pd.read_csv(args.savepath + "wanted_features.csv")
    else:
        # import one donorfile and extract features from columns
        data = pd.read_csv(args.datapath + "any_donor.tsv.gz")
        wanted_features = list(data.columns.values)
        
        # create wanted features list
        skippable_cols = ["Unnamed: 0", "id","NUCLEUS METCENTER-Y","NUCLEUS METCENTER-X",
        "CELL METCENTER-X","CELL METCENTER-Y", "NUCLEUS NUMBER OF COMPONENTS", "CELL NUMBER OF COMPONENTS",
        "well","fov","donor","plate"]

        # remove metadata
        wanted_features = [ele for ele in wanted_features if ele not in skippable_cols]

        # remove brightfield features
        wanted_features = [x for x in wanted_features if "Brightfield" not in x]
        pd.DataFrame(wanted_features).to_csv(args.savepath + "wanted_features.csv")

    # mappings due to sklearn model not accepting str categories
    plates_codes = {"plate1":1, "plate2":2, "plate3":3} # dictionary of all plate names to numerical

    wells_codes = {"H8":1, "F8":2, "F20":3, "D18":4, "D6":5, "D20":6, "F16":7, "B6":8, "B12":9, "B4":10, "H2":11, "B16":12, "D8":13, "J4":14, 
    "H12":15, "F14":16, "H10":17, "J14":18, "J16":19, "D4":20, "D2":21, "H4":22, "J20":23, "F2":24, "H14":25, "J6":26, "B2":27, "J8":28, 
    "F10":29, "D16":30, "H18":31, "F4":32, "F18":33, "F12":34, "D10":35, "B18":36, "J18":37, "F6":38, "B20":39, "J12":40, "H20":41, 
    "H16":42, "H6":43, "B8":44, "B10":45, "B14":46, "D14":47, "D12":48, "J10":49, "J2":50} # dictionary of all well mappings to numerical

    # create uncorrected features
    create_uncorrected_features(wanted_features=wanted_features, savepath=args.savepath, donorlist=donorlist, datapath=args.datapath)

    # create list of outliers
    extract_outliers(wanted_features=wanted_features,savepath=args.savepath)
    indices_to_remove = np.loadtxt(args.savepath+"indices_to_remove.csv", dtype=float).astype(int)

    # create corrected features for 10,000 cells of each donor
    create_corrected_features(wanted_features=wanted_features, savepath=args.savepath, indices_to_remove=indices_to_remove, donorlist=donorlist, plates_codes=plates_codes, 
                              wells_codes=wells_codes, n=args.n, feature_location=args.feature_location, name_of_run=args.name_of_run)
    
    # create dataframe for umap, called pca_df
    pca_df = create_combined_df(wanted_features=wanted_features, savepath=args.savepath, feature_location=args.feature_location,
                                name_of_run=args.name_of_run)
    
    # create umaps
    alpha=0.1
    size=3
    create_umaps(savepath=args.savepath, size=size, alpha=alpha, name_of_run=args.name_of_run, pca_df=pca_df)

if __name__ == "__main__":
    main()
