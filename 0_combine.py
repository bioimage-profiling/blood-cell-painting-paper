'''
Read and combine FOV-wise BIAS features into donor-wise features.
Clean nan values.

Input: FOV features, lookup table for plates
Output: Concatenated features for each donor
'''

from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class combine:

    def __init__(self):
        pass
    
    def get_names(self, filedir):
        # get filenames
        filedatas = pd.DataFrame({'filepath': filedir.glob('*.csv')})

        # extract well and fov info from filenames
        wells, fovs = [], []
        for filename in filedatas.filepath:
            filename = os.path.basename(filename)
            plate, well, timepoint, fov, channel, plane, _, _= filename.split('_')
            wells.append(well[1:])
            fovs.append(fov[1:])
        wells = np.asarray(wells)
        fovs = np.asarray(fovs)

        filedatas['well'] = wells
        filedatas['fov'] = fovs
        
        return filedatas
        
    def open_combine(self, filedatas):
        # open all separate csv files and concatenate
        alldf = []
        for i in tqdm(range(filedatas.shape[0]), desc='Image CSV files', leave=False):
            # meta
            well = filedatas.iloc[i].well
            fov = filedatas.iloc[i].fov
            filepath = filedatas.iloc[i].filepath
            # open csv
            tempdf = pd.read_csv(filepath, low_memory=False)
            # tqdm.write(f'with shape {tempdf.shape}')
            # add well and fov
            tempdf['well'] = [well]*tempdf.shape[0]
            tempdf['fov'] = [fov]*tempdf.shape[0]
            # add to alldf
            alldf.append(tempdf)
        alldf = pd.concat(alldf)
        
        print(f'Open measurements into one df')
        
        return alldf
        
    def cleanup(self, df, feat_cols) -> pd.DataFrame:

        # make sure all feature columns are in correct dtype
        for col in feat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # remove nan rows
        df = df.dropna()
        # remove other nan rows if exist
        weird_nan = '-nan(ind)'
        for col in df.columns[(df==weird_nan).sum() > 0]:
            df.drop(df.index[df[col] == weird_nan], inplace=True)
        print('Done clean up!')
        
    def get_donor_wells(self, start_count, original, dfrow) -> pd.DataFrame:
        # Define donor wells
        donorwells = pd.DataFrame()
        
        # only for the very first plate
        if original:
            k = 0
            for i in range(2,12,1):
                k += 1
                wells = []
                for c in 'BDFHJ':
                    wells.append(f'{c}{i}')
                donorwells[f'donor_{k}'] = wells
            
            print(donorwells)   
        # all other plates
        columnlist = []
        for i in range(dfrow.donor_start, dfrow.donor_start+dfrow.n_donors):
            columnlist.append(f'donor_{i}')
        if dfrow.n_controls == 1:
            columnlist = columnlist + ['Control']
        elif dfrow.n_controls == 2:
            control_cols = [int(k) for k in dfrow.control_col.split(',')]
            if control_cols[0] == 20 and control_cols[1] == 22:
                columnlist =  columnlist + ['Control_1'] + ['Control_2']
            elif control_cols[0] == 2 and control_cols[1] == 22:
                columnlist = ['Control_1'] + columnlist + ['Control_2']
        elif dfrow.n_controls == 0 and dfrow.donor_start == 138:
            columnlist = ["AML_1", "donor_138", "AML_2", "donor_139", "AML_3", "AML_4", "donor_140", "AML_5", "donor_141", "AML_6", "Control",]
        elif dfrow.n_controls == 0 and dfrow.donor_start == 109:
            columnlist = ["MOLM_13","donor_109","donor_110","donor_111","donor_112","donor_113","donor_114","donor_115","donor_116","donor_117","Control"]
        else:
            assert False, 'Undefined number of controls'
            
        k = 0
        for i in range(2,23,2):
            wells = []
            for c in 'BDFHJ': # for plate rows B D F H J.
                wells.append(f'{c}{i}')
            donorwells[columnlist[k]] = wells
            k += 1
            
        print(donorwells)
        
    def assign_donor(self, df, donorwelldf) -> pd.DataFrame:
        
        # assign donor names
        df['donor'] = df.well
        for donor_id in donorwelldf.columns:
            df['donor'] = df['donor'].apply(lambda x: donor_id if x in list(donorwelldf[donor_id]) else x)
        print('Assigned donor information.')
        
def main(dir_to_features_of_one_plate, path_to_save_combined_df, donor_start_count, dfrow, original=False):
    
    path_to_save_combined_df.mkdir(exist_ok=True, parents=True)
    
    combine_pipeline = combine()
    # make sure donor wells are correct
    donorwelldf = combine_pipeline.get_donor_wells(start_count=donor_start_count, original=original, dfrow=dfrow)
    
    # give directory to bias features
    filedatas = combine_pipeline.get_names(filedir=dir_to_features_of_one_plate)
    # give path to save combined csv
    combined_measurements = combine_pipeline.open_combine(filedatas=filedatas)
    # define columns
    meta_cols = ['well', 'fov', 'id']
    feat_cols = combined_measurements.columns[~combined_measurements.columns.isin(meta_cols)]
    # clean up
    combined_measurements = combine_pipeline.cleanup(df=combined_measurements, feat_cols=feat_cols)
    # assign donor info
    combined_measurements = combine_pipeline.assign_donor(df=combined_measurements, donorwelldf=donorwelldf)
    # assign plate info
    combined_measurements['plate'] = [dfrow.Dir_name.split(' ')[0]]*combined_measurements.shape[0]
    
    # save each donor
    donors = [f'donor_{i}' for i in range(dfrow.donor_start, dfrow.donor_start+dfrow.n_donors)]
    for donor in tqdm(donors):
        combined_measurements.loc[combined_measurements.donor == donor].to_csv(path_to_save_combined_df / f'{donor}.csv', index=False)
    
    for col in combined_measurements.donor.unique():
        if 'Control' in col:
            combined_measurements.loc[combined_measurements.donor == col].to_csv(path_to_save_combined_df / f'{combined_measurements.plate.unique()[0]}_{col}.csv', index=False)
    
if __name__ == "__main__":
    
    path_to_lookup_table = Path('...')
    path_to_fov_features = Path('...')
    output_path = Path('...')
    
    
    df = pd.read_csv(path_to_lookup_table, low_memory=False)
    for col in ['donor_start', 'donor_end', 'n_controls', 'n_donors']:
        df[col] = df[col].astype('uint')
    for col in ['Dir_name', 'donors', 'control_col', 'plate_type', 'comment']:
        df[col] = df[col].astype('string')
        
    for idx, row in tqdm(df.iterrows(), leave=True):
        main(
            dir_to_features_of_one_plate=path_to_fov_features / row.Dir_name, 
            path_to_save_combined_df=output_path, 
            donor_start_count=row.donor_start,
            dfrow=row
            )