from pathlib import Path
import time
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import sklearn
import sklearn.preprocessing

def normalize_w_scaler(df, scaler, metacols):
    featcols = df.columns[~df.columns.isin(metacols)]
    featcols = [col for col in featcols if 'Brightfield' not in col] # remove brightfield features
    
    df[featcols] = scaler.transform(df[featcols].to_numpy())
    return df

def create_scaler(control_path, metacols):
    # open df
    controldf = pd.read_csv(control_path, low_memory=False)
    # set featcols
    featcols = controldf.columns[~controldf.columns.isin(metacols)]
    # create scaler
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(controldf[featcols].to_numpy())
    # return scaler
    return scaler
    
def main(dfrow, params):
    metacols = params['metacols']+params['dropcols']
    
    # Open control and create scaler
    controllist = list(params['dir_controls'].glob(f'{dfrow.plate}*.csv'))
    if len(controllist) == 0: raise ValueError('No controls')
    control_path = controllist[0]
    scaler = create_scaler(control_path, metacols)    
    # Open, scale, and save each donor
    for d_id in tqdm(range(dfrow.donor_start, dfrow.donor_end+1), leave=False, desc=dfrow.plate):
        df = pd.read_csv(params['dir_features'] / f'donor_{d_id}.csv', low_memory=False)
        if 'index' in df.columns: df.drop(columns=['index'], inplace=True)
        df = normalize_w_scaler(df.copy(), scaler, metacols)
        df.to_csv(params['path_to_save'] / f'donor_{d_id}.csv', index=False)
    
if __name__ == '__main__':
    parameters = {
                    'dir_lookup_table'  : Path('...'),
                    'dir_features'      :Path('...'), 
                    'dir_controls'      :Path('...'),
                    'path_to_save'      :Path('...'), 
                    'metacols'          : ['donor', 'plate', 'well', 'fov', 'id'],
                    'dropcols'          : ['NUCLEUS METCENTER-X', 'NUCLEUS METCENTER-Y', 'NUCLEUS NUMBER OF COMPONENTS', 
                                           'CELL METCENTER-X', 'CELL METCENTER-Y', 'CELL NUMBER OF COMPONENTS'],
                }
    
    '''
    For complete donors
    '''
    df = pd.read_csv(parameters['dir_lookup_table'], low_memory=False)
    
    for col in ['donor_start', 'donor_end', 'n_controls', 'n_donors']:
        df[col] = df[col].astype('uint')
    for col in ['Dir_name', 'donors', 'control_col', 'plate_type', 'comment']:
        df[col] = df[col].astype('string')
        
    df['plate'] = df.Dir_name
    df = df[(df.plate_type!='control')].copy()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), leave=True):
        main(dfrow=row, params=parameters)