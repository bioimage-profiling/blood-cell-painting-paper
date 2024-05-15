'''
Input: donorwise dataframes
Output: UMAP plot, UMAP coordinates with meta-columns
'''

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sklearn
import umap

from harmony import harmonize
from combat.pycombat import pycombat

def open_and_concat_df(filenamelist: list, base_path: Path, parameters):
    
    df = []
    for di in filenamelist:
        donordf = pd.read_csv(base_path / di, low_memory=False)
            
        if parameters['sample'] == 'sample':
            donordf = donordf.sample(n=parameters['sample_size'], random_state=9)
            
        df.append(donordf)
    df = pd.concat(df)
        
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)
    print('Opened dataframe with size:', df.shape)
    return df

def normalize_df(df, featcols, parameters):
    if parameters['normalize_func'] == 'normalize':
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(parameters['normlow'],parameters['normhigh']))
    elif parameters['normalize_func'] == 'scale':
        scaler = sklearn.preprocessing.StandardScaler()

    temp_normalized = scaler.fit_transform(df[featcols])
    df_temp = pd.DataFrame(temp_normalized, columns=featcols, index=df.index)
    df[featcols] = df_temp
    print('Normalized with function:', scaler)
    return df

def harmony_batch_correction(df, metacols, featcols, parameters):
    df['plate_well'] = df['plate'].astype(str) + '_' + df['well'].astype(str)
    df[featcols] = harmonize(df[featcols].to_numpy(), df[metacols+['plate_well']], batch_key=parameters['batch_column'])
    print('Batch normalized with harmony', df.shape)

    df.drop(columns=['plate_well'], inplace=True)
    
    for grpn, group in df.groupby(by=['plate', 'donor']):
        group.to_csv(f'../data_harmony_batch_corrected/{'_'.join(grpn)}.csv', index=False)
        print(f'Saved {grpn}')
    print('Saved batch normalized')

    return df

def combat_batch_correction(df, featcols, parameters):
    df['plate_well'] = df['plate'].astype(str) + '_' + df['well'].astype(str)
    # combat batch effect removal <= plate effect removal
    batch_column = parameters['batch_column']
    df[featcols] = pycombat(df[featcols].transpose(), df[batch_column].to_list()).transpose()
    print('Batch normalized with pycombat', df.shape)

    df.drop(columns=['plate_well'], inplace=True)

    return df

def save_meta_coord(df, mapper, parameters, name):
    embeddings = mapper.embedding_
    df['umap_x'] = embeddings[:,0]
    df['umap_y'] = embeddings[:,1]
    df.to_csv(parameters['output_dir'] / f"{name}.csv", index=False)
    print(f'A csv including umap_x & umap_y has been saved')

def make_umap(df, metacols, featcols, parameters):
    print('Starting UMAP')
    n_neighbors = parameters['n_neighbors']
    min_dist    = parameters['min_dist']
    plotname    = f'umap_nn{n_neighbors}_md{min_dist}'

    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, spread=1.0).fit(df[featcols])
    print('Fitting done!')

    # save coordinates with metacols
    save_meta_coord(df[metacols].copy(), mapper, parameters, plotname)
    print('UMAP coordinates are saved.')

def main(parameters, all_donors_id):
    
    # open dataframes
    bigdf = open_and_concat_df(all_donors_id, parameters['input_dir'], parameters)

    # Get column names that do not contain "brightfield"
    columns_without_brightfield = [col for col in bigdf.columns if 'Brightfield' not in col]
    
    # Select columns without "brightfield"
    bigdf = bigdf[columns_without_brightfield].copy()

    metacols = parameters['metacols'] + parameters['dropcols']
    featcols = bigdf.columns[~bigdf.columns.isin(metacols)].to_list()

    if parameters['normalize']:
        bigdf = normalize_df(bigdf, featcols, parameters)

    if parameters['batch_correct'] == 'harmony':
        bigdf = harmony_batch_correction(bigdf, metacols, featcols, parameters)
    elif parameters['batch_correct'] == 'combat':
        bigdf = combat_batch_correction(bigdf, featcols, parameters)
    else:
        print('No batch correction method used!')

    if parameters['make_umap']:
        make_umap(bigdf, metacols, featcols, parameters)


if __name__ == '__main__':
    parameters = {}
    # std scale or normalize between values
    parameters['normalize']     = True
    parameters['normalize_func']= 'scale' # scale or normalize
    parameters['normlow']       = 0.01 # if normalize low threshold
    parameters['normhigh']      = 0.99 # if normalize high threshold
    # path lookup table
    parameters['lookup__dir']   = Path('...')
    # input basedir
    parameters['input_dir']     = Path('...')
    # output directory
    parameters['output_dir']    = Path('...')
    # parameters for UMAP
    parameters['make_umap']     = True
    parameters['n_neighbors']   = 100
    parameters['min_dist']      = 0.01
    # specify columns that is not used for measurement
    parameters['dropcols']      = ['NUCLEUS METCENTER-X', 'NUCLEUS METCENTER-Y', 'NUCLEUS NUMBER OF COMPONENTS',
                                   'CELL METCENTER-X', 'CELL METCENTER-Y', 'CELL NUMBER OF COMPONENTS']
    # metadata columns
    parameters['metacols']      = ['donor', 'plate', 'well', 'fov', 'id']
    # batch correction modes
    parameters['batch_correct'] = 'harmony' # 'harmony', 'combat', or None
    parameters['batch_column']  ='plate' # plate, plate_well, plate_well_fov, well, fov, etc
    # random sample from each donor
    parameters['sample_size'] = 10000
    
    parameters['output_dir'].mkdir(exist_ok=True, parents=True)

    '''For complete features'''
    # Specify donor dataframes
    lookup_df = pd.read_csv(parameters['lookup__dir'], low_memory=False)
    for col in ['donor_start', 'donor_end', 'n_controls', 'n_donors']:
        lookup_df[col] = lookup_df[col].astype('uint')
    
    file_name_list = []
    for idx, row in lookup_df.iterrows():
        file_name_list += [f"donor_{did}.csv" for did in range(row.donor_start, row.donor_end+1)]

    main(parameters, file_name_list)