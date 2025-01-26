import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import json
from os.path import join, basename
from os import getcwd, makedirs
from classifiers import classify_nir, classify_mir1, classify_mir2, classify_nmir, classify_w
from utils import angular_distance, convert_to_degrees, get_adql_query, tap, generate_index

  
def analyse_area(ra, dec, radius, use_gps, out_dir, i):

    #---------INPUTS--------
    print(f'\nArea: {ra}, {dec}, {radius * 60}')
    print(f'Use UKIDSS: {use_gps}')
    ra, dec = convert_to_degrees(ra, dec)
    radius = radius / 60

    # Retrieve data from **UKIDSS GPS**
    if use_gps:
    #---------------------------------------------UKIDSS-------------------------------------------------
        print('\nRetrieve data from UKIDSS...')
        def adql_query(n):
            return f"""
            SELECT TOP {n} UGPS, RAICRS, DEICRS, Jmag, e_Jmag, Hmag, e_Hmag, Kmag1, e_Kmag1
            FROM "II/316/gps6"
            WHERE 1=CONTAINS(
                POINT('ICRS', RAICRS, DEICRS),
                CIRCLE('ICRS', {ra}, {dec}, {radius})
            )
            AND (e_Jmag IS NOT NULL OR e_Hmag IS NOT NULL OR e_Kmag1 IS NOT NULL)
            AND (Jmag <= 19.77 OR Hmag <= 19.00 OR Kmag1 <= 18.05) 
            AND pN < 0.33
            AND Kflag1 < 64
            """        

        n = 10000
        while True:
            job = tap.launch_job(adql_query(n), maxrec=-1)
            result = job.get_results()
            if len(result) == n:
                n *= 2
                print(f'*** TRY {n}')
            else:
                break
        # Convert result to a Pandas DataFrame
        df = result.to_pandas()
        
        print(f'Found rows: {len(df)}')
        df.replace([None, ''], np.nan, inplace=True)
        df.loc[
            (df['Kmag1'] > 18.05) 
            | df['Kmag1'].isna() 
            | df['e_Kmag1'].isna(), 
            ['Kmag1', 'e_Kmag1']
        ] = np.nan

        df.loc[(
            (df['Jmag'] > 19.77) 
            | df['Jmag'].isna() 
            | df['e_Jmag'].isna() 
            | df['Kmag1'].isna() 
            | df['e_Kmag1'].isna()
        ),
            ['Jmag', 'e_Jmag']
        ] = np.nan

        df.loc[(
            (df['Hmag'] > 19.00) 
            | df['Hmag'].isna() 
            | df['e_Hmag'].isna()
            | df['Kmag1'].isna() 
            | df['e_Kmag1'].isna()
            | df['Jmag'].isna() 
            | df['e_Jmag'].isna() 
        ), 
            ['Hmag', 'e_Hmag']
        ] = np.nan
        
        print('Convert to 2MASS format')
        df['J'] = 1.073 * (df['Jmag'] - df['Kmag1']) + df['Kmag1'] - 0.01
        df['H'] = 1.062*(df['Hmag'] - df['Kmag1']) + df['Kmag1'] + 0.004 * (df['Jmag'] - df['Kmag1']) + 0.019
        df['K'] = df['Kmag1'] + 0.002
        df.loc[~df['Jmag'].isna(), 'K'] = df['K'] + 0.004 * (df['Jmag'] - df['Kmag1'])
        df.rename(columns={
            'e_Kmag': 'e_Kmag1',
            'RAICRS': 'ra',
            'DEICRS': 'de',
        }, inplace=True)
        del df['Jmag'], df['Hmag'], df['Kmag1']

    #---------------------------------------------2MASS-------------------------------------------------
    print('\nRetrieve data from 2MASS...')
    def adql_query(n):
        return f"""
            SELECT TOP {n} *
            FROM "II/246/out"
            WHERE 1=CONTAINS(
            POINT('ICRS', RAJ2000, DEJ2000),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
            )
            AND (e_Jmag IS NOT NULL OR e_Hmag IS NOT NULL OR e_Kmag IS NOT NULL)
            AND (Jmag <= 16.5 OR Hmag <= 15.8 OR Kmag <= 15.0)
        """
    n = 10000
    while True:
        job = tap.launch_job(adql_query(n), maxrec=-1)
        result = job.get_results()
        if len(result) == n:
            n *= 2
        else:
            break

    # Convert result to a Pandas DataFrame
    df_2MASS = result.to_pandas()
    print(f'Found rows: {len(df_2MASS)}')

    df_2MASS.loc[(df_2MASS['Jmag'] > 16.5) | df_2MASS['e_Jmag'].isna(), ['Jmag', 'e_Jmag']] = np.nan
    df_2MASS.loc[(df_2MASS['Hmag'] > 15.8) | df_2MASS['e_Hmag'].isna(), ['Hmag', 'e_Hmag']] = np.nan
    df_2MASS.loc[(df_2MASS['Kmag'] > 15.0) | df_2MASS['e_Kmag'].isna(), ['Kmag', 'e_Kmag']] = np.nan

    df_2MASS.rename(columns={
            'Jmag': 'J',
            'Hmag': 'H',
            'Kmag': 'K',
            'RAJ2000': 'ra',
            'DEJ2000': 'de',
        }, inplace=True)


    # ----------------------------------Make replacements from 2MASS-------------------------------------
    if use_gps:
        print('Make replacements in UKIDSS from 2MASS...')
        _2mass_cols = ['_2MASS', '2J', '2H', '2K', '2e_Jmag', '2e_Hmag', '2e_Kmag']
        for k in _2mass_cols:
            df.loc[:, k] = None
            
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if row['J'] < 13.25 or row['H'] < 12.75 or row['K'] < 12:
                distances = angular_distance(row['ra'], row['de'], df_2MASS['ra'], df_2MASS['de'])
                closest = np.where(distances < 1/3600)
                select = df_2MASS.iloc[closest]
                if len(select) > 0:
                    if not np.isnan(select['K'].idxmin()):
                        select = select.loc[select['K'].idxmin()]
                    elif not np.isnan(select['H'].idxmin()):
                        select = select.loc[select['H'].idxmin()]
                    elif not np.isnan(select['J'].idxmin()):
                        select = select.loc[select['J'].idxmin()]
                    
                    if select['_2MASS'] in list(df['_2MASS']):
                        [[ind]] = np.where(df['_2MASS'] == select['_2MASS'])
                        clear_prev = False

                        if np.isnan(df.iloc[ind]['K']):
                            if not np.isnan(row['K']):
                                clear_prev = True
                            elif np.isnan(df.iloc[ind]['H']):
                                if not np.isnan(row['H']):
                                    clear_prev = True
                                elif row['J'] < df.iloc[ind]['J']:
                                    clear_prev = True
                            elif not np.isnan(row['H']) and row['H'] < df.iloc[ind]['H']:
                                clear_prev = True
                        elif not np.isnan(row['K']) and row['K'] < df.iloc[ind]['K']:
                            clear_prev = True
                        
                        if clear_prev:
                            df.loc[df.index[ind], _2mass_cols] = None
                            if row['J'] < 13.25 and not np.isnan(select['J']):
                                df.loc[index, ['_2MASS', '2J', '2e_Jmag']] = select[['_2MASS', 'J', 'e_Jmag']]
                            if row['H'] < 12.75 and not np.isnan(select['H']):
                                df.loc[index, ['_2MASS', '2H', '2e_Hmag']] = select[['_2MASS', 'H', 'e_Hmag']]
                            if row['K'] < 12 and not np.isnan(select['K']):
                                df.loc[index, ['_2MASS', '2K', '2e_Kmag']] = select[['_2MASS', 'K', 'e_Kmag']]
                            
                    else:
                        if row['J'] < 13.25 and not np.isnan(select['J']):
                            df.loc[index, ['_2MASS', '2J', '2e_Jmag']] = select[['_2MASS', 'J', 'e_Jmag']]
                        if row['H'] < 12.75 and not np.isnan(select['H']):
                            df.loc[index, ['_2MASS', '2H', '2e_Hmag']] = select[['_2MASS', 'H', 'e_Hmag']]
                        if row['K'] < 12 and not np.isnan(select['K']):
                            df.loc[index, ['_2MASS', '2K', '2e_Kmag']] = select[['_2MASS', 'K', 'e_Kmag']]
        df['K'] = df.apply(lambda row: row['2K'] if pd.notna(row['2K']) else row['K'], axis=1)
        df['H'] = df.apply(lambda row: row['2H'] if pd.notna(row['2H']) else row['H'], axis=1)
        df['J'] = df.apply(lambda row: row['2J'] if pd.notna(row['2J']) else row['J'], axis=1)
        df['e_Jmag'] = df.apply(lambda row: row['2e_Jmag'] if pd.notna(row['2e_Jmag']) else row['e_Jmag'], axis=1)
        df['e_Hmag'] = df.apply(lambda row: row['2e_Hmag'] if pd.notna(row['2e_Hmag']) else row['e_Hmag'], axis=1)
        df['e_Kmag'] = df.apply(lambda row: row['2e_Kmag'] if pd.notna(row['2e_Kmag']) else row['e_Kmag1'], axis=1)
    else:
        df = df_2MASS
        df = df[['_2MASS', 'ra', 'de', 'e_Jmag', 'e_Hmag', 'e_Kmag', 'J', 'H', 'K']]


    # ------------------------------------------Remove Classical Be stars-----------------------------------------
    # If object have J–K < 0.6 and H–K < 0.3, we remove it from the list
    df = df[~((df['J'] - df['K'] < 0.6) & (df['H'] - df['K'] < 0.3))]
    print(f'Remove Classical Be stars, left rows: {len(df)}')


    # ------------------------------------------Specify NIR class-----------------------------------------
    print('Specify NIR class')
    df = df.copy()
    df.loc[:, 'NIR_Class'] = df.apply(classify_nir, axis=1)


    # ------------------------------------------GLIMPSE-----------------------------------------
    print('\nRetrieve data from GLIMPSE...')
    df_GLIMPSE = get_adql_query('II/293/glimpse', ra, dec, radius, 'glimpse', c=True)
    print(f'Found rows: {len(df_GLIMPSE)}')

    # [3.6]–[5.8] < 0.75×([4.5]–[8.0]-1),  [3.6]–[5.8] < 1.5, [4.5]–[8.0] > 1
    df_GLIMPSE = df_GLIMPSE[~((
        df_GLIMPSE['_3_6mag'] - df_GLIMPSE['_5_8mag'] 
        < 0.75 * (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 1)
    ) & (
        df_GLIMPSE['_3_6mag'] - df_GLIMPSE['_5_8mag'] < 1.5
    ) & (
        df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] > 1
    ))]

    # [4.5]–[8.0] > 0.5, [4.5] > 13.5+([4.5]–[8.0]-2.3)/0.4, and [4.5] > 13.5
    df_GLIMPSE = df_GLIMPSE[~((
        df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] > 0.5
    ) & (
        df_GLIMPSE['_4_5mag'] > 13.5 + (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 2.3) / 0.4
    ) & (
        df_GLIMPSE['_4_5mag'] > 13.5
    ))]

    # [4.5]> 14 +([4.5]-[8.0]-0.5);
    # [4.5]> 14.5 -([4.5]-[8.0]-1.2)/0.3;
    # [4.5]> 14.5
    df_GLIMPSE = df_GLIMPSE[~((
        df_GLIMPSE['_4_5mag'] > 14 + (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 0.5)
    ) | (
    #     True
        df_GLIMPSE['_4_5mag'] > 14.5 - (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 1.2) / 0.3
    ) | (
    #     True
        df_GLIMPSE['_4_5mag'] > 14.5
    ))]

    print(f'Filtering, rows left: {len(df_GLIMPSE)}')

    print('Specify MIR1 class')
    df_GLIMPSE.loc[:, 'MIR1_Class'] = df_GLIMPSE.apply(classify_mir1, axis=1)
    glimpse_cols_to_keep = [
        'GLIMPSE', 'glimpse_ra', 'glimpse_de', 
        '_3_6mag', 'e_3_6mag', '_4_5mag', 'e_4_5mag', '_5_8mag', 'e_5_8mag', '_8_0mag', 'e_8_0mag',
        'F_3_6_', 'e_F_3_6_', 'F_4_5_', 'e_F_4_5_', 'F_5_8_', 'e_F_5_8_', 'F_8_0_', 'e_F_8_0_',
        'MIR1_Class'
    ]
    df_GLIMPSE = df_GLIMPSE[glimpse_cols_to_keep]


    # ------------------------------------------MIPSGAL-----------------------------------------
    print('\nRetrieve data from MIPSGAL...')
    df_MIPSGAL = get_adql_query('J/AJ/149/64/catalog', ra, dec, radius, 'mipsgal')
    mipsgal_cols_to_keep = ['MIPSGAL', 'mipsgal_ra', 'mipsgal_de', '__24_', 'e__24_' ]
    df_MIPSGAL = df_MIPSGAL[mipsgal_cols_to_keep]
    print(f'Found rows: {len(df_MIPSGAL)}')

    print('Merge MIPSGAL to GLIMPS')
    for col in list(df_MIPSGAL.columns):
        df_GLIMPSE.loc[:, col] = None

    for index, row in tqdm(df_GLIMPSE.iterrows(), total=len(df_GLIMPSE)):
        distances = angular_distance(row['glimpse_ra'], row['glimpse_de'], df_MIPSGAL['mipsgal_ra'], df_MIPSGAL['mipsgal_de'])
        if len(distances) > 0:
            closest = np.where((distances < 3/3600) & (distances == distances.min()))
            select = df_MIPSGAL.iloc[closest]
            if len(select) > 0:
                if select.iloc[0]['MIPSGAL'] not in list(df_GLIMPSE['MIPSGAL']):
                    df_GLIMPSE.loc[index, list(df_MIPSGAL.columns)] = select[list(df_MIPSGAL.columns)].iloc[0]
                else:
                    [[ind]] = np.where(df_GLIMPSE['MIPSGAL'] == select.iloc[0]['MIPSGAL'])
                    prev = df_GLIMPSE.iloc[ind]
                    dist_prev = angular_distance(prev['glimpse_ra'], prev['glimpse_de'], prev['mipsgal_ra'], prev['mipsgal_de'])
                    dist_curr = distances.min()
                    if dist_prev > dist_curr:
                        df_GLIMPSE.loc[df_GLIMPSE.index[ind], list(df_MIPSGAL.columns)] = None
                        df_GLIMPSE.loc[index, list(df_MIPSGAL.columns)] = select[list(df_MIPSGAL.columns)].iloc[0]

    print('Remove AGB contaminants')
    df_GLIMPSE = df_GLIMPSE[~((df_GLIMPSE['_4_5mag'] > 7.8) & (df_GLIMPSE['_8_0mag'] - df_GLIMPSE['__24_'] < 2.5))]


    # -------------------------------------------------MERGINGS--------------------------------------
    print('Merge GLIMPS and UKIDSS')
    for k in list(df_GLIMPSE.columns):
        df.loc[:, k] = None

    for index, row in tqdm(df_GLIMPSE.iterrows(), total=len(df_GLIMPSE)):
        distances = angular_distance(row['glimpse_ra'], row['glimpse_de'], df['ra'], df['de'])
        if len(distances) > 0:
            closest = np.where(distances < 1.2 / 3600)
            
            select = df.iloc[closest]
            if len(select) > 0:
                idx = None
                if not np.isnan(select['K'].idxmin()):
                    idx = select['K'].idxmin()
                elif not np.isnan(select['H'].idxmin()):
                    idx = select['H'].idxmin()              
                elif not np.isnan(select['J'].idxmin()):
                    idx = select['J'].idxmin()   
                
                if idx is not None:
                    if df.loc[idx, 'GLIMPSE'] is not None:
                        prev = df.loc[idx]
                        dist_prev = angular_distance(prev['glimpse_ra'], prev['glimpse_de'], prev['ra'], prev['de'])
                        dist_cur = angular_distance(row['glimpse_ra'], row['glimpse_de'], prev['ra'], prev['de'])
                        if dist_prev > dist_cur:
                            df.loc[idx, list(df_GLIMPSE.columns)] = row[list(df_GLIMPSE.columns)]
                    else:
                        df.loc[idx, list(df_GLIMPSE.columns)] = row[list(df_GLIMPSE.columns)]

    print('Merge rest of MIPSGAL and UKIDSS')
    for index, row in tqdm(df_MIPSGAL.iterrows(), total=len(df_MIPSGAL)):
        distances = angular_distance(row['mipsgal_ra'], row['mipsgal_de'], df['ra'], df['de'])
        if len(distances) > 0:
            closest = np.where(distances < 3 / 3600)
            
            select = df.iloc[closest]
            if len(select) > 0:
                idx = None
                if not np.isnan(select['K'].idxmin()):
                    idx = select['K'].idxmin()
                elif not np.isnan(select['H'].idxmin()):
                    idx = select['H'].idxmin()              
                elif not np.isnan(select['J'].idxmin()):
                    idx = select['J'].idxmin()   
                
                if idx is not None:
                    if df.loc[idx, 'MIPSGAL'] is not None:
                        prev = df.loc[idx]
                        dist_prev = angular_distance(prev['mipsgal_ra'], prev['mipsgal_de'], prev['ra'], prev['de'])
                        dist_cur = angular_distance(row['mipsgal_ra'], row['mipsgal_de'], prev['ra'], prev['de'])
                        if dist_prev > dist_cur:
                            df.loc[idx, list(df_MIPSGAL.columns)] = row[list(df_MIPSGAL.columns)]
                    else:
                        df.loc[idx, list(df_MIPSGAL.columns)] = row[list(df_MIPSGAL.columns)]     

    print('Concat the rest from GLIMPSE and MIPSGAL')
    for index, row in tqdm(df_GLIMPSE.iterrows(), total=len(df_GLIMPSE)):
        if row['GLIMPSE'] not in list(df['GLIMPSE']):
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    for index, row in tqdm(df_MIPSGAL.iterrows(), total=len(df_MIPSGAL)):
        if row['MIPSGAL'] not in list(df['MIPSGAL']):
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print('Specify MIR2 and NMIR')
    df['MIR2_Class'] = df.apply(classify_mir2, axis=1)
    df['NMIR_Class'] = df.apply(classify_nmir, axis=1)


    # -------------------------------------------------ALLWISE--------------------------------------
    print('\nRetrieve data from ALLWISE...')
    df_ALLWISE = get_adql_query('II/328/allwise', ra, dec, radius, 'allwise')
    allwise_cols_to_keep = ['AllWISE', 'allwise_ra', 'allwise_de', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 
                    'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag']
    df_ALLWISE = df_ALLWISE[allwise_cols_to_keep]
    print(f'Found rows: {len(df_ALLWISE)}')
    df_ALLWISE = df_ALLWISE[(df_ALLWISE['e_W1mag'] < 0.2) & (df_ALLWISE['e_W2mag'] < 0.2) & (df_ALLWISE['e_W3mag'] < 0.2) & (df_ALLWISE['e_W4mag'] < 0.2)]
    print(f'Filtering, rows left: {len(df_ALLWISE)}')
    print('Specify W class')
    df_ALLWISE['W_Class'] = df_ALLWISE.apply(classify_w, axis=1)

    print('Merge ALLWISE to UKIDSS')
    for k in list(df_ALLWISE.columns):
        df.loc[:, k] = None

    for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
        distances = angular_distance(row['allwise_ra'], row['allwise_de'], df['ra'], df['de'])
        if len(distances) > 0:
            closest = np.where(distances < 3 / 3600)
            
            select = df.iloc[closest]
            if len(select) > 0:
                idx = None
                if not np.isnan(select['K'].idxmin()):
                    idx = select['K'].idxmin()
                elif not np.isnan(select['H'].idxmin()):
                    idx = select['H'].idxmin()              
                elif not np.isnan(select['J'].idxmin()):
                    idx = select['J'].idxmin()   
                
                if idx is not None:
                    if df.loc[idx, 'AllWISE'] is not None:
                        prev = df.loc[idx]
                        dist_prev = angular_distance(prev['allwise_ra'], prev['allwise_de'], prev['ra'], prev['de'])
                        dist_cur = angular_distance(row['allwise_ra'], row['allwise_de'], prev['ra'], prev['de'])
                        if dist_prev > dist_cur:
                            df.loc[idx, list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]
                    else:
                        df.loc[idx, list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]

    print('Continue to merge ALLWISE via GLIMPSE and MIPSGAL')
    df['glimpse_ra'] = df['glimpse_ra'].fillna(np.nan)
    df['glimpse_de'] = df['glimpse_de'].fillna(np.nan)
    df['mipsgal_ra'] = df['mipsgal_ra'].fillna(np.nan)
    df['mipsgal_de'] = df['mipsgal_de'].fillna(np.nan)

    for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
        if row['AllWISE'] in list(df['AllWISE']):
            continue
        distances = angular_distance(row['allwise_ra'], row['allwise_de'], df['glimpse_ra'], df['glimpse_de'])
        if len(distances) > 0:
            [closest] = np.where((distances < 3 / 3600) & (distances == distances.min()))
            if len(closest) > 0:
                if not df.iloc[closest[0]]['AllWISE']:
                    df.iloc[closest[0], list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]

    for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
        if row['AllWISE'] in list(df['AllWISE']):
            continue
        distances = angular_distance(row['allwise_ra'], row['allwise_de'], df['mipsgal_ra'], df['mipsgal_de'])
        if len(distances) > 0:
            [closest] = np.where((distances < 3 / 3600) & (distances == distances.min()))
            if len(closest) > 0:
                if not df.iloc[closest[0]]['AllWISE']:
                    df.iloc[closest[0], list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]

    print('Attach the rest of ALLWISE')
    for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
        if row['AllWISE'] not in list(df['AllWISE']):
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df[[
        'UGPS' if use_gps else '_2MASS', 'ra', 'de', 'J', 'e_Jmag', 'H', 'e_Hmag', 'K', 'e_Kmag', 
        'GLIMPSE', 'glimpse_ra', 'glimpse_de', '_3_6mag', 'e_3_6mag', '_4_5mag', 'e_4_5mag', '_5_8mag', 'e_5_8mag', '_8_0mag', 'e_8_0mag',  
        'MIPSGAL', 'mipsgal_ra', 'mipsgal_de', '__24_', 'e__24_',  
        'AllWISE', 'allwise_ra', 'allwise_de', 'W1mag', 'e_W1mag', 'W2mag', 'e_W2mag', 'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag', 
        'NIR_Class', 'MIR1_Class', 'MIR2_Class', 'NMIR_Class', 'W_Class'
    ]]
    
    df.rename(columns={
        '_3_6mag': '3_6mag',
        '_4_5mag': '4_5mag',
        '_5_8mag': '5_8mag',
        '_8_0mag': '8_0mag',
        '__24_': '_24',
        '_2MASS': '2MASS',
        'e__24_': 'e_24',
    }, inplace=True)

    condition_columns = ['NIR_Class', 'MIR1_Class', 'MIR2_Class', 'NMIR_Class', 'W_Class']
    df[condition_columns] = df[condition_columns].replace('', None)

    df.index = df.apply(generate_index, axis=1)

    # Separate the rows where any of the specified columns is not None
    df_not_none = df.dropna(subset=condition_columns, how='all')
    # The remaining rows where all specified columns are None
    df_others = df.loc[df.index.difference(df_not_none.index)]
    df_others.drop(columns=condition_columns, inplace=True)
    
    df_not_none.to_excel(out_dir + f'_{i}_class.xlsx')
    df_others.to_excel(out_dir + f'_{i}_no_class.xlsx')

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    if "output_dir" not in data or not data['output_dir'] or data['output_dir'] == '':
        out_dir = join(getcwd(), 'output')
    else:
        out_dir = data['output_dir']
    makedirs(out_dir, exist_ok=True)
    use_gps = data['use_gps']

    for i, (ra, dec, radius) in enumerate(data['data']):
        analyse_area(ra, dec, radius, use_gps, join(out_dir, basename(sys.argv[1]).replace('.json', '')), i)

