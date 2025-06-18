import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import json
from os.path import join, basename
from os import getcwd, makedirs
from classifiers import classify_nir, classify_mir1, classify_mir2, classify_nmir, classify_w
from utils import angular_distance, convert_to_degrees, get_adql_query, tap, generate_index
from helpers import convert_2mass_format, filter_glimpse, filter_ukidss, replace_from_2mass

  
def analyse_area(ra, dec, radius, use_gps, out_dir, i):

    #---------INPUTS--------
    print(f'\nArea: {ra}, {dec}, {radius * 60}')
    print(f'Use UKIDSS: {use_gps}')
    ra, dec = convert_to_degrees(ra, dec)
    radius = radius / 60

    # Retrieve data from **UKIDSS GPS**
    base = '2MASS'
    if use_gps:
    #---------------------------------------------UKIDSS-------------------------------------------------
        base = 'UKIDSS'
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

        if len(df) == 0:
            base = '2MASS'
            print('\n', 'No data from UKIDSS, get 2MASS as a base')
            use_gps = False
        else:
            df = filter_ukidss(df)
            df = convert_2mass_format(df)
        

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

    df_2MASS.loc[(df_2MASS['Jmag'] > 16.5) | df_2MASS['e_Jmag'].isna(), ['Jmag', 'e_Jmag']] = pd.NA
    df_2MASS.loc[(df_2MASS['Hmag'] > 15.8) | df_2MASS['e_Hmag'].isna(), ['Hmag', 'e_Hmag']] = pd.NA
    df_2MASS.loc[(df_2MASS['Kmag'] > 15.0) | df_2MASS['e_Kmag'].isna(), ['Kmag', 'e_Kmag']] = pd.NA

    df_2MASS.rename(columns={
            'Jmag': 'J',
            'Hmag': 'H',
            'Kmag': 'K',
            'RAJ2000': 'ra',
            'DEJ2000': 'de',
        }, inplace=True)


    # ----------------------------------Make replacements from 2MASS-------------------------------------
    if use_gps:
        df = replace_from_2mass(df, df_2MASS)
    else:
        df = df_2MASS
        df = df[['_2MASS', 'ra', 'de', 'e_Jmag', 'e_Hmag', 'e_Kmag', 'J', 'H', 'K']]


    # ------------------------------------------Remove Classical Be stars-----------------------------------------
    df = df[~((df['J'] - df['K'] < 0.6) & (df['H'] - df['K'] < 0.3))]
    print(f'Remove Classical Be stars, left rows: {len(df)}')


    # ------------------------------------------Specify NIR class-----------------------------------------
    print('Specify NIR class')
    df = df.copy()
    df.loc[:, 'NIR_Class'] = df.apply(classify_nir, axis=1)


    # ------------------------------------------GLIMPSE-----------------------------------------
    
    df_GLIMPSE = get_adql_query(
        'II/293/glimpse', ra, dec, radius, 'glimpse', c=True,
        cols_to_keep=[
            'GLIMPSE', 'glimpse_ra', 'glimpse_de', 
            '_3_6mag', 'e_3_6mag', '_4_5mag', 'e_4_5mag', '_5_8mag', 'e_5_8mag', '_8_0mag', 'e_8_0mag',
            'F_3_6_', 'e_F_3_6_', 'F_4_5_', 'e_F_4_5_', 'F_5_8_', 'e_F_5_8_', 'F_8_0_', 'e_F_8_0_'
        ]
    )
    print(f'Found rows: {len(df_GLIMPSE)}')

    df_GLIMPSE = filter_glimpse(df_GLIMPSE)
    print(f'Filtering, rows left: {len(df_GLIMPSE)}')

    if len(df_GLIMPSE) > 0:
        print('Specify MIR1 and NMIR classes')
        df_GLIMPSE.loc[:, 'MIR1_Class'] = df_GLIMPSE.apply(classify_mir1, axis=1)

        # ------------------------------------------MIPSGAL-----------------------------------------
    df_MIPSGAL = get_adql_query(
        'J/AJ/149/64/catalog', ra, dec, radius, 'mipsgal',
        cols_to_keep=['MIPSGAL', 'mipsgal_ra', 'mipsgal_de', '__24_', 'e__24_' ]
    )

    if len(df_GLIMPSE) > 0 and len(df_MIPSGAL) > 0:
        print('Merge MIPSGAL to GLIMPS')
        for col in list(df_MIPSGAL.columns):
            df_GLIMPSE.loc[:, col] = pd.NA

        for index, row in tqdm(df_GLIMPSE.iterrows(), total=len(df_GLIMPSE)):
            distances = angular_distance(row['glimpse_ra'], row['glimpse_de'], df_MIPSGAL['mipsgal_ra'], df_MIPSGAL['mipsgal_de'])
            if len(distances) > 0:
                closest = np.where((distances < 3 / 3600) & (distances == distances.min()))
                select = df_MIPSGAL.iloc[closest]
                if len(select) > 0:
                    value = select.iloc[0]['MIPSGAL']
                    if pd.notna(value) and value not in df_GLIMPSE['MIPSGAL'].dropna().values:
                        df_GLIMPSE.loc[index, list(df_MIPSGAL.columns)] = select[list(df_MIPSGAL.columns)].iloc[0]
                    else:
                        [[ind]] = np.where(df_GLIMPSE['MIPSGAL'] == select.iloc[0]['MIPSGAL'])
                        prev = df_GLIMPSE.iloc[ind]
                        dist_prev = angular_distance(prev['glimpse_ra'], prev['glimpse_de'], prev['mipsgal_ra'], prev['mipsgal_de'])
                        dist_curr = distances.min()
                        if dist_prev > dist_curr:
                            df_GLIMPSE.loc[df_GLIMPSE.index[ind], list(df_MIPSGAL.columns)] = pd.NA
                            df_GLIMPSE.loc[index, list(df_MIPSGAL.columns)] = select[list(df_MIPSGAL.columns)].iloc[0]

        print('Remove AGB contaminants')
        df_GLIMPSE = df_GLIMPSE[~((df_GLIMPSE['_4_5mag'] > 7.8) & (df_GLIMPSE['_8_0mag'] - df_GLIMPSE['__24_'] < 2.5))]


    # -------------------------------------------------MERGINGS--------------------------------------
    if len(df_GLIMPSE) > 0:
        print(f'Merge GLIMPS and {base}')
        for col in list(df_GLIMPSE.columns):
            df.loc[:, col] = pd.NA

        for index, row in tqdm(df_GLIMPSE.iterrows(), total=len(df_GLIMPSE)):
            distances = angular_distance(row['glimpse_ra'], row['glimpse_de'], df['ra'], df['de'])
            if len(distances) > 0:
                [closest] = np.where(distances < 1.2 / 3600)
                
                select = df.iloc[closest]
                if len(select) > 0:
                    idx = None
                    if pd.notna(select['K'].idxmin()):
                        idx = select['K'].idxmin()
                    elif pd.notna(select['H'].idxmin()):
                        idx = select['H'].idxmin()              
                    elif pd.notna(select['J'].idxmin()):
                        idx = select['J'].idxmin()   
                    
                    if idx is not None:
                        if pd.notna(df.loc[idx, 'GLIMPSE']):
                            prev = df.loc[idx]
                            dist_prev = angular_distance(prev['glimpse_ra'], prev['glimpse_de'], prev['ra'], prev['de'])
                            dist_cur = angular_distance(row['glimpse_ra'], row['glimpse_de'], prev['ra'], prev['de'])
                            if dist_prev > dist_cur:
                                df.loc[idx, list(df_GLIMPSE.columns)] = row[list(df_GLIMPSE.columns)]
                        else:
                            df.loc[idx, list(df_GLIMPSE.columns)] = row[list(df_GLIMPSE.columns)]
        df.loc[:, 'NMIR_Class'] = df.apply(classify_nmir, axis=1)


    if len(df_MIPSGAL) > 0:
        print(f'Merge rest of MIPSGAL and {base}')
        for index, row in tqdm(df_MIPSGAL.iterrows(), total=len(df_MIPSGAL)):
            distances = angular_distance(row['mipsgal_ra'], row['mipsgal_de'], df['ra'], df['de'])
            if len(distances) > 0:
                closest = np.where(distances < 3 / 3600)
                
                select = df.iloc[closest]
                if len(select) > 0:
                    idx = None
                    if pd.notna(select['K'].idxmin()):
                        idx = select['K'].idxmin()
                    elif pd.notna(select['H'].idxmin()):
                        idx = select['H'].idxmin()              
                    elif pd.notna(select['J'].idxmin()):
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
    
    if len(df_GLIMPSE) > 0 and len(df_MIPSGAL) > 0:
        print('Specify MIR2 class')
        df['MIR2_Class'] = df.apply(classify_mir2, axis=1)
     

    print('Concat the rest from GLIMPSE and MIPSGAL')
    for index, row in tqdm(df_GLIMPSE.iterrows(), total=len(df_GLIMPSE)):
        value = row['GLIMPSE']
        if pd.notna(value) and value not in df['GLIMPSE'].dropna().values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    for index, row in tqdm(df_MIPSGAL.iterrows(), total=len(df_MIPSGAL)):
        value = row['MIPSGAL']
        if pd.notna(value) and value not in df['MIPSGAL'].dropna().values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)



    # -------------------------------------------------ALLWISE--------------------------------------
    df_ALLWISE = get_adql_query(
        'II/328/allwise', ra, dec, radius, 'allwise',
        cols_to_keep=[
            'AllWISE', 'allwise_ra', 'allwise_de', 
            'W1mag', 'W2mag', 'W3mag', 'W4mag', 'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag']
    )

    df_ALLWISE = df_ALLWISE[(df_ALLWISE['e_W1mag'] < 0.2) & (df_ALLWISE['e_W2mag'] < 0.2) & (df_ALLWISE['e_W3mag'] < 0.2) & (df_ALLWISE['e_W4mag'] < 0.2)]
    print(f'Filtering, rows left: {len(df_ALLWISE)}')
    print('Specify W class')
    df_ALLWISE['W_Class'] = df_ALLWISE.apply(classify_w, axis=1)

    print(f'Merge ALLWISE to {base}')
    for col in list(df_ALLWISE.columns):
        df.loc[:, col] = pd.NA

    for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
        distances = angular_distance(row['allwise_ra'], row['allwise_de'], df['ra'], df['de'])
        if len(distances) > 0:
            closest = np.where(distances < 3 / 3600)
            
            select = df.iloc[closest]
            if len(select) > 0:
                idx = None
                if pd.notna(select['K'].idxmin()):
                    idx = select['K'].idxmin()
                elif pd.notna(select['H'].idxmin()):
                    idx = select['H'].idxmin()              
                elif pd.notna(select['J'].idxmin()):
                    idx = select['J'].idxmin()   
                
                if idx is not None:
                    if pd.notna(df.loc[idx, 'AllWISE']):
                        prev = df.loc[idx]
                        dist_prev = angular_distance(prev['allwise_ra'], prev['allwise_de'], prev['ra'], prev['de'])
                        dist_cur = angular_distance(row['allwise_ra'], row['allwise_de'], prev['ra'], prev['de'])
                        if dist_prev > dist_cur:
                            df.loc[idx, list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]
                    else:
                        df.loc[idx, list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]

    print('Continue to merge ALLWISE via GLIMPSE and MIPSGAL')

    if len(df_GLIMPSE) > 0:
        for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
            value = row['AllWISE']
            if pd.notna(value) and value in df['AllWISE'].dropna().values:
                continue
            distances = angular_distance(row['allwise_ra'], row['allwise_de'], df['glimpse_ra'], df['glimpse_de'])
            if len(distances) > 0:
                [closest] = np.where((distances < 3 / 3600) & (distances == distances.min()))
                if len(closest) > 0:
                    if not df.iloc[closest[0]]['AllWISE']:
                        df.iloc[closest[0], list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]

    if len(df_MIPSGAL) > 0:
        for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
            value = row['AllWISE']
            if pd.notna(value) and value in df['AllWISE'].dropna().values:
                continue
            distances = angular_distance(row['allwise_ra'], row['allwise_de'], df['mipsgal_ra'], df['mipsgal_de'])
            if len(distances) > 0:
                [closest] = np.where((distances < 3 / 3600) & (distances == distances.min()))
                if len(closest) > 0:
                    if not df.iloc[closest[0]]['AllWISE']:
                        df.iloc[closest[0], list(df_ALLWISE.columns)] = row[list(df_ALLWISE.columns)]

    print('Attach the rest of ALLWISE')
    for index, row in tqdm(df_ALLWISE.iterrows(), total=len(df_ALLWISE)):
        value = row['AllWISE']
        if pd.notna(value) and value in df['AllWISE'].dropna().values:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    cols_include = [
        x for x in [
        'UGPS' if use_gps else '_2MASS', 'ra', 'de', 'J', 'e_Jmag', 'H', 'e_Hmag', 'K', 'e_Kmag', 
        'GLIMPSE', 'glimpse_ra', 'glimpse_de', '_3_6mag', 'e_3_6mag', '_4_5mag', 'e_4_5mag', '_5_8mag', 'e_5_8mag', '_8_0mag', 'e_8_0mag',  
        'MIPSGAL', 'mipsgal_ra', 'mipsgal_de', '__24_', 'e__24_',  
        'AllWISE', 'allwise_ra', 'allwise_de', 'W1mag', 'e_W1mag', 'W2mag', 'e_W2mag', 'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag', 
        'NIR_Class', 'MIR1_Class', 'MIR2_Class', 'NMIR_Class', 'W_Class'
    ] if x in df]
    df = df[cols_include]
    
    df.rename(columns={
        '_3_6mag': '3_6mag',
        '_4_5mag': '4_5mag',
        '_5_8mag': '5_8mag',
        '_8_0mag': '8_0mag',
        '__24_': '_24',
        '_2MASS': '2MASS',
        'e__24_': 'e_24',
    }, inplace=True)

    class_columns = [x for x in ['NIR_Class', 'MIR1_Class', 'MIR2_Class', 'NMIR_Class', 'W_Class'] if x in df]

    df.index = df.apply(generate_index, axis=1)

    # Separate the rows where any of the specified columns is not None
    df_not_none = df.dropna(subset=class_columns, how='all')
    # The remaining rows where all specified columns are None
    df_others = df.loc[df.index.difference(df_not_none.index)]
    df_others.drop(columns=class_columns, inplace=True)
    
    df_not_none.to_csv(out_dir + f'_{i}_class.csv')
    df_others.to_csv(out_dir + f'_{i}_no_class.csv')

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

