import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import angular_distance


def filter_glimpse(df_GLIMPSE):
    df_GLIMPSE = df_GLIMPSE.copy()

    condition1 = ~((df_GLIMPSE['_3_6mag'] - df_GLIMPSE['_5_8mag'] < 0.75 * (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 1))
               & (df_GLIMPSE['_3_6mag'] - df_GLIMPSE['_5_8mag'] < 1.5)
               & (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] > 1))

    condition2 = ~((df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] > 0.5)
                & (df_GLIMPSE['_4_5mag'] > 13.5 + (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 2.3) / 0.4)
                & (df_GLIMPSE['_4_5mag'] > 13.5))

    condition3 = ~((df_GLIMPSE['_4_5mag'] > 14 + (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 0.5))
                | (df_GLIMPSE['_4_5mag'] > 14.5 - (df_GLIMPSE['_4_5mag'] - df_GLIMPSE['_8_0mag'] - 1.2) / 0.3)
                | (df_GLIMPSE['_4_5mag'] > 14.5))

    df_GLIMPSE = df_GLIMPSE[condition1 & condition2 & condition3]

    return df_GLIMPSE


def get_idx_min(select):
    idx = None
    if select['K'].notna().any():
        idx = select['K'].idxmin()
    elif select['H'].notna().any():
        idx = select['H'].idxmin()
    elif select['J'].notna().any():
        idx = select['J'].idxmin()
    return idx

def replace_from_2mass(df, df_2MASS, base):
    print('Make replacements in UKIDSS from 2MASS...')

    j_thr, h_thr, k_thr = {
        'UGPS': [13.25, 12.75, 12],
        'VVV': [10.5, 10.0, 9.5],
    }[base]

    _2mass_cols = ['_2MASS', '2J', '2H', '2K', '2e_Jmag', '2e_Hmag', '2e_Kmag']
    for k in _2mass_cols:
        df.loc[:, k] = pd.NA
        
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row['J'] < j_thr or row['H'] < h_thr or row['K'] < k_thr:
            distances = angular_distance(row['ra'], row['de'], df_2MASS['ra'], df_2MASS['de'])
            [closest] = np.where((distances < 1 / 3600) & (distances == distances.min()))

            if len(closest) > 0:
                select = df_2MASS.iloc[closest]
                select = select.loc[get_idx_min(select)]
                

                value = select['_2MASS']
                should_replace = False
                if pd.notna(value) and value in df['_2MASS'].dropna().values:
                    [[ind]] = np.where(df['_2MASS'] == select['_2MASS'])
                    clear_prev = False

                    if np.isnan(df.iloc[ind]['K']):
                        if pd.notna(row['K']):
                            clear_prev = True
                        elif pd.notna(df.iloc[ind]['H']):
                            if pd.notna(row['H']):
                                clear_prev = True
                            elif row['J'] < df.iloc[ind]['J']:
                                clear_prev = True
                        elif pd.notna(row['H']) and row['H'] < df.iloc[ind]['H']:
                            clear_prev = True
                    elif pd.notna(row['K']) and row['K'] < df.iloc[ind]['K']:
                        clear_prev = True
                    
                    if clear_prev:
                        df.loc[df.index[ind], _2mass_cols] = None
                        should_replace = True                        
                else:
                    should_replace = True

                if should_replace:
                    if row['J'] < j_thr and pd.notna(select['J']):
                        df.loc[index, ['_2MASS', '2J', '2e_Jmag']] = select[['_2MASS', 'J', 'e_Jmag']]
                    if row['H'] < h_thr and pd.notna(select['H']):
                        df.loc[index, ['_2MASS', '2H', '2e_Hmag']] = select[['_2MASS', 'H', 'e_Hmag']]
                    if row['K'] < k_thr and pd.notna(select['K']):
                        df.loc[index, ['_2MASS', '2K', '2e_Kmag']] = select[['_2MASS', 'K', 'e_Kmag']]

    df['K'] = df.apply(lambda row: row['2K'] if pd.notna(row['2K']) else row['K'], axis=1)
    df['H'] = df.apply(lambda row: row['2H'] if pd.notna(row['2H']) else row['H'], axis=1)
    df['J'] = df.apply(lambda row: row['2J'] if pd.notna(row['2J']) else row['J'], axis=1)
    if base == 'UGPS':
        df['e_Jmag'] = df.apply(lambda row: row['2e_Jmag'] if pd.notna(row['2e_Jmag']) else row['e_Jmag'], axis=1)
        df['e_Hmag'] = df.apply(lambda row: row['2e_Hmag'] if pd.notna(row['2e_Hmag']) else row['e_Hmag'], axis=1)
        df['e_Kmag'] = df.apply(lambda row: row['2e_Kmag'] if pd.notna(row['2e_Kmag']) else row['e_Kmag1'], axis=1)
    elif base == 'VVV':
        df['e_Jmag'] = df.apply(lambda row: row['2e_Jmag'] if pd.notna(row['2e_Jmag']) else row['e_Jmag3'], axis=1)
        df['e_Hmag'] = df.apply(lambda row: row['2e_Hmag'] if pd.notna(row['2e_Hmag']) else row['e_Hmag3'], axis=1)
        df['e_Kmag'] = df.apply(lambda row: row['2e_Kmag'] if pd.notna(row['2e_Kmag']) else row['e_Ksmag3'], axis=1)
        del df['e_Ksmag3'], df['e_Hmag3'], df['e_Jmag3']

    return df

def filter_ukidss(df):
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
    ),
        ['Jmag', 'e_Jmag']
    ] = np.nan

    df.loc[(
        (df['Hmag'] > 19.00) 
        | df['Hmag'].isna() 
        | df['e_Hmag'].isna()
        | df['Kmag1'].isna() 
        | df['Jmag'].isna() 
    ), 
        ['Hmag', 'e_Hmag']
    ] = np.nan

    return df

def filter_vvv(df):
    df.replace([None, ''], np.nan, inplace=True)
    df.loc[(
        (df['Ksmag3'] > 18.0) 
        | df['Ksmag3'].isna() 
        | df['e_Ksmag3'].isna()
        | df['Hmag3'].isna() 
        | df['e_Hmag3'].isna()
        | df['Ksperrbits'] > 256
    ),
        ['Ksmag3', 'e_Ksmag3']
    ] = np.nan

    df.loc[(
        (df['Jmag3'] > 19.5) 
        | df['Jmag3'].isna() 
        | df['e_Jmag3'].isna() 
        | df['Hmag3'].isna() 
        | df['e_Hmag3'].isna()
        | df['Jperrbits'] > 256
    ),
        ['Jmag3', 'e_Jmag3']
    ] = np.nan

    df.loc[(
        (df['Hmag3'] > 19.00) 
        | df['Hmag3'].isna() 
        | df['e_Hmag3'].isna()
        | df['Ksmag3'].isna() 
        | df['e_Ksmag3'].isna()
        | df['Hperrbits'] > 256
    ), 
        ['Hmag3', 'e_Hmag3']
    ] = np.nan

    return df

def filter_allwise(df):
    df.replace([None, ''], np.nan, inplace=True)
    df.loc[(
        df['e_W1mag'].isna() | ~(df['chi2W1'] < (df['snr1'] - 3) / 7)
    ),
        ['W1mag', 'e_W1mag']
    ] = np.nan
    df.loc[df['e_W2mag'].isna(),
        ['W2mag', 'e_W2mag']
    ] = np.nan
    df.loc[(
        df['e_W3mag'].isna() 
        | ~(
            (df['snr3'] >= 5) 
            & (
                (df['chi2W3'] < (df['snr3'] - 8) / 8) | 
                ((df['chi2W3'] < 1.15) & (df['chi2W3'] > 0.45))
            )
        )
    ),
        ['W3mag', 'e_W3mag']
    ] = np.nan
    
    df.loc[(
        df['e_W4mag'].isna()
        | ~(df['chi2W4'] < (2 * df['snr4'] - 20) / 10)
    ),
        ['W4mag', 'e_W4mag']
    ] = np.nan

    return df

def convert_2mass_format(df, base):
    print('Convert to 2MASS format')
    if base == 'UGPS':
        df['J'] = 1.073 * (df['Jmag'] - df['Kmag1']) + df['Kmag1'] - 0.01
        df['H'] = 1.062 * (df['Hmag'] - df['Kmag1']) + df['Kmag1'] + 0.004 * (df['Jmag'] - df['Kmag1']) + 0.019
        df['K'] = df['Kmag1'] + 0.002
        df.loc[~df['Jmag'].isna(), 'K'] = df['K'] + 0.004 * (df['Jmag'] - df['Kmag1'])
        del df['Jmag'], df['Hmag'], df['Kmag1']

    elif base == 'VVV':
        df['J'] = df['Jmag3'] + 0.07 * (df['Jmag3'] - df['Hmag3'])
        df['H'] = df['Hmag3'] + 0.01 * (df['Hmag3'] - df['Ksmag3'])
        df['K'] = df['Ksmag3'] + 0.01 * (df['Hmag3'] - df['Ksmag3'])
        del df['Jmag3'], df['Hmag3'], df['Ksmag3']
    return df

def ugps_adql_query(n, ra, dec, radius):
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

def vvv_adql_query(n, ra, dec, radius):
    return f"""
        SELECT TOP {n} iauname, RAJ2000, DEJ2000, Jmag3, e_Jmag3, Hmag3, e_Hmag3, Ksmag3, e_Ksmag3, Jperrbits, Hperrbits, Ksperrbits, mClass
        FROM "II/348/vvv2"
        WHERE 1=CONTAINS(
            POINT('ICRS', RAJ2000, DEJ2000),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        AND (e_Jmag3 IS NOT NULL OR e_Hmag3 IS NOT NULL OR e_Ksmag3 IS NOT NULL)
        AND (Jmag3 <= 19.5 OR Hmag3 <= 19.0 OR Ksmag3 <= 18.0) 
        AND mClass <> 0
    """    


def generate_index(row):
    """
    Generate an index string based on the first non-null value in the order:
    UGPS / 2MASS, GLIMPSE, MIPSGAL, AllWISE.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: Generated index string.
    """
    base = '2MASS'
    if 'UGPS' in row:
        base = 'UGPS'
    if 'VVV' in row:
        base = 'VVV'
        
        
    if pd.notna(row[base]):
        return f"{base}_{row[base]}"
    elif 'GLIMPSE' in row and pd.notna(row['GLIMPSE']):
        return f"GLIMPSE_{row['GLIMPSE']}"
    elif 'MIPSGAL' in row and pd.notna(row['MIPSGAL']):
        return f"MIPSGAL_{row['MIPSGAL']}"
    elif 'AllWISE' in row and pd.notna(row['AllWISE']):
        return f"AllWISE_{row['AllWISE']}"
    else:
        return "UNKNOWN"
    
def _2mass_adql_query(n, ra, dec, radius):
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

def fin_class(row):
    class_cols = ["NIR_Class", "MIR1_Class", "MIR2_Class", "NMIR_Class", "W_Class"]
    
    _1 = 0
    _2 = 0
    for cls in class_cols:
        if pd.notna(row[cls]):
            if row[cls].endswith("II"):
                _2 += 2 if cls == "W_Class" else 1
            else:
                _1 += 2 if cls == "W_Class" else 1
    
    if _1 == _2:
        return "Class_I_II"
    elif _1 > _2:
        return "Class_I"
    elif _2 > _1:
        return "Class_II"
