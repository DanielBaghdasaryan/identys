from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astroquery.utils.tap.core import Tap
from collections.abc import Iterable
import pandas as pd

# Initialize TAP client
tap = Tap(url="http://tapvizier.u-strasbg.fr/TAPVizieR/tap")

def convert_to_degrees(alpha, delta):
    """
    Convert Right Ascension (RA) and Declination (Dec) from sexagesimal format to degrees.

    Args:
        alpha (str): Right Ascension in 'HH MM SS' format.
        delta (str): Declination in 'DD MM SS' format.

    Returns:
        tuple: (RA in degrees, Dec in degrees)
    """
    ra = hms_to_degrees(alpha)
    dec = dms_to_degrees(delta)
    return ra, dec

def hms_to_degrees(hms_str):
    """
    Convert hours, minutes, seconds (HMS) to degrees.

    Args:
        hms_str (str): HMS formatted string (e.g. '12 34 56').

    Returns:
        float: Degrees equivalent.
    """
    hours, minutes, seconds = map(float, hms_str.split())
    return 15 * (hours + minutes / 60 + seconds / 3600)

def dms_to_degrees(dms_str):
    """
    Convert degrees, arcminutes, arcseconds (DMS) to decimal degrees.

    Args:
        dms_str (str): DMS formatted string (e.g. '12 34 56').

    Returns:
        float: Decimal degrees.
    """
    degrees, arcminutes, arcseconds = map(float, dms_str.split())
    sign = -1 if degrees < 0 else 1
    return sign * (abs(degrees) + arcminutes / 60 + arcseconds / 3600)

def get_adql_query(name, ra, dec, radius, prefix, start=10000, c=False, cols_to_keep=[]):
    """
    Retrieve data from the TAP service using an ADQL query.

    Args:
        name (str): Catalog name.
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.
        radius (float): Search radius in degrees.
        prefix (str): Prefix for column renaming.
        start (int, optional): Initial query limit. Defaults to 10,000.
        c (bool, optional): Include additional filtering condition. Defaults to False.
        cols_to_keep (List(str)): columns included in final output

    Returns:
        pd.DataFrame: Query results in a pandas DataFrame.
    """
    print(f'\nRetrieve data from {prefix.upper()}...')
    n = start
    while True:
        adql_query = f"""
            SELECT TOP {n} *
            FROM "{name}"
            WHERE 1=CONTAINS(
              POINT('ICRS', RAJ2000, DEJ2000),
              CIRCLE('ICRS', {ra}, {dec}, {radius})
            ) {"AND C='C'" if c else ""}
        """
        job = tap.launch_job(adql_query, maxrec=-1)
        result = job.get_results()
        if len(result) == n:
            n *= 2
        else:
            break
    result = result.to_pandas().rename(columns={
        'RAJ2000': f'{prefix}_ra',
        'DEJ2000': f'{prefix}_de',
    })
    result[f'{prefix}_ra'] = result[f'{prefix}_ra'].fillna(np.nan)
    result[f'{prefix}_de'] = result[f'{prefix}_de'].fillna(np.nan)

    print(f'Found rows: {len(result)}')

    if len(result) > 0 and len(cols_to_keep) > 0:
        result = result[cols_to_keep]
    return result

def angular_distance(ra1, dec1, ra2, dec2):
    """
    Calculate the angular distance between two celestial coordinates.

    Args:
        ra1 (float): RA of the first coordinate in degrees.
        dec1 (float): Dec of the first coordinate in degrees.
        ra2 (float or list): RA of the second coordinate(s) in degrees.
        dec2 (float or list): Dec of the second coordinate(s) in degrees.

    Returns:
        float or np.ndarray: Angular separation in degrees.
    """
    if isinstance(ra2, Iterable):
        ra2 = np.array(ra2)
        dec2 = np.array(dec2)

    coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
    
    # Calculate the separations
    separations = coord1.separation(coord2)
    
    return separations.deg

def generate_index(row):
    """
    Generate an index string based on the first non-null value in the order:
    UGPS / 2MASS, GLIMPSE, MIPSGAL, AllWISE.

    Args:
        row (pd.Series): A row from the DataFrame.

    Returns:
        str: Generated index string.
    """
    base = 'UGPS' if 'UGPS' in row else '2MASS'
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
