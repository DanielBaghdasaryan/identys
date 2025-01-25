import numpy as np
import pandas as pd

def classify_nir(row):
    """
    Classify an object into NIR (Near Infrared) categories based on magnitude values.

    Args:
        row (pd.Series): DataFrame row containing the magnitudes J, H, and K.

    Returns:
        str: The NIR classification label.
    """
    j_k = row['J'] - row['K'] if pd.notna(row['J']) and pd.notna(row['K']) else np.nan
    j_h_h_k = (
        (row['J'] - row['H']) - 1.6984 * (row['H'] - row['K']) + 0.09262 
        if pd.notna(row['J']) and pd.notna(row['H']) and pd.notna(row['K']) 
        else np.nan
    )

    if j_k > 3:
        return "NIR_Class_I"
    elif j_h_h_k < 0:
        return "NIR_Class_II"
    return ""


def classify_mir1(row):
    """
    Classify an object into MIR1 (Mid Infrared) categories.

    Args:
        row (pd.Series): DataFrame row containing the magnitudes.

    Returns:
        str: The MIR1 classification label.
    """
    diff_36_45 = row['_3_6mag'] - row['_4_5mag'] if pd.notna(row['_3_6mag']) and pd.notna(row['_4_5mag']) else np.nan
    diff_58_80 = row['_5_8mag'] - row['_8_0mag'] if pd.notna(row['_5_8mag']) and pd.notna(row['_8_0mag']) else np.nan

    if (diff_36_45 > 0.8 and diff_58_80 > 0.2) or (diff_36_45 >= 0.4 and diff_58_80 > 1.1):
        return "MIR1_Class_I"
    elif diff_58_80 > 1.1 and diff_36_45 < 0.4:
        return "MIR1_Class_I_II"
    elif 0 < diff_36_45 < 0.8 and 0.4 < diff_58_80 < 1.1:
        return "MIR1_Class_II"
    return ""


def classify_mir2(row):
    """
    Classify an object into MIR2 categories.

    Args:
        row (pd.Series): DataFrame row containing the magnitudes.

    Returns:
        str: The MIR2 classification label.
    """
    diff_36_58 = row['_3_6mag'] - row['_5_8mag'] if pd.notna(row['_3_6mag']) and pd.notna(row['_5_8mag']) else np.nan
    diff_80_24 = row['_8_0mag'] - row['__24_'] if pd.notna(row['_8_0mag']) and pd.notna(row['__24_']) else np.nan

    if diff_36_58 > 1.5 and diff_80_24 > 2.4:
        return "MIR2_Class_I"
    elif diff_36_58 > 0.3 and diff_80_24 > 2:
        return "MIR2_Class_II"
    return ""


def classify_nmir(row):
    """
    Classify an object into NMIR (Near and Mid Infrared) categories.

    Args:
        row (pd.Series): DataFrame row containing the magnitudes.

    Returns:
        str: The NMIR classification label.
    """
    k_36 = row['K'] - row['_3_6mag'] if pd.notna(row['K']) and pd.notna(row['_3_6mag']) else np.nan
    diff_36_45 = row['_3_6mag'] - row['_4_5mag'] if pd.notna(row['_3_6mag']) and pd.notna(row['_4_5mag']) else np.nan

    is_yso = (
        k_36 + 2.85714 * diff_36_45 - 0.78857 > 0
        and (diff_36_45 - 0.08) * 3.55 - (k_36 - 0.45) * 0.87 > 0
        and (diff_36_45 - 0.2) * 3.5 - k_36 * 2.3 < 0
    ) if pd.notna(k_36) and pd.notna(diff_36_45) else False

    if is_yso:
        if (diff_36_45 - 1) * 2.59 + k_36 * 0.92 >= 0:
            return "NMIR_Class_I"
        return "NMIR_Class_II"
    return ""


def classify_w(row):
    """
    Classify an object into W (WISE infrared) categories.

    Args:
        row (pd.Series): DataFrame row containing the magnitudes and errors.

    Returns:
        str: The W classification label.
    """
    diff_w1_w2 = row['W1mag'] - row['W2mag'] if pd.notna(row['W1mag']) and pd.notna(row['W2mag']) else np.nan
    diff_w2_w3 = row['W2mag'] - row['W3mag'] if pd.notna(row['W2mag']) and pd.notna(row['W3mag']) else np.nan
    diff_w2_w4 = row['W2mag'] - row['W4mag'] if pd.notna(row['W2mag']) and pd.notna(row['W4mag']) else np.nan

    sigma1 = (
        (row['e_W1mag'] ** 2 + row['e_W2mag'] ** 2) ** 0.5
        if pd.notna(row['e_W1mag']) and pd.notna(row['e_W2mag'])
        else np.nan
    )
    sigma2 = (
        (row['e_W2mag'] ** 2 + row['e_W3mag'] ** 2) ** 0.5
        if pd.notna(row['e_W2mag']) and pd.notna(row['e_W3mag'])
        else np.nan
    )
    condition = (
        row['W1mag'] - row['W3mag'] + 1.7 * (row['W3mag'] - row['W4mag']) - 4.3
        if pd.notna(row['W1mag']) and pd.notna(row['W3mag']) and pd.notna(row['W4mag'])
        else np.nan
    )

    if diff_w1_w2 > 1 and diff_w2_w3 > 2 and diff_w2_w4 > 4:
        return "W_Class_I"
    elif (diff_w1_w2 - sigma1 > 0.25) and (diff_w2_w3 - sigma2 > 1) and condition > 0:
        return "W_Class_II"
    return ""
