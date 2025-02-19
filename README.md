# IdentYS: Young Stellar Objects detection and classification

## Overview

IdentYS is a Python-based tool for retrieving and analyzing astronomical data from catalogs like 2MASS, UKIDSS GPS, GLIMPSE, MIPSGAL, and ALLWISE. It helps astronomers identify and classify young stellar objects (YSOs) in star-forming regions using NIR and MIR photometric data, focusing on Class I and II evolutionary stages.

## Features

- **Data Retrieval:**

  - Retrieve data from the following infrared astronomical catalogues:
    - 2MASS
    - UKIDSS GPS
    - GLIMPSE
    - MIPSGAL
    - ALLWISE
  - Ensure sample purity by filtering data based on specific criteria.

- **Contamination Exclusion:**

  - Remove potential contamination sources such as:
    - Be stars
    - Asymptotic Giant Branch (AGB) stars
    - Star-forming galaxies
    - Narrow- and broad-line active galactic nuclei (AGNs)

- **YSO Evolutionary Stage Classification:**

  - Classify YSOs based on Near-Infrared (NIR) and Mid-Infrared (MIR) photometric data using the following color-color diagrams:
    - **NIR:** (J-H) vs. (H-K)
    - **MIR1:** ([3.6] - [4.5]) vs. ([5.8] - [8.0])
    - **MIR2:** ([3.6] - [4.5]) vs. ([8.0] - [24])
    - **NMIR:** (K - [3.6]) vs. ([3.6] - [4.5])
    - **W:** (W1 - W2) vs. (W2 - W3)

- **Result Export:**

  - Export processed data to Excel files, including:
    - Source designations
    - Astrometric data
    - Photometric parameters from the catalogues
  - Separate files for sources located in the star-forming region but not identified as YSO candidates.

## Installation Guide

### Prerequisites

This project has been tested on **Python 3.11** and **Python 3.8**. If it does not run correctly with your Python version, you can create a new environment using Conda:
```bash
conda create --name identys python=3.11
conda activate identys
```

### Clone the Repository

```bash
git clone https://github.com/DanielBaghdasaryan/identys.git
cd identys
```

### Install Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

To execute the script with an example input file:

```bash
python main.py example.json
```

### JSON Input Structure

The JSON input file should have the following structure:

```json
{
    "output_dir": "",
    "use_gps": true,
    "data":[
        ["19 13 27.85", "+10 53 36.7", 1.9]
    ]
}
```

- `output_dir`: Specifies the folder for saving output files. If left empty, results will be saved in the `<current dir>/output` folder.
- `use_gps`: Determines whether to use UKIDSS GPS data. If set to `false`, 2MASS data will be used as the base. If set `true`, UKIDSS data will be used as a base and 2MASS data will only update UKIDSS data. However, it will automatically get 2MASS as a base if the data from UKIDSS is empty.
- `data`: A list of areas in the format `[ra, dec, radius in arcmin]`. The process will run for each area and generate files named:
  - `<json name>_<area index>_class.csv`: Contains classified objects.
  - `<json name>_<area index>_no_class.csv`: Contains non-classified objects.

## Output

The script will generate Excel files with:

- Source designations
- Astrometric data
- Photometric parameters
- Evolutionary stage classifications
- Non-YSO candidates with relevant data

## License

This project is licensed under the MIT License.

## Acknowledgements

Data retrieved from the VizieR database and processed using Astropy and Pandas libraries.


