import pandas as pd
from pathlib import Path

################################################################################
def create_output_directory(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

################################################################################
def read_samples(samples_file):
    if samples_file.endswith('.tsv'):
        sep = '\t'
        tabular = True
    elif samples_file.endswith('.csv'):
        sep = ','
        tabular = True
    elif samples_file.endswith('.xlsx'):
        sep = None
        tabular = False
    else:
        raise ValueError(f"Samples file must be either a .tsv, .csv, or .xlsx file: {samples_file}")
    df_samples = pd.read_csv(samples_file, sep=sep) if tabular else pd.read_excel(samples_file)

    # Check the number of columns
    num_columns = df_samples.shape[1]
    if num_columns not in [2, 3, 4]:
        raise ValueError(f"Samples file must have either 2, 3, or 4 columns, found {num_columns} columns")
    
    if num_columns == 4:
        expected_headers = ['Sample Name', 'Sample Group', 'File', 'Normalization']
    elif num_columns == 3:
        expected_headers = ['Sample Name', 'Sample Group', 'File']
    else:
        expected_headers = ['Sample Name', 'File']

    # Check for headers or assign them if missing
    if list(df_samples.columns) != expected_headers:
        if list(df_samples.columns) == list(range(num_columns)):
            # If there are no headers, assign the expected headers
            df_samples.columns = expected_headers
        else:
            # If headers are present but incorrect, raise an error
            raise ValueError(f"Column headers must be {expected_headers}, found {list(df_samples.columns)}")

    assert not df_samples.isnull().any().any(), "There are missing values in the input samples file. Please fill them in."
    
    in_files = df_samples['File'].values
    sample_names = df_samples['Sample Name'].values
    sample_groups = df_samples['Sample Group'].values if 'Sample Group' in df_samples.columns else np.repeat(0, len(sample_names))
    normalization = df_samples['Normalization'].values if 'Normalization' in df_samples.columns else None
    
    return df_samples, in_files, sample_names, sample_groups, normalization