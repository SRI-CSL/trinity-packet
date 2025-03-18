## Data Access

You can access the processed data at [https://zenodo.org/records/15046995](https://zenodo.org/records/15046995). These processed datasets contain the payloads and labels already so you do not need to install any files from the CIC-IDS-2017 and UNSW-NB15 repositories. However, these datasets should be cited when using this dataset. From the processed data, one can generate training and testing splits. We recommend training on balanced data and inferring on unbalanced data. This improves training time by produces a more realistic test time environment where the amount of benign traffic far exceeds the amount of malicious traffic.

## Data Manipulation

Consult the Jupyter notebook for converting the processed data into a set of training and testing splits. The data is saved in a ``temp'' folder. Note, each training and testing split will produce a large number of files (~90GB for CIC).

## Code

We provide the code for transforming the raw PCAP files and flow-level datasets into the processed packet ones. However, we provide the output from this code as well for ease of use. The code for this processing is in netflow.context_pcap.preprocess.py. The output from ```pcap_to_dataframe(meta_filename)``` is the processed data we publish at [https://zenodo.org/records/15046995](https://zenodo.org/records/15046995).