## SAFE-NID: Self-Attention with Normalizing-Flow Encodings for Network Intrusion Detection

Machine learning models are increasingly adopted to monitor network traffic and detect intrusions. In this work, we introduce SAFE-NID, a novel machine learning approach for real-time packet-level traffic monitoring and intrusion detection that includes a safeguard to detect zero day attacks as  out-of-distribution inputs. Unlike traditional models, which falter against zero-day attacks and concept drift, SAFE-NID leverages a lightweight encoder-only transformer architecture combined with a novel normalizing flows-based safeguard. This safeguard not only quantifies uncertainty but also identifies out-of-distribution (OOD) inputs, enabling robust performance in dynamic threat landscapes. Our generative model learns class-conditional representations of the internal features of the deep neural network. We demonstrate the effectiveness of our approach by converting publicly available network flow-level intrusion datasets into packet-level ones. We release the labeled packet-level versions of these datasets with over 50 million packets each and describe the challenges in creating these datasets. We withhold from the training data certain attack categories to simulate zero-day attacks. Existing deep learning models, which achieve an accuracy of over 99% when detecting known attacks, only correctly classify 1% of the novel attacks. Our proposed transformer architecture with normalizing flows model safeguard achieves an area under the receiver operating characteristic curve of over 0.97 in detecting these novel inputs, outperforming existing combinations of neural architectures and model safeguards. The additional latency in processing each packet by the safeguard is a small fraction of the overall inference task. This dramatic improvement in detecting zero-day attacks and distribution shifts emphasizes SAFE-NID’s novelty and utility as a reliable and efficient safety monitoring tool for real-world network intrusion detection.

### Installation

The Jupyter notebooks and configuration files assume installing ```trinity-packet``` in the home directory. Update the configuration files and notebooks if that is not the case.

```
git clone https://github.com/SRI-CSL/trinity-packet.git
conda env create -f environment.yml 
conda activate trinity_packet_env
```

### Data

```data/README.md``` provides information for downloading the datasets used in our SAFE-NID paper. We host the data on zenodo and it includes the labeled payloads. The notebook ```notebooks/data_processing.ipynb``` contains the code to generate training and testing splits. In our paper, we generate ten training and testing splits for each dataset. Note, each training and testing split creates a large number of auxillary files in a ```temp``` directory (created from where the ```cwd```). For CIC-IDS-2017, each split creates ~90GB of extra files.

### Config Files

We use config files to contain all relevant information on each dataset and experiment. We provide example config files in ```SAFE-NID/configs```. These example configs assume that ```trinity-packet``` was installed in the home directory. The config files have headers that begin with ```#``` followed by lists of attributes. 

#### Dataset Config Files

The dataset config files contain file location information for the dataset. With the processed data provided on zenodo, one does not need to worry about the ```raw datasets``` or ```flow datasets``` attributes in the config file. No install of those datasets from CIC-IDS-2017 or UNSW-NB15 is required.

| Header | Explanation | 
| --- | --- |
| prefix | Name to refer to dataset by (should match base name of file) |
| raw datasets | Location of PCAP file from CIC-IDS-2017 and UNSW-NB15 and the format (PCAP or PCAPNG) |
| flow datasets | Location of flow summaries from CIC-IDS-2017 and UNSW-NB15 | 
| flow preprocessor | Data source (CIC-IDS) or (UNSW-NB15) - changes the date/time processing | 
| processed datasets | Location of the processed datasets that combine flow labels to packet payloads | 
| category mapping | A mapping from names used in processed datasets to formatted text | 

#### Experiment Config Files

The experiment config file contains information regarding an OOD experiment. 

| Header | Explanation |
| --- | --- |
| prefix | Name to refer to experiment by (should match base name of file) |
| ID examples | The in-distribution samples used for training. The first "word" is the dataset config location and the second is the packet category (e.g., Benign or DDoS). |
| OOD examples | The out-of-distribution samples added during inference. The first "word" is the dataset config location and the second is the packet category (e.g., FTP-Patator or Infiltration). |
| network architecture | The type of neural network to use. Provided networks are ```ContextFNNBatchNormDropout```, ```ContextCNNBatchNormDropout```, and ```ContextTransformer```. |
| payload | The length of the payload in bytes (paper uses 1500). Payloads are truncated beyond this value. |
| target | The target of the discriminative classifier. For ```packet_label```, the NN will predict benign/attack for packets. For ```category_label```, the NN will predict with attack granularity. |
| nepochs | The number of training epochs (paper uses 20 for FNN and CNN, 6 for Transformer). |
| header context | Add header context to the payloads. Paper uses header context. |

### Code 

We provide all code for the data generation and experimentation from our paper in ```code/netflow```. We also provide Jupyter notebooks for an easier start. ```data_processing.ipynb``` contains the functions to take the processed datasets that we provide at [https://zenodo.org/records/15046995](https://zenodo.org/records/15046995) and create training and testing splits. ```experiment_pipeline.ipynb``` contains the entire pipeline discussed in our paper from training a discriminative classifier, training a model safeguard, inferring on ID and OOD samples, and identifying the OOD samples.

### Citation

Please cite our work if using our code or data:

Please also cite the original UNSW-NB15 and CIC-IDS-2017 datasets:

Moustafa, N. and Slay, J., 2015, November. UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In 2015 military communications and information systems conference (MilCIS) (pp. 1-6). IEEE.

Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., 2018. Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSp, 1(2018), pp.108-116.