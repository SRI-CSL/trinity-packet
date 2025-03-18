import os
import sys 
import dpkt 
import time 
import pickle
import socket



import numpy as np
import pandas as pd 
from datetime import datetime, timedelta, timezone
from dateutil import tz


from netflow.context_pcap.dataset import ContextPCAPDataset



def cic_ids_time_parser(str_time):
    """
    Time parser specific to the CIC-IDS-2017 dataset. Different datasets require different parsing 
    functions based on the time format. For the CIC-IDS-2017 dataset, attack times are given 
    in the New Brunswick timezone GMT-3 during the summer months. AM/PM are not given but since 
    all traffic occurs between 9am and 5pm, we add 12 to afternoon hours.

    @param str_time: the string representation of the time. 
    """
    # get the date and time of the attack 
    date, time = str_time.split()
    day, month, year = date.split('/')
    
    # determine if seconds are given or not
    if time.count(':') == 1:
        hour, minute = time.split(':')
        second, precision_in_seconds = 0, 60
    else:
        hour, minute, second = time.split(':')
        precision_in_seconds = 1

    # if the hour is in the afternoon (less than 8pm since some attacks 
    # begin in the 8th hours), add 12 to convert to military time
    hour = int(hour)
    if hour < 8: hour += 12
    # add three to the hour to get the time in GMT format to match the
    # times in dpkt pcap format. current times are in New Brunswick time
    # in summer months which has GMT-3. we do not need to worry about the 
    # day changing since the last attacks occurs at 5pm ADT (8pm GMT)
    hour += 3
    # convert into a timestamp to get the float value for return
    timestamp = datetime(
        day = int(day), 
        month = int(month), 
        year = int(year), 
        hour = int(hour), 
        minute = int(minute), 
        second = int(second),
        tzinfo = timezone.utc,
    )
    
    # convert to a float value and return the precision in seconds
    return timestamp.timestamp(), precision_in_seconds

def cic_ids_flow_processor(dataset, data, flow_labels, previous_flows):
    """
    Process a given flow from CIC-IDS and add hashes to the flow labels. Returns the
    number of processed flows. 

    @param dataset: ContextPCAPDataset variable with category mapping attribute
    @param data: pandas dataframe with relevant flow information 
    @param flow_labels: a dictionary of flow hashes to start_time, end_time, category, etc.
    @param previous_flows: number of previous flows seen
    """
    # count the number of flows processed 
    nflows = 0

    # remove leading and trailing whitespace from column names 
    data.columns = data.columns.str.strip()

    # go through every flow in the CSV file
    for (src_ip, src_port, dest_ip, dest_port, protocol, timestamp, duration, category) in zip(
        data['Source IP'],
        data['Source Port'],
        data['Destination IP'],
        data['Destination Port'],
        data['Protocol'],
        data['Timestamp'],
        data['Flow Duration'],
        data['Label']
    ):
        # TCP is protocol 6 and UDP is protocol 17 by the Internet Assigned Numbers Authority (IANA)
        if not protocol == 6 and not protocol == 17: continue 
        if protocol == 6: protocol = 'TCP'
        else: protocol = 'UDP'
        
        # get the start time as an integer conditioned on whether or not seconds are included
        start_time, precision = cic_ids_time_parser(timestamp)
        # CIC-IDS-2017 flow durations in microseconds (cannot add since the integral values are milliseconds)
        end_time = (datetime.fromtimestamp(start_time) + timedelta(microseconds = int(duration))).timestamp()

        # update the category name if a category mapping exists 
        if hasattr(dataset, 'category_mapping'):
            category = dataset.category_mapping[category]
        
        # create a list of this hash tuple if it doesn't already exist
        hash_tuple = (src_ip, src_port, dest_ip, dest_port, protocol)
        if not hash_tuple in flow_labels:
            flow_labels[hash_tuple] = []
        flow_labels[hash_tuple].append((start_time, end_time, precision, category, previous_flows + nflows))
        # since these flows are bidirectional, we also need to hash the reverse flow information
        # when iterating through dpkt, it will give (src_ip, src_port, dest_ip, dest_port, protocol)
        # but the CSV file only considers the src_ip as the initiator of the connection, will miss 
        # all response packets 
        hash_tuple = (dest_ip, dest_port, src_ip, src_port, protocol)
        if not hash_tuple in flow_labels:
            flow_labels[hash_tuple] = []
        flow_labels[hash_tuple].append((start_time, end_time, precision, category, previous_flows + nflows))

        # increment the number of flows seen 
        nflows += 1

    return nflows

def unsw_nb15_flow_processor(dataset, data, flow_labels, previous_flows):
    """
    Process a given flow from UNSW-NB15 and add hashes to the flow labels. Returns the
    number of processed flows. 

    @param dataset: ContextPCAPDataset variable with category mapping attribute
    @param data: pandas dataframe with relevant flow information 
    @param flow_labels: a dictionary of flow hashes to start_time, end_time, category, etc.
    @param previous_flows: number of previous flows seen
    """
    # count the number of flows processed 
    nflows = 0

    # remove leading and trailing whitespace from column names 
    data.columns = data.columns.str.strip()
                                  
    # go through every flow in the CSV file
    for (src_ip, src_port, dest_ip, dest_port, protocol, start_time, end_time, category) in zip(
        data['srcip'],
        data['sport'],
        data['dstip'],
        data['dsport'],
        data['proto'],
        data['Stime'],
        data['Ltime'],
        data['attack_cat'],
    ):
        # TCP is protocol 6 and UDP is protocol 17 by the Internet Assigned Numbers Authority (IANA)
        if not protocol == 'tcp' and not protocol == 'udp': continue 
        protocol = protocol.upper()
        
        # remove white space around category and add 'Benign' for empty attacks (benign traffic)
        category = category.strip()
        if not len(category): category = 'Benign'
        
        # update the category name if a category mapping exists 
        if hasattr(dataset, 'category_mapping'):
            category = dataset.category_mapping[category]
        
        # precision is zero second for the UNSW-NB15 datasets since the start and end 
        # time are both given (in seconds)
        precision = 0
        
        # create a list of this hash tuple if it doesn't already exist
        hash_tuple = (src_ip, src_port, dest_ip, dest_port, protocol)
        if not hash_tuple in flow_labels:
            flow_labels[hash_tuple] = []
        flow_labels[hash_tuple].append((start_time, end_time, precision, category, previous_flows + nflows))
        # since these flows are bidirectional, we also need to hash the reverse flow information
        # when iterating through dpkt, it will give (src_ip, src_port, dest_ip, dest_port, protocol)
        # but the CSV file only considers the src_ip as the initiator of the connection, will miss 
        # all response packets 
        hash_tuple = (dest_ip, dest_port, src_ip, src_port, protocol)
        if not hash_tuple in flow_labels:
            flow_labels[hash_tuple] = []
        flow_labels[hash_tuple].append((start_time, end_time, precision, category, previous_flows + nflows))

        # increment the number of flows seen 
        nflows += 1

    return nflows

def generate_flow_labels_hash(dataset):
    """
    Generate a hash for flows to labels with the given tuple:
    (source_ip, destination_ip, source_port, destination_port).

    @param dataset: ContextPCAPDataset to generate hashes for
    """
    # create a dictionary of flow hashes and a counter of the number of flows seen 
    flow_labels = {}
    nflows = 0

    # read all of the flow CSV files 
    for flow_filename in dataset.flow_datasets:
        flow_start_time = time.time()
        sys.stdout.write('Parsing flow file {}...'.format(flow_filename))

        if dataset.flow_preprocessor == 'CIC-IDS':
            data = pd.read_csv(flow_filename, delimiter = ',', encoding = 'cp1252')
            
            nflows += cic_ids_flow_processor(dataset, data, flow_labels, nflows)
        elif dataset.flow_preprocessor == 'UNSW-NB15':
            data = pd.read_csv(flow_filename, dtype = {'attack_cat': str})
            
            # attack category left blank for normal traffic
            data['attack_cat'] = data['attack_cat'].fillna('Benign')

            # update column dtypes for ports and clean data 
            dropped_ports = ['-', '0x20205321']
            data = data[~data['sport'].isin(dropped_ports)]
            data = data[~data['dsport'].isin(dropped_ports)]
            
            data.reset_index(drop = True, inplace = True)
            
            # convert hexadecimal ports into base 10 before column conversion 
            hex_ports = {
                '0x000b': 11, 
                '0x000c': 12,
                '0xc0a8': 49320, 
                '0xcc09': 52233,
            }
            # replace the hex ports 
            data['sport'] = data['sport'].replace(hex_ports)
            data['dsport'] = data['dsport'].replace(hex_ports)
            
            data['sport'] = data['sport'].astype(np.int32)
            data['dsport'] = data['dsport'].astype(np.int32)
            
            nflows += unsw_nb15_flow_processor(dataset, data, flow_labels, nflows)
        else:
            assert ('Unknown flow preprocessor attribute.')

        sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - flow_start_time))
        sys.stdout.flush()

    # since the times are often at coarser resolution than the microsecond durations, sort by end time
    # the last (start_time, end_time) that satisfies the constraint start_time <= timestamp <= end_time
    # is the correct flow 
    for hash_tuple in flow_labels:
        flow_labels[hash_tuple].sort(key = lambda x: x[1])

    # save the hash filename
    hash_filename = '{}/flow_labels_hash.pkl'.format(dataset.temp_directory)
    with open(hash_filename, 'wb') as fd:
        pickle.dump(flow_labels, fd, protocol = pickle.HIGHEST_PROTOCOL)

def inet_to_str(inet):
    """
    Convert an inet object to a string representation. 

    @param inet: inet network address
    """
    # return either IPv4 or IPv6 address
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

def read_pcap_data(pcap_file, flow_labels):
    """
    Given a pcap class, read the pcap data packet by packet.

    @param pcap: type PCAPFile that contains filenames and types
    @param flow_labels: labels for each flow identified by tuple (src ip, src port, dest ip, dest port)
    """
    # store all packets into an array to save into a dataframe later
    packets = []
    
    # read this PCAP file and return the TCP/UDP packets
    with open(pcap_file.filename, 'rb') as pcap_fd:
        pcap_start_time = time.time()
        sys.stdout.write('Reading PCAP file {}...'.format(pcap_file.filename))
        # read either PCAP or PCAP next generation formats
        if pcap_file.pcap_type == 'PCAP':
            pcap = dpkt.pcap.Reader(pcap_fd)
        elif pcap_file.pcap_type == 'PCAPNG':
            pcap = dpkt.pcapng.Reader(pcap_fd)

        npackets = 0
        ndropped = 0
        # read each packet in the pcap file 
        for timestamp, packet in pcap:
            # Linux cooked capture 
            if pcap.datalink() == dpkt.pcap.DLT_LINUX_SLL:
                eth = dpkt.sll.SLL(packet)
            # ethernet format
            else:
                eth = dpkt.ethernet.Ethernet(packet)

            # skip any non IP IPv6 packets
            if not isinstance(eth.data, dpkt.ip.IP) and not isinstance(eth.data, dpkt.ip6.IP6):
                continue 

            # get the IP data 
            ip = eth.data 
            # get relevant fields from the IP packet header (convert to string representation)
            src_ip = inet_to_str(ip.src)
            dest_ip = inet_to_str(ip.dst)
            # get specific attributes from the header based on the protocol type (IP v IPv6)
            if isinstance(eth.data, dpkt.ip.IP): 
                ttl = ip.ttl
                total_length = ip.len
                internet_layer_protocol = 'IPv4'
            elif isinstance(eth.data, dpkt.ip6.IP6):
                ttl = ip.hlim
                total_length = ip.plen
                internet_layer_protocol = 'IPv6'
            else:
                assert ('Unknown internet layer protocol.')
            
            # skip any non UDP and TCP protocols (n.b., get_proto has a bug and does not recognize
            # certain protocols, so we need to skip the others here)
            if not isinstance(ip.data, dpkt.tcp.TCP) and not isinstance(ip.data, dpkt.udp.UDP):
                continue 
            
            # get the protocol name 
            transport_layer_protocol = ip.get_proto(ip.p).__name__
            transport_layer_protocol = transport_layer_protocol.upper()
            
            # count the number of packets before any drops
            npackets += 1

            # get the data from the transport layer and convert to bytes
            transport = ip.data 
            data = transport.data 

            # get the source and destination ports 
            src_port = transport.sport 
            dest_port = transport.dport 

            # get relevant header information for TCP 
            if isinstance(ip.data, dpkt.tcp.TCP):
                # need to divide by 2 raised to the bit location to get 0 and 1 values
                cwr_flag = (transport.flags & dpkt.tcp.TH_CWR) // 128 
                ece_flag = (transport.flags & dpkt.tcp.TH_ECE) // 64
                urg_flag = (transport.flags & dpkt.tcp.TH_URG) // 32
                ack_flag = (transport.flags & dpkt.tcp.TH_ACK) // 16
                psh_flag = (transport.flags & dpkt.tcp.TH_PUSH) // 8
                rst_flag = (transport.flags & dpkt.tcp.TH_RST) // 4
                syn_flag = (transport.flags & dpkt.tcp.TH_SYN) // 2
                fin_flag = (transport.flags & dpkt.tcp.TH_FIN)
            # all flags have zero value for UDP 
            else:
                cwr_flag = 0
                ece_flag = 0
                urg_flag = 0
                ack_flag = 0
                psh_flag = 0
                rst_flag = 0
                syn_flag = 0
                fin_flag = 0

            # get the label for this packet 
            hash_tuple = (src_ip, src_port, dest_ip, dest_port, transport_layer_protocol)
            # skip tuples that do not appear in the flow data
            # for the CIC-IDS-2017 dataset, this will include all IPv6 traffic
            if not hash_tuple in flow_labels:
                ndropped += 1
                continue

            # go through possible hashed label values until identifying the proper flow
            # based on the flow's start and end times. there can be multiple matches since the 
            # start time resolution is in seconds and the end time resolution is in microseconds
            # so find the last flow that fits the conditions (must be the correct one)
            packet_category = None
            packet_flow_id = None
            # the start time precision is only to the second (at best), we allow a small buffer 
            # for flows to be considered. if all flows in the buffer have the same category, the 
            # packet receives that category label 
            buffer_categories = set()
            buffer_flow_ids = set()
            for (start_time, end_time, precision, category, flow_id) in flow_labels[hash_tuple]:
                # consider all possible flows that can fall in this time frame 
                if start_time <= timestamp and timestamp <= end_time + precision: 
                    buffer_categories.add(category)
                    buffer_flow_ids.add(flow_id)
            
            # continue if there are overlapping flows with different category labels
            if len(buffer_categories) == 1: 
                packet_category = list(buffer_categories)[0]
                # take the first flow ID (not a perfect method)
                packet_flow_id = list(buffer_flow_ids)[0]
            else:
                ndropped += 1
                continue
            
            # only Benign category is labeled as non-attack
            packet_label = 0 if packet_category == 'Benign' else 1
            
            # get the hex representation of the payload
            if isinstance(data, bytes):
                payload = data.hex()
            else:
                payload = (data.pack()).hex()
                
            # save the payload length (for pruning later)
            payload_length = len(payload)

            packets.append([
                src_ip, 
                src_port,
                dest_ip, 
                dest_port,
                timestamp,
                internet_layer_protocol,
                transport_layer_protocol, 
                ttl, 
                total_length, 
                cwr_flag, 
                ece_flag,
                urg_flag, 
                ack_flag, 
                psh_flag, 
                rst_flag, 
                syn_flag, 
                fin_flag, 
                payload,
                payload_length,
                packet_flow_id,
                packet_category, 
                packet_label,
            ])
        
        sys.stdout.write('read {} packets (dropped: {}) in {:0.2f} seconds.\n'.format(npackets, ndropped, time.time() - pcap_start_time))
        sys.stdout.flush()

    return packets

def pcap_to_dataframe(meta_filename):
    """
    Convert the raw pcap files into a pandas dataframe.

    @param meta_filename: meta file corresponding to a PCAP dataset
    """
    # read the ContextPCAPDataset file 
    dataset = ContextPCAPDataset(meta_filename)

    # create the hash for the flows if they do not already exist 
    hash_filename = '{}/flow_labels_hash.pkl'.format(dataset.temp_directory)
    if not os.path.exists(hash_filename):
        generate_flow_labels_hash(dataset)

    with open(hash_filename, 'rb') as fd:
        flow_labels = pickle.load(fd)

    # create a new dataframe with these columns/header attributes
    columns = [
        'src_ip', 
        'src_port',
        'dest_ip', 
        'dest_port',
        'epoch',
        'internet_layer_protocol',
        'transport_layer_protocol', 
        'ttl', 
        'total_length', 
        'cwr_flag', 
        'ece_flag',
        'urg_flag', 
        'ack_flag', 
        'psh_flag', 
        'rst_flag', 
        'syn_flag', 
        'fin_flag', 
        'payload',
        'payload_length',
        'packet_flow_id',
        'packet_category',
        'packet_label',
    ]

    # create a new dataframe for every PCAP file 
    for pcap_file in dataset.raw_datasets:
        # get timing statistics 
        start_time = time.time()
        
        # get the suffix for the PCAP file
        suffix = pcap_file.filename.split('/')[-1].split('.')[0]
        
        # read the pcap data
        packets = read_pcap_data(pcap_file, flow_labels)
        
        # create the dataframe 
        processed_data = pd.DataFrame(packets, columns = columns)

        # set the dtypes for some of the columns 
        processed_data['src_port'] = processed_data['src_port'].apply(lambda x: int(x))
        processed_data['dest_port'] = processed_data['dest_port'].apply(lambda x: int(x))
        processed_data['epoch'] = processed_data['epoch'].apply(lambda x: float(x))
        processed_data['ttl'] = processed_data['ttl'].apply(lambda x: int(x))
        processed_data['total_length'] = processed_data['total_length'].apply(lambda x: int(x))
        processed_data['cwr_flag'] = processed_data['cwr_flag'].apply(lambda x: int(x))
        processed_data['ece_flag'] = processed_data['ece_flag'].apply(lambda x: int(x))
        processed_data['urg_flag'] = processed_data['urg_flag'].apply(lambda x: int(x))
        processed_data['ack_flag'] = processed_data['ack_flag'].apply(lambda x: int(x))
        processed_data['psh_flag'] = processed_data['psh_flag'].apply(lambda x: int(x))
        processed_data['rst_flag'] = processed_data['rst_flag'].apply(lambda x: int(x))
        processed_data['syn_flag'] = processed_data['syn_flag'].apply(lambda x: int(x))
        processed_data['fin_flag'] = processed_data['fin_flag'].apply(lambda x: int(x))
        processed_data['payload_length'] = processed_data['payload_length'].apply(lambda x: int(x))
        processed_data['packet_flow_id'] = processed_data['packet_flow_id'].apply(lambda x: int(x))
        processed_data['packet_label'] = processed_data['packet_label'].apply(lambda x: int(x))

        processed_data.reset_index(drop = True, inplace = True)
        
        # create the output directory if it doesn't exist 
        output_directory = 'processed_data/{}'.format(dataset.prefix)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok = True)
        
        # save the full dataframe to disk
        save_time = time.time()
        output_filename = '{}/{}.pkl.gz'.format(output_directory, suffix)
        sys.stdout.write('Saving {}...'.format(output_filename))
        processed_data.to_pickle(output_filename)
        sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - save_time))
                        
        sys.stdout.write('Completed preprocessing for {} in {:0.2f} seconds.\n\n'.format(pcap_file.filename, time.time() - start_time))
        sys.stdout.flush()

def augment_processed_data(meta_filename):
    """
    Create new processed datasets by removing the zero payloads and forcing unique payload 
    values. 

    @param meta_filename: meta file corresponding to a PCAP dataset
    """
    # read the ContextPCAPDataset file 
    dataset = ContextPCAPDataset(meta_filename)

    # read the processed data
    read_time = time.time()
    data = dataset.read_processed_data()
    print ('Read dataset in {:0.2f} seconds.'.format(time.time() - read_time))

    # create an empty array of the unseen payloads
    packets = []

    # iterate over all payloads in the data 
    payload_time = time.time()
    for payload, data_by_payload in data.groupby('payload'):
        # skip over payloads that correspond to both benign and attack 
        labels = pd.unique(data_by_payload['packet_label'])
        if not len(labels) == 1: continue 

        # get any packet category (since we don't care about attack type)
        packet_category = pd.unique(data_by_payload['packet_category'])[0]
        packet_label = labels[0]

        packets.append([
            payload,
            packet_category,
            packet_label,
        ])

    columns = [
        'payload',
        'packet_category',
        'packet_label',
    ]

    payload_reduced_data = pd.DataFrame(packets, columns = columns)

    output_filename = '{}-payload_reduced.pkl'.format(dataset.processed_datasets[0].split('.')[0])
    payload_reduced_data.to_pickle(output_filename)
    print ('Created reduced payload dataset in {:0.2f} seconds.'.format(time.time() - payload_time))

    # create a new meta file for the reduced dataset 
    reduced_meta_filename = '{}-payload_reduced.data'.format(dataset.filename.split('.')[0])
    with open(reduced_meta_filename, 'w') as fd:
        fd.write('# prefix\n')
        fd.write('{}-PAYLOAD-REDUCED\n'.format(dataset.prefix))
        fd.write('# raw datasets\n')
        for raw_dataset in dataset.raw_datasets:
            fd.write('{} {}\n'.format(raw_dataset.filename, raw_dataset.pcap_type))
        fd.write('# flow datasets\n')
        for flow_dataset in dataset.flow_datasets:
            fd.write('{}\n'.format(flow_dataset))
        fd.write('# processed datasets\n')
        for processed_data in dataset.processed_datasets:
            # no compression for reduced payload files 
            fd.write('{}-payload_reduced.pkl\n'.format(processed_data.split('.')[0]))
        fd.write('# category mapping\n')
        for category, mapping in dataset.category_mapping.items():
            fd.write('{}: {}\n'.format(category, mapping))

    return reduced_meta_filename