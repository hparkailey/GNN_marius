### requires following arguments:
# 1. dataset name (after running MARIUS preprocessor and creating a bin file)
# 2. partition_num = total_nodes/nodes_per_page

import sys
import os
import json
import argparse
import pymetis
import numpy as np
import humanfriendly
from itertools import groupby


def read_config_file(config_file):
    with open(config_file, "r") as reader:
        return json.load(reader)

def read_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, help="The config file containing the details for the simulation")
    parser.add_argument("--data_name", required=True, type=str, help="The directory containing MARIUS preprocessed data")
    parser.add_argument("--partition_num",default=-10,type=int,help="Number of partitions to run METIS")
    return parser.parse_args()

def get_nodes_per_page(config):
    feature_stats = config["features_stats"]
    feature_size = np.dtype(feature_stats["feature_size"]).itemsize
    page_size = 1.0 * humanfriendly.parse_size(feature_stats["page_size"])
    feature_dimension = int(feature_stats["feature_dimension"])
    return int(page_size/(feature_size * feature_dimension))

def main():
    arguments = read_arguments()
    config = read_config_file(arguments.config_file)
    data_name = arguments.data_name#sys.argv[1]
    part_num = arguments.partition_num #int(sys.argv[2])

    path_for_bin_data = os.path.join("datasets",data_name,"edges","train_edges.bin")

    with open(path_for_bin_data, "rb") as reader:
        edges_bytes = reader.read()

    #edge list 
    edges_flaten_arr = np.frombuffer(edges_bytes, dtype=np.int32)
    edge_list = np.array(edges_flaten_arr.reshape((-1, 2)))

    #convert edge list into adj list 
    duplicated_edge_list = np.vstack([edge_list, edge_list[:, ::-1]])
    adj_dict={}
    for k,g in groupby(np.array(duplicated_edge_list), lambda e: e[0]):
        if k not in adj_dict:
            adj_dict[k]=[]
        adj_dict[k].extend([pair[1] for pair in g])

    adj_list = [nodes for nodes in adj_dict.values()]
    total_nodes_num = len(adj_list)
    nodes_per_page = get_nodes_per_page(config)
    if part_num<0:
        part_num= max(int(total_nodes_num/nodes_per_page),2)
        print("Partitioning with: ", part_num)


    #run metis 
    n_cuts, membership = pymetis.part_graph(part_num, adjacency=adj_list)
    print("METIS output N_cuts :",n_cuts)

    #write output 
    path_for_partitions = os.path.join("datasets",data_name,"partitions")
    output_fname = os.path.join(path_for_partitions,"membership_dict.json")
    if not os.path.exists(path_for_partitions):
        os.makedirs(path_for_partitions)

   
    mem_dict = dict(zip(range(len(adj_list)), membership))
    with open(output_fname,"w") as j_file:
            json.dump(mem_dict, j_file)
            print("Membership written sucesssfully at ", output_fname)


if __name__ == "__main__":
    main()