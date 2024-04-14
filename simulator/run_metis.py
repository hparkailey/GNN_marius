### Take in arguments:
# 1. dataset
# 2. partition_num

import sys
import os
import json
import pymetis
import numpy as np
from itertools import groupby


data_name = sys.argv[1]
part_num = sys.argv[2]

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

#run metis 
n_cuts, membership = pymetis.part_graph(part_num, adjacency=adj_list)
print("METIS output N_cuts :",n_cuts)

#write output 
path_for_partitions = os.path.join("datasets",data_name,"partitions")
output_fname = os.path.join(path_for_partitions,"membership_dict.json")
if not os.path.exists(path_for_partitions):
    os.makedirs(path_for_partitions)

elif  os.path.exists(output_fname):
    mem_dict = dict(zip(range(len(adj_list)), membership))
    with open(output_fname,"w") as j_file:
            json.dump(mem_dict, j_file)
            print("Membership written sucesssfully at ", output_fname)
else:
    print("Membership file already exists at: ", output_fname)
