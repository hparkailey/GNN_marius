from marius.data import Batch, DENSEGraph, MariusGraph
from marius.data.samplers import LayeredNeighborSampler

import subprocess
import torch
import os
import numpy as np
import time
import traceback
import threading
from collections import defaultdict
import pymetis
import json 

from .metrics import *

class DatasetLoader:
    SAVE_DIR = "datasets"
    EDGES_PATH = "edges/train_edges.bin"

    def __init__(self, config):
        self.config = config
        self.name = config["dataset_name"]
        self.sampling_depth = config["sampling_depth"]
        os.makedirs(DatasetLoader.SAVE_DIR, exist_ok=True)
        self.save_dir = os.path.join(DatasetLoader.SAVE_DIR, self.name)
        if not os.path.exists(self.save_dir):
            self.create_dataset()
        self.load_dataset()
        self.metrics = MetricTracker()
        self.max_batches_to_load = 75

    def create_dataset(self):
        command_to_run = f"marius_preprocess --dataset {self.name} --output_directory {self.save_dir}"
        print("Running command", command_to_run)
        subprocess.check_output(command_to_run, shell=True)

    def load_dataset(self):
        # Load the file
        edges_path = os.path.join(self.save_dir, DatasetLoader.EDGES_PATH)
        with open(edges_path, "rb") as reader:
            edges_bytes = reader.read()

        # Create the adjacency map
        edges_flaten_arr = np.frombuffer(edges_bytes, dtype=np.int32)
        edges_arr = edges_flaten_arr.reshape((-1, 2))
        self.edge_list = torch.tensor(edges_arr, dtype = torch.int64)
        self.total_nodes = torch.max(self.edge_list).item() + 1
        # divide total_nodes/SunGraphSampler.nodes_per_page to get the total number of partitions in running METIS 

        print("EDGE LIST: ", self.edge_list)
        path_for_partitions = os.path.join(self.SAVE_DIR,self.name,"partitions")
        n_cuts, membership = pymetis.part_graph(2, adjacency=self.edge_list)
        print(membership)
        if not os.path.exists(path_for_partitions):
            os.makedirs(path_for_partitions)

        membership_fname = "membership_dict.json"
        if not os.path.exists(os.path.join(path_for_partitions,membership_fname)):
            mem_dict = dict(zip(range(len(self.edge_list)), membership))
            with open(os.path.join(path_for_partitions,membership_fname),"w") as j_file:
                json.dump(mem_dict, j_file)
                print("Membership written sucesssfully at ", path_for_partitions)
        else:
            print("Membership file already exists at: ", path_for_partitions)
        
        print("Number of METIS cuts: ", n_cuts)

    # def edge_list_to_adj_list(edge_list):
    #     adj_list = []
    #     for edge_elem in edge_list:
    #         if edge_elm[0] in 
    
    
    def get_num_nodes(self):
        return self.total_nodes
    
    def get_edges(self):
        return self.edge_list
    
    def get_num_edges(self):
        return self.edge_list.size(0)
    
    def get_nodes_sorted_by_incoming(self):
        return torch.argsort(torch.bincount(self.edge_list[ : , 1]), descending=True)

    def get_average_neighbors(self):
        outgoing_nodes = self.edge_list[ : , 0]
        outgoing_unique_nodes = torch.unique(outgoing_nodes)
        return outgoing_nodes.size(0)/outgoing_unique_nodes.size(0)

    def get_average_incoming(self):
        incoming_nodes = self.edge_list[ : , 1]
        incoming_unique_nodes = torch.unique(incoming_nodes)
        return incoming_nodes.size(0)/incoming_unique_nodes.size(0)
    
    def get_metrics(self):
        return self.metrics.get_metrics()

    def get_values_to_log(self):
        return {
            "Average Node Out Degree": str(round(self.get_average_neighbors(), 2)),
            "Average Node In Degree": str(round(self.get_average_incoming(), 2)),
        }