from .metrics import *
from marius.data import *
from marius.data.samplers import *
from .in_mem_storage import *

import torch
import numpy as np
import humanfriendly
import math
import time

class SubgraphSampler:
    def __init__(self, data_loader, config):
        self.config = config
        self.data_loader = data_loader
        self.sampling_depth = config["sampling_depth"]
        self.metrics = MetricTracker()
        self.initialize()
    
    def initialize(self):
        # Create the graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edges = self.data_loader.get_edges()
        total_nodes = self.data_loader.get_num_nodes()
        self.current_graph = MariusGraph(edges, edges[torch.argsort(edges[:, -1])], total_nodes)
        self.current_graph.to(self.device)

        # Create the features_config
        feature_stats = self.config["features_stats"]
        feature_size = np.dtype(feature_stats["feature_size"]).itemsize
        page_size = 1.0 * humanfriendly.parse_size(feature_stats["page_size"])
        feature_dimension = int(feature_stats["feature_dimension"])
        self.nodes_per_page = int(page_size/(feature_size * feature_dimension))

        # Determine the in memory nodes
        in_memory_nodes = torch.empty((0,), dtype = torch.int64).to(self.device)
        if "top_percent_in_mem" in self.config:
            percent_in_memory = float(self.config["top_percent_in_mem"])
            num_nodes_in_memory = int((total_nodes * percent_in_memory)/100.0)
            in_memory_nodes = self.data_loader.get_nodes_sorted_by_incoming()[ : num_nodes_in_memory]
        in_memory_nodes = in_memory_nodes.to(self.device)
        self.nodes_in_memory = in_memory_nodes.numel()
        print("Have", self.nodes_in_memory, "in memory nodes on device", in_memory_nodes.device)

        # Create the sampler
        sampling_depth = int(self.config["sampling_depth"])
        levels = [-1 for _ in range(sampling_depth)]
        self.sampler = LayeredNeighborSampler(self.current_graph, levels, in_memory_nodes)

    def perform_sampling_for_nodes(self, batch):        
        # Get all nodes
        batch = batch.to(self.device)
        nodes_to_load = 1.0 * self.sampler.getNeighborsNodes(batch)
        nodes_pages = torch.floor(nodes_to_load/self.nodes_per_page)
        unique_pages = torch.unique(nodes_pages)
        return True, unique_pages.numel()/batch.numel()

    def get_values_to_log(self):
        total_pages = int(math.ceil(self.data_loader.get_num_nodes() / (1.0 * self.nodes_per_page)))
        return {
            "Nodes Per Page" : self.nodes_per_page,
            "Total Pages" : total_pages,
            "Nodes In Memory" : self.nodes_in_memory
        }
    
    def get_metrics(self):
        metrics = self.metrics.get_metrics()
        metrics["avg_scaling_factor"]  = self.sampler.getAvgScalingFactor()
        metrics["avg_in_mem_nodes_removed"]  = self.sampler.getAvgPercentRemoved()
        return metrics
