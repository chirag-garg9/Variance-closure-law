import torch
from ogb.nodeproppred import PygNodePropPredDataset

class ArxivDomainSplit:
    def __init__(self, dataset_name="ogbn-arxiv"):
        # We load the raw dataset without ToSparseTensor for AD compatibility
        self.dataset = PygNodePropPredDataset(name=dataset_name)
        self.data = self.dataset[0]
        
        # Temporal Domain Split
        self.years = self.data.node_year.squeeze()
        self.t1_mask = self.years <= 2017
        self.t2_mask = self.years >= 2018
        
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes

    def get_task_data(self):
        # Returns raw data containing edge_index, and the task-specific masks
        return self.data, self.t1_mask, self.t2_mask