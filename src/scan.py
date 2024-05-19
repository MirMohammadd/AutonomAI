from torch_geometric.nn import GCNConv,GATConv
import torch
from sklearn.cluster import KMeans
import numpy as np


leaky_relu = lambda v,slope=.01: torch.where(v>0,v,slope*v)

class KmCluster(KMeans):
    def __init__(self,k,data):
        super().__init__(k)
        self.k = k
        self.dataSet = data
    
    def _dataPreProcess(self):
        if isinstance(self.dataSet,np.ndarray):
            mask = ~np.isnan(self.dataSet)
            notNan = self.dataSet[mask]
            return notNan
        else:
            return self.dataSet
    
    def _init_model(self):
        self.fit(self.k)
        return
    
    def _get_labels(self):
        return (self.labels_,self.cluster_centers_)
    
    def pred(self):
        self.predict()

class VGAEncoder(torch.nn.Module):
    def __init__(self,in_ch,out_ch):
        super(VGAEncoder,self).__init__()
        self.conv1 = GCNConv(in_ch, 2 * out_ch,cached=True)
        self.conv_mu = GCNConv(2 * out_ch, out_ch, cached=True)
        self.conv_logstd = GCNConv(2 * out_ch, out_ch, cached=True)

    def forward(self,x,edge_index):
        x = self.conv1(x,edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GAAEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAAEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, cached=True,heads=1)
        self.conv2 = GATConv(2 * out_channels, out_channels, cached=True,heads=1)

