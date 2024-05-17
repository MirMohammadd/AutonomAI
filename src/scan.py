from torch_geometric.nn import GCNConv,GATConv
import torch

leaky_relu = lambda v,slope=.01: torch.where(v>0,v,slope*v)

class VGAEncoder(torch.nn.Module):
    def __init__(self,in_ch,out_ch):
        super(VGAEncoder,self).__init__()
        self.conv1 = GCNConv(in_ch, 2 * out_ch,cached=True)
        self.conv_mu = GCNConv(2 * out_ch, out_ch, cached=True)
        self.conv_logstd = GCNConv(2 * out_ch, out_ch, cached=True)

    def forward(self,x,edge_index):
        x = self.conv1(x,edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    