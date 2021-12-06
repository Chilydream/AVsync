import torch
import torch.nn as nn
import torch.nn.functional as F


# 能大致看懂
#

class NetVLAD(nn.Module):
    """NetVLAD layer implementation

    Args:
        num_clusters : int
            The number of clusters
        dim : int
            Dimension of descriptors
        alpha : float
            Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool
            If true, descriptor-wise L2 normalization is applied to input.
    """

    def __init__(self, num_clusters=10, dim=512, alpha=100.0, normalize_input=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)

        # 中心点是随机生成的
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        # x: (N, C, H, W), H * W 对应论文中的N，表示局部特征的数目，C对应论文中的D，表示特征的维度

        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # (N, C, H, W) -> (N, num_clusters, H, W) -> (N, num_clusters, H * W)

        soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, H * W)

        x_flatten = x.view(N, C, -1)
        # (batch, feature_dim ,向量数量)

        # 计算残差

        a = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)
        #  (batch_size , num_clusters, feature_dim ,向量数量)

        b = self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        #  (num_clusters, feature_dim, 向量数量)

        residual = a - b
        # 4维的向量
        #  (N, num_clusters, feature_dim, 向量数量)

        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
        # 貌似就是做了一个mask

        vlad = residual.sum(dim=-1)
        # (N, num_clusters, C, H * W) -> (N, num_clusters, C)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad
