import torch


# 参考论文:Utterance-level Aggregation For Speaker Recognition In The Wild
# library/speaker_recognition/3_VGG-Speaker-Recognition/src/model.py


class VladPooling(torch.nn.Module):
    def __init__(self, k_centers, feature_size):
        super(VladPooling, self).__init__()
        self.k_centers = k_centers
        self.feature_dim = feature_size

        # 暂时初始化为0向量
        self.cluster = torch.nn.Parameter(torch.zeros((self.k_centers, self.feature_dim)))
        # 然后初始化为正交向量,每行向量两两之间正交
        torch.nn.init.orthogonal_(self.cluster)

    def forward(self, feat_channel_first, cluster_score_channel_first):
        # pytorch中CNN的标准输出是（Batch，Channel，Height，Width）,下面将Channel于Width的位置调换:
        feat = feat_channel_first.permute(0, 3, 2, 1)
        # (batch,width,height,feature_dim)

        cluster_score = cluster_score_channel_first.permute(0, 3, 2, 1)
        # (batch,width,height,cluster_dim)

        assert feat.shape[-1] == self.feature_dim
        assert cluster_score.shape[-1] == self.k_centers

        max_cluster_score, _ = torch.max(cluster_score, dim=3, keepdim=True)
        # (batch,width,height,1)

        exp_cluster_score = torch.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / torch.sum(exp_cluster_score, dim=-1, keepdim=True)
        # (batch,width,height,10)

        A = A.unsqueeze(-1)
        # (batch,width,height,10,1)

        feat_broadcast = feat.unsqueeze(-2)
        # (batch,width,height,1,512)

        feat_res = feat_broadcast - self.cluster
        # (batch,width,height,10,512)

        weighted_res = A * feat_res
        # (batch,width,height,10,512)

        cluster_res = torch.sum(weighted_res, dim=[1, 2])
        # (batch,clusters,512)

        # cluster_l2 = K.l2_normalize(cluster_res, -1)
        cluster_l2 = torch.nn.functional.normalize(cluster_res, dim=-1, p=2)
        outputs = torch.reshape(cluster_l2, [-1, int(self.k_centers) * self.feature_dim])

        # (batch,5120)
        return outputs


if __name__ == "__main__":
    batch = 4
    width = 2
    height = 2
    cluster_dim = 10
    feature_size = 512
    cluster_score = torch.rand((batch, cluster_dim, width, height))
    feat = torch.rand((batch, feature_size, width, height))
    layer = VladPooling(cluster_dim, feature_size)
    layer(feat, cluster_score)
