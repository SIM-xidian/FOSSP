import torch
import torch.nn as nn
import torch.nn.functional as F


# 构造center loss
class CenterLoss(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def nt_xent_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2  # 同一图片内部不同patch也是负样本
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)  # 同一图片内部不同patch也是负样本，除了自己和自己
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= n
    return loss
