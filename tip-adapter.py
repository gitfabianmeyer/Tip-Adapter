import torch
import torch.nn.functional as F
import torch.nn as nn


class WeightAdapter(nn.Module):
    def __init__(self,
                 clip_model,
                 train_features_path,
                 cls_num,
                 shots):
        super().__init__()

        # TODO why no bias
        self.linear1 = nn.Linear(1024, cls_num * shots, bias=False).to(clip_model.dtype)
        self.linear1.weight = nn.Parameter(torch.load(train_features_path).t())


