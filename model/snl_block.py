import torch
import torch.nn as nn


class ImprovedSNL(nn.Module):
    def __init__(self, in_channels, transfer_channels, stage_num=2):
        super(ImprovedSNL, self).__init__()
        self.in_channels = in_channels
        self.transfer_channels = transfer_channels
        self.stage_num = stage_num
        self.transform_t = nn.Conv2d(in_channels, transfer_channels, kernel_size=1, stride=1, bias=False)
        self.transform_p = nn.Conv2d(in_channels, transfer_channels, kernel_size=1, stride=1, bias=False)
        self.row_transform = nn.Conv2d(in_channels, transfer_channels, kernel_size=1, stride=1, bias=False)
        self.column_transform = nn.Conv2d(in_channels, transfer_channels, kernel_size=1, stride=1, bias=False)
        self.w1 = nn.Conv2d(transfer_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.w2 = nn.Conv2d(transfer_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def getAtt(self, x):
        t = self.transform_t(x)
        p = self.transform_p(x)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        m = torch.bmm(torch.relu(t), torch.relu(p))
        m += m.permute(0, 2, 1)
        m_hat = m / 2
        degree = torch.sum(m_hat, dim=2)
        degree[degree != 0] = torch.sqrt(1.0 / degree[degree != 0])
        affinity_matrix = m_hat * degree.unsqueeze(1)
        affinity_matrix *= degree.unsqueeze(2)
        
        return affinity_matrix

    def stage(self, x):
        affinity_matrix = self.getAtt(x)
        
        column_features = self.column_transform(x)
        b, c, h, w = column_features.size()
        column_features = column_features.view(b, c, -1)
        column_features = torch.bmm(column_features, affinity_matrix).contiguous().view(b,c,h,w)
        column_features = self.w1(column_features)
        
        row_features = self.row_transform(x)
        b, c, h, w = row_features.size()
        row_features = row_features.view(b, c, -1).permute(0, 2, 1)
        row_features = torch.bmm(affinity_matrix, row_features).permute(0, 2, 1).contiguous().view(b,c,h,w)
        row_features = self.w2(row_features)
        
        output = column_features + row_features
        output = self.bn(output)
        output = output + x

        return output

    def forward(self, x):
        for stage in range(self.stage_num):
            x = self.stage(x)

        return x






        





