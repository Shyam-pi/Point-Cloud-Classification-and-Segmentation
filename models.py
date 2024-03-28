import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, point_feat=False, in_channels = 3, num_points = 10000):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_maxpool = nn.MaxPool1d(num_points)
        self.point_feat = point_feat

    def forward(self, points):
        x = points
        B = x.size()[0]
        x = x.view(B,3,-1)
        # print(x.shape)
        pt_feat = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(pt_feat)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = torch.max(x, 2, keepdim=True)[0]
        x = self.global_maxpool(x)

        if self.point_feat:
            return x, pt_feat
        
        else:
            return x


# ------ TO DO ------

class cls_model(nn.Module):
    def __init__(self, num_points = 10000, num_classes=3):
        super(cls_model, self).__init__()
        self.feat = PointNetEncoder(num_points=num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points
        B = x.size()[0]
        x = x.view(B,3,-1)
        x = self.feat(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        return x



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_points = 10000, num_classes=6):
        super(seg_model, self).__init__()
        # self.conv1 = nn.Conv1d(3, 64, 1)
        # self.conv2 = nn.Conv1d(64, 128, 1)
        # self.conv3 = nn.Conv1d(128, 1024, 1)
        self.feat = PointNetEncoder(point_feat=True, in_channels=3, num_points=num_points)

        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.num_classes = num_classes
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points
        B = x.size()[0]
        num_points = x.size()[1]
        x = x.view(B,3,-1)
        x, pt_feat = self.feat(x)

        x = x.repeat(1, 1, num_points)

        # Concatenate the global feature vector with the per-point features
        x = torch.cat([pt_feat, x], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.view(B,self.num_classes, num_points)
        return x


