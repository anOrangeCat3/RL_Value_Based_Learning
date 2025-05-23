import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):
    '''
    only 1 hidden layer
    '''
    def __init__(self, 
                 obs_dim:int, 
                 action_dim:int,
                 hidden_dim:int=64 )->None:
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x

class VANet(torch.nn.Module):
    def __init__(self,
                 obs_dim:int,
                 action_dim:int,
                 hidden_dim:int=64)->None:
        super(VANet, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self,
                x:torch.Tensor)->torch.Tensor:
        # A=self.fc_A(F.relu(self.fc1(x)))
        # V=self.fc_V(F.relu(self.fc1(x)))
        # Q=V+A-A.mean(dim=1).view(-1,1)
        # return Q
        # 检查输入维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # 添加batch维度
        
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(dim=1).view(-1,1)
        
        # 如果输入是单个样本，去掉batch维度
        return Q.squeeze(0) if len(x.shape) == 1 else Q
        