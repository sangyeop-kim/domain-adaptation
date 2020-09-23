import torch
from torch import nn
import torch.nn.functional as F

class Iterative_MultiTaskNet(nn.Module) :
    def __init__(self, encoding_kernel, attn_kernel):
        super(Iterative_MultiTaskNet, self).__init__()
        self.kernel = encoding_kernel
        self.hidden = attn_kernel
        
        # time-series encoding
        self.conv = nn.Conv2d(1, self.kernel, (65,3), bias = False)
        
        # convolution attention 
        self.attn_conv = nn.Conv2d(1, self.hidden, (self.kernel, 1), bias = False) # bias 지우기
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc_recipe1_1 = nn.Linear(self.kernel * self.hidden, 128)
        self.fc_recipe1_2 = nn.Linear(128, 128)
        self.fc_recipe1_3 = nn.Linear(128, 2)
        
        self.fc_recipe2_1 = nn.Linear(self.kernel * self.hidden, 128)
        self.fc_recipe2_2 = nn.Linear(128, 128)
        self.fc_recipe2_3 = nn.Linear(128, 2)
        
    def forward(self, x, num):
        ## Embedding ##
        batch, time, variable = x.size()
        x = x.view(-1,1,variable,time)
        x = F.tanh(self.conv(x))
        x = torch.transpose(x, 1,2)
        original_x = x
        x = self.attn_conv(x)
        x = F.softmax(x, dim = -1)
        x = self.dropout(x)
        
        x = torch.squeeze(x, dim = 2)
        x = torch.transpose(x, 1,2)
        original_x = torch.squeeze(original_x, dim = 1)
        output = torch.bmm(original_x, x)
        output = torch.flatten(output, start_dim=1)
        
        self.fc_recipe1_1.weight.requires_grad=True
        self.fc_recipe1_2.weight.requires_grad=True
        self.fc_recipe1_3.weight.requires_grad=True
        self.fc_recipe2_1.weight.requires_grad=True
        self.fc_recipe2_2.weight.requires_grad=True
        self.fc_recipe2_3.weight.requires_grad=True
        if num == 1 :
            ## Recipe1 Network ##
            self.fc_recipe2_1.weight.requires_grad=False
            self.fc_recipe2_2.weight.requires_grad=False
            self.fc_recipe2_3.weight.requires_grad=False
            
            output = F.relu(self.fc_recipe1_1(output))
            output = F.relu(self.fc_recipe1_2(output))
            output = self.fc_recipe1_3(output)
        else :
            ## Recipe2 Network ##
            self.fc_recipe1_1.weight.requires_grad=False
            self.fc_recipe1_2.weight.requires_grad=False
            self.fc_recipe1_3.weight.requires_grad=False
            output = F.relu(self.fc_recipe2_1(output))
            output = F.relu(self.fc_recipe2_2(output))
            output = self.fc_recipe2_3(output)
        return F.log_softmax(output, dim=1)
    
    
class SingleTaskNet(nn.Module) :
    def __init__(self, encoding_kernel, attn_kernel):
        super(SingleTaskNet, self).__init__()
        self.kernel = encoding_kernel
        self.hidden = attn_kernel
        self.conv = nn.Conv2d(1, self.kernel, (65,3), bias = False)
        #########
        self.attn_conv = nn.Conv2d(1, self.hidden, (self.kernel, 1), bias = False) # bias 지우기
        self.dropout = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(self.kernel * self.hidden, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x, lengths):
        ## Embedding ##
        batch, time, variable = x.size()
        x = x.view(-1,1,variable,time)
        x = F.tanh(self.conv(x))
        x = torch.transpose(x, 1,2)
        original_x = x
        x = self.attn_conv(x)
        x = F.softmax(x, dim = -1)
        x = self.dropout(x)
        
        x = torch.squeeze(x, dim = 2)
        x = torch.transpose(x, 1,2)
        original_x = torch.squeeze(original_x, dim = 1)
        output = torch.bmm(original_x, x)
        output = torch.flatten(output, start_dim=1)
        
        ## Network ##
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return F.log_softmax(output, dim=1)