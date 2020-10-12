import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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


class HardMultiTaskNet(nn.Module) :
    def __init__(self, encoding_kernel, attn_kernel):
        super(HardMultiTaskNet, self).__init__()
        self.kernel = encoding_kernel
        self.hidden = attn_kernel
        
        # time-series encoding
        self.conv = nn.Conv2d(1, self.kernel, (65,3), bias = False)
        
        # convolution attention 
        self.attn_conv = nn.Conv2d(1, self.hidden, (self.kernel, 1), bias = False) # bias 지우기
        self.dropout = nn.Dropout2d(p=0.5, )
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

        output1 = F.relu(self.fc_recipe1_1(output))
        output1 = F.relu(self.fc_recipe1_2(output1))
        output1 = self.fc_recipe1_3(output1)

        output2 = F.relu(self.fc_recipe2_1(output))
        output2 = F.relu(self.fc_recipe2_2(output2))
        output2 = self.fc_recipe2_3(output2)
        return [F.log_softmax(output1, dim=1), F.log_softmax(output2, dim=1)]


class SoftMultiTaskNet(nn.Module) :
    def __init__(self, encoding_kernel, attn_kernel):
        super(SoftMultiTaskNet, self).__init__()
        self.kernel = encoding_kernel
        self.hidden = attn_kernel
        initial_weight = 2.5
        self.share_embed1 = Parameter(torch.tensor([initial_weight]))
        self.share_embed2 = Parameter(torch.tensor([initial_weight]))
        self.share_attn1 = Parameter(torch.tensor([initial_weight]))
        self.share_attn2 = Parameter(torch.tensor([initial_weight]))
        
        ############################ model1 ############################
        # time-series encoding
        self.conv1 = nn.Conv2d(1, self.kernel, (65,3), bias = False)
        # convolution attention 
        self.attn_conv1 = nn.Conv2d(1, self.hidden, (self.kernel, 1), bias = False) # bias 지우기
        self.dropout1 = nn.Dropout2d(p=0.5)
        # FCN
        self.fc_recipe1_1 = nn.Linear(self.kernel * self.hidden, 128)
        self.fc_recipe1_2 = nn.Linear(128, 128)
        self.fc_recipe1_3 = nn.Linear(128, 2)
        
        ############################ model2 ############################
        # time-series encoding
        self.conv2 = nn.Conv2d(1, self.kernel, (65,3), bias = False)
        # convolution attention 
        self.attn_conv2 = nn.Conv2d(1, self.hidden, (self.kernel, 1), bias = False) # bias 지우기
        self.dropout2 = nn.Dropout2d(p=0.5)
        # FCN
        self.fc_recipe2_1 = nn.Linear(self.kernel * self.hidden, 128)
        self.fc_recipe2_2 = nn.Linear(128, 128)
        self.fc_recipe2_3 = nn.Linear(128, 2)
        
        
    def forward(self, x1, x2):
        
        ## Embedding 1 ##
        batch, time, variable = x1.size()
        x1 = x1.view(-1,1,variable,time)
        x1_1 = F.tanh(self.conv1(x1))
        x1_2 = F.tanh(self.conv2(x1))
        
        ## Embedding 2 ##
        x2 = x2.view(-1,1,variable,time)
        x2_1 = F.tanh(self.conv2(x2))
        x2_2 = F.tanh(self.conv1(x2))
        
        ## Sharing Parameter ##
        x1 = x1_1 * F.sigmoid(self.share_embed1) + x2_2 * (1 - F.sigmoid(self.share_embed2))
        x2 = x2_1 * F.sigmoid(self.share_embed2) + x1_2 * (1 - F.sigmoid(self.share_embed1))

        
        ## Attention ##
        x1 = torch.transpose(x1, 1, 2)
        original_x1 = x1
        x1_1 = self.attn_conv1(x1)
        x1_2 = self.attn_conv2(x1)
        
        x2 = torch.transpose(x2, 1, 2)
        original_x2 = x2
        x2_1 = self.attn_conv2(x2)
        x2_2 = self.attn_conv1(x2)
    
        ## Sharing Parameter ##
        x1 = x1_1 * F.sigmoid(self.share_attn1) + x2_2 * (1 - F.sigmoid(self.share_attn2))
        x2 = x2_1 * F.sigmoid(self.share_attn2) + x1_2 * (1 - F.sigmoid(self.share_attn1))
        
        ## attention sum & dropout ##
        x1 = F.softmax(x1, dim = -1)
        x1 = self.dropout1(x1)
        x2 = F.softmax(x2, dim = -1)
        x2 = self.dropout2(x2)
        
        ## FCN 1 ##
        x1 = torch.squeeze(x1, dim = 2)
        x1 = torch.transpose(x1, 1, 2)
        original_x1 = torch.squeeze(original_x1, dim = 1)
        output1 = torch.bmm(original_x1, x1)
        output1 = torch.flatten(output1, start_dim=1)

        output1 = F.relu(self.fc_recipe1_1(output1))
        output1 = F.relu(self.fc_recipe1_2(output1))
        output1 = self.fc_recipe1_3(output1)
        
        ## FCN 2 ##
        x2 = torch.squeeze(x2, dim = 2)
        x2 = torch.transpose(x2, 1, 2)
        original_x2 = torch.squeeze(original_x2, dim = 1)
        output2 = torch.bmm(original_x2, x2)
        output2 = torch.flatten(output2, start_dim=1)

        output2 = F.relu(self.fc_recipe2_1(output2))
        output2 = F.relu(self.fc_recipe2_2(output2))
        output2 = self.fc_recipe2_3(output2)
        

        return [F.log_softmax(output1, dim=1), F.log_softmax(output2, dim=1)]