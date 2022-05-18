import torch
import numpy as np
import torch.nn as nn
from timeit import default_timer
from torch.optim import Adam
import time
import torch.nn.functional as F


#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ResFNO(dataT, dataA, dataTair,task='T', x_index=0, ntrain=50):

    
    ################################################################
    ntest = 200 - ntrain # 
    
    batch_size = 10
    learning_rate = 0.001
    epochs = 300
    step_size = 50
    gamma = 0.5
    
    ############ parameters of Fourier layer ##########
    modes = 16   # number of frequncy modes
    width = 64   # size of channel
    
    ############# load data ####################################
    nx = dataA.shape[1]
    x_data = torch.from_numpy(dataTair.astype(np.float32)).to(device) 
    if task=='A': # A  Degree of cure
        y_data = torch.from_numpy(dataA[:,x_index,:].astype(np.float32)).to(device) 
    if task=='T': # T Temperature
        y_data = torch.from_numpy(dataT[:,x_index,:].astype(np.float32)).to(device) 
    
    
    # Normalization
    norm_x = RangeNormalizer(x_data)
    x_data = norm_x.encode(x_data)
    
    if task=='T':  # Cure of degree donot need normlization
        norm_y = RangeNormalizer(y_data)
        y_data = norm_y.encode(y_data) 
    
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    s = y_data.shape[1]
    x_train = x_train.reshape(ntrain,s,1)
    x_test = x_test.reshape(ntest,s,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # model
    
    model = FNO1d(modes, width, task).to(device) 
    
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = LpLoss(size_average=False)
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device) , y.to(device) 
            optimizer.zero_grad()
            out = model(x)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
                 
            '''
            This is a small trick, 
            minimize the second derivative to maintain smoothness
            '''
            
            d1 = out[:, 1:, :] - out[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            ld2 = torch.max(abs(d2))
            
            loss = l2 + 0.5*ld2
            l2.backward() 
            
            train_l2 += l2.item()
            optimizer.step()
            
        scheduler.step()
        model.eval()
        

        test_l2 = 0.0
        ET_max = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device) , y.to(device) 
                out = model(x)
                test_l2 += myloss((out).view(batch_size, -1), (y).view(batch_size, -1)).item()
                ET_max = torch.max(torch.abs(norm_y.decode(out)[:,:,0] - norm_y.decode(y)))
                
        train_l2  /= ntrain
        test_l2   /= ntest
        train_error[ep] = train_l2
        test_error[ep] = test_l2
        ET_list[ep] = ET_max
        
        t2 = default_timer()
        
        if ep%10 == 0:
            time.sleep(0.0002)
            print('\rtask:', task, 'x=', 
                  x_index, '/', nx, 
                  ', Epoch:', ep, 
                  '/',  str(epochs), ' dt=', np.round(t2-t1, 5),  
                  ' L2_train=', np.round(train_l2, 5), 
                  ' L2_test=',np.round(test_l2, 5),
                   ' ld2 =',np.round(ld2.cpu().detach().numpy(), 5),
                  end = '\r')
        

    with torch.no_grad():
        
        pre_test  = model(x_test)
        pre_train = model(x_train)
        
        x_test    = norm_x.decode(x_test)
        x_train   = norm_x.decode(x_train)
        
        if task=='T': 
            pre_test  = norm_y.decode(pre_test)
            pre_train = norm_y.decode(pre_train)
            y_test    = norm_y.decode(y_test)
            y_train   = norm_y.decode(y_train)
    
    
    # Output: model, loss, prediction results
    model_output = model.state_dict()
    loss_dict = {'train_error':train_error,
                 'test_error' :test_error,
                 'deltaT'     :ET_list}
    
    out_data_dict = {
                'pre_test' : pre_test.cpu().detach().numpy(),
                'pre_train': pre_train.cpu().detach().numpy(),
                'x_test'   : x_test.cpu().detach().numpy(),
                'x_train'  : x_train.cpu().detach().numpy(),
                'y_test'   : y_test.cpu().detach().numpy(),
                'y_train'  : y_train.cpu().detach().numpy(),
                }
    
    return model_output, loss_dict, out_data_dict

def Predict(model, dataTair, dataT, x_index, T_index):
        # Normalization

    x_data_ori = torch.from_numpy(dataTair.astype(np.float32)).to(device) 
    y_data_ori = torch.from_numpy(dataT[:,x_index,:].astype(np.float32)).to(device) 
    # Normalization
    norm_x = RangeNormalizer(x_data_ori)
    x_data = norm_x.encode(x_data_ori)
    
    norm_y = RangeNormalizer(y_data_ori)
    
    x_input  = x_data[T_index].reshape(1, x_data.shape[1], 1)
    y_pre = model(x_input)
    
    y_pre  = norm_y.decode(y_pre)
    
    x_input = x_data_ori[T_index].detach().numpy().reshape(-1)
    T_Pre   = y_pre.detach().numpy().reshape(-1)
    T_Real  = y_data_ori[T_index].detach().numpy().reshape(-1)
    
    return x_input, T_Pre, T_Real
    
################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # Tensor operation
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier modes
        x_ft = torch.fft.rfft(x) 

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        # Abandon high frequency modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, task):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.task = task
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x_in1 = x 
        
        x = torch.cat((x, grid), dim=-1)
        
        ##############################   Fourier Residual Layer #################
        x_in_ft = torch.fft.rfft(x_in1, axis=-2) 
        x_in_ft[:, self.modes1:, :] = 0
        x_ifft = torch.fft.irfft(x_in_ft, n = x_in1.size(-2), axis=-2)
        ########################################################################
        
        x = self.fc0(x) 
        
        x = x.permute(0, 2, 1) 
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x) 

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2   

        x = x.permute(0, 2, 1) 
        x = self.fc1(x)
        x = F.gelu(x)
        
        x = self.fc2(x) 
        x = self.fc3(x) 
        
        
        ##############################   Fourier Residual Layer #################
        if self.task=='T':
            x = x  + x_ifft 
        ########################################################################
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


