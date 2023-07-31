import torch
import torch.nn as nn





'''
batch norm   vs    layer norm

https://m.blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=baek2sm&logNo=222176799509&categoryNo=99&proxyReferer=

'''


class GCNLayer(nn.Module):
    
    def __init__(self, 
                 hidden_size, 
                 activation=nn.LeakyReLU(),
                 skip_connection = True):
        super().__init__()

        self.linear_transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.activation = activation
        self.skip_connection = skip_connection

        # nn.init.xavier_uniform_(self.linear_transform.weight)

    def forward(self, x, adj):
        out = self.layernorm(x)
        out = self.linear_transform(out)
        out = self.activation(torch.matmul(adj, out))  ######## memory issue / shape is shie : [1, 181318, 21, 87] => 181318  should be near 200
        if self.skip_connection == True:
            out = out + x
        return out, adj
     

class CustomSequential(nn.Sequential):
    # since nn.Sequential class does not provide multiple input arguments and returns.
    def forward(self, *x):
        for block in self._modules.values():
            x = block(*x)
        return x



class ReadOut(nn.Module):
    
    def __init__(self, hidden_size, activation=nn.LeakyReLU(), sum_axis=-2):
        super().__init__()

        self.linear = nn.Linear(hidden_size, 
                                hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = activation
        self.sum_axis = sum_axis

    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, self.sum_axis) # -2
        out = self.activation(out)
        return out
    
class Predictor(nn.Module):
    
    def __init__(self, hidden_size, final_output, activation=nn.LeakyReLU()):
        super().__init__()

        self.linear = nn.Linear(hidden_size,
                                final_output)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = activation
        
    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out
    

    
class GCNNet(nn.Module):
    
    def __init__(self, 
                 hidden_size, 
                 final_output,
                 n_layer,
                 activation
    ):
        super().__init__()

        self.GCNEncoder = CustomSequential(
            *[GCNLayer(hidden_size = hidden_size,
                       activation = activation) for _ in range(n_layer)]
        )

        self.readout = ReadOut(hidden_size, activation=activation)

        self.pred1 = Predictor(hidden_size,
                              final_output,
                              activation=activation)
        
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, adj):
        z, _ = self.GCNEncoder(x, adj)  # [bs, frame, node, feature] - [32, 200, 21, 87]
        z = self.readout(z) # bs, frame, feature
        z = self.pred1(z) # [bs, frame, str_num]
        z = self.logSoftmax(z)
        return z
     



class GCNNetCNN(nn.Module):
    
    def __init__(self, 
                 hidden_size, 
                 final_output,
                 n_layer,
                 activation
    ):
        super().__init__()

        self.GCNEncoder = CustomSequential(
            *[GCNLayer(hidden_size = hidden_size,
                       activation = activation) for _ in range(n_layer)]
        )

        self.readout = ReadOut(hidden_size, activation=activation)

        self.pred1 = Predictor(hidden_size,
                              final_output,
                              activation=activation)
        
        self.logSoftmax = nn.LogSoftmax(dim=-1)

        self.Conv1 = nn.Conv1d(hidden_size, hidden_size, 3, 2, 1)
        self.Conv2 = nn.Conv1d(hidden_size, hidden_size, 3, 2, 1)
        self.activation = activation


    def forward(self, x, adj):
        z, _ = self.GCNEncoder(x, adj)  # [bs, frame, node, feature] - [32, 200, 21, 87]
        z = self.readout(z) # bs, frame, feature
        z = z.permute(0,2,1) # bs, feature, frame
        z = self.activation(self.Conv1(z)) # bs, 87, 100
        z = self.activation(self.Conv2(z)) # bs, 87, 50
        z = z.permute(0,2,1) # bs, 50, 87
        z = self.pred1(z) # [bs, frame, str_num]
        z = self.logSoftmax(z)
        return z
     
