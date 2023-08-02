import torch
import torch.nn as nn


class FrameEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.cnn1 = nn.Sequential(
                    nn.Conv1d(in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size = 3,
                    stride=2,
                    padding=1),
                    nn.LeakyReLU()
            )
        self.cnn2 = nn.Sequential(
                    nn.Conv1d(in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size = 3,
                    stride=2,
                    padding=1),
                    nn.LeakyReLU()
            )

        
    def forward(self, x):
        x = x.permute(0,2,1)
        z = self.cnn1(x)
        z = self.cnn2(z)
        z = z.permute(0,2,1)
        return z


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, dk = 128):        # dk 128 means 4heads when Q,K,V's shape as 512
        # K=V. It does not depend on number of modalities
        # but for example if K has n modalities, shape of K will be [bs, n, 128] 

        w = torch.bmm(Q,K.transpose(1,2))        # Q : [bs, 1, 128]  // K : [bs, 2, 128] // => w : [bs, 1, 2] 
        w = self.softmax(w/(dk**.5))
        c = torch.bmm(w, V)                      # C : [bs, 1, 128]
        return c


class MultiHead(nn.Module):
    def __init__(self, hidden_size, n_splits=4):  # 512 / 4 = 128
        '''
            hidden_size : last tensor shape
            n_splits : number of heads

            when 'query' is text hidden_size will be 768 since it shape as [bs, seq, 768]

        '''
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # projection
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, Q, K, V, mask=None):
        QWs = self.Q_linear(Q).split(self.hidden_size//self.n_splits, dim=-1)    # [bs, seq, 768] -> ([bs, seq, 192], [bs,seq,192] , ..)
        KWs = self.K_linear(K).split(self.hidden_size//self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size//self.n_splits, dim=-1)

        QWs = torch.cat(QWs, dim=0)  # ([bs, seq, 192])*4 -> [4bs, seq, 192]
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        
        c = self.attn(
            QWs, KWs, VWs,
            # mask=mask,
            dk=self.hidden_size//self.n_splits,
        )

        c = c.split(Q.size(0), dim=0)         # [4bs, seq, 192] => ([bs, seq, 192])*4
        c = self.linear(torch.cat(c, dim=-1)) # [bs, seq, 768]
        return c



    @torch.no_grad()
    def _create_positional_encoding_(self, hidden_size, max_length):
        '''
        this is for training

        예) pos = 3(word) // dim_idx = 2 = 2*i
                    pos
            sin( ---------  )
                    10^4(2*i/d)

        returning : [max_length, hs]
        '''
        empty = torch.FloatTensor(max_length, hidden_size).zero_()
        
        pos = torch.arange(0, max_length).unsqueeze(-1).float()    # |max_length, 1|
        dim = torch.arange(0, hidden_size//2).unsqueeze(0).float() # |1, hidden_size//2|
        
        empty[:, 0::2] = torch.sin(pos/1e+4**dim.div(float(hidden_size)))
        empty[:, 1::2] = torch.cos(pos/1e+4**dim.div(float(hidden_size)))

        return empty




class EncoderBlock(nn.Module):


    def __init__(self, hidden_size, n_splits, dropout_p=0.1, use_leaky_relu=False):
        super().__init__()

        self.attn_layernorm = nn.LayerNorm(hidden_size)
        self.multihead = MultiHead(hidden_size, n_splits)
        self.attn_dropout = nn.Dropout(dropout_p)
        # residual connection
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size),
        )
        # residual
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        x = self.attn_layernorm(x)
        multihead = self.attn_dropout(self.multihead(
                                                    Q=x,
                                                    K=x,
                                                    V=x,
                                                    mask=mask))
        z = x + multihead

        result = z + self.fc_dropout(self.fc(self.fc_norm(z)))

        return result, mask
    

class CustomSequential(nn.Sequential):
    # since nn.Sequential class does not provide multiple input arguments and returns.
    def forward(self, *x):
        for block in self._modules.values():
            x = block(*x)

        return x



class Transformer(nn.Module):

    def __init__(
            self,
            # input_size,
            # output_size,
            hidden_size,
            n_splits,
            last_size,
            max_length=512,
            dropout_p=0.1,
            num_enc_layer=6,
            # num_dec_layer=6,
            use_leaky_relu=False,
            ):

        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits
        self.dropout_p = dropout_p
        self.num_enc_layer = num_enc_layer
        self.max_length = max_length

        self.embed_frame = FrameEmbedding(hidden_size)
        # self.embed_enc = nn.Embedding(input_size, hidden_size)
        # self.embed_dec = nn.Embedding(output_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_p)
        # self.pos_enc = self._create_positional_encoding_(hidden_size,max_length)


        self.Encoder = CustomSequential(
                            *[EncoderBlock(hidden_size, 
                                            n_splits, 
                                            dropout_p, 
                                            use_leaky_relu,
                                            ) for _ in range(num_enc_layer)])


        # self.Decoder = CustomSequential(
        #                     *[DecoderBlock(hidden_size, 
        #                                     n_splits, 
        #                                     dropout_p, 
        #                                     use_leaky_relu,
        #                                     ) for _ in range(num_dec_layer)])
        
        self.generator = nn.Sequential(
                                nn.LayerNorm(hidden_size),
                                nn.Linear(hidden_size, last_size),
                                nn.LogSoftmax(dim = -1)        
                                )

    @torch.no_grad()
    def _create_positional_encoding_(self, hidden_size, max_length):
        '''
        this is for training

        예) pos = 3(word) // dim_idx = 2 = 2*i
                    pos
            sin( ---------  )
                    10^4(2*i/d)

        returning : [max_length, hs]
        '''
        empty = torch.FloatTensor(max_length, hidden_size).zero_()
        
        pos = torch.arange(0, max_length).unsqueeze(-1).float()    # |max_length, 1|
        dim = torch.arange(0, hidden_size//2).unsqueeze(0).float() # |1, hidden_size//2|
        
        empty[:, 0::2] = torch.sin(pos/1e+4**dim.div(float(hidden_size)))
        empty[:, 1::2] = torch.cos(pos/1e+4**dim.div(float(hidden_size)))

        return empty

    def _positional_encoding_(self, x, init_pos = 0):
        '''
        x = |bs, n, hs|        
        '''
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
        x = x+pos_enc.to(x.device)

        return x

    # me
    @torch.no_grad()
    def _generate_mask_(self, x, length):
        '''
        x : x[0] tensor
        length : x[1] length
        '''
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]
            else:
                # If length of sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()
        # |mask| = (batch_size, max_length)

        return mask




    def forward(self, x):
        '''
        x = (|bs, frame, features|
        y = |bs,m|
        frame_length : [lengths]

        '''        
 
        # for encoder
        with torch.no_grad():
            mask=None
            # mask = self._generate_mask_(x[0], frame_length) # |batch, n|
            # mask_enc = mask.unsqueeze(1).expand(*x[0].size(), mask.size(-1)) # |batch, 1, n| expand to [|batch, n| * n(mask.size(-1))]
            # mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1)) # |bs, m, n|

        # z = self.emb_dropout(
        #         self._positional_encoding_(
        #             self.embed_frame(x)))
        

        z = self.emb_dropout(
                    self.embed_frame(x))

        # z, _ = self.Encoder(z, mask_enc) # z = |bs, n, n|
        z, _ = self.Encoder(z, mask) # z = |bs, n, n|

        # for decoder
        with torch.no_grad():
            future_mask = None
            # future_mask = torch.triu(x[0].new_ones([y.size(1), y.size(1)]), diagonal = 1).bool()
            # future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
        
        # dz = self.emb_dropout(
        #         self._positional_encoding_(
        #             self.embed_dec(y)))


        #x, key_value, prev_tensor, mask, future_mask
        # dz, _, _, _, _ = self.Decoder(dz, z, None, mask_dec, future_mask)
        # dz = self.generator(dz)
        # dz = |bs, m, output_size|

        # return dz

        z = self.generator(z)
        return z

if __name__ == "__main__":
    a = torch.randn([32, 200, 40])
    y = torch.randint(low=0,high=10, size=[32]).long()
    transformer = Transformer(hidden_size=40,
                              n_splits=4,
                              last_size=59)
    z = transformer(a, y, y)
    print(1)