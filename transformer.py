import math

import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.d_model= d_model

    def forward(self,inputs):
        """
        (seq_len,id) -> Tensor shape: (Batch,seq_len,d_model)
        """
        return self.embedding(inputs) * (self.d_model**0.5) # 增加数值稳定性
class PositionalEncoding(nn.Module):
    """
    Tensor shape: (Batch,seq_len,d_model) -> Tensor shape: (Batch,seq_len,d_model) , random value
    """
    def __init__(self,seq_len,d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        pos = torch.arange(seq_len).unsqueeze(1) # (seq_len,1)
        div_term = 1/(10000**(torch.arange(0,d_model,step=2)/d_model)).unsqueeze(0) # (1,d_model//2)
        pe = torch.zeros(seq_len,d_model)
        pe[:,0::2] = torch.sin(pos*div_term) # (seq_len,d_model)
        pe[:,1::2] = torch.cos(pos*div_term) # X^T.dot(Y) 横向量点乘列向量->矩阵
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,embedding_output):
        """
        embedding_output:  (Batch,seq_len,d_model)
        """
        len_x= embedding_output.shape[1]
        return embedding_output + self.pe[:,:len_x,:]
class Multi_HeadAttention(nn.Module):
    def __init__(self,d_model=512,h=8,dropout=0.1):
        super().__init__()
        self.h = h
        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model,d_model)
        self.Wo = nn.Linear(d_model,d_model)
        assert d_model%h ==0, '选取的h,d_model有误'
        self.dropout = nn.Dropout(dropout)


    def attention(self,q,k,v,mask=None):
        '''
        没写dropout
        Parameters
        ----------
        h 放在最前面, 让attention看最后两项 , 一共有 Batch*h 个矩阵
        q (Batch,h,seq_len,d_k)
        k (Batch,h,seq_len,d_k)  k.T->(Batch,h,d_k,seq_len)
        v (Batch,h,seq_len,d_k)

        Returns ()
        -------
        '''
        d_k = q.shape[-1]
        tmp = q @ k.transpose(-1,-2) / math.sqrt(d_k) # (Batch,h,seq_len,seq_len)
        if mask is not None: # 潜在的错误??
            tmp.masked_fill_(mask==0,-math.inf) # 负无穷
        attention = torch.softmax(tmp,dim=-1)
        attention = self.dropout(attention)
        return attention @ v # dim=-1是计算每一行的softmax?

    def forward(self,query,key,value,mask=None):
        batch, seq_len, d_model = query.shape
        d_k = d_model // self.h

        k_batch ,k_seq_len,k_d_model = key.shape

        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        # (Batch,seq_len,d_model) -> (Batch,seq_len,h,d_k) -> (Batch,h,seq_len,d_k)
        query = query.reshape(batch,seq_len,self.h,d_k).transpose(1,2)
        key = key.reshape(k_batch,k_seq_len,self.h,d_k).transpose(1,2)
        value = value.reshape(k_batch,k_seq_len,self.h,d_k).transpose(1,2)

        # Ensure the intermediate shapes are correct
        assert query.shape == (batch, self.h, seq_len, d_k), f"Unexpected query shape: {query.shape}"
        assert key.shape == (k_batch, self.h, k_seq_len, d_k), f"Unexpected key shape: {key.shape}"
        assert value.shape == (k_batch, self.h, k_seq_len, d_k), f"Unexpected value shape: {value.shape}"

        attention_score = self.attention(query,key,value,mask) # (Batch,h,seq_len,d_k)

        attention_score = attention_score.transpose(-1,-2).reshape(batch,seq_len,d_model) # (这里不转置一下有问题吗?)

        return self.Wo(attention_score)

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.01):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):  # (Batch,seq_len,d_model)
        y = self.layernorm(x)
        return x + self.dropout(sublayer(y))
class FeedForward(nn.Module):
    def  __init__(self,d_model,dropout=0.01,d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class EncoderBlock(nn.Module):
    def __init__(self,d_model,h,dropout=0.1):
        super().__init__()
        self.self_attention = Multi_HeadAttention(d_model,h,dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model,dropout) for _ in range(2)])
        self.feed = FeedForward(d_model,dropout)

    def forward(self,x,enc_mask=None):
        x = self.residual_connections[0](x,lambda x:self.self_attention(x,x,x,enc_mask))
        x = self.residual_connections[1](x,self.feed)
        return x

class Encoder(nn.Module):
    def __init__(self,d_model=512,h=8,N=6,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model,h,dropout) for _ in range(N)])

    def forward(self,x,mask=None):
        for layer in self.layers:
            x = layer(x,mask)
        return  x
class DecoderBlock(nn.Module):
    def __init__(self,d_model=512,h=8,dropout=0.1):
        super().__init__()
        self.self_attention = Multi_HeadAttention(d_model,h,dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model,dropout) for _ in range(3)])
        self.cross_attention = Multi_HeadAttention(d_model,h,dropout)
        self.feed =FeedForward(d_model,dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda i:self.self_attention(i, i, i, tgt_mask))
        x = self.residual_connections[1](x, lambda i:self.cross_attention(i, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed)
        return x

class Decoder(nn.Module):
    def __init__(self,N=6,d_model=512,h=8,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model,h,dropout) for _ in range(N)])

    def forward(self, x, encoder_input, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_input, src_mask, tgt_mask)
        return x

class Projection(nn.Module):
    def __init__(self,vocab_size,d_model=512):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        # x -> (Batch,seq_len,d_model) ->(Batch,seq_len,vocab_size) -> (Batch,seq_len)
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self,vocab_size,src_seq_len=512,tgt_seq_len=128,d_model=256,h=8,N=6,dropout=0.01):
        super().__init__()
        self.enc_embedding = InputEmbedding(d_model,vocab_size)
        self.dec_embedding = InputEmbedding(d_model,vocab_size)

        self.enc_pos_enc = PositionalEncoding(src_seq_len, d_model)
        self.dec_pos_enc = PositionalEncoding(tgt_seq_len, d_model)

        self.enc = Encoder(d_model,h,N,dropout)
        self.dec = Decoder(N,d_model,h,dropout)
        self.proj = Projection(vocab_size,d_model)

    def encode(self,x,mask):
        x = self.enc_embedding(x)
        x = self.enc_pos_enc(x)
        return self.enc(x,mask)

    def decode(self, x, enc_output, src_mask, tgt_mask):
        x = self.dec_embedding(x)
        x = self.dec_pos_enc(x)
        return self.dec(x, enc_output, src_mask, tgt_mask)

    def project(self,x):
        return self.proj(x)



def mask_shapes():
    q = torch.randn(2, 10, 512)  # [B,S,D]
    mask = torch.ones(2, 1, 1, 10).bool()  # 正确形状
    attn = Multi_HeadAttention()
    output = attn(q, q, q, mask)  # 应无报错
    return output


def gradient_flow():
    # model = Transformer(3) # 只会三个词
    # optimizer = torch.optim.Adam(model.parameters())
    #
    # # 模拟训练步骤
    # loss = model(...).mean()
    # loss.backward()
    #
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(f"梯度中断：{name}")
    pass



def bulid_transformer(vocab_size,seq_len,d_model=512,h=8,N=6,dropout=0.1):
    transformer = Transformer(vocab_size,seq_len,d_model,h,N,dropout)
    return transformer

if __name__ == '__main__':

    a = mask_shapes()
    gradient_flow()
    bulid_transformer(3,512)
    print(a,a.shape)