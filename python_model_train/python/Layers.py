
import torch 
import torch.nn as nn
import math


def sinusoidal_2d_position_encoding( height_patches, width_patches, embed_dim):
    """
    Returns:
        PE shape = [height_patches * width_patches, embed_dim]
    Example:
        height_patches = 30
        width_patches  = 60
        embed_dim      = 256
    Output:
        [1800, 256]
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4")
    half_dim = embed_dim // 2
    row_dim = half_dim
    col_dim = half_dim
    # Row encoding
    row_pe = torch.zeros(height_patches, row_dim)
    row_position = torch.arange(height_patches,dtype=torch.float32).unsqueeze(1)
    row_div_term = torch.exp(torch.arange(0, row_dim, 2, dtype=torch.float32)*(-math.log(10000.0) / row_dim))
    row_pe[:, 0::2] = torch.sin(row_position * row_div_term)
    row_pe[:, 1::2] = torch.cos(row_position * row_div_term)
    # Column encoding
    col_pe = torch.zeros(width_patches, col_dim)
    col_position = torch.arange(width_patches,dtype=torch.float32).unsqueeze(1)
    col_div_term = torch.exp(torch.arange(0, col_dim,2,dtype=torch.float32)*(-math.log(10000.0) / col_dim))
    col_pe[:, 0::2] = torch.sin(col_position * col_div_term)
    col_pe[:, 1::2] = torch.cos(col_position * col_div_term)
    # Combine row + column
    pe = torch.zeros(height_patches, width_patches, embed_dim)
    for row in range(height_patches):
        for col in range(width_patches):
            pe[row, col] = torch.cat(
                [row_pe[row], col_pe[col]]
            )
    pe = pe.reshape(height_patches * width_patches,embed_dim)
    return pe

class input_to_transformer_end(nn.Module):
    def __init__(self, image_wid, height_patch, width_patch, embd_dim):
        super().__init__()
        self.a = nn.Linear(image_wid, embd_dim)
        poe = sinusoidal_2d_position_encoding(height_patch,width_patch, embd_dim)
        self.register_buffer("poe", poe)

    def forward(self, x):
        inter = self.a(x)
        return inter + self.poe

class Transformer_encode(nn.Module):
    def __init__(self, emd_dim):
        super().__init__()
        self.weight_RMS1 = nn.Parameter(torch.ones(emd_dim, dtype=torch.float32))
        self.weight_RMS2 = nn.Parameter(torch.ones(emd_dim, dtype=torch.float32))
        self.MLPlayer1 = nn.Linear(emd_dim,1728)
        self.MLPlayero = nn.Linear(1728, emd_dim)
        self.query = nn.Linear(emd_dim, emd_dim)
        self.key = nn.Linear(emd_dim, emd_dim)
        self.value = nn.Linear(emd_dim, emd_dim)
        self.output_project = nn.Linear(emd_dim,emd_dim)
        self.act = nn.SiLU()
        '''input(copy) -> RMSnorm -> attention  -> Residual add(attention out + copy)(copy2) -> MLP -> Residual add( MLP output + copy2)'''
    def forward(self, x):
        y = RMSnorm(x, self.weight_RMS1)
        y = attention_en(y,self.key, self.query, self.value, self.output_project)
        y = y + x
        x = RMSnorm(y, self.weight_RMS2)
        x = self.MLPlayer1(x)
        x = self.act(x)
        x = self.MLPlayero(x)
        y = y + x
        return y

def RMSnorm(x,w):
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x/rms)*w

def attention_en(x, wk, wq, wv, wo):
    # input(x)-> calculate xk,xq,xv - > divide in head -> calculate attention score -> applyu softmax - > calculate output -> output projection
    xk = wk(x)#[B,N,D]
    xq = wq(x)
    xv = wv(x)
    B = xk.shape[0]
    N = xk.shape[1]
    D = xk.shape[2]
    num_heads = 6
    head_dim = D//num_heads
    xq = xq.reshape(B, N, num_heads, head_dim)
    xk = xk.reshape(B, N, num_heads, head_dim)
    xv = xv.reshape(B, N, num_heads, head_dim)
    xq = xq.transpose(1, 2)#[B,num_heads N, head_dim]
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    scores = xq @ xk.transpose(-2, -1)#[B,num_heads N, head_dim]*[B,num_heads, head_dim,N]
    scores = scores / (head_dim ** 0.5)#[B,num_heads, N, N]
    attention = torch.softmax(scores, dim=-1)
    out = attention @ xv#[B,num_heads, N, N]*[B,num_heads N, head_dim]
    out = out.transpose(1, 2)
    out = out.reshape(B, N, D)
    out = wo(out)
    return out

def cross_attention(query, image_feature, cq,cv, ck,co):
    xq = cq(query)
    xv = cv(image_feature)
    xk = ck(image_feature)
    B = image_feature.shape[0]
    N = image_feature.shape[1]
    D = image_feature.shape[2]
    BQ = xq.shape[0]
    numofquer  = query.shape[-2]
    numofhead = 6
    headdim = D//numofhead
    xv = xv.reshape(B,N,numofhead,headdim)
    xk = xk.reshape(B,N,numofhead,headdim)
    xq = xq.reshape(BQ,numofquer, numofhead,headdim)
    xv =xv.transpose(1,2)
    xk = xk.transpose(1,2)
    xq = xq.transpose(1,2)
    scores = xq@xk.transpose(-2,-1)
    scores = scores / (headdim ** 0.5)
    attention = torch.softmax(scores, dim=-1)
    out = attention @ xv
    out = out.transpose(1,2)
    out = out.reshape(B,numofquer, D)
    out = co(out)
    return out

class Transformer_decoder(nn.Module):
    #Input (query, image feature)(copy of q)->RMS norm -> self attention(query) - Residual add(query+self_attention out)(copy2)
    # - > RMS norm ->cross attention(normalized2 , image feature) - > Residual add(copy3) -> RMS norm -> MLP->residual add
    def __init__(self, embd):
        super().__init__()
        self.self_query = nn.Linear(embd,embd)
        self.self_key = nn.Linear(embd,embd)
        self.self_value = nn.Linear(embd,embd)
        self.self_outproj = nn.Linear(embd,embd)
        self.cross_query = nn.Linear(embd,embd)
        self.cross_value = nn.Linear(embd,embd)
        self.cross_key = nn.Linear(embd,embd)
        self.cross_outproj = nn.Linear(embd,embd)
        self.weight_RMS1 = nn.Parameter(torch.ones(embd, dtype=torch.float32))
        self.weight_RMS2 = nn.Parameter(torch.ones(embd, dtype=torch.float32))
        self.weight_RMS3 = nn.Parameter(torch.ones(embd, dtype=torch.float32))
        self.MLPlayer1 = nn.Linear(embd,1728)
        self.MLPlayero = nn.Linear(1728, embd)
        self.act = nn.SiLU()

    def forward(self, x, img_featur):
        x = x.unsqueeze(0)
        y = RMSnorm(x, self.weight_RMS1)
        y = attention_en(y,self.self_key,self.self_query,  self.self_value, self.self_outproj)
        y = y + x
        x = RMSnorm(y, self.weight_RMS2)
        x = cross_attention(x, img_featur,self.cross_query ,self.cross_value, self.cross_key, self.cross_outproj)
        x = x + y
        y = RMSnorm(x, self.weight_RMS3)
        y = self.MLPlayer1(y)
        y = self.act(y)
        y = self.MLPlayero(y)
        y = y + x
        return y

class classification_head(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(emb_dim, num_classes + 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class BoxHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(emb_dim, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
