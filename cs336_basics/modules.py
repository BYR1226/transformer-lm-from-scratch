import torch
import torch.nn as nn
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)#这里注意矩阵需要转置
        self.weight = nn.Parameter(weight_tensor)#需要放进参数方便更新
        sigma = (2/(in_features+out_features))**(0.5)
        #截断正态初始化
        nn.init.trunc_normal_(self.weight, mean = 0, std=sigma, a = -3*sigma, b = 3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.T#矩阵乘法y=xW.t,无偏置

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        g_tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(g_tensor)
        # self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        #performing RMSNorm
        rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim = True)+self.eps)
        #dim=-1表示对最后一维d_model做平方均值计算，keepdim = True表示这一维做完均值处理后不删去
        result = (x/rms)*self.weight
        result = result.to(in_dtype)
        return result

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)#矩阵的实际形状是d_ff行,d_model列;x的最后一维可以视为d_model行，1列
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)#但是torch里相乘是反过来的,y=xW.T,x视作一行d_model列
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.w1(x)
        z3 = self.w3(x)
        activated = z1*torch.sigmoid(z1)
        mixed = activated*z3
        out = self.w2(mixed)
        return out

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        k = torch.arange(d_k//2, device=device)#k是分组编号,0,1,2,3（假如d_k=8）
        theta_k = theta**(-2*k/d_k)#每组的旋转频率的核心公式
        i = torch.arange(max_seq_len, device=device)#i是位置下标
        freqs = torch.outer(i, theta_k)
        self.register_buffer("cos",torch.cos(freqs),persistent=False)
        self.register_buffer("sin",torch.sin(freqs),persistent=False)#避免把能按公式重算的表存进 state_dict

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_reshape = x.reshape(*x.shape[:-1],self.d_k//2,2)#把最后一维d_k拆成d_k//2组,每组2个数,同时保留前面任意数量的维度
        a = x_reshape[...,0]
        b = x_reshape[...,1]
        # RoPE旋转公式
        a_rotated = a*cos-b*sin
        b_rotated = a*sin+b*cos

        #把a和b在最后一维堆叠，再展平
        paired = torch.stack([a_rotated, b_rotated],dim=-1)
        result = paired.reshape(x.shape)
        return result

def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:
    max_vals = x.max(dim = dimension, keepdim = True).values#keepdim=True能保留原有维度，方便广播计算
    shifted = x - max_vals
    exp_vals = torch.exp(shifted)
    exp_sums = exp_vals.sum(dim = dimension, keepdim = True)
    return exp_vals/exp_sums    #归一化

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2,-1))/(d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, -float('inf'))
    scores = softmax(scores, dimension = -1)
    return scores @ V

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta:float | None = None,
        max_seq_len: int  |None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None and max_seq_len is not None:
            assert self.head_dim % 2 == 0
            self.rope=RoPE(theta, d_k = self.head_dim, max_seq_len = max_seq_len, device=device)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        #将q,k,v拆分为多头
        q = q.reshape(*q.shape[:-1],self.num_heads,self.head_dim)
        q = q.transpose(-3,-2)
        k = k.reshape(*k.shape[:-1],self.num_heads,self.head_dim)
        k = k.transpose(-3,-2)
        v = v.reshape(*v.shape[:-1],self.num_heads,self.head_dim)
        v = v.transpose(-3,-2)

        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        if self.rope is not None:
            token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        #保留下三角+对角线
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype = torch.bool))

        head_outputs = scaled_dot_product_attention(q, k, v, mask)
        head_outputs = head_outputs.transpose(-3,-2)
        head_outputs = head_outputs.reshape(x.shape)

        result = self.wo(head_outputs)
        return  result

class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.swi_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.swi = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #第一层:MHA
        normed_x = self.attn_norm(x)
        attn_output = self.attn(normed_x, token_positions)
        h = x + attn_output
        #第二层:SwiGLU
        normed_h = self.swi_norm(h)
        swi_output = self.swi(normed_h)
        result = h + swi_output
        return result

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        #创建多个block层,并保存进nn.ModuleList
        self.blocks = nn.ModuleList([
            transformer_block(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                theta = theta,
                max_seq_len = context_length,
                device = device,
                dtype = dtype,
            )
            for _ in range(num_layers)
        ])
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.final_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)#如果以后要处理“从非零位置开始”的序列，再显式传 token_positions
        x = self.final_norm(x)
        x = self.final_linear(x)
        # 这里不做 softmax，是因为模型返回的是每个候选词的原始分数（logits）。
        # 训练时，交叉熵损失会直接使用 logits，以更稳定的方式完成所需的概率计算；
        # 如果先在模型里做 softmax，再交给交叉熵，反而不合适。等到生成文本、需要按概率采样时，再对 logits 做 softmax。
        return x
