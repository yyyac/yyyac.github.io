# Transformer 以及 pyorch 实现

## Pad mask

```python
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
```

`seq_k.data.eq(0)` 返回大小和 `seq_k` 相同的 tensor。将 `seq_k` 值为 0 的位置返回 True，否则返回 False。例如输入为：`seq_k=[1,2,3,4,0]`，返回 `[F,F,F,F,T]`。后续在注意力机制计算时通过 `scores.masked_fill_(attn_mask, -1e9)`，将 `True` 填充为 `-inf`。


## Masked Self-Attention

```python
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]
```

在训练时，是把正确结果传入 Decoder。由于采用了 Self-Attention，模型会关注到所有信息，但模型不应知道当前时刻之后的信息，所有要进行 mask 处理。通过计算得到 Scaled Scores 后，只需再生成一个下三角全为 0，上三角全为负无穷的矩阵，然后与 Scaled Scores 相加即可。之后再做 softmax，就能将负无穷变为 0，因此当前时刻之后的信息也就被掩盖了。

## FeedForward Layer

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]
```

两个线性层加一个激活函数，之后是残差连接和 LayerNorm。

## ScaledDotProductAttention

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn
```

通过 Q 和 K 计算得到 scores，然后对 scores 进行 mask，之后进行 softmax 后再与 V 相乘，得到 context。

## MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  
                                                                           
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn
```

在多头注意力机制中，“其他头”实际上是通过对查询（$Q$）、键（$K$）、和值（$V$）向量的多次线性变换和分割操作实现的。每个头的计算实际上都是独立的，但在代码中它们是通过一次批量操作实现的。因此，我们从代码的角度可以看到多个头是如何实现的，而不用单独列出每个头。以下详细解释如何观察和理解“其他头”的存在：

**不同头的生成方式**

在代码中，多头是通过 `self.W_Q`、`self.W_K` 和 `self.W_V` 这三个线性层生成的。每个线性层会将输入维度从 `d_model`（如 512 维）转换为 `n_heads * d_k`（如 $8 \times 64 = 512$ 维），并通过 `view` 和 `transpose` 将其划分成多个头。具体代码如下：

```python
# Q: [batch_size, n_heads, len_q, d_k]
Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
# K: [batch_size, n_heads, len_k, d_k]
K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
# V: [batch_size, n_heads, len_v, d_v]
V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
```

这里的 `.view(batch_size, -1, n_heads, d_k).transpose(1, 2)` 操作，将 $Q$、$K$、$V$ 从形状 `(batch_size, seq_len, n_heads * d_k)` 转换为 `(batch_size, n_heads, seq_len, d_k)`。这样一来，`n_heads` 维度就代表了“多个头”。

例如，假设 `batch_size=2`、`seq_len=10`、`n_heads=8`、`d_k=64`，则此时 $Q$、$K$、$V$ 的形状将为 `(2, 8, 10, 64)`。在这种形状下，每个头独立地占据了 `n_heads` 维度的一个位置，因此第一个头的 Q 向量在 `Q[:, 0, :, :]`，第二个头的 Q 向量在 `Q[:, 1, :, :]`，以此类推。

**其他头是如何并行计算的**

虽然代码中没有单独列出每个头的具体计算，但因为我们将 $Q$、$K$、$V$ 的形状改成了 `(batch_size, n_heads, seq_len, d_k)`，PyTorch 可以自动在 `n_heads` 维度上并行计算每个头的注意力。具体在 `ScaledDotProductAttention` 中，代码如下：

```python
# scores : [batch_size, n_heads, len_q, len_k]
scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
```

这里的矩阵乘法 `torch.matmul(Q, K.transpose(-1, -2))` 会在 `n_heads` 维度上进行并行操作，计算每个头的注意力分数。这样，每个头的 Q 和 K 都会分别进行矩阵乘法，得到自己的注意力分数，不需要单独写出每个头的计算。

**观察多头的输出**

每个头的注意力计算完成后，得到的 `context` 的形状为 `[batch_size, n_heads, len_q, d_v]`。我们可以在 `n_heads` 维度上看到每个头的输出：

```python
# context 的形状为 [batch_size, n_heads, len_q, d_v]
context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
```

此时，`context[:, i, :, :]` 就表示第 $i$ 个头的输出。比如，如果 `context` 的形状是 `(2, 8, 10, 64)`，则：

- `context[:, 0, :, :]` 是第一个头的输出。
- `context[:, 1, :, :]` 是第二个头的输出。
- ……直到 `context[:, 7, :, :]` 表示第八个头的输出。

**合并多头输出**

在多头注意力机制中，所有头的输出最终会通过 `transpose` 和 `reshape` 操作拼接到一起，形成一个完整的输出。在代码中，合并操作如下：

```python
# context: [batch_size, len_q, n_heads * d_v]
context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
```

- `transpose(1, 2)` 将 `n_heads` 维度移动到 `len_q` 的后面。
- `reshape(batch_size, -1, n_heads * d_v)` 将所有头的输出拼接在一起，形成形状 `[batch_size, len_q, n_heads * d_v]`。

这样，多个头的输出被合并成一个大的输出，后续再通过线性层映射到 `d_model` 维度。

**总结**

- 每个头的 $Q$、$K$、$V$ 是通过线性层 `self.W_Q`、`self.W_K` 和 `self.W_V` 生成的，并通过 `view` 和 `transpose` 进行分头处理。
- `n_heads` 维度表示每个头的独立空间，通过在 `n_heads` 维度上的并行计算，可以在代码中同时处理所有头，而不需要分别计算。
- 多头的输出最终通过 `transpose` 和 `reshape` 拼接在一起，并通过线性层映射回 `d_model` 维度。

因此，虽然没有单独列出每个头的计算过程，但所有头的计算在 `n_heads` 维度上是并行完成的，通过观察 `n_heads` 维度的大小（如 8）就可以理解“多个头”的存在。