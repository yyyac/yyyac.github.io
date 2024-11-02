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