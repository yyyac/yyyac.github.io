# Informer

## 背景及不足

许多实际应用需要长序列实际序列的预测，例如电力消耗计划

目前 Transformer 具有较强的长距离依赖能力，但传统的 Transformer 仍存在以下不足：

- Self-Attention 平方级的计算复杂度。
- 堆叠多层网络，内存占用瓶颈。
- step-by-step 解码预测，速度较慢。

## Informer 改进

- 提出 ProbSparse Self-Attention，筛选出最重要的 query，使复杂度降低到 $O(LlogL)$
- 提出 Self-Attention Distilling，减少维度和网络参数量。
- 提出 Generative Style Decoder，一步得到所有预测结果。

左图展示了与短期预测相比，LTSF 可以预测更长的序列；右图表明随着预测序列长度增加，从 $L=48$ 开始，MSE 迅速增大推理速度下降。

<figure markdown=span> ![](images/time-past.jpg) </figure>

## Informer 架构

<figure markdown=span> ![](images/Informer.jpg) </figure>

- Encoder 接受大量长序列输入。模型采用了 ProbSparse Self-Attention 代替了 Transformer 中的 Self-Attention。并且 Encoder 在堆叠时采用了 Self-Attention Distilling。
- Decoder 同样接受长序列输入，预测部分用 0 进行 padding。结果处理后直接输出所有预测结果。

## 预处理（以 ETTh1 为例）

