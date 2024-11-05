# Informer

## 背景及不足

许多实际应用需要长序列实际序列的预测，例如电力消耗计划

目前Transformer具有较强的长距离依赖能力，但传统的Transformer仍存在以下不足：

- Self-Attention平方级的计算复杂度。
- 堆叠多层网络，内存占用瓶颈。
- step-by-step解码预测，速度较慢。

## Informer改进

- 提出ProbSparse Self-Attention，筛选出最重要的query，使复杂度降低到$O(LlogL)$
- 提出Self-Attentio Distilling，减少维度和网络参数量。
- 提出Generative Style Decoder，一步得到所有预测结果。