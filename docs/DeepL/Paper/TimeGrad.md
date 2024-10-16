# Time Grad

**自回归**：是一种常用于时间序列预测的模型。一个简单的自回归模型可以由如下公式表示。核心思想是通过过去的时间步值来解释当前时间步的值，从而捕捉时间序列中的依赖性。

$$
x_{t}=\phi_{1} x_{t-1}+\phi_{2} x_{t-2}+\cdots+\phi_{p} x_{t-p}+\epsilon_{t}
$$


**自回归的优势**：自回归模型通过递归地使用前一个时间步的预测结果来生成下一个时间步的预测，这对于时间序列数据中的趋势和周期性变化非常有效。它能够逐步地调整每个时间步的预测，以反映时间序列中的复杂依赖关系。

TimeGrad 目标是学习一个条件分布模型，用来预测多变量时间序列未来时间步的分布，给定过去的数据和协变量，公式如下，其中协变量在所有时间点上是已知的。

$$
q\left(\mathbf{x}_{t_{0}: T}^{0} \mid \mathbf{x}_{1: t_{0}-1}^{0}, \mathbf{c}_{1: T}\right)=\prod_{t = t_{0}}^{T} q\left(\mathbf{x}_{t}^{0} \mid \mathbf{x}_{1: t-1}^{0}, \mathbf{c}_{1: T}\right)
$$

公式中的每个时间步的预测 $q\left(\mathbf{x}_{t_{0}: T}^{0} \mid \mathbf{x}_{1: t_{0}-1}^{0}, \mathbf{c}_{1: T}\right)$ 表明当前时间步的值不仅依赖于协变量，还依赖于之前的所有时间步。即预测是通过递归的方式完成的，这就形成了自回归机制。

将多变量时间序列的实体表示为 $x_{i, t}^{0} \in \mathbb{R}$，其中 $i \in\{1, \ldots, D\}$ 且 $t$ 为时间索引。因此 $t$ 时刻的多变量向量表示为 $x_t^{0} \in \mathbb{R}$。任务是预测在未来某个给定时间步的多变量分布。接下来考虑时间序列 $t \in [1, T]$，该序列是从训练数据的完整时间序列历史中采样的，将这段连续的序列分为上下文窗口（大小为 $[1,t_0]$ 和预测区间 $[t_0,T]$）

为了建模时间动态性，采样了 RNN 结构，该结构利用 LSTM 或 GRU 来编码时间点 t 之前的时序列，以在给定协变量 $c_t$ 的情况下，更新隐状态：

$$
\mathbf{h}_{t}=\operatorname{RNN}_{\theta}\left(\operatorname{concat}\left(\mathbf{x}_{t}^{0}, \mathbf{c}_{t}\right), \mathbf{h}_{t-1}\right)
$$

![Forward](images/Time%20Grad%20forward.jpg)

下图为 TimeGrad 原理图，RNN 产生每个时间点的隐状态，再通过 DDPM 由隐状态产生时间序列的值。

![time Grad](images/Time%20Grad.jpg)