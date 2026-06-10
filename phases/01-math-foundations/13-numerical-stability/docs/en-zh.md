# 数值稳定性（Numerical Stability）

> 浮点数是"漏水"的抽象。它会在训练时悄悄咬你一口，而你毫无察觉。

**类型：** 动手构建
**语言：** Python
**前置知识：** 阶段 1，第 01-04 课
**时间：** ~120 分钟

## 学习目标（Learning Objectives）

- 使用极大值减法技巧实现数值稳定的 softmax 和对数-求和-指数（log-sum-exp）
- 识别浮点数计算中的上溢（overflow）、下溢（underflow）和灾难性抵消（catastrophic cancellation）
- 使用中心有限差分验证解析梯度与数值梯度的一致性
- 解释为什么 bfloat16 比 float16 更适合训练，以及损失缩放（loss scaling）如何防止梯度下溢

## 问题背景（The Problem）

你的模型训练了三小时，然后损失变成了 NaN。你加了一行打印语句。第 9,000 步时 logits 还正常。第 9,001 步它们变成了 `inf`。到第 9,002 步每个梯度都是 `nan`，训练彻底死亡。

又或者：你的模型训练完成了，但准确率比论文声称的低 2%。你检查了一切。架构匹配、超参数匹配、数据匹配。问题在于论文用了 float32，而你用了 float16 却没有做正确的缩放。三十二位的累积舍入误差悄悄吞噬了你的准确率。

再或者：你从头实现了交叉熵损失（cross-entropy loss）。它在小 logits 时正常工作。当 logits 超过 100，它返回 `inf`。softmax 发生了上溢，因为 `exp(100)` 超出了 float32 能表示的范围。每个 ML 框架都用一个两行的小技巧处理了这个问题。而你并不知道这个技巧的存在。

数值稳定性不是一个理论问题。它是成功训练与静默失败之间的分水岭。你最终要调试的每一个严重的 ML bug 归根结底都与浮点数有关。

## 核心概念（The Concept）

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点数。一个浮点数有三个部分：符号位（sign bit）、指数（exponent）和尾数（mantissa / significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（多少位有效数字）。指数决定范围（一个数可以有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 给你大约 7 位十进制精度。这意味着它可以区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002。超过 7 位后，全都是舍入噪声。

float16 给你大约 3 位精度。它能表示的最大数是 65,504。这在 ML 中小得令人不安——logits、梯度和激活值常常超过这个值。

bfloat16 是 Google 针对 float16 范围问题的解决方案。它拥有与 float32 相同的 8 位指数（相同范围，最大 3.4e38），但只有 7 位尾数（精度比 float16 还低）。对于训练神经网络，范围比精度更重要，所以 bfloat16 通常是赢家。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法用二进制浮点数精确表示。在二进制中，它是一个无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 的影响在于：

1. 像 `if loss < threshold` 这样的损失比较可能给出错误结果
2. 累积大量小数值（数千步的梯度更新）会偏离真实总和
3. 如果你用 `==` 比较浮点数，校验和和可重现性测试会失败

解决办法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消（Catastrophic Cancellation）

当你相减两个几乎相等的浮点数时，有效数字相互抵消，只剩下舍入噪声被提升为前导数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

一次减法就带来了 19% 的相对误差。在 ML 中，这种情况会在以下情形发生：

- 计算均值很大的数据的方差：$\mathbb{E}[x^2] - \mathbb{E}[x]^2$，当 $\mathbb{E}[x]$ 很大时
- 相减几乎相等的对数概率
- 使用太小的 $\epsilon$ 计算有限差分梯度

解决办法：重新排列公式，避免相减大而几乎相等的数。对于方差，使用 Welford 算法或先中心化数据。对于对数概率，全程在对数空间中计算。

### 上溢和下溢（Overflow and Underflow）

**上溢**发生在结果太大而无法表示时。**下溢**发生在结果太小（比最小可表示正数更接近零）时。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中上溢的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则冲向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。如果不使用正确的技巧，`log(exp(x))` 的组合就是一个雷区。

### 对数-求和-指数技巧（Log-Sum-Exp Trick）

直接计算 $\log(\sum \exp(x_i))$ 在数值上很危险。如果任何一个 $x_i$ 很大，$\exp(x_i)$ 会溢出。如果所有 $x_i$ 都非常负，每个 $\exp(x_i)$ 都会下溢为零，而 $\log(0) = -\infty$。

技巧：在取指数之前减去最大值。

$$
\log\left(\sum e^{x_i}\right) = \max(x) + \log\left(\sum e^{x_i - \max(x)}\right)
$$

为什么有效：减去 $\max(x)$ 后，最大的指数项是 $e^0 = 1$。不可能再发生上溢。至少有一项是 1，因此总和至少为 1，$\log(1) = 0$。不可能再下溢到 $-\infty$。

证明：

$$
\begin{aligned}
\log\left(\sum e^{x_i}\right)
&= \log\left(\sum e^{x_i - c + c}\right) \\[15pt]
&= \log\left(\sum e^{x_i - c} \cdot e^c\right) \\[15pt]
&= \log\left(e^c \cdot \sum e^{x_i - c}\right) \\[15pt]
&= c + \log\left(\sum e^{x_i - c}\right)
\end{aligned}
$$

令 $c = \max(x)$，则消除了上溢的可能。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型（Mixture of Gaussians）
- 变分推断（Variational Inference）

### 为什么 Softmax 需要极大值减法技巧

Softmax 将 logits 转换为概率：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}
$$

如果不使用技巧，logits 为 [100, 101, 102] 时会引发上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用技巧，减去 $\max(x) = 102$：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同，计算安全可靠。这不是优化，而是正确性的必要条件。

### NaN 和 Inf：检测与预防

`nan`（Not a Number）和 `inf`（无穷大）在计算中像病毒一样传播。梯度更新中一个 `nan` 会让权重变成 `nan`，进而使后续所有输出都变成 `nan`。一步之内训练就彻底死亡。

`inf` 的出现方式：
- 大正数的 `exp()` 计算
- 除以零：$1.0 / 0.0$
- 累积过程中的 `float32` 上溢

`nan` 的出现方式：
- $0.0 / 0.0$
- $\infty - \infty$
- $\infty \times 0$
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及已有 `nan` 的运算

**检测方法：**

```python
import math

# 检查是否出现 nan/inf，用于监控训练过程中的数值异常
# 在每次前向传播后调用，以便第一时间发现数值问题
math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

**预防策略：**

1. 对 `exp()` 的输入做裁剪：$\exp(\text{clamp}(x, -80, 80))$
2. 给分母加上 epsilon：$x / (y + \text{1e-8})$
3. 在 `log()` 内部加上 epsilon：$\log(x + \text{1e-8})$
4. 使用稳定的实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪防止权重爆炸
6. 调试时在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查（Numerical Gradient Checking）

反向传播得到的**解析梯度**（analytical gradients）可能存在 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

$$
\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
$$

这是 $O(h^2)$ 精度，远比只有 $O(h)$ 的前向差分 $\frac{f(x+h) - f(x)}{h}$ 好。

$h$ 的选择：太大则近似误差大；太小则灾难性抵消会破坏结果。通常取 $h = 10^{-5}$ 到 $10^{-7}$。

检查方法：计算解析梯度与数值梯度之间的相对差异。

$$
\text{relative_error} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|, 10^{-8})}
$$

经验法则：
- $\text{relative_error} < 10^{-7}$：完美，梯度正确
- $\text{relative_error} < 10^{-5}$：可接受，很可能正确
- $\text{relative_error} > 10^{-3}$：有问题
- $\text{relative_error} > 1$：梯度完全错误

在实现新层或新损失函数时，务必检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练（Mixed Precision Training）

现代 GPU 拥有专门的硬件（Tensor Cores），可以比 float32 快 2-8 倍的速度计算 float16 矩阵乘法。混合精度训练利用了这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题：梯度通常非常小（$10^{-8}$ 或更小）。Float16 将任何低于约 $6 \times 10^{-8}$ 的值下溢为零。你的模型因为所有梯度更新都是零而停止学习。

解决办法是**损失缩放**（loss scaling）：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

**动态损失缩放**自动调整缩放因子。从一个较大的值（65536）开始。如果梯度上溢到 `inf`，减半。如果连续 N 步没有发生上溢，加倍。

### bfloat16 对比 float16：为什么 bfloat16 在训练中胜出

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 位尾数 vs 7 位），但范围有限（最大约 65,504）。bfloat16 精度更低，但与 float32 有相同的范围（最大约 3.4e38）。

对于训练神经网络：

- 激活值和 logits 在训练尖峰期间经常超过 65,504。float16 会溢出；bfloat16 可以处理。
- 使用 float16 时需要损失缩放，但 bfloat16 通常不需要，因为它的范围覆盖了梯度的幅度谱。
- bfloat16 是 float32 的简单截断：丢掉尾数的低 16 位。转换简单且指数部分无损。

float16 在推理场景中更受青睐（此时值有界，精度更重要）。bfloat16 在训练中更受青睐（此时范围更重要）。这就是 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16 的原因。

### 梯度裁剪（Gradient Clipping）

**梯度爆炸**（exploding gradients）发生在梯度通过多层网络呈指数增长时（常见于 RNN、深层网络和 Transformer）。单次大梯度可以在一步之内破坏所有权重。

两种裁剪方式：

**按值裁剪（Clip by value）：** 独立裁剪每个梯度元素。

$$
\text{grad} = \text{clamp}(\text{grad}, -\text{max\_val}, \text{max\_val})
$$

简单但可能改变梯度向量的方向。

**按范数裁剪（Clip by norm）：** 缩放整个梯度向量，使其范数不超过阈值。

$$
\text{if } ||\text{grad}|| > \text{max\_norm}: \quad \text{grad} = \text{grad} \times \frac{\text{max\_norm}}{||\text{grad}||}
$$

保留了梯度的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的。它是标准选择。

典型取值：Transformer 中 `max_norm=1.0`，强化学习中 `max_norm=0.5`，较简单网络中 `max_norm=5.0`。

梯度裁剪不是一个 hack。它是一个安全机制。没有它，一个离群批次就能产生大到足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批归一化（Batch Normalization）、层归一化（Layer Normalization）和 RMS 归一化通常被表述为有助于训练收敛的正则化方法。它们同时也是数值稳定器。

没有归一化，激活值在各层之间会呈指数增长或收缩：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层重新居中和缩放激活值：

$$
\text{LayerNorm}(x) = \frac{x - \mu(x)}{\sigma(x) + \epsilon} \cdot \gamma + \beta
$$

$\epsilon$（通常为 $10^{-5}$）防止所有激活值相同时除零。可学习参数 $\gamma$ 和 $\beta$ 让网络在需要时可以恢复任意尺度。

这使数值在整个网络中保持在安全范围内，既防止了前向传播中的上溢，也防止了反向传播中的梯度爆炸。

### 常见 ML 数值 Bug

**Bug：几轮 epoch 后损失变成 NaN。**
原因：logits 变得太大，softmax 上溢。或者学习率太高导致权重发散。
解决：使用稳定 softmax（极大值减法），降低学习率，加入梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近于均匀概率。通常意味着梯度消失或模型根本没在学习。
解决：检查数据标签是否正确，验证损失函数，检查是否存在死亡 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度训练没有正确的损失缩放。梯度下溢静默地将小更新置零。
解决：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死亡 ReLU 神经元（所有输入为负），或 float16 下溢。
解决：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一张 GPU 上正常，在另一张上结果不同。**
原因：非确定性的浮点累积顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不可交换。
解决：接受微小差异（$10^{-6}$），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：`exp()` 在损失计算中返回 `inf`。**
原因：原始 logits 在没有极大值减法技巧的情况下传入 `exp()`。
解决：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 $6 \times 10^{-8}$ 的梯度幅度或高于 65,504 的激活值。
解决：使用混合精度加损失缩放（AMP），或改用 bfloat16。

## 动手实现（Build It）

### 步骤 1：演示浮点精度限制

```python
# 演示浮点数精度问题：0.1 + 0.2 != 0.3
# 这是因为 0.1 在二进制中无限循环，float32 截断尾数后产生表示误差
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素版与稳定版 softmax

```python
import math

# 朴素版 softmax：直接计算 exp(x_i) / sum(exp(x_j))
# 问题：当 logits 较大时（如 > 88），exp() 在 float32 中会溢出为 inf
def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

# 稳定版 softmax：先减去最大值再计算 exp
# 原理：softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
# 减去最大值后，最大指数项为 exp(0) = 1，不可能溢出
def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

# 安全的 logits：两种方法结果一致
safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

# 危险的 logits：朴素版会返回 [nan, nan, nan]
# 因为 exp(100) 远超 float32 上限 3.4e38，exp(100) = inf，softmax 结果全为 nan
dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤 3：实现稳定的 log-sum-exp

```python
# 朴素版 log-sum-exp：直接计算 log(sum(exp(x)))
# 存在双重风险：大 x 导致 exp 上溢，小 x 导致 exp 下溢为 0 后 log(0) = -inf
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

# 稳定版 log-sum-exp：提取最大值因子
# log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
# 保证 exp 参数 ≤ 0，最大项为 exp(0) = 1，既不上溢也不下溢
def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
# 朴素版 exp(500) 已远超 float32 上限，返回 inf
# 稳定版 exp(500-502) = exp(-2) ≈ 0.135，安全
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤 4：实现稳定的交叉熵

```python
# 朴素版交叉熵：组合朴素 softmax 和对数
# 前一步 softmax 如果溢出，整个损失计算都会失败
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

# 稳定版交叉熵：直接计算 log-probability，避免中间 softmax 数值问题
# 利用 log-softmax(x_i) = x_i - max(x) - log(sum(exp(x - max(x))))
# 这种形式等价于 log-sum-exp 技巧，全程数值安全
def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
# 使用中心有限差分法计算数值梯度
# 原理：f'(x) ≈ (f(x+h) - f(x-h)) / (2h)，精度 O(h²)
# 相比前向差分 (f(x+h)-f(x))/h 的 O(h) 精度，中心差分误差小得多
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

# 对比解析梯度与数值梯度，验证反向传播实现是否正确
# 按相对误差评估：rel_error < 1e-5 通常认为梯度正确
def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

# 示例：f(x, y) = x² + 3xy + y³，解析推导后验证
def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 实际应用（Use It）

### 混合精度模拟

```python
import struct

# 模拟 float32 → float16 精度损失效果
# 使用 struct.pack 做实际格式转换，观察精度损失量级
def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

# 模拟 bfloat16 的截断效果：将 float32 尾数低 16 位清零
# bfloat16 只是 float32 的简单截断，保留全部 8 位指数，范围与 float32 相同
def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
# 按范数裁剪梯度：将梯度向量缩放到不超过 max_norm
# 相比按值裁剪，按范数裁剪保留梯度方向，不影响更新方向
# 这是防止梯度爆炸的标准方法，对 Transformer 和 RNN 尤其关键
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
# 验证方向是否保留：裁剪前后每个分量与第一个分量的比值应相同
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
# 每次前向传播后检查张量中是否出现 nan/inf
# 因为 nan/inf 会在后续计算中病毒式传播，及早发现能精确定位问题来源
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况演示见 `code/numerical.py`。

## 交付物（Ship It）

本课产出：
- `code/numerical.py`：包含稳定的 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断 NaN/Inf 和训练中数值问题的提示

这些稳定实现在阶段 3 构建训练循环和阶段 4 实现注意力机制时会再次出现。

## 练习题（Exercises）

1. **灾难性抵消**：用朴素公式 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 在 float32 下计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法计算。比较两者与真实方差（0.6667）的误差。

2. **精度搜寻**：找到 Python 中使 `1.0 + x == 1.0` 成立的最小正 float32 值 $x$。这就是机器 epsilon（machine epsilon）。验证它等于 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况**：测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值非常负（-1000）。验证在朴素版失败的情况下它给出正确结果。

4. **对神经网络层做梯度检查**：实现一个线性层 $y = Wx + b$ 及其解析反向传播。用 `numerical_gradient` 验证一个 $3 \times 2$ 权重矩阵的正确性。

5. **损失缩放实验**：模拟 float16 训练：生成 [1e-9, 1e-3] 范围内的随机梯度，转换为 float16，测量有多少比例变为零。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回来，再次测量零值比例。

## 关键术语（Key Terms）

| 术语（English） | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | "浮点数标准" | 定义二进制浮点数格式、舍入规则和特殊值（inf、nan）的国际标准。每颗现代 CPU 和 GPU 都实现它。 |
| Machine epsilon | "精度极限" | 在给定浮点格式中使 $1.0 + e \neq 1.0$ 的最小值 $e$。对于 float32，约为 $1.19 \times 10^{-7}$。 |
| Catastrophic cancellation | "减法导致的精度损失" | 相减两个几乎相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| Overflow | "数字太大" | 结果超过最大可表示值，变为 inf。$\exp(89)$ 在 float32 中溢出。 |
| Underflow | "数字太小" | 结果比最小可表示正数更接近零，变为 0.0。$\exp(-104)$ 在 float32 中下溢。 |
| Log-sum-exp trick | "先减去最大值" | 通过提取 $\exp(\max(x))$ 因子来计算 $\log(\sum \exp(x))$，防止上溢和下溢。用于 softmax、交叉熵和对数概率计算。 |
| Stable softmax | "不会爆炸的 softmax" | 在取指数之前减去 $\max(\text{logits})$。数值结果相同，不可能溢出。 |
| Gradient checking | "验证反向传播" | 将反向传播的解析梯度与有限差分的数值梯度对比，以发现实现 bug。 |
| Mixed precision | "float16 前向，float32 反向" | 在速度关键的操作中使用低精度浮点数，在数值敏感的操作中使用高精度浮点数。典型加速比为 2-3 倍。 |
| Loss scaling | "防止梯度下溢" | 在反向传播前将损失乘以一个大常数，使梯度保持在 float16 可表示范围内，然后在权重更新前除以相同常数。 |
| bfloat16 | "脑浮点格式" | Google 的 16 位格式，8 位指数（与 float32 相同范围），7 位尾数（比 float16 精度更低）。训练场景中更受青睐。 |
| Gradient clipping | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "不是数字" | 来自未定义运算（0/0、$\infty - \infty$、$\sqrt{-1}$）的特殊浮点值。在所有后续算术中传播。 |
| Inf | "无穷大" | 来自上溢或除零的特殊浮点值。可以与 NaN 组合产生 NaN（$\infty - \infty$、$\infty \times 0$）。 |
| Numerical gradient | "暴力法导数" | 通过计算 $f(x+h)$ 和 $f(x-h)$ 并除以 $2h$ 来近似导数。慢但可靠，适合验证。 |

## 延伸阅读（Further Reading）

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic（Goldberg 1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)——权威参考，内容密集但完整
- [Mixed Precision Training（Micikevicius et al., 2018）](https://arxiv.org/abs/1710.03740)——NVIDIA 论文，引入用于 float16 训练的损失缩放
- [AMP：Automatic Mixed Precision（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html)——PyTorch 中混合精度的实践指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16)——为什么 Google 为 TPU 选择这种格式
- [Kahan 求和（Wikipedia）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)——减少浮点求和中舍入误差的算法
