# 概率与分布

> 概率是 AI 表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 1，第 01-04 课
**时间：** ~75 分钟

## 学习目标

- 从零实现伯努利(Bernoulli)、类别(Categorical)、泊松(Poisson)、均匀(Uniform)和正态(Normal)分布的 PMF 与 PDF
- 计算期望值(Expected Value)和方差(Variance)，并使用中心极限定理(CLT)解释为什么高斯分布(Gaussian)无处不在
- 构建带有数值稳定性技巧（减去最大 logit）的 Softmax 和 Log-Softmax 函数
- 从 logits 计算交叉熵损失(Cross-Entropy Loss)，并将其与负对数似然联系起来

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50,000 个候选中选择下一个词。一个扩散模型通过从学习到的分布中采样来生成图像。这些都是概率的实际应用。

模型做出的每一个预测都是一个概率分布。每一个损失函数都在衡量预测分布与真实分布之间的差距。每一次训练步骤都在调整参数，使一个分布更接近另一个分布。没有概率，你无法阅读任何一篇 ML 论文，无法调试任何一个模型，也无法理解为什么你的训练损失是 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率将事件映射到 0 到 1 之间的数字。

抛硬币：

$$
S = \{H, T\}, \quad P(H) = 0.5, \quad P(T) = 0.5
$$

单次掷骰子：

$$
S = \{1, 2, 3, 4, 5, 6\}, \quad P(\text{偶数}) = P(\{2, 4, 6\}) = \frac{3}{6} = 0.5
$$

三条公理定义了所有概率：
1. 对任意事件 A，$P(A) \geq 0$
2. $P(S) = 1$（某事件总会发生）
3. 当 A 和 B 互斥时，$P(A \text{ 或 } B) = P(A) + P(B)$

其他所有内容（贝叶斯定理、期望、分布）都源自这三条规则。

### 条件概率与独立性

$P(A|B)$ 是在 B 已发生的条件下 A 发生的概率。

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

示例：一副扑克牌

$$
P(\text{King} | \text{Face card}) = \frac{P(\text{King} \cap \text{Face card})}{P(\text{Face card})}
= \frac{4/52}{12/52}
= \frac{4}{12} = \frac{1}{3}
$$

两个事件独立时，已知一个事件对另一个事件不提供任何信息。

独立条件：$P(A|B) = P(A)$，等价于 $P(A \cap B) = P(A) \cdot P(B)$

抛硬币是独立的。不放回地抽牌则不是。

### 概率质量函数与概率密度函数

离散随机变量具有概率质量函数(PMF)。每个结果都有一个可以直接读出的具体概率。

PMF：$P(X = k)$

公平骰子：

$$
P(X = 1) = \frac{1}{6}, \quad P(X = 2) = \frac{1}{6}, \quad \ldots, \quad P(X = 6) = \frac{1}{6}
$$

所有概率之和为 1。

连续随机变量具有概率密度函数(PDF)。单点处的密度不是概率。概率来自对区间上的密度进行积分。

PDF：$f(x)$

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

$f(x)$ 可以大于 1（密度，不是概率），$\int_{-\infty}^{+\infty} f(x) \, dx = 1$

这种区别在 ML 中很重要。分类输出是 PMF（离散选择），VAE 的潜空间使用 PDF（连续）。

### 常见分布

**伯努利分布(Bernoulli)：** 一次试验，两个结果。用于建模二分类。

$$
P(X = 1) = p, \quad P(X = 0) = 1 - p
$$
$$
\text{均值} = p, \quad \text{方差} = p(1-p)
$$

**类别分布(Categorical)：** 一次试验，k 个结果。用于建模多分类（softmax 输出）。

$$
P(X = i) = p_i, \quad \sum p_i = 1
$$

例如：$P(\text{猫}) = 0.7, \; P(\text{狗}) = 0.2, \; P(\text{鸟}) = 0.1$

**均匀分布(Uniform)：** 所有结果等概率。用于随机初始化。

离散均匀分布：$P(X = k) = \frac{1}{n}, k \in \{1, \ldots, n\}$

连续均匀分布：$f(x) = \frac{1}{b-a}, x \in [a, b]$

**正态分布(Normal/Gaussian)：** 钟形曲线。由均值 $\mu$ 和方差 $\sigma^2$ 参数化。

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

标准正态分布：$\mu = 0, \sigma = 1$

- 68% 的数据落在 1 个标准差内
- 95% 的数据落在 2 个标准差内
- 99.7% 的数据落在 3 个标准差内

**泊松分布(Poisson)：** 固定区间内稀有事件的计数。用于建模事件发生率。

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$
$$
\text{均值} = \lambda, \quad \text{方差} = \lambda
$$

### 期望值与方差

期望值是结果的加权平均值。

离散：$E[X] = \sum x_i \cdot P(X = x_i)$

连续：$E[X] = \int x \cdot f(x) \, dx$

方差衡量围绕均值的分散程度。

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$
$$
\text{标准差} = \sqrt{\text{Var}(X)}
$$

在 ML 中，期望值以损失函数的形式出现（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练过程噪声大。

### 联合分布与边缘分布

联合分布 $P(X, Y)$ 描述两个随机变量一起出现的情况。

联合 PMF 示例（X = 天气，Y = 是否带伞）：

| | Y=0（不带伞） | Y=1（带伞） | 边缘分布 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘分布 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一个变量求和得到：

$$
P(X = x) = \sum_{y} P(X = x, Y = y)
$$

上表中行和列的合计值就是边缘分布。

### 为什么正态分布无处不在

中心极限定理(CLT)：大量独立随机变量的和（或平均值）收敛到正态分布，无论原始分布是什么。

掷 1 颗骰子：均匀分布（平坦）

2 颗骰子的平均值：三角形（中间凸起）

30 颗骰子的平均值：近乎完美的钟形曲线

这对**任何起始分布**都成立。

这就是为什么：
- 测量误差近似正态（大量微小的独立误差源叠加）
- 神经网络中的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态（大量样本梯度的和）
- 正态分布是给定均值和方差下的最大熵分布

### 对数概率

原始概率会导致数值问题。将许多小概率相乘会迅速下溢(underflow)为零。

句子概率 = $P(\text{word}_1) \times P(\text{word}_2) \times \ldots \times P(\text{word}_n)$

$$
= 0.01 \times 0.003 \times 0.02 \times \ldots \rightarrow 0.0 \;(\text{约 30 项后下溢})
$$

对数概率解决了这个问题。乘法变成了加法。

$$
\log P(\text{句子}) = \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n)
$$
$$
= -4.6 + (-5.8) + (-3.9) + \ldots \rightarrow \text{有限值（不下溢）}
$$

规则：
- $\log(a \cdot b) = \log(a) + \log(b)$
- 对数概率始终 $\leq 0$（因为 $0 < P \leq 1$）
- 越负表示越不可能
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将其转换为合法的概率分布。

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

性质：
- 所有输出都在 $(0, 1)$ 之间
- 所有输出之和为 1
- 保持输入的相对顺序
- $\exp()$ 放大 logits 之间的差异

Softmax 技巧：在指数运算前减去最大 logit，防止溢出。

```
原始 logits: z = [100, 101, 102]
exp(102) → 溢出

减去最大值: z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1（安全）

计算结果相同，无溢出。
```

Log-Softmax 将 softmax 和 log 合并为一步，具有更好的数值稳定性。PyTorch 内部使用它来计算交叉熵损失。

### 采样

采样指从分布中随机抽取数值。在 ML 中：
- Dropout 随机采样要置零的神经元
- 数据增强随机采样变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布中采样需要逆变换采样、拒绝采样或重参数化技巧（用于 VAE）等技术。

## 动手实现

### Step 1：概率基础

```python
import math
import random

# 计算阶乘，用于后续组合数和泊松分布
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# 计算组合数 C(n, k) = n! / (k! * (n-k)!)，用于概率计算
def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

# 条件概率公式 P(A|B) = P(A and B) / P(B)
def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

# 验证从一副扑克牌中抽到 King 的条件概率：已知抽到人头牌
p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2：从零实现 PMF 和 PDF

```python
# 伯努利分布 PMF：单次试验中结果为 1 的概率为 p，结果为 0 的概率为 1-p
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

# 类别分布 PMF：按概率向量 probs 返回类别 k 的概率
def categorical_pmf(k, probs):
    return probs[k]

# 泊松分布 PMF：lambda 是单位区间内的平均事件数
# 注意：lambda 较大时 lambda**k 可能溢出，实际中改用 scipy 的优化版本
def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

# 连续均匀分布 PDF：在 [a, b] 区间内密度恒定，外部为 0
def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

# 正态分布 PDF：mu 控制中心位置，sigma 控制分布宽度
# 系数 1/(sigma * sqrt(2pi)) 确保积分结果为 1
# 指数部分衡量 x 距离均值的偏差平方，sigma 越大曲线越扁平
def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3：期望值和方差

```python
# 计算离散随机变量的期望值：每个结果值乘以其概率后求和
# 期望值是概率加权平均，也是损失函数的基本形式
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

# 计算方差：衡量分布围绕均值的分散程度
# 使用 Var(X) = E[(X - mu)^2] = sum(p * (v - mu)^2)
# 方差越大，结果的不确定性越高；在 ML 中对应梯度噪声和模型稳定性
def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

# 验证公平骰子的期望值应为 3.5，方差约为 2.917
die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4：从分布中采样

```python
# 伯努利采样：产生一个 0/1 随机值，1 的概率为 p
# 这是 Dropout 的基础——每个神经元以概率 p 保留
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

# 类别采样：使用累积分布(CDF)逆变换方法
# 先构建累积概率向量，再生成 [0,1) 均匀随机数做查找
# 当类别数很大时（如语言模型词汇表），可用二分搜索加速
def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

# Box-Muller 变换：通过两个均匀随机数生成正态分布样本
# 比逆变换更高效，因为正态分布的 CDF 没有闭式逆函数
# 生成标准正态样本 z，再通过 mu + sigma * z 得到目标分布
def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5：Softmax 和对数概率

```python
# Softmax：将任意实数 logits 转换为合法的概率分布
# 数值稳定技巧：先减去最大值，防止 exp(大数) 溢出
# 这不会改变结果，因为 softmax 对输入平移不变
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

# Log-Softmax：直接计算 log(softmax(z))，避免先算 softmax 再取 log
# 在 log-sum-exp 中保持 max_logit 项确保数值稳定
# PyTorch 的 cross-entropy 损失内部使用这一实现
def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

# 交叉熵损失：正确类别 log-probability 的负值
# 等价于最大化正确类别的对数似然（最大似然估计）
def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6：中心极限定理演示

```python
# 中心极限定理演示：从任意分布 dist_fn 中抽取 n_samples 个样本
# 重复 n_averages 次，计算每次的样本均值
# 无论 dist_fn 是什么分布，这些均值的分布将趋近正态
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7：可视化

```python
import matplotlib.pyplot as plt

# 绘制正态分布 PDF 曲线：在均值附近生成密集采样点
# 用于直观展示正态分布的形状及其与 sigma 的关系
xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整的实现代码和所有可视化见 `code/probability.py`。

## 实用工具

使用 NumPy 和 SciPy，上述所有功能都可以一行代码实现：

```python
import numpy as np
from scipy import stats

# 使用 SciPy 的正态分布：一行代码完成采样、PDF 计算和 CDF 查询
# loc 指定均值，scale 指定标准差
normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

# 使用 SciPy 的 softmax 和 log_softmax（数值稳定版本）
logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零实现了这些函数。现在你知道了库函数背后在做什么。

## 练习

1. 为指数分布实现逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 比较来验证。

2. 为两个有偏骰子构建联合分布表。计算边缘分布并检查两个骰子是否独立。

3. 计算一个 5 类分类器的交叉熵损失，它输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别索引为 3。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接收对数概率列表，返回最可能的序列、总对数概率和等价的原始概率。用一个 50 词的句子测试，其中每个词的概率为 0.01。

## 关键术语

| Term | 通俗说法 | 真正含义 |
|------|---------|---------|
| Sample space（样本空间） | "所有可能性" | 实验所有可能结果构成的集合 S |
| PMF（概率质量函数） | "概率函数" | 给出每个离散结果确切概率的函数，所有概率之和为 1 |
| PDF（概率密度函数） | "概率曲线" | 连续变量的密度函数。在区间上积分得到概率 |
| Conditional probability（条件概率） | "给定某条件下的概率" | $P(A\|B) = P(A \cap B) / P(B)$。贝叶斯思维和贝叶斯定理的基础 |
| Independence（独立性） | "它们互不影响" | $P(A \cap B) = P(A) \cdot P(B)$。知道一个事件对另一个事件不提供任何信息 |
| Expected value（期望值） | "平均值" | 所有结果的概率加权和。损失函数就是一个期望值 |
| Variance（方差） | "分散程度" | 与均值的期望平方偏差。方差大意味着估计噪声大、不稳定 |
| Normal distribution（正态分布） | "钟形曲线" | $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$。由于 CLT 无处不在 |
| Central Limit Theorem（中心极限定理） | "均值变正态" | 大量独立样本的均值收敛到正态分布，无论原始分布是什么 |
| Joint distribution（联合分布） | "两个变量一起" | $P(X, Y)$ 描述 X 和 Y 所有结果组合的概率 |
| Marginal distribution（边缘分布） | "对另一个变量求和" | $P(X) = \sum_y P(X, Y)$。从联合分布恢复单个变量的分布 |
| Log probability（对数概率） | "概率的对数" | $\log P(x)$。将乘积变为加法，防止长序列的数值下溢 |
| Softmax | "将分数转为概率" | $\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum \exp(z_j)}$。将实数 logits 映射到合法的概率分布 |
| Cross-entropy（交叉熵） | "损失函数" | $-\sum p_{\text{true}} \cdot \log(p_{\text{pred}})$。衡量两个分布的差异，越小越好 |
| Logits | "原始模型输出" | softmax 之前的未归一化分数。名称源自 logistic 函数 |
| Sampling（采样） | "随机抽取" | 根据概率分布生成数值。模型生成输出的基本方式 |

## 延伸阅读

- [3Blue1Brown：中心极限定理究竟是什么？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 关于均值为何趋向正态的可视化证明
- [Stanford CS229 概率论复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖此处及更多内容的精炼参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性很重要以及如何实现
