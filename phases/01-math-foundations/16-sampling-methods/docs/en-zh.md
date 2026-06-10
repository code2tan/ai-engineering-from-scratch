# 采样方法（Sampling Methods）

> 采样是 AI 探索可能性空间的方式。

**类型：** 动手构建
**语言：** Python
**前置知识：** 阶段 1，第 06-07 课（概率、贝叶斯定理）
**时间：** ~120 分钟

## 学习目标（Learning Objectives）

- 仅使用均匀随机数从头实现逆 CDF、拒绝采样和重要性采样
- 构建用于语言模型 token 生成的温度采样、top-k 采样和 top-p（核）采样
- 解释重参数化技巧（reparameterization trick）及其如何使 VAE 中的采样可反向传播
- 运行 Metropolis-Hastings MCMC 从未归一化的目标分布中采样

## 问题背景（The Problem）

一个语言模型处理完你的提示后，产生了一个包含 50,000 个 logits 的向量，每个对应词汇表中的一个 token。现在它必须选一个。怎么选？

如果它总是选择概率最高的 token，每一次响应都相同——确定性的、无聊的。如果它均匀随机选择，输出就是胡言乱语。答案存在于这两个极端之间，而那个"之间"就由采样控制。

采样不限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE 通过从学习到的分布中采样并通过随机性反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡洛方法估计没有闭式解的积分。MCMC 算法探索无法枚举的高维后验分布。

每一个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课从均匀随机数出发，逐步构建到驱动现代 LLM 和生成模型的所有主要采样方法。

## 核心概念（The Concept）

### 为什么采样很重要

采样在 AI 和机器学习中扮演四种基本角色：

**生成。** 语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接影响创造力、连贯性和多样性。温度、top-k 和核采样是工程师每天调节的旋钮。

**训练。** 随机梯度下降采样小批量。Dropout 采样要停用的神经元。数据增强采样随机变换。重要性采样重新加权样本以减少强化学习（PPO、TRPO）中的梯度方差。

**估计。** ML 中的许多量没有闭式解。数据分布上的期望损失、基于能量的模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本求平均来近似所有这些量。

**探索。** MCMC 算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。汤普森采样在多臂老虎机中平衡探索与利用。

核心挑战：你只能直接从简单的分布（均匀、正态）中采样。对于其他分布，你需要一种方法将简单样本转换为目标分布的样本。

### 均匀随机采样

每种采样方法都从这里开始。均匀随机数生成器产生 $[0, 1)$ 范围内的值，其中每个等长子区间具有相等的概率。

$$
U \sim \text{Uniform}(0, 1)
$$

$$
P(a \le U \le b) = b - a \quad \text{for } 0 \le a \le b \le 1
$$

- $\mathbb{E}[U] = 0.5$
- $\text{Var}(U) = 1/12$

要从 $n$ 个项目的离散集合中均匀采样，生成 $U$ 并返回 $\lfloor n \cdot U \rfloor$。要从连续区间 $[a, b]$ 采样，计算 $a + (b - a) \cdot U$。

关键洞察：一个均匀随机数恰好包含从一个分布中产生一个样本所需的随机性。秘诀在于找到正确的变换。

### 逆 CDF 法（逆变换采样）

累积分布函数（CDF）将值映射为概率：

$$
F(x) = P(X \le x)
$$

- $F$ 是非递减的
- $F(-\infty) = 0$
- $F(+\infty) = 1$
- $F$ 将实数线映射到 $[0, 1]$

逆 CDF 将概率映射回值。如果 $U \sim \text{Uniform}(0, 1)$，则 $X = F^{-1}(U)$ 服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 $F^{-1}$ 的闭式表达式时，这种方法完美工作。对于正态分布，没有闭式的逆 CDF，因此我们使用其他方法（Box-Muller 或数值近似）。

**离散版本：** 对于离散分布，将 CDF 构建为累积和，生成 $U$，然后找到累积和超过 $U$ 的第一个索引。这就是第 06 课中 `sample_categorical` 的工作方式。

### 拒绝采样（Rejection Sampling）

当你无法求逆 CDF 但可以计算目标 PDF（最多差一个常数）时，拒绝采样有效。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

约束 $M$ 越紧，接受率越高。在低维度（1-3）中，拒绝采样效果很好。在高维度中，接受率呈指数下降，因为大多数提议体积被拒绝。这就是拒绝采样的维度灾难。

**示例：从截断正态分布中采样。** 在截断范围内使用均匀提议。包络线 $M$ 是该范围内正态 PDF 的最大值。

**示例：从半圆中采样。** 在边界矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡洛计算 $\pi$ 的方式：接受率等于面积比 $\pi / 4$。

### 重要性采样（Importance Sampling）

有时你不需要来自目标分布 $p(x)$ 的样本。你需要估计在 $p(x)$ 下的期望，而你拥有来自另一个分布 $q(x)$ 的样本。

$$
\text{目标：估计 } \mathbb{E}_p[f(x)] = \int f(x) p(x) \, dx
$$

改写：

$$
\mathbb{E}_p[f(x)] = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_q[f(x) w(x)]
$$

其中 $w(x) = p(x) / q(x)$ 是重要性权重。

$$
\text{估计量：} \quad \mathbb{E}_p[f(x)] \approx \frac{1}{N} \sum f(x_i) w(x_i) \quad \text{其中 } x_i \sim q(x)
$$

这在强化学习中至关重要。在 PPO 中，你在旧策略 $\pi_{\text{old}}$ 下收集轨迹，但想优化新策略 $\pi_{\text{new}}$。重要性权重是 $\pi_{\text{new}}(a|s) / \pi_{\text{old}}(a|s)$。PPO 对这些权重进行裁剪，防止新策略偏离旧策略太远。

重要性采样估计量的方差取决于 $q$ 与 $p$ 的相似程度。如果 $q$ 与 $p$ 差异很大，少数样本会获得巨大权重并主导估计。自归一化重要性采样通过除以权重之和来缓解这个问题：

$$
\mathbb{E}_p[f(x)] \approx \frac{\sum w_i f(x_i)}{\sum w_i}
$$

### 蒙特卡洛估计（Monte Carlo Estimation）

蒙特卡洛估计通过对随机样本求平均来近似积分。大数定律保证收敛。

$$
\text{目标：估计 } I = \int_D g(x) \, dx
$$

方法：
1. 从 $D$ 中均匀采样 $x_1, \dots, x_N$
2. $I \approx (\text{Volume of } D / N) \cdot \sum g(x_i)$

误差：$O(1 / \sqrt{N})$，与维度无关。

这就是蒙特卡洛方法在高维度（其中基于网格的积分不可能）中占主导地位的原因。

**估计 $\pi$：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

$$
\mathbb{E}[f(X)] \approx \frac{1}{N} \sum f(x_i) \quad \text{其中 } x_i \sim p(x)
$$

样本均值收敛到真实期望。估计量的方差 $= \text{Var}(f(X)) / N$。

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC 构建一个平稳分布为目标分布 $p(x)$ 的马尔可夫链。经过足够多步后，链中的样本（近似）是来自 $p(x)$ 的样本。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称提议（$q(x'|x) = q(x|x')$），该比率简化为 $p(x')/p(x)$。这就是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保了细致平衡（detailed balance）：处于 $x$ 并移动到 $x'$ 的概率等于处于 $x'$ 并移动到 $x$ 的概率。细致平衡意味着 $p(x)$ 是链的平稳分布。

**实际考虑：**
- Burn-in（预热期）：丢弃链达到平衡之前的早期样本
- Thinning（稀疏化）：每 $k$ 个样本保留一个以减少自相关
- 提议步长：太小则链移动缓慢（高接受率、慢探索）；太大则大多数提议被拒绝（低接受率、卡住不动）
- 高维高斯提议的最优接受率约为 0.234

### Gibbs 采样

Gibbs 采样是多变量分布 MCMC 的一个特例。它不是一次性在所有维度上提议移动，而是每次从其条件分布中更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求你能从每个条件分布 $p(x_i | x_{-i})$ 中采样。对于许多模型这很直接：
- 贝叶斯网络：条件分布由图结构决定
- 高斯混合模型：条件分布是高斯分布
- Ising 模型：每个自旋的条件分布仅取决于其邻居

接受率始终为 1（每个提议都被接受），因为从精确的条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs 采样混合缓慢，因为一次只更新一个变量无法在分布中做大对角线移动。

### 温度采样（用于 LLM）

语言模型为词汇表中的每个 token 输出 logits $z_1, \dots, z_V$。Softmax 将其转换为概率。温度在 softmax 之前重新缩放 logits：

$$
p_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}
$$

- $T = 1.0$：标准 softmax（原始分布）
- $T \to 0$：argmax（确定性的，总是选最高 logit）
- $T \to \infty$：均匀分布（所有 token 等可能）
- $T < 1.0$：锐化分布（更自信，更少多样性）
- $T > 1.0$：展平分布（更不自信，更多多样性）

**为什么有效。** 用 $T < 1$ 除以 logits 会放大它们之间的差异。如果 $z_1 = 2$ 且 $z_2 = 1$，除以 $T = 0.5$ 得到 $z_1/T = 4$ 和 $z_2/T = 2$，差距变得更大。经过 softmax 后，最高 logit 的 token 获得更大的份额。

**实践中：**
- $T = 0.0$：贪心解码，最适合事实型问答
- $T = 0.3-0.7$：略微有创意，适合代码生成
- $T = 0.7-1.0$：平衡，适合一般对话
- $T = 1.0-1.5$：创意写作、头脑风暴
- $T > 1.5$：越来越随机，很少有用

温度不会改变哪些 token 是可能的。它改变分配给每个 token 的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的 $k$ 个 token，然后重新归一化并从该受限集中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型选择词汇分布长尾中极不可能的 token（拼写错误、无意义内容）。问题在于：$k$ 是固定的，与上下文无关。当模型很自信（一个 token 有 95% 概率）时，$k = 40$ 仍然允许 39 个备选。当模型不确定（概率分散在 1000 个 token 上）时，$k = 40$ 切断了合理的选项。

### Top-p（核）采样

Top-p 采样动态调整候选集大小。它不是保留固定数量的 token，而是保留累积概率超过 $p$ 的最小 token 集。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型自信时，核采样保留很少的 token（可能 2-3 个）。当模型不自信时，它保留很多（可能 200 个）。这种自适应行为是核采样通常比 top-k 产生更好文本的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：良好的通用设置
- 温度 0.0（贪心）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan et al.（2018）原始论文设置

Top-k 和 top-p 可以结合使用。先应用 top-k，然后在剩余集合上应用 top-p。

### 重参数化技巧（用于 VAE）

变分自编码器（VAE）通过将输入编码为潜空间中的分布、从该分布采样、然后将样本解码回来进行学习。问题在于：你无法通过采样操作进行反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧将随机性与参数分离：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这是因为 $\mathcal{N}(\mu, \sigma^2)$ 与 $\mu + \sigma \cdot \mathcal{N}(0, 1)$ 具有相同的分布。关键洞察：将随机性移到无参数的源（$\epsilon$）中，然后样本就是参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 $\mu$ 和 $\log(\sigma^2)$
2. 采样 $\epsilon \sim \mathcal{N}(0, 1)$
3. 计算 $z = \mu + \sigma \cdot \epsilon$
4. 解码 $z$ 以重建输入
5. 通过步骤 4、3、2、1 反向传播（因为步骤 3 是可微的）

没有重参数化技巧，VAE 就无法用标准的反向传播训练。这个单一洞察使 VAE 变得实用。

### Gumbel-Softmax（可微的分类采样）

重参数化技巧适用于连续分布（高斯分布）。对于离散的分类分布，我们需要不同的方法。Gumbel-Softmax 提供了分类采样的可微近似。

**Gumbel-Max 技巧（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

$$
y_i = \frac{\exp((\log(p_i) + g_i) / \tau)}{\sum \exp((\log(p_j) + g_j) / \tau)}
$$

$\tau$（温度）控制近似的程度：
- $\tau \to 0$：接近独热向量（硬分类）
- $\tau \to \infty$：接近均匀分布 $(1/k, \dots, 1/k)$
- $\tau = 1.0$：软近似

Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（软独热）而不是硬独热。梯度通过 softmax 流动。在训练的前向传播中，你可以使用"直通"估计器：前向用硬 argmax，反向用软 Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散潜变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层采样（Stratified Sampling）

标准蒙特卡洛采样可能会在样本空间中偶然留下空白。分层采样通过将空间划分为层（strata）并从每层中采样来强制均匀覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样总是具有比标准蒙特卡洛更低或相等的方差：

$$
\text{Var(stratified)} \le \text{Var(standard Monte Carlo)}
$$

当 $f(x)$ 平滑变化时改进最大。对于分段常数函数，分层采样是精确的。

**应用：**
- 数值积分（拟蒙特卡洛）
- 训练数据划分（确保每折中的类别平衡）
- 带有分层的重要性采样（结合两种技术）
- NeRF（神经辐射场）沿相机射线使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程将一个图像在 $T$ 步中逐渐添加高斯噪声，直到变为纯噪声。反向过程学习去噪，逐步恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本章方法的联系：
- 每个去噪步骤都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度 $\{\alpha_t\}$ 控制一种温度退火形式
- 训练使用蒙特卡洛估计来近似 ELBO（evidence lower bound）
- 扩散模型中的祖先采样是一个马尔可夫链（每一步仅取决于当前状态）

整个图像生成过程是迭代采样：从噪声开始，在每一步，基于已学习的去噪模型采样一个稍微少一点噪声的版本。

## 动手实现（Build It）

### 步骤 1：均匀分布和逆 CDF 采样

```python
import math
import random

# 从 [a, b] 区间均匀采样
def sample_uniform(a, b):
    return a + (b - a) * random.random()

# 使用逆 CDF 法从指数分布采样
# 指数分布 CDF：F(x) = 1 - exp(-lambda * x)
# 求逆得：x = -ln(1 - u) / lambda，等价于 -ln(u) / lambda
def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数样本并验证均值是否为 $1/\lambda$。

### 步骤 2：拒绝采样

```python
# 拒绝采样：当无法直接求逆 CDF 时，使用一个简单的提议分布
# target_pdf：目标分布的概率密度函数（可未归一化）
# proposal_sample：从提议分布中采样的函数
# proposal_pdf：提议分布的概率密度函数
# M：包络常数，确保 p(x) <= M * q(x) 对所有 x 成立
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        # 接受概率为 p(x) / (M * q(x))，越接近则接受率越高
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中抽样。通过绘制样本直方图验证形状。

### 步骤 3：重要性采样

```python
# 使用重要性采样估计 E_p[f(x)]
# 从提议分布 q 中采样，但用权重 w = p(x)/q(x) 校正分布偏移
# 用于 RL（PPO、TRPO）和贝叶斯推断中无法直接从中采样时的期望估计
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议分布估计正态分布下的 $\mathbb{E}[X^2]$。与已知答案 $\mu^2 + \sigma^2$ 对比。

### 步骤 4：蒙特卡洛估计 $\pi$

```python
# 通过蒙特卡洛方法估计圆周率 pi
# 原理：单位圆面积与正方形面积之比为 pi/4
# 随机点落在圆内的概率 = (pi * 1^2) / (2^2) = pi/4
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤 5：Metropolis-Hastings MCMC

```python
# Metropolis-Hastings 算法：构建马尔可夫链以从目标分布中采样
# 即使目标分布只有未归一化的概率密度也能工作
# target_log_pdf：目标分布的对数概率密度（可差一个常数）
# proposal_sample：提议分布采样函数（给定当前状态 x 提出新状态）
# proposal_log_pdf：提议分布的对数概率密度
# x0：初始状态，burn_in：预热步数（链达到平稳分布前丢弃的样本）
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        # 计算接受比率的对数形式以避免数值溢出
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯分布的混合）中采样。可视化链的轨迹。

### 步骤 6：Gibbs 采样

```python
# Gibbs 采样：一次更新一个变量的条件分布
# 适用于高维分布中条件分布易于采样的情况
# 每次更新都从条件分布精确采样，因此接受率始终为 1
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤 7：温度采样

```python
# 数值稳定的 softmax：先减去最大值防止 exp 溢出
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

# 温度采样：温度 T 控制分布的"锐利"程度
# T < 1：放大 logits 差异，输出更确定
# T > 1：缩小 logits 差异，输出更多样
def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组 token logits 的输出分布。

### 步骤 8：Top-k 和 top-p 采样

```python
# Top-k 采样：仅保留概率最高的 k 个 token，然后重新归一化后采样
# 防止模型从长尾中选出极不可能的 token
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

# Top-p（核）采样：动态保留累积概率超过 p 的最小 token 集
# 模型自信时保留极少 token，模型不确定时保留大量 token
# 自适应行为使其通常优于固定 k 的 top-k
def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤 9：重参数化技巧

```python
# 重参数化采样：将随机性从参数中分离
# z = mu + sigma * epsilon，其中 epsilon ~ N(0, 1)
# 使采样过程可微，梯度可通过 mu 和 sigma 反向传播
# VAE 训练的核心技巧
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

# 重参数化的梯度：dz/dmu = 1，dz/dsigma = epsilon
# 由于 epsilon 是无参数的随机数，梯度可以自由流通
def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度可以通过重参数化样本流动，但不能通过直接采样流动。

### 步骤 10：Gumbel-Softmax

```python
# Gumbel 噪声采样：g = -log(-log(u))，用于 Gumbel-Max 技巧
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

# Gumbel-Softmax：分类采样的可微近似
# temperature 控制近似程度：->0 趋近独热，->inf 趋近均匀
# 用于离散潜变量 VAE、神经架构搜索、硬注意力机制
def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出趋近于独热向量。

完整实现及所有可视化内容见 `code/sampling.py`。

## 实际应用（Use It）

使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

# 使用 NumPy 内置的采样器高效生成指数分布样本
rng = np.random.default_rng(42)

# 从指数分布生成 10000 个样本，验证样本均值接近理论值
exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

# SciPy stats 模块提供完整的分布函数接口
from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

# 温度采样的高效 NumPy 实现
logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
# 数值稳定的 softmax：减去最大值
exp_scaled = np.exp(scaled - scaled.max())
probs = exp_scaled / exp_scaled.sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 使用专用库：
- PyMC：完整的贝叶斯建模，包含 NUTS（自适应 HMC）
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

你从头实现了这些方法。现在你知道库调用在做什么了。

## 练习题（Exercises）

1. 为柯西分布实现逆 CDF 采样。CDF 为 $F(x) = 0.5 + \arctan(x)/\pi$。生成 10,000 个样本并将直方图与真实 PDF 对比。注意重尾特征（远离中心的极值）。

2. 使用拒绝采样从 Beta(2, 5) 分布生成样本，用 Uniform(0, 1) 作为提议分布。绘制接受的样本与真实 Beta PDF 的对比图。理论接受率是多少？

3. 使用 1,000、10,000 和 100,000 个样本的蒙特卡洛方法估计 $\int_0^\pi \sin(x) \, dx$。比较每级的误差。验证误差按 $O(1/\sqrt{N})$ 规模缩放。

4. 实现 Metropolis-Hastings 从二维分布 $p(x, y) \propto \exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8y)/2)$ 中采样。绘制样本和链轨迹。尝试不同的提议标准差。

5. 构建完整的文本生成演示：给定一个包含 10 个词的词汇表及其 logits，用以下方法生成 20 个 token 的序列：(a) 贪心，(b) 温度=0.7，(c) top-k=3，(d) top-p=0.9。比较 5 次运行中的输出多样性。

## 关键术语（Key Terms）

| 术语（English） | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Sampling | "抽取随机值" | 根据概率分布生成值。所有生成式 AI 背后的机制 |
| Uniform distribution | "所有值等可能" | $[a, b]$ 中每个值都有相同的概率密度 $1/(b-a)$。所有采样方法的起点 |
| Inverse CDF | "概率变换" | $F^{-1}(U)$ 将均匀样本转换为任意已知 CDF 的分布的样本。精确且高效 |
| Rejection sampling | "提议后接受/拒绝" | 从简单提议分布生成，按目标/提议比例接受。精确但浪费样本 |
| Importance sampling | "重新加权样本" | 用 $q(x)$ 的样本通过权重 $p(x)/q(x)$ 估计 $p(x)$ 下的期望。RL 中 PPO 的核心 |
| Monte Carlo | "随机样本求平均" | 将积分近似为样本均值。误差 $O(1/\sqrt{N})$，与维度无关 |
| MCMC | "收敛的随机游走" | 构建平稳分布为目标分布的马尔可夫链。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "接受上坡，有时下坡" | 提议移动，基于密度比接受。细致平衡确保收敛到目标分布 |
| Gibbs sampling | "一次一个变量" | 从条件分布中更新每个变量，固定其他变量。接受率 100% |
| Temperature | "置信度旋钮" | softmax 前用 T 除 logits。T<1 锐化（更自信），T>1 展平（更多样化） |
| Top-k sampling | "保留 k 个最好的" | 将除最高概率的 k 个 token 外的全部置零，重新归一化后采样。候选集大小固定 |
| Nucleus sampling (top-p) | "保留概率大的" | 保留累积概率超过 p 的最小 token 集。自适应候选集大小 |
| Reparameterization trick | "将随机性移到外部" | 将 $z$ 写为 $\mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,1)$。使采样可微。VAE 训练的核心 |
| Gumbel-Softmax | "软分类采样" | 使用 Gumbel 噪声 + 带温度的 softmax 的分类采样可微近似 |
| Stratified sampling | "强制覆盖" | 将样本空间划分为层，从每层中采样。方差总是不高于朴素蒙特卡洛 |
| Burn-in | "预热期" | 链达到平稳分布之前丢弃的初始 MCMC 样本 |
| Detailed balance | "可逆性条件" | $p(x) \cdot T(x \to y) = p(y) \cdot T(y \to x)$。$p$ 是马尔可夫链平稳分布的充分条件 |
| Diffusion sampling | "迭代去噪" | 从噪声开始应用学习到的去噪步骤生成数据。每一步都是一个条件采样操作 |

## 延伸阅读（Further Reading）

- [Holbrook（2023）：Metropolis-Hastings 算法](https://arxiv.org/abs/2304.07010)——MCMC 基础的详细教程
- [Jang, Gu, Poole（2017）：使用 Gumbel-Softmax 的分类重参数化](https://arxiv.org/abs/1611.01144)——原始 Gumbel-Softmax 论文
- [Holtzman et al.（2020）：神经文本退化的奇妙案例](https://arxiv.org/abs/1904.09751)——核（top-p）采样论文
- [Kingma & Welling（2014）：自编码变分贝叶斯](https://arxiv.org/abs/1312.6114)——引入重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel（2020）：去噪扩散概率模型](https://arxiv.org/abs/2006.11239)——DDPM 将采样与图像生成联系起来
