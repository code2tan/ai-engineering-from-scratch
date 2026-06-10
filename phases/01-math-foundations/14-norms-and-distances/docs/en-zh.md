# 范数与距离（Norms and Distances）

> 你的距离函数定义了"相似"的含义。选错了，下游的一切都会出问题。

**类型：** 动手构建
**语言：** Python
**前置知识：** 阶段 1，第 01 课（线性代数直觉）、第 02 课（向量、矩阵与运算）
**时间：** ~90 分钟

## 学习目标（Learning Objectives）

- 从头实现 L1、L2、余弦（cosine）、马氏（Mahalanobis）、Jaccard 和编辑距离（edit distance）函数
- 针对给定的 ML 任务选择合适的距离度量，并解释为什么其他方案会失败
- 建立 L1/L2 范数与 LASSO/Ridge 正则化及其几何约束区域的联系
- 展示同一数据集在不同度量下如何产生不同的最近邻

## 问题背景（The Problem）

你有两个向量。可能是词嵌入（word embeddings），可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点在一个度量下是最近邻，在另一个度量下可能相隔很远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——都依赖这个选择。选错了，你的模型就在为错误的目标优化。

没有通用的最佳距离。L2 适用于空间数据。余弦相似度主导 NLP。Jaccard 处理集合。编辑距离处理字符串。马氏距离考虑相关性。Wasserstein 度量移动概率质量。每种距离都编码了关于"相似"含义的不同假设。

本课从头实现每一种主要的距离函数，展示何时使用哪种工具，并演示相同的数据如何因度量的不同而产生完全不同的最近邻。

## 核心概念（The Concept）

### 范数：衡量向量的"大小"

范数衡量向量的"大小"。两个向量之间的任何距离函数都可以写成它们之差的范数：$d(a, b) = ||a - b||$。所以理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数对所有分量的绝对值求和。

$$
||x||_1 = |x_1| + |x_2| + \cdots + |x_n|
$$

之所以称为曼哈顿距离，是因为它衡量你在只能沿着坐标轴移动的城市网格中要走多远。没有对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用 L1：
- 高维稀疏数据（文本特征、独热编码）
- 需要对离群值具有鲁棒性（单个巨大差异不会主导结果）
- 特征选择问题（L1 正则化促进稀疏性）

与 L1 正则化（Lasso）的关系：在损失函数中加入 $||w||_1$ 会惩罚权重绝对值的总和。这会将小权重推向精确为零，实现自动特征选择。L1 惩罚在权重空间中创建菱形约束区域，而菱形的角落在坐标轴上——恰好是某些权重为零的位置。

与损失函数的关系：平均绝对误差（MAE）是预测值与目标值之间 L1 距离的平均值。它线性地惩罚所有误差，相比 MSE 对离群值更鲁棒。

### L2 范数（欧几里得距离）

L2 范数是直线距离。分量平方和的平方根。

$$
||x||_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}
$$

这是你在几何课上学到的距离。n 维空间中的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用 L2：
- 低到中等维度的连续数据
- 各特征尺度相当时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与 L2 正则化（Ridge）的关系：在损失函数中加入 $||w||_2^2$ 会惩罚大的权重。与 L1 不同，它不会将权重推向零，而是成比例地将所有权重向零收缩。L2 惩罚创建圆形约束区域，因此在坐标轴上没有角。权重变得很小，但极少精确为零。

与损失函数的关系：均方误差（MSE）是 L2 距离平方的平均值。平方使得对大误差的惩罚重于小误差。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：通用家族

L1 和 L2 是 Lp 范数的特例：

$$
||x||_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^{1/p}
$$

不同的 $p$ 值产生不同形状的"单位球"（到原点距离为 1 的所有点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（切比雪夫距离）

当 $p$ 趋近于无穷时，Lp 范数收敛到最大绝对值分量。

$$
||x||_\infty = \max(|x_1|, |x_2|, \dots, |x_n|)
$$

两点之间的距离由它们差异最大的单个维度决定。所有其他维度被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-infinity：
- 当任何单个维度的最坏偏差至关重要时
- 棋盘游戏（国际象棋中的国王按 L-infinity 移动：向任何方向走一步都算 1）
- 制造公差（每个维度必须在规格内）

### 余弦相似度与余弦距离

余弦相似度衡量两个向量之间的角度，忽略它们的大小。

$$
\text{cos\_sim}(a, b) = \frac{a \cdot b}{||a||_2 \cdot ||b||_2}
$$

范围从 -1（方向相反）到 +1（方向相同）。垂直向量的余弦相似度为 0。

余弦距离将其转换为距离：$\text{cosine\_distance} = 1 - \text{cosine\_similarity}$。范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦相似度主导 NLP 和嵌入（embeddings）：在文本中，文档长度不应影响相似度。一篇关于猫的文档如果是另一篇的两倍长，仍应被认为是"相似的"。余弦相似度忽略大小（长度），只关心方向。两篇词分布相同但长度不同的文档指向上相同方向，余弦相似度为 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF 向量、词嵌入、句子嵌入）
- 任何幅度是噪声、方向是信号的应用场景
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是使用余弦或点积）

### 点积相似度与余弦相似度

两个向量的点积：

$$
a \cdot b = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n = ||a|| \cdot ||b|| \cdot \cos(\theta)
$$

余弦相似度是点积除以两者的模长。当两个向量都已单位归一化（模长 = 1）时，点积和余弦相似度是同一个东西。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

它们的区别在于：点积包含模长的信息。模长更大的向量得到更高的点积分值。这在某些检索系统中很重要——你可能希望"热门"项目排名更高。模长扮演了一种隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 需要纯方向相似度时用余弦相似度
- 模长携带有意义信息时用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）让你在两者之间选择
- 如果你的嵌入已经是 L2 归一化的，选哪个都一样

### 马氏距离（Mahalanobis Distance）

欧几里得距离对所有维度一视同仁。但如果你的特征相关或尺度不同，L2 会给出误导性结果。

马氏距离考虑了数据的协方差结构。

$$
d_M(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)}
$$

其中 $S$ 是数据的协方差矩阵。

直观理解：马氏距离先对数据做去相关和归一化（白化），然后在变换后的空间中计算 L2 距离。如果 $S$ 是单位矩阵（特征不相关且方差为 1），马氏距离退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 离群值检测（距离均值马氏距离大的点是离群点）
- 特征尺度和相关性各不相同的分类任务
- 有足够数据估计可靠的协方差矩阵时
- 制造质量控制（多变量过程监控）

### Jaccard 相似度（适用于集合）

Jaccard 相似度衡量两个集合之间的重叠。

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

范围从 0（无重叠）到 1（相同集合）。Jaccard 距离 $= 1 -$ Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签、类别或特征的集合
- 基于词存在性（而非词频）的文档相似度
- 近似重复检测（MinHash 近似 Jaccard）
- 比较二值特征向量（存在/不存在数据）
- 评估分割模型（Intersection over Union = Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离计算将一个字符串转换为另一个所需的最小单字符操作次数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

使用动态规划计算。填充一个矩阵，其中条目 $(i, j)$ 是字符串 A 的前 $i$ 个字符与字符串 B 的前 $j$ 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查和纠错
- DNA 序列比对（带权重操作）
- 模糊字符串匹配
- 混乱文本数据的去重

### KL 散度（不是距离，但被当作距离使用）

KL 散度衡量一个概率分布与另一个的差异。在第 09 课中有详细覆盖，但它属于这里的讨论，因为人们尽管它不是距离，仍将其当作"距离"使用。

$$
D_{KL}(P || Q) = \sum p(x) \log \frac{p(x)}{q(x)}
$$

关键性质：KL 散度**不对称**。

$$
D_{KL}(P || Q) \neq D_{KL}(Q || P)
$$

这意味着它不符合距离度量的基本要求。它也不满足三角不等式。它是一个散度（divergence），而非距离。

前向 KL（$D_{KL}(P || Q)$）是"均值寻求型"：$Q$ 试图覆盖 $P$ 的所有模态。
反向 KL（$D_{KL}(Q || P)$）是"模态寻求型"：$Q$ 聚焦于 $P$ 的单个模态。

KL 散度的出现场景：
- VAE（ELBO 中的 KL 项将潜变量分布推向先验）
- 知识蒸馏（学生模型试图匹配教师模型的分布）
- RLHF（KL 惩罚使微调后的模型靠近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein 距离（推土机距离）

Wasserstein 距离衡量将一个概率分布变换为另一个所需的最小"功"。可以这样想：如果一个分布是一堆土，另一个是一个坑，你需要移动多少土以及移动多远？

$$
W(P, Q) = \inf_{\gamma \in \Gamma} \mathbb{E}[d(x, y)]
$$

对于一维分布，它简化为累积分布函数绝对差的积分：

$$
W_1(P, Q) = \int |\text{CDF}_P(x) - \text{CDF}_Q(x)| \, dx
$$

Wasserstein 距离为什么重要：
- 它是真正的度量（对称、满足三角不等式）
- 即使在分布不重叠时也能提供梯度（KL 散度此时会趋于无穷大）
- 这一性质使其成为 Wasserstein GAN（WGAN）的核心，解决了原始 GAN 的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输（optimal transport）问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离

| 任务 | 最佳距离 | 原因 |
|------|---------|------|
| 文本相似度 | Cosine | 幅度是噪声，方向是含义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度相当 |
| 稀疏高维特征 | L1 | 鲁棒，不会放大罕见的巨大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合值，而非向量值 |
| 字符串匹配 | Edit distance | 操作与人类编辑直觉对应 |
| 离群值检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 比较分布 | KL divergence | 度量用 Q 替代 P 所损失的信息量 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | Cosine 或 Dot product | 嵌入训练将意义编码在方向上 |
| 推荐系统 | Dot product | 幅度可编码流行度或置信度 |
| DNA 序列 | Weighted edit distance | 替换成本因核苷酸对而异 |
| 制造质控 | L-infinity | 任何维度的最坏偏差都很关键 |

### 与损失函数的联系

损失函数是应用于预测值与目标值之间的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化在损失函数中加入权重的范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 产生稀疏性而 L2 不产生：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最可能在角落处触到菱形——那里有一个权重为零。它们则在光滑点处触到圆形——那里两个权重都不为零。

### 最近邻搜索

每个距离函数都隐含一个最近邻搜索问题：给定一个查询点，在数据集中找到最近的点。

在包含 $n$ 个点、$d$ 维的数据集中，精确最近邻搜索的复杂度是每次查询 $O(n \cdot d)$。对于大数据集来说太慢了。

近似最近邻（ANN）算法用少量精度换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代向量数据库中的主导算法。它构建一个多层图，其中每个节点连接到其近似最近邻。搜索从顶层（稀疏、长跳）开始，逐层下降到底层（密集、短跳）。

## 动手实现（Build It）

### 步骤 1：所有范数和距离函数

完整实现见 `code/distances.py`。每个函数仅使用基本的 Python 数学从头构建。

### 步骤 2：相同数据、不同距离、不同邻居

`distances.py` 中的演示创建了一个数据集，选择一个查询点，并展示最近邻如何随距离度量而变化。在 L1 下"最近"的点在 L2 或余弦下可能不是最近的。

### 步骤 3：嵌入相似度搜索

代码包含一个模拟的嵌入相似度搜索，使用余弦相似度和 L2 距离分别查找与查询最相似的"文档"，展示排序可以不同。

## 实际应用（Use It）

最常见的实际用途：在向量数据库中查找相似项目。

```python
import numpy as np

# 计算矩阵中所有行向量两两之间的余弦相似度
# 先对每行做 L2 归一化，然后点积即为余弦相似度矩阵
# 处理零向量：将零范数替换为 1 避免除零
def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

# 模拟 1000 个 768 维的嵌入（如 BERT/RoBERTa 的输出维度）
embeddings = np.random.randn(1000, 768)

# 计算余弦相似度矩阵，每行表示该样本与所有样本的相似度
sim_matrix = cosine_similarity_matrix(embeddings)

# 找与第 0 个样本最相似的 5 个样本（排除自身）
query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索向量数据库时，底层就是这样的机制。嵌入模型将文本映射为向量。向量数据库计算查询向量与每个存储向量之间的余弦相似度（或点积），使用 ANN 算法避免检查全部向量。

## 练习题（Exercises）

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对于任意两个点，始终有 L-inf $\le$ L2 $\le$ L1。证明为什么这个顺序是必然的。

2. 创建两个向量，使余弦相似度高（$> 0.9$）但 L2 距离大（$> 10$）。从几何上解释发生了什么。然后再创建两个向量，使余弦相似度低（$< 0.3$）但 L2 距离小（$< 0.5$）。

3. 实现一个函数，输入数据集和查询点，分别使用 L1、L2、余弦和马氏距离返回最近邻。找到一个数据集使四种距离对最近邻的判断各不相同。

4. 用 CDF 方法手工计算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。然后计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的。哪个更大，为什么？

5. 实现 MinHash 用于近似 Jaccard 相似度。生成 100 个随机集合，计算所有对的精确 Jaccard 值，并与使用 50、100 和 200 个哈希函数的 MinHash 近似值比较。绘制近似误差图。

## 关键术语（Key Terms）

| 术语（English） | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Norm | "向量的大小" | 将向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且仅零向量时为零 |
| L1 norm | "曼哈顿距离" | 各分量绝对值之和。在优化中产生稀疏性。对离群值鲁棒 |
| L2 norm | "欧几里得距离" | 各分量平方和的平方根。欧几里得空间中的直线距离 |
| Lp norm | "广义范数" | 各分量绝对值的 $p$ 次幂之和的 $p$ 次根。L1 和 L2 是特例 |
| L-infinity norm | "最大范数"或"切比雪夫距离" | 最大绝对值分量。Lp 在 $p \to \infty$ 时的极限 |
| Cosine similarity | "向量间的夹角" | 点积除以两向量模长。范围 -1 到 +1。忽略向量长度 |
| Cosine distance | "1 减余弦相似度" | 将余弦相似度转换为距离。范围 0 到 2 |
| Dot product | "未归一化的余弦" | 各分量乘积之和。等于余弦相似度乘以两向量模长 |
| Mahalanobis distance | "感知相关性的距离" | 在使用数据协方差矩阵进行了白化（去相关和归一化）空间中的 L2 距离 |
| Jaccard similarity | "集合重叠度" | 交集大小除以并集大小。适用于集合，不适用于向量 |
| Edit distance | "Levenshtein 距离" | 将一个字符串转换为另一个所需的最少插入、删除和替换次数 |
| KL divergence | "分布间的距离" | 并非真正的距离（不对称）。衡量用 Q 编码 P 时多出的比特数 |
| Wasserstein distance | "推土机距离" | 将质量从一个分布搬运到另一个的最小做功。真正的度量 |
| Approximate nearest neighbor | "ANN 搜索" | 比精确搜索快得多的近似最近邻算法（HNSW、LSH、IVF） |
| HNSW | "向量数据库算法" | 分层可导航小世界图。用于快速近似最近邻搜索的多层图 |
| L1 regularization | "Lasso" | 在损失中加入权重的 L1 范数。将权重推向零（稀疏性） |
| L2 regularization | "Ridge"或"权重衰减" | 在损失中加入权重的 L2 范数平方。将权重向零收缩，不产生稀疏性 |
| Elastic Net | "L1 + L2" | 结合 L1 和 L2 正则化。比单独使用任一种更能处理相关特征组 |

## 延伸阅读（Further Reading）

- [FAISS：高效相似度搜索库](https://github.com/facebookresearch/faiss)——Meta 用于十亿级 ANN 搜索的库
- [Wasserstein GAN（Arjovsky et al., 2017）](https://arxiv.org/abs/1701.07875)——将推土机距离引入 GAN 的论文
- [位置敏感哈希（Indyk & Motwani, 1998）](https://dl.acm.org/doi/10.1145/276698.276876)——奠基性的 ANN 算法
- [词向量的高效估计（Mikolov et al., 2013）](https://arxiv.org/abs/1301.3781)——Word2Vec，余弦相似度从此成为嵌入的默认度量
- [sklearn.neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html)——scikit-learn 中距离度量和最近邻算法的实践指南
