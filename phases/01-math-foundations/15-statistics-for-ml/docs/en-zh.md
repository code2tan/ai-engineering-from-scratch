# 机器学习统计学（Statistics for Machine Learning）

> 统计学是你判断模型是真有效还是纯属运气的方法。

**类型：** 动手构建
**语言：** Python
**前置知识：** 阶段 1，第 06 课（概率与分布）、第 07 课（贝叶斯定理）
**时间：** ~120 分钟

## 学习目标（Learning Objectives）

- 从头计算描述性统计量、皮尔逊/斯皮尔曼相关系数和协方差矩阵
- 执行假设检验（t 检验、卡方检验），正确解读 p 值和置信区间
- 使用自助法（bootstrap）重采样为任意指标构建置信区间，无需分布假设
- 使用效应量区分统计显著性与实际显著性

## 问题背景（The Problem）

你训练了两个模型。模型 A 在你的测试集上得分为 0.87。模型 B 得分为 0.89。你部署了模型 B。三周后，生产指标比之前还差。发生了什么？

模型 B 实际上并没有优于模型 A。那 0.02 的差异只是噪声。你的测试集太小，或者方差太高，或者两者兼有。你发布了一个用改进包装起来的随机波动。

这种情况屡见不鲜。Kaggle 排行榜的剧烈波动。无法复现的论文。基于几百个样本就宣布胜利者的 A/B 测试。根本原因都是一样的：有人跳过了统计学。

统计学为你提供了区分信号和噪声的工具。它告诉你差异是否真实、你应该有多大的信心、以及你需要多少数据才能信任一个结果。每一个 ML 流水线、每一次模型比较、每一个实验都需要统计学。没有它，你就是在猜。

## 核心概念（The Concept）

### 描述性统计：概括你的数据

在建模之前，你需要知道数据长什么样。描述性统计将数据集压缩为几个捕捉其形状的数字。

**集中趋势度量**回答"中间在哪里？"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点。中位数是半程标记。当它们偏离时，你的分布是有偏的。收入分布中均值 $\gg$ 中位数（亿万富翁带来的右偏）。训练中的损失分布通常均值 $\ll$ 中位数（简单样本带来的左偏）。

**离散程度度量**回答"数据有多分散？"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数**将排序后的数据分为 100 等份。第 25 百分位（Q1）意味着 25% 的值低于该点。第 50 百分位是中位数。第 75 百分位是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 中，你需要关注推理延迟的百分位数、预测置信度的分布以及理解误差分布。一个平均误差低但 P99 误差可怕的模型在安全关键应用中可能毫无用处。

**样本统计量与总体统计量。** 从样本计算方差时，除以 $(n-1)$ 而不是 $n$。这是贝塞尔校正（Bessel's correction）。它补偿了样本均值不是真实总体均值这一事实。用 $n$ 做分母时，你系统性地低估了真实方差。用 $(n-1)$ 时，估计是无偏的。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实践中：如果 $n$ 很大（数千个样本），差异可以忽略。如果 $n$ 很小（几十个样本），就需要关注了。

### 相关性：变量如何共同变化

相关性衡量两个变量之间线性关系的强度和方向。

**皮尔逊相关系数（Pearson correlation coefficient）**衡量线性关联：

$$
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{n \cdot s_x \cdot s_y}
$$

- $r = +1$：完全正线性关系
- $r = -1$：完全负线性关系
- $r = 0$：无线性关系（但可能存在非线性关系！）

范围：$[-1, 1]$

皮尔逊相关假设关系是线性的，且两个变量大致呈正态分布。它对离群值敏感。单个极端点就能将 $r$ 从 0.1 拖到 0.9。

**斯皮尔曼秩相关系数（Spearman rank correlation）**衡量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用哪种：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** 相关不等于因果。冰淇淋销售和溺水死亡相关，是因为两者在夏季都增加。你模型的准确率和参数量相关，但增加参数并不会自动提高准确率（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差衡量它们如何共同变化：

$$
\text{Cov}(X, Y) = \frac{1}{n} \sum (x_i - \bar{x})(y_i - \bar{y})
$$

- $\text{Cov}(X, Y) > 0$：$X$ 和 $Y$ 倾向于同增
- $\text{Cov}(X, Y) < 0$：$X$ 增加时 $Y$ 倾向于减少
- $\text{Cov}(X, Y) = 0$：无线性同向变动

对于 $d$ 个特征，协方差矩阵 $C$ 是一个 $d \times d$ 矩阵，其中 $C[i][j] = \text{Cov}(\text{feature}_i, \text{feature}_j)$。对角线元素 $C[i][i]$ 是每个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的联系。** PCA 对协方差矩阵进行特征分解。特征向量是主成分（最大方差方向）。特征值告诉你每个成分捕捉了多少方差。这正是第 10 课的内容，但你现在明白为什么协方差矩阵是分解的正确对象：它编码了数据中所有两两之间的线性关系。

**与相关性的联系。** 相关矩阵是标准化变量（每个除以其标准差）的协方差矩阵。相关性将协方差归一化，使所有值落在 $[-1, 1]$ 范围内。

### 假设检验

假设检验是在不确定性下做决策的框架。从一个主张出发，收集数据，然后判断数据是否与主张一致。

**基本设定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值**是在假设 H0 为真的前提下，观察到与现有数据一样极端的数据的概率。它**不是** H0 为真的概率。这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间**给出参数的一个合理取值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度告诉你精度。宽区间意味着高不确定性。窄区间意味着你的估计是精确的（但如果数据有偏差，不一定准确）。

### t 检验

t 检验比较均值。有几种变体。

**单样本 t 检验：** 总体均值是否与假设值不同？

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

自由度为 $n - 1$。

**双样本 t 检验（独立样本）：** 两个组的均值是否不同？

$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2 / n_1 + s_2^2 / n_2}}
$$

这是 Welch t 检验，不假设方差相等。除非你有特殊原因要假设方差相等，否则始终使用 Welch 检验。

**配对 t 检验：** 当测量值成对出现时（同一模型在相同数据划分上评估）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在 ML 中，配对 t 检验很常见：你在相同的 10 个交叉验证折上运行两个模型，然后逐对比较它们的得分。

### 卡方检验

卡方检验检查观测频数是否与期望频数匹配。适用于分类数据。

$$
\chi^2 = \sum \frac{(\text{observed} - \text{expected})^2}{\text{expected}}
$$

```
Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 模型的 A/B 测试

ML 中的 A/B 测试与网页 A/B 测试不同。模型比较有特定的挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性与实际显著性

一个结果可以是统计显著的，但在实际中毫无意义。只要有足够的数据，即使是微不足道的差异也会变得统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量**量化差异有多大，与样本量无关：

$$
\text{Cohen's d} = \frac{\text{mean}_1 - \text{mean}_2}{\text{pooled\_std}}
$$

- $d = 0.2$：小效应
- $d = 0.5$：中效应
- $d = 0.8$：大效应

始终同时报告 p 值和效应量。p 值告诉你差异是否真实。效应量告诉你它是否重要。

### 多重比较问题

当你检验很多假设时，有些会纯属巧合地"显著"。如果你在 $\alpha = 0.05$ 下测试 20 件事，即使一切都没有问题，也预计会有 1 个假阳性。

$$
P(\text{at least one false positive}) = 1 - (1 - \alpha)^m
$$

- $m = 20$ 次检验，$\alpha = 0.05$：
- $P(\text{假阳性}) = 1 - 0.95^{20} = 0.64$
- 你有 64% 的概率至少出现一个假阳性。

**Bonferroni 校正：** 将 $\alpha$ 除以检验次数。

$$
\text{Adjusted alpha} = \alpha / m = 0.05 / 20 = 0.0025
$$

仅在 p 值 $< 0.0025$ 时拒绝 H0。保守但简单。在检验独立时有效。

在 ML 中，当你跨多个指标比较模型、测试许多超参数配置或在多个数据集上评估时，这一点很重要。

### 自助法（Bootstrap Methods）

自助法通过有放回地重采样数据来估计统计量的抽样分布。不需要对底层分布做任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**自助法置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么自助法对 ML 很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用于模型比较的自助法：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对 t 检验更稳健，因为它没有任何分布假设。

### 参数检验与非参数检验

**参数检验**假设特定的分布（通常是正态分布）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验**不假设任何分布：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在 ML 实验中，你通常样本量较小（5 或 10 个交叉验证折），因此 Wilcoxon 符号秩检验等非参数检验通常比 t 检验更适合。

### 中心极限定理：实际意义

CLT 指出，随着 $n$ 增大，样本均值的分布趋近于正态分布，不论底层总体分布如何。

$$
\text{If } X_1, X_2, \dots, X_n \text{ are iid with mean } \mu \text{ and variance } \sigma^2:
$$

$$
\bar{X} \sim \text{Normal}(\mu, \sigma^2 / n) \quad \text{as } n \to \infty
$$

在大多数情况下，$n \ge 30$ 时成立。对于高度偏斜的分布，可能需要 $n \ge 100$。

**为什么这对 ML 很重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不能做什么：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 论文中的常见统计错误

1. **在训练集上测试。** 保证过拟合。始终保留模型在训练期间从未见过的数据。

2. **没有置信区间。** 报告单一准确率数值而不给出不确定性，使得结果不可复现也无法验证。

3. **忽略多重比较。** 测试 50 种配置且不进行校正就报告最佳的那个，会夸大假阳性率。

4. **混淆统计显著性与实际显著性。** 在 0.01% 的准确率提升上得到 p = 0.001 是没有意义的。

5. **在不平衡数据上使用准确率。** 在 99% 负类别的数据集上获得 99% 准确率意味着模型什么也没学到。使用精确率、召回率、F1 或 AUC。

6. **挑选指标。** 只报告你的模型获胜的指标。诚实的评估报告所有相关指标。

7. **在训练/测试划分之间泄露信息。** 在划分前做归一化，或用未来数据预测过去。

8. **小测试集且没有方差估计。** 在 100 个样本上评估然后声称 2% 的提升是噪声，不是信号。

9. **数据不独立时假设独立性。** 同一患者的医学影像、同一文档的多个句子。组内的观测是相关的。

10. **P 值操纵（p-hacking）。** 尝试不同的检验、子集或排除标准直到得到 p < 0.05。结果是搜索过程的人为产物。

## 动手实现（Building It）

你将实现：

1. **从头实现描述性统计量**（均值、中位数、众数、标准差、百分位数、IQR）
2. **相关函数**（皮尔逊和斯皮尔曼，含协方差矩阵）
3. **假设检验**（单样本 t 检验、双样本 t 检验、卡方检验）
4. **自助法置信区间**（适用于任意统计量，无需假设）
5. **A/B 测试模拟器**（生成数据、检验、检查 I 类和 II 类错误）
6. **统计显著性 vs 实际显著性演示**（展示大 n 让一切变得"显著"）

全部从头实现，仅使用 `math` 和 `random`。不依赖 numpy，不依赖 scipy。

## 关键术语（Key Terms）

| 术语（English） | 定义 |
|---|---|
| Mean | 值之和除以数量。对离群值敏感。 |
| Median | 排序后的中间值。对离群值鲁棒。 |
| Standard deviation | 方差的平方根。以原始单位衡量离散程度。 |
| Percentile | 给定百分比的数据落在其下的值。 |
| IQR | 四分位距。Q3 减 Q1。中间 50% 数据的离散程度。 |
| Pearson correlation | 衡量两个变量之间的线性关联。范围 $[-1, 1]$。 |
| Spearman correlation | 使用秩次衡量单调关联。 |
| Covariance matrix | 所有特征两两协方差构成的矩阵。 |
| Null hypothesis | 无效应或无差异的默认假设。 |
| p-value | 在原假设为真时观察到与现有数据一样极端的结果的概率。 |
| Confidence interval | 在给定置信水平下参数的合理取值范围。 |
| t-test | 检验均值是否有显著差异。使用 t 分布。 |
| Chi-squared test | 检验观测频数与期望频数是否有显著差异。 |
| Effect size | 差异的大小，与样本量无关。常用 Cohen's d。 |
| Bonferroni correction | 将显著性阈值除以检验次数以控制假阳性。 |
| Bootstrap | 有放回重采样以估计抽样分布。 |
| Type I error | 假阳性。原假设为真时拒绝原假设。 |
| Type II error | 假阴性。原假设为假时未能拒绝原假设。 |
| Statistical power | 正确拒绝假的原假设的概率。Power = 1 减 II 类错误率。 |
| Central limit theorem | 随着样本量增大，样本均值收敛到正态分布。 |
| Parametric test | 假设数据服从特定分布（通常是正态分布）。 |
| Non-parametric test | 不假设任何分布。基于秩次或符号。 |
