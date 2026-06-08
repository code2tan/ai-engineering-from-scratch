# 向量、矩阵与运算（Vectors, Matrices & Operations）

> 每一个神经网络都不过是矩阵乘法再加上一些额外步骤。

**类型：** 实践
**语言：** Python, Julia
**前置条件：** 阶段 1，第 01 课（线性代数直觉）
**预计时间：** ~60 分钟

## 学习目标

- 构建一个 Matrix 类，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法和矩阵乘法，并能说明各自的应用场景
- 仅使用手写 Matrix 类实现一个单层稠密神经网络（`relu(W @ x + b)`）
- 解释广播（broadcasting）规则以及偏置加法在神经网络框架中如何工作

## 问题背景

你想构建一个神经网络。阅读代码时你看到了这样一行：

```
output = activation(weights @ input + bias)
```

其中的 `@` 是矩阵乘法。`weights`（权重）是一个矩阵，`input`（输入）是一个向量。如果你不了解这些运算的含义，这一行代码就像魔法。但如果你了解，它就是神经网络单层前向传播的全部——由三个运算完成。

模型处理的每张图像都是一个像素值矩阵。每个词嵌入（word embedding）都是一个向量。每个神经网络的每一层都是一个矩阵变换。要构建 AI 系统，你必须熟练掌握矩阵运算——就像不理解变量就无法写代码一样。

本课程致力于从零构建这种熟练度。

## 概念解析

### 向量（Vectors）：有序数字列表

向量是一个具有方向和长度的有序数字列表。在 AI 中，向量表示数据点、特征或参数。

```
v = [3, 4]        -- 二维向量
w = [1, 0, -2]    -- 三维向量
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。其长度（模）为 5（即 3-4-5 直角三角形）。

### 矩阵（Matrices）：数字网格

矩阵是一个二维网格，由行和列组成。一个 $m \times n$ 的矩阵有 $m$ 行和 $n$ 列。

```
A = | 1  2  3 |     -- 2×3 矩阵（2 行，3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入向量变换为输出向量。一个具有 784 个输入和 128 个输出的层，使用一个 $128 \times 784$ 的权重矩阵。

### 为什么形状（shape）很重要

矩阵乘法有一个严格规则：`(m x n) @ (n x p) = (m x p)`。内维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  权重矩阵      输入向量       输出向量

内维度：784 = 784  -- 匹配，合法
```

如果你在 PyTorch 中遇到形状不匹配的错误，原因就在这里。

### 运算总览

| 运算 | 作用 | 神经网络中的应用 |
|------|------|-----------------|
| 加法（Addition） | 逐元素合并 | 将偏置加到输出 |
| 标量乘法（Scalar Multiply） | 缩放每个元素 | 学习率 × 梯度 |
| 矩阵乘法（Matrix Multiply） | 变换向量 | 层的前向传播 |
| 转置（Transpose） | 行列互换 | 反向传播 |
| 行列式（Determinant） | 单一数值概括 | 判断是否可逆 |
| 逆矩阵（Inverse） | 撤销变换 | 求解线性系统 |
| 单位矩阵（Identity） | 不做任何操作的矩阵 | 初始化、残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别经常困扰初学者。

**逐元素乘法（Element-wise）**：对应位置相乘。两个矩阵必须形状相同。

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
*
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
=
\begin{bmatrix}
5 & 12 \\
21 & 32
\end{bmatrix}
$$

**矩阵乘法（Matrix Multiply）**：行与列的点积。内维度必须匹配。

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
@
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
=
\begin{bmatrix}
1\times5+2\times7 & 1\times6+2\times8 \\
3\times5+4\times7 & 3\times6+4\times8
\end{bmatrix}
=
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

不同的运算，不同的规则，得到不同的结果。

### 广播（Broadcasting）

当你将一个偏置向量加到输出矩阵上时，两者的形状不匹配。广播机制会将较小的数组拉伸到匹配的大小。

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
+
\begin{bmatrix}
10 & 20 & 30
\end{bmatrix}
$$

广播机制将向量沿行方向复制：

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
+
\begin{bmatrix}
10 & 20 & 30 \\
10 & 20 & 30
\end{bmatrix}
=
\begin{bmatrix}
11 & 22 & 33 \\
14 & 25 & 36
\end{bmatrix}
$$

每个现代框架都会自动执行广播。理解这一点可以避免"形状看起来不对但代码却能运行"时的困惑。

## 动手构建

### 第 1 步：Vector 类

```python
class Vector:
    def __init__(self, data):
        # 将输入数据转为列表，确保每份数据都是独立的副本
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        # 向量加法：对应位置元素分别相加
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        # 向量减法：对应位置元素分别相减
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        # 标量乘法：每个元素都乘以标量值
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        # 点积（内积）：对应位置相乘后求和，结果是一个标量
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        # 向量模长（L2 范数）：各分量平方和再开方
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：Matrix 类及核心运算

```python
class Matrix:
    def __init__(self, data):
        # 将输入数据转换为列表的列表，确保每行都是独立可变的副本
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        # 矩阵加法：对应位置元素相加，要求两矩阵形状完全相同
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        # 矩阵减法：对应位置元素相减
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        # 标量乘法：每个元素都乘以标量值
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        # Hadamard 积（逐元素乘法）：对应位置元素相乘
        # 与矩阵乘法不同，不涉及行与列之间的点积运算
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        # 矩阵乘法：结果矩阵 (i, j) 位置的元素 = 第 i 行与第 j 列的点积
        # 严格约束：self.cols 必须等于 other.rows（内维度匹配）
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        # 转置：行列互换，形状从 (m, n) 变为 (n, m)
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        # 递归计算行列式（Laplace 展开）
        # 1×1 矩阵：直接返回唯一元素
        # 2×2 矩阵：ad - bc（对角线乘积之差）
        # 更大矩阵：按第一行展开，递归计算余子式
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            # 构造余子式：去掉第 0 行和第 j 列后得到的子矩阵
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            # (-1)^j 为符号因子，a[0][j] 为展开元素，minor.determinant() 为余子式
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        # 2×2 矩阵的逆：1/det * [[d, -b], [-c, a]]
        # 行列式为 0 时矩阵奇异（不可逆），对应的线性变换会将维度压扁
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        # 创建 n×n 单位矩阵：对角线上为 1，其余位置为 0
        # 单位矩阵在矩阵乘法中相当于数字 1——任何矩阵乘以单位矩阵等于自身
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：验证效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)                     # 矩阵加法
print("A @ B =", A.matmul(B).data)                 # 矩阵乘法
print("A^T =", A.transpose().data)                 # 转置
print("det(A) =", A.determinant())                 # 行列式
print("A^-1 =", A.inverse_2x2().data)              # 逆矩阵

I = Matrix.identity(2)                              # 2×2 单位矩阵
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data) # 验证逆矩阵：A @ A^{-1} 应得到单位矩阵
```

### 第 4 步：连接神经网络

```python
import random

# 创建 3×1 输入向量（3 个特征值）
inputs = Matrix([[0.5], [0.8], [0.2]])

# 创建 2×3 权重矩阵：将 3 维输入映射到 2 维输出空间
# 权重初始化为 [-1, 1] 之间的随机值，打破对称性使各神经元学到不同特征
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])

# 偏置向量：每个输出维度一个偏置，让层可以在输入全为零时仍有非零输出
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    # ReLU 激活函数：将所有负值截断为 0，正值保持不变
    # 相比 sigmoid/tanh，ReLU 计算简单且能缓解深层网络中的梯度消失问题
    return Matrix([[max(0, val) for val in row] for row in m.data])

# 前向传播：output = relu(W @ x + b)
# 策略说明：先做线性变换（Wx + b），再通过 ReLU 引入非线性。
# 如果没有非线性激活函数，多层网络将退化为单层线性变换。
pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")     # 预期：(3, 1) — 3 个特征，1 个样本
print(f"Weight shape: {weights.shape}")   # 预期：(2, 3) — 2 个输出神经元，3 个输入特征
print(f"Output shape: {output.shape}")    # 预期：(2, 1) — 2 个输出值
print(f"Output: {output.data}")           # 最终激活值
```

这就是一个单层稠密层：`output = relu(W @ x + b)`。每个神经网络中的每个稠密层都在做完全相同的事情。

## 实际应用

NumPy 能用更少的代码实现上述所有功能，且速度快数个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)                            # 矩阵加法
print("A * B (element-wise) =\n", A * B)             # 逐元素乘法（Hadamard 积）
print("A @ B (matrix multiply) =\n", A @ B)          # 矩阵乘法
print("A^T =\n", A.T)                                # 转置
print("det(A) =", np.linalg.det(A))                  # 行列式
print("A^-1 =\n", np.linalg.inv(A))                  # 逆矩阵
print("I =\n", np.eye(2))                            # 单位矩阵

# 神经网络单层前向传播：与上面手写 Matrix 版本的数学运算完全相同
# 使用 NumPy 的随机初始化、矩阵乘法和广播机制，一行完成线性变换 + 激活
inputs = np.random.randn(3, 1)                       # 3 维输入，标准正态分布初始化
weights = np.random.randn(2, 3)                      # 2×3 权重矩阵
bias = np.array([[0.1], [0.1]])                      # 偏置项
output = np.maximum(0, weights @ inputs + bias)      # ReLU(Wx + b) — 一行完成前向传播

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__` 方法。NumPy 使用用 C 和 Fortran 编写的优化 BLAS 例程实现它。数学完全相同，速度快 100 倍以上。

NumPy 中的广播（Broadcasting）：

```python
# NumPy 自动广播：将 1D 偏置向量沿行方向复制，与 2D 矩阵的每一行相加
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动将一维偏置广播到两行。这就是每个神经网络框架中偏置加法的实现方式。

## 交付物

本课程产出一个用于通过几何直觉教授矩阵运算的提示词。参见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类是阶段 3 第 10 课中构建的微型神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认得到单位矩阵。用三个不同的 2×2 矩阵尝试。当行列式为 0 时会发生什么？

2. **实现 3×3 逆矩阵。** 使用伴随矩阵法（adjugate method）扩展 Matrix 类，使其能计算 3×3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 验证结果。

3. **构建两层网络。** 仅使用你的 Matrix 类（不使用 NumPy），创建一个两层神经网络：输入 (3) → 隐藏层 (4) → 输出 (2)。初始化随机权重，运行一次前向传播，验证所有形状是否正确。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 向量（Vector） | "一个箭头" | 一组有序数字。在 AI 中：高维空间中的一个点。 |
| 矩阵（Matrix） | "一个数字表格" | 一种线性变换，将向量从一个空间映射到另一个空间。 |
| 矩阵乘法（Matrix Multiply） | "就是把数字乘起来" | 第一个矩阵的每一行与第二个矩阵的每一列之间的点积。顺序很重要。 |
| 转置（Transpose） | "翻转一下" | 交换行和列，将 $m \times n$ 的矩阵变为 $n \times m$。在反向传播中至关重要。 |
| 行列式（Determinant） | "从矩阵算出来的某个数" | 衡量矩阵缩放面积（2D）或体积（3D）的程度。为零意味着变换压扁了一个维度。 |
| 逆矩阵（Inverse） | "撤销矩阵变换" | 能逆转原矩阵变换的矩阵。仅当行列式不为零时存在。 |
| 单位矩阵（Identity Matrix） | "那个无聊的矩阵" | 矩阵世界中相当于乘以 1。用于残差连接（ResNet）。 |
| 广播（Broadcasting） | "魔法般地修正形状" | 通过沿缺失维度重复，将较小的数组拉伸到与较大数组匹配。 |
| 逐元素（Element-wise） | "普通的乘法" | 对应位置相乘。两个数组必须形状相同（或可广播）。 |

## 延伸阅读

- [3Blue1Brown：线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) - 本课程涉及的每种运算的可视化直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的精确广播规则
- [Stanford CS229 线性代数复习资料](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的线性代数简明参考
