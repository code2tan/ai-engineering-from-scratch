# 计算图反向传播——零基础入门

> 写给完全没接触过的自己。只用加减乘除讲清楚。

---

## 第一步：什么是"梯度"？

### 一个简单的场景

你有两个数字：$x = 3$，$y = 2 \times x = 6$。

现在把 $x$ 增加一丁点儿（比如增加 0.1），$y$ 会变多少？

```
x = 3.0  →  y = 2 × 3.0 = 6.0
x = 3.1  →  y = 2 × 3.1 = 6.2
```

$x$ 增加了 0.1，$y$ 增加了 0.2。$y$ 的增加量是 $x$ 增加量的 **2 倍**。

**这个"2"就是梯度（导数）。** 它告诉我们：输入变化一单位，输出变化多少单位。

用数学表示：$\dfrac{dy}{dx} = 2$

**求导过程：** $y = 2x$，$y$ 对 $x$ 求导 → $\dfrac{dy}{dx} = \dfrac{d(2x)}{dx} = 2$。系数 2 直接就是导数。

### 第二个场景

$x = 3$，$y = x \times 5 = 15$。

$x$ 增加 0.1 → $y$ 增加 0.5。比例是 5。

所以 $\dfrac{dy}{dx} = 5$。

**求导过程：** $y = 5x$ → $\dfrac{dy}{dx} = \dfrac{d(5x)}{dx} = 5$。

### 规律

当你做 $y = a \times x$（一个数乘以 x）时，梯度就是这个 **a**。

从求导公式来看：

$$
y = a \cdot x  \quad\rightarrow\quad  \frac{dy}{dx} = \frac{d(a \cdot x)}{dx} = a
$$

因为 $a$ 是常数，所以 $a \cdot x$ 对 $x$ 求导就是 $a$。

具体例子：
- $y = 2x$ → $\dfrac{dy}{dx} = 2$
- $y = 5x$ → $\dfrac{dy}{dx} = 5$
- $y = wx$（$w$ 是随便一个数）→ $\dfrac{dy}{dx} = w$

很好，这就是全部的基础。记住了。

---

## 第二步：什么是"局部梯度"？

用一个有两个输入的算式：

$$a = x_1 \times x_2$$

比如 $x_1 = 3$，$x_2 = 2$，那 $a = 3 \times 2 = 6$。

现在问两个问题：

1. **如果只改变 $x_1$**（$x_2$ 不变），$a$ 变多少？
2. **如果只改变 $x_2$**（$x_1$ 不变），$a$ 变多少？

### 问题 1

$x_1 = 3$，$x_1$ 增加 0.1 → $x_1 = 3.1$，代入 $a = 3.1 \times 2 = 6.2$

$a$ 从 6 变成了 6.2，增加了 **0.2**。

0.2 = 0.1 × 2 = x₁ 的变化量 × **x₂ 的值**

所以：$\dfrac{da}{dx_1} = x_2 = 2$

### 问题 2

$x_2 = 2$，$x_2$ 增加 0.1 → $x_2 = 2.1$，代入 $a = 3 \times 2.1 = 6.3$

$a$ 从 6 变成了 6.3，增加了 **0.3**。

0.3 = 0.1 × 3 = x₂ 的变化量 × **x₁ 的值**

所以：$\dfrac{da}{dx_2} = x_1 = 3$

### 重要发现

**乘法的求导公式（偏导数）：**

$$
a = x_1 \cdot x_2
$$

把 $x_2$ 当作常数，对 $x_1$ 求导：$\displaystyle \frac{\partial a}{\partial x_1} = \frac{\partial (x_1 \cdot x_2)}{\partial x_1} = x_2$

把 $x_1$ 当作常数，对 $x_2$ 求导：$\displaystyle \frac{\partial a}{\partial x_2} = \frac{\partial (x_1 \cdot x_2)}{\partial x_2} = x_1$

因为"对 $x_1$ 求导"时，$x_2$ 是**系数**，系数直接就是导数。

所以：

```
乘法运算 a = x₁ × x₂ 的局部梯度：
  ∂a/∂x₁ = x₂  （x₁ 的梯度是对方的数值）
  ∂a/∂x₂ = x₁  （x₂ 的梯度是对方的数值）
```

不是对称的——谁求导，谁的梯度就是对**方**的值。

再看加法：

$$a = x_1 + x_2$$

$x_1$ 增加 0.1 → $a$ 增加 0.1。所以 $da/dx_1 = 1$。
$x_2$ 增加 0.1 → $a$ 增加 0.1。所以 $da/dx_2 = 1$。

**加法的求导公式（偏导数）：**

$$
a = x_1 + x_2
$$

对 $x_1$ 求导：$\displaystyle \frac{\partial a}{\partial x_1} = \frac{\partial (x_1 + x_2)}{\partial x_1} = 1$

对 $x_2$ 求导：$\displaystyle \frac{\partial a}{\partial x_2} = \frac{\partial (x_1 + x_2)}{\partial x_2} = 1$

因为 $x_1 + x_2$ 中 $x_1$ 的系数就是 1，所以对 $x_1$ 求导得 1。

```
加法运算 a = x₁ + x₂ 的局部梯度：
  ∂a/∂x₁ = 1
  ∂a/∂x₂ = 1
```

加法更简单：梯度永远都是 1。

---

## 第三步：什么是"链式法则"？

现在把两步运算连起来：

$$a = 2 \times x$$
$$y = a + 1$$

代入就是 $y = 2x + 1$。

### 你想求 dy/dx

直接算也很简单：$y = 2x + 1$，所以 $dy/dx = 2$。

但假如我们用"分步"的方法来做——这就是链式法则：

```
第 1 步：求 dy/da   （y 对 a 的梯度）
第 2 步：求 da/dx   （a 对 x 的梯度）
第 3 步：把它们乘起来
```

### 实际计算

写出每一步的求导公式：

**第 1 步的导数：**
$$
y = a + 1 \quad\rightarrow\quad \frac{dy}{da} = \frac{d(a + 1)}{da} = 1
$$
加法的导数是 1。

**第 2 步的导数：**
$$
a = 2 \cdot x \quad\rightarrow\quad \frac{da}{dx} = \frac{d(2x)}{dx} = 2
$$
系数乘法的导数是系数 2。

**第 3 步：链式法则——把两步的导数乘起来：**
$$
\frac{dy}{dx} = \frac{dy}{da} \times \frac{da}{dx} = 1 \times 2 = 2
$$

答案和直接算出来一样：**2**。

### 链式法则公式

$$\frac{dy}{dx} = \frac{dy}{da} \times \frac{da}{dx}$$

"把梯度沿着链乘起来。"

---

## 第四步：用完整的计算图演示

现在我们把上面两步画成图：

```
         x = 3
          │
          ▼
         乘 2 ──→ a = 6
          │
          ▼
         加 1 ──→ y = 7
```

每个箭头是一个运算，每个节点存一个中间结果。

### 前向传播

从上往下读：从输入 $x$ 开始一路计算到输出 $y$。

```
x = 3
  → a = 2 × 3 = 6
    → y = 6 + 1 = 7
```

**前向传播** = 正常计算，得出最终结果。

### 反向传播

从下往上读：从输出 $y$ 开始，把梯度往回传。

**第一步：从 y 出发**

$dy/dy = 1$。输出对自己的梯度永远是 1。

**第二步：经过加法节点**

运算：$y = a + 1$

求导公式：$\displaystyle \frac{dy}{da} = \frac{d(a+1)}{da} = 1$

加法的局部梯度 = 1。上游梯度 $dy/dy = 1$ 传到这里：

$$
\frac{dy}{da} = \frac{dy}{dy} \times 1 = 1 \times 1 = 1
$$

```
          上游梯度: 1（从 dy/dy 来）
                │
                ▼
         加 1 ──→  y = a + 1
                ↑
           局部梯度: dy/da = d(a+1)/da = 1
```

**第三步：经过乘法节点**

运算：$a = 2 \cdot x$

求导公式：$\displaystyle \frac{da}{dx} = \frac{d(2x)}{dx} = 2$

乘法的局部梯度 = 系数 2。上游梯度 $dy/da = 1$ 传到这里：

$$
\frac{dy}{dx} = \frac{dy}{da} \times \frac{da}{dx} = \frac{dy}{da} \times 2 = 1 \times 2 = 2
$$

```
          上游梯度: 1（从 dy/da 来）
                │
                ▼
         乘 2 ──→  a = 2x
                ↑
           局部梯度: da/dx = d(2x)/dx = 2
                │
                ▼
           最终梯度: dy/dx = 1 × 2 = 2 -> x
```

### 最终结果

$dy/dx = 2$。也就是说：**x 每增加 1 个单位，y 增加 2 个单位。**

验证：$x = 3 \to y = 7$；$x = 4 \to y = 9$。确实增加了 2。

---

## 第五步：再加一个节点——三个节点

$$a = x_1 \times x_2$$
$$b = a + 1$$
$$y = \text{relu}(b)$$

其中 $\text{relu}(b) = \max(0, b)$。

### 前向传播

```
x₁ = 2    x₂ = 3
   \       /
    ×     /      → a = 2 × 3 = 6
      a = 6
       │
       │       b = 1
       │      /
       + ←───/    → b = 6 + 1 = 7
       │
      relu         → y = max(0, 7) = 7
```

每一层都是前一层的输入。

### 反向传播：一步一步推

> 绿色数字 = 该节点的**局部梯度**（公式固定，取决于运算类型）
> 紫色数字 = 传到该节点的**上游梯度**（从上一层来的）
> **最终梯度** = 上游梯度 × 局部梯度

---

**第 1 层：relu 节点**

| 项目 | 值 | 说明 |
|:----|:--:|:-----|
| 运算 | $y = \text{relu}(b)$ | 取最大值 |
| 输入 | $b = 7$ | |
| 输出 | $y = 7$ | |
| 局部梯度 | $1$ | 因为 $b > 0$，relu 的梯度 = 1。如果 $b < 0$，梯度 = 0 |
| 上游梯度 | $dy/dy = 1$ | 输出对自己的梯度 |
| 传给 b | $1 \times 1 = 1$ | $dy/db = 1$ |

**relu 的求导公式：**

$$
\text{relu}(b) = \max(0, b)
$$

$$
\frac{d}{db}\,\text{relu}(b) = \begin{cases}
1 & \text{如果 } b > 0 \\[5pt]
0 & \text{如果 } b < 0
\end{cases}
$$

**relu 的局部梯度为什么这么算？**
- $y = \max(0, b)$
- 如果 $b > 0$，relu 的输出就是 $b$ 本身。所以 $db$ 增加多少，$dy$ 就增加多少——梯度 = 1。
- 如果 $b < 0$，输出永远是 0。不管 $b$ 怎么变，输出都不变——梯度 = 0。

**本例计算：**
$$
\frac{dy}{db} = \frac{dy}{dy} \times \frac{d\,\text{relu}(b)}{db} = 1 \times 1 = 1
$$

---

**第 2 层：加法节点**

| 项目 | 值 | 说明 |
|:----|:--:|:-----|
| 运算 | $b = a + 1$ | 两个数相加 |
| 局部梯度 | $db/da = 1$ | 加法的梯度固定为 1 |
| 上游梯度 | $dy/db = 1$ | 从 relu 传来的梯度 |
| 传给 a | $1 \times 1 = 1$ | $dy/da = 1$ |

**加法的求导公式：**

$$
b = a + 1 \quad\rightarrow\quad \frac{db}{da} = \frac{d(a+1)}{da} = 1
$$

**本例计算：**

$$
\frac{dy}{da} = \frac{dy}{db} \times \frac{db}{da} = 1 \times 1 = 1
$$

---

**第 3 层：乘法节点**

| 项目 | 值 | 说明 |
|:----|:--:|:-----|
| 运算 | $a = x_1 \times x_2$ | 两个数相乘 |
| 局部梯度 | $da/dx_1 = x_2 = 3$ | x₁ 的梯度 = 对方的值 |
| | $da/dx_2 = x_1 = 2$ | x₂ 的梯度 = 对方的值 |
| 上游梯度 | $dy/da = 1$ | 从加法传来的梯度 |
| 传给 x₁ | $1 \times 3 = 3$ | $dy/dx_1 = 3$ |
| 传给 x₂ | $1 \times 2 = 2$ | $dy/dx_2 = 2$ |

**乘法的求导公式（偏导数）：**

$$
a = x_1 \cdot x_2
$$

对 $x_1$ 求偏导（把 $x_2$ 当常数）：$\displaystyle \frac{\partial a}{\partial x_1} = \frac{\partial (x_1 \cdot x_2)}{\partial x_1} = x_2 = 3$

对 $x_2$ 求偏导（把 $x_1$ 当常数）：$\displaystyle \frac{\partial a}{\partial x_2} = \frac{\partial (x_1 \cdot x_2)}{\partial x_2} = x_1 = 2$

**本例计算（应用链式法则）：**

$$
\frac{dy}{dx_1} = \frac{dy}{da} \times \frac{\partial a}{\partial x_1} = 1 \times 3 = 3
$$

$$
\frac{dy}{dx_2} = \frac{dy}{da} \times \frac{\partial a}{\partial x_2} = 1 \times 2 = 2
$$

---

### 完整传播路径（含求导公式）

每一层的传播计算：

$$
\frac{dy}{dy} = 1 \quad\xrightarrow{\text{通过 relu}}\quad \frac{dy}{db} = \frac{dy}{dy} \cdot \frac{d\,\text{relu}(b)}{db} = 1 \times 1 = 1
$$

$$
\frac{dy}{db} = 1 \quad\xrightarrow{\text{通过加法}}\quad \frac{dy}{da} = \frac{dy}{db} \cdot \frac{d(a+1)}{da} = 1 \times 1 = 1
$$

$$
\frac{dy}{da} = 1 \quad\xrightarrow{\text{通过乘法}}\quad \frac{dy}{dx_1} = \frac{dy}{da} \cdot \frac{\partial (x_1 x_2)}{\partial x_1} = 1 \times x_2 = 3
$$

$$
\frac{dy}{da} = 1 \quad\xrightarrow{\text{通过乘法}}\quad \frac{dy}{dx_2} = \frac{dy}{da} \cdot \frac{\partial (x_1 x_2)}{\partial x_2} = 1 \times x_1 = 2
$$

用图来看就是：

```
dy/dy = 1
   │
   ▼
relu 节点:    dy/dy = 1
              × d(relu(b))/db = 1（因 b > 0）
              ↓
              dy/db = 1 × 1 = 1
   │
   ▼
加法节点:     dy/db = 1
              × d(a+1)/da = 1
              ↓
              dy/da = 1 × 1 = 1
   │
   ▼
乘法节点:     dy/da = 1
              × ∂(x₁·x₂)/∂x₁ = x₂ = 3  →  dy/dx₁ = 1 × 3 = 3
              × ∂(x₁·x₂)/∂x₂ = x₁ = 2  →  dy/dx₂ = 1 × 2 = 2
```

### 手动验证

直接对公式 $y = \text{relu}(x_1 \times x_2 + 1)$ 求导：

因为 $x_1 \times x_2 + 1 = 7 > 0$，relu 就是恒等映射。

$$\frac{dy}{dx_1} = \frac{d}{dx_1}(x_1 \times x_2 + 1) = x_2 = 3$$
$$\frac{dy}{dx_2} = \frac{d}{dx_2}(x_1 \times x_2 + 1) = x_1 = 2$$

和反向传播的结果一样。

---

## 核心规律（必须记住）

> **每个节点只做一件事：把上游梯度 × 自己的局部梯度，传给子节点。**

```
               上游梯度（从更高层的节点来）
                    │
                    ▼
        ┌─────────────────────┐
        │     当前运算节点      │
        │                     │
        │  局部梯度（固定公式）  │
        │  加法：梯度 = 1       │
        │  乘法：梯度 = 对方的值 │
        │  relu: 梯度 = 0 或 1 │
        └─────────────────────┘
                    │
                    ▼
             上游梯度 × 局部梯度
             传给更低层的节点
```

**节点不需要知道整个网络的结构。** 它只需要知道：
1. 我的运算类型是什么？（加法？乘法？relu？）
2. 上游传来的梯度是多少？

然后就可以用自己的局部梯度公式算出该往下传多少。

---

## 为什么叫"反向传播"？

因为方向和前向传播完全相反：

```
前向传播方向（从输入到输出）：
x₁, x₂  →  乘法  →  加法  →  relu  →  y

反向传播方向（从输出到输入）：
y  →  relu  →  加法  →  乘法  →  x₁, x₂
```

就像沿着绳子往回摸，每摸到一个节点就做一次"上游梯度 × 局部梯度"。

---

## 对应代码

以上所有计算在 Value 引擎中就是以下逻辑：

```python
# 加法：局部梯度为 1
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        # 上游梯度等分传到两个输入（因为加法的局部梯度 = 1）
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

# 乘法：局部梯度 = 对方的值
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        # 对 self 的梯度 = other 的值 × 上游梯度
        # 对 other 的梯度 = self 的值 × 上游梯度
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out

# relu：b > 0 时梯度为 1，否则为 0
def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')
    def _backward():
        self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out

# 反向传播引擎
def backward(self):
    self.grad = 1.0        # 输出对自己的梯度 = 1
    for v in reversed(topo):   # 逆拓扑序遍历
        v._backward()      # 每个节点：上游梯度 × 局部梯度
```

运行验证：

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2           # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"dy/dx1 = {x1.grad}")   # 输出: 3.0
print(f"dy/dx2 = {x2.grad}")   # 输出: 2.0
```

---

## 一句话总结

> **梯度是"变化的比例"。链式法则是"把比例乘起来"。计算图反向传播就是：从输出往回走，每到一个节点就做一次"上游梯度 × 局部梯度"。**
