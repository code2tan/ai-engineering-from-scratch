# AI Engineering 学习路线图

> 目标人群：LLM Agent 开发工程师（Java 背景转 AI）
> 总课时：约 100 小时核心内容
> 学习方式：读文档 → 手写实现 → 调库验证 → 做 quiz → 记笔记

---

## 第一阶段：打底（约 15 小时）

### Phase 1 — 数学基础（选 6 课）

| # | 课程 | 目录 | 为什么学 |
|---|------|------|---------|
| 01 | 线性代数直觉 | `phases/01-math-foundations/01-linear-algebra-intuition/` | 向量、矩阵、点积 → embedding、attention 的底层语言 |
| 02 | 向量矩阵运算 | `phases/01-math-foundations/02-vectors-matrices-operations/` | 矩阵乘法 → 神经网络前向传播就是矩阵乘法 |
| 04 | 微积分与梯度 | `phases/01-math-foundations/04-calculus-for-ml/` | 梯度下降的数学基础 |
| 05 | 链式法则与自动微分 | `phases/01-math-foundations/05-chain-rule-and-autodiff/` | 反向传播的数学本质 |
| 06 | 概率与分布 | `phases/01-math-foundations/06-probability-and-distributions/` | softmax、交叉熵、采样的基础 |
| 09 | 信息论 | `phases/01-math-foundations/09-information-theory/` | 熵、KL 散度 → 损失函数为什么这么设计 |

### Phase 2 — ML 基础（选 4 课）

| # | 课程 | 目录 | 为什么学 |
|---|------|------|---------|
| 01 | 什么是机器学习 | `phases/02-ml-fundamentals/01-what-is-machine-learning/` | 监督/无监督/强化学习分类，过拟合/欠拟合概念 |
| 02 | 线性回归 | `phases/02-ml-fundamentals/02-linear-regression/` | 最简单的"模型训练"全流程，理解什么叫"拟合" |
| 09 | 模型评估 | `phases/02-ml-fundamentals/09-model-evaluation/` | 交叉验证、精确率/召回率/F1 → Agent 评估也要用 |
| 10 | 偏差与方差 | `phases/02-ml-fundamentals/10-bias-variance/` | 过拟合/欠拟合的本质 → 理解 LLM 为什么也会过拟合 |

### Phase 3 — 深度学习核心（选 6 课）

| # | 课程 | 目录 | 为什么学 |
|---|------|------|---------|
| 01 | 感知机 | `phases/03-deep-learning-core/01-the-perceptron/` | 神经网络的起点，理解"神经元"是什么 |
| 02 | 多层网络与前向传播 | `phases/03-deep-learning-core/02-multi-layer-networks/` | 层叠起来就是深度网络 |
| 03 | 🔥 反向传播 | `phases/03-deep-learning-core/03-backpropagation/` | **整个深度学习的灵魂**，必须手写一遍 |
| 05 | 损失函数 | `phases/03-deep-learning-core/05-loss-functions/` | MSE、交叉熵、对比损失 → 模型在优化什么 |
| 06 | 优化器 | `phases/03-deep-learning-core/06-optimizers/` | SGD、Adam、AdamW → 模型怎么更新参数 |
| 11 | PyTorch 入门 | `phases/03-deep-learning-core/11-intro-to-pytorch/` | 从此以后用 PyTorch 代替手写 numpy |

---

## 第二阶段：深入（约 35 小时）

### Phase 7 — Transformers 深度剖析（选 9 课）

| # | 课程 | 目录 | 为什么学 |
|---|------|------|---------|
| 01 | 为什么需要 Transformer | `phases/07-transformers-deep-dive/01-why-transformers/` | RNN 的问题 → Transformer 解决了什么 |
| 02 | 🔥 Self-Attention 手写 | `phases/07-transformers-deep-dive/02-self-attention-from-scratch/` | **Attention 的核心**，Q、K、V 怎么算 |
| 03 | Multi-Head Attention | `phases/07-transformers-deep-dive/03-multi-head-attention/` | 多个 attention 头并行，捕获不同关系 |
| 04 | 位置编码 | `phases/07-transformers-deep-dive/04-positional-encoding/` | Sinusoidal、RoPE → 模型怎么知道词的位置 |
| 05 | 完整 Transformer | `phases/07-transformers-deep-dive/05-full-transformer/` | Encoder + Decoder 完整架构 |
| 07 | GPT 因果语言模型 | `phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/` | 你每天调用的 GPT 底层就长这样 |
| 11 | Mixture of Experts | `phases/07-transformers-deep-dive/11-mixture-of-experts/` | MoE → 大模型省计算的核心技术 |
| 12 | 🔥 KV Cache & Flash Attention | `phases/07-transformers-deep-dive/12-kv-cache-flash-attention/` | **推理加速的核心**，做 Agent 必须懂 |
| 14 | 从零构建 Transformer | `phases/07-transformers-deep-dive/14-build-a-transformer-capstone/` | 综合实战 |

### Phase 10 — 从零构建 LLM（全 22 课）

| # | 课程 | 目录 | 深度 |
|---|------|------|------|
| 01 | Tokenizer 概览 | `phases/10-llms-from-scratch/01-tokenizers/` | 🔥 重点 |
| 02 | 从零构建 Tokenizer | `phases/10-llms-from-scratch/02-building-a-tokenizer/` | 🔥 重点 |
| 03 | 预训练数据管线 | `phases/10-llms-from-scratch/03-data-pipelines/` | 重点 |
| 04 | 🔥 预训练 Mini GPT | `phases/10-llms-from-scratch/04-pre-training-mini-gpt/` | 🔥 重点 |
| 05 | 分布式训练 | `phases/10-llms-from-scratch/05-scaling-distributed/` | 了解概念 |
| 06 | 🔥 SFT 指令微调 | `phases/10-llms-from-scratch/06-instruction-tuning-sft/` | 🔥 重点 |
| 07 | 🔥 RLHF | `phases/10-llms-from-scratch/07-rlhf/` | 🔥 重点 |
| 08 | 🔥 DPO | `phases/10-llms-from-scratch/08-dpo/` | 🔥 重点 |
| 09 | Constitutional AI | `phases/10-llms-from-scratch/09-constitutional-ai-self-improvement/` | 了解概念 |
| 10 | LLM 评估 | `phases/10-llms-from-scratch/10-evaluation/` | 重点 |
| 11 | 🔥 量化 INT8/GPTQ/AWQ/GGUF | `phases/10-llms-from-scratch/11-quantization/` | 🔥 重点 |
| 12 | 🔥 推理优化 | `phases/10-llms-from-scratch/12-inference-optimization/` | 🔥 重点 |
| 13 | 完整 LLM 管线 | `phases/10-llms-from-scratch/13-building-complete-llm-pipeline/` | 重点 |
| 14 | 开源模型架构巡礼 | `phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/` | 了解 |
| 15 | 推测解码 | `phases/10-llms-from-scratch/15-speculative-decoding-eagle3/` | 了解 |
| 16 | Differential Attention | `phases/10-llms-from-scratch/16-differential-attention-v2/` | 了解 |
| 17 | Native Sparse Attention | `phases/10-llms-from-scratch/17-native-sparse-attention/` | 了解 |
| 18 | Multi-Token Prediction | `phases/10-llms-from-scratch/18-multi-token-prediction/` | 了解 |
| 19 | DualPipe 并行 | `phases/10-llms-from-scratch/19-dualpipe-parallelism/` | 了解 |
| 20 | DeepSeek-V3 架构 | `phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/` | 了解 |
| 21 | Jamba 混合架构 | `phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/` | 了解 |
| 22 | Async/Hogwild 推理 | `phases/10-llms-from-scratch/22-async-hogwild-inference/` | 了解 |

### Phase 11 — LLM 工程（全 17 课）

| # | 课程 | 目录 | 深度 |
|---|------|------|------|
| 01 | 🔥 Prompt Engineering | `phases/11-llm-engineering/01-prompt-engineering/` | 🔥 重点 |
| 02 | Few-Shot / CoT | `phases/11-llm-engineering/02-few-shot-cot/` | 重点 |
| 03 | 结构化输出 | `phases/11-llm-engineering/03-structured-outputs/` | 重点 |
| 04 | Embeddings | `phases/11-llm-engineering/04-embeddings/` | 重点 |
| 05 | Context Engineering | `phases/11-llm-engineering/05-context-engineering/` | 重点 |
| 06 | 🔥 RAG | `phases/11-llm-engineering/06-rag/` | 🔥 重点 |
| 07 | 高级 RAG | `phases/11-llm-engineering/07-advanced-rag/` | 重点 |
| 08 | 🔥 LoRA 微调 | `phases/11-llm-engineering/08-fine-tuning-lora/` | 🔥 重点 |
| 09 | 🔥 Function Calling | `phases/11-llm-engineering/09-function-calling/` | 🔥 重点 |
| 10 | LLM 评估 | `phases/11-llm-engineering/10-evaluation/` | 重点 |
| 11 | 缓存与成本 | `phases/11-llm-engineering/11-caching-cost/` | 重点 |
| 12 | Guardrails | `phases/11-llm-engineering/12-guardrails/` | 重点 |
| 13 | 生产级 LLM App | `phases/11-llm-engineering/13-production-app/` | 重点 |
| 14 | 🔥 MCP 协议 | `phases/11-llm-engineering/14-model-context-protocol/` | 🔥 重点 |
| 15 | Prompt Caching | `phases/11-llm-engineering/15-prompt-caching/` | 重点 |
| 16 | LangGraph 状态机 | `phases/11-llm-engineering/16-langgraph-state-machines/` | 重点 |
| 17 | Agent 框架对比 | `phases/11-llm-engineering/17-agent-framework-tradeoffs/` | 重点 |

---

## 第三阶段：专业（约 50 小时）

### Phase 13 — 工具与协议（选 20 课）

| # | 课程 | 目录 | 深度 |
|---|------|------|------|
| 01 | Tool Interface | `phases/13-tools-and-protocols/01-the-tool-interface/` | 重点 |
| 02 | Function Calling 深挖 | `phases/13-tools-and-protocols/02-function-calling-deep-dive/` | 🔥 重点 |
| 03 | 并行与流式 Tool Call | `phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls/` | 重点 |
| 04 | 结构化输出 | `phases/13-tools-and-protocols/04-structured-output/` | 重点 |
| 05 | Tool Schema 设计 | `phases/13-tools-and-protocols/05-tool-schema-design/` | 重点 |
| 06 | 🔥 MCP 基础 | `phases/13-tools-and-protocols/06-mcp-fundamentals/` | 🔥 重点 |
| 07 | 🔥 构建 MCP Server | `phases/13-tools-and-protocols/07-building-an-mcp-server/` | 🔥 重点 |
| 08 | 🔥 构建 MCP Client | `phases/13-tools-and-protocols/08-building-an-mcp-client/` | 🔥 重点 |
| 09 | MCP 传输层 | `phases/13-tools-and-protocols/09-mcp-transports/` | 重点 |
| 10 | MCP Resources & Prompts | `phases/13-tools-and-protocols/10-mcp-resources-and-prompts/` | 了解 |
| 11 | MCP Sampling | `phases/13-tools-and-protocols/11-mcp-sampling/` | 了解 |
| 12 | MCP Roots & Elicitation | `phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/` | 了解 |
| 13 | MCP Async Tasks | `phases/13-tools-and-protocols/13-mcp-async-tasks/` | 了解 |
| 14 | MCP Apps | `phases/13-tools-and-protocols/14-mcp-apps/` | 了解 |
| 15 | MCP 安全 I — Tool 投毒 | `phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/` | 重点 |
| 16 | MCP 安全 II — OAuth 2.1 | `phases/13-tools-and-protocols/16-mcp-security-oauth-2-1/` | 了解 |
| 17 | MCP 网关与注册中心 | `phases/13-tools-and-protocols/17-mcp-gateways-and-registries/` | 了解 |
| 18 | MCP 生产认证 | `phases/13-tools-and-protocols/18-mcp-auth-production/` | 了解 |
| 19 | 🔥 A2A 协议 | `phases/13-tools-and-protocols/19-a2a-protocol/` | 🔥 重点 |
| 22 | Skills & Agent SDKs | `phases/13-tools-and-protocols/22-skills-and-agent-sdks/` | 重点 |

### Phase 14 — Agent 工程（选 30 课）

**核心循环（必学 6 课）**

| # | 课程 | 目录 |
|---|------|------|
| 01 | 🔥 Agent Loop | `phases/14-agent-engineering/01-the-agent-loop/` |
| 02 | ReWOO / Plan-Execute | `phases/14-agent-engineering/02-rewoo-plan-and-execute/` |
| 03 | Reflexion 反思 | `phases/14-agent-engineering/03-reflexion-verbal-rl/` |
| 04 | Tree of Thoughts | `phases/14-agent-engineering/04-tree-of-thoughts-lats/` |
| 05 | Self-Refine / CRITIC | `phases/14-agent-engineering/05-self-refine-and-critic/` |
| 06 | Tool Use & Function Calling | `phases/14-agent-engineering/06-tool-use-and-function-calling/` |

**记忆系统（必学 3 课）**

| # | 课程 | 目录 |
|---|------|------|
| 07 | Memory — MemGPT | `phases/14-agent-engineering/07-memory-virtual-context-memgpt/` |
| 08 | Memory Blocks | `phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/` |
| 09 | 混合记忆 — Mem0 | `phases/14-agent-engineering/09-hybrid-memory-mem0/` |

**规划与模式（必学 3 课）**

| # | 课程 | 目录 |
|---|------|------|
| 11 | HTN / Evolutionary Planning | `phases/14-agent-engineering/11-planning-htn-and-evolutionary/` |
| 12 | Anthropic Workflow Patterns | `phases/14-agent-engineering/12-anthropic-workflow-patterns/` |
| 28 | 编排模式 | `phases/14-agent-engineering/28-orchestration-patterns/` |

**框架（选学，按实际使用的框架选 2-3 课）**

| # | 课程 | 目录 |
|---|------|------|
| 13 | LangGraph | `phases/14-agent-engineering/13-langgraph-stateful-graphs/` |
| 14 | AutoGen | `phases/14-agent-engineering/14-autogen-actor-model/` |
| 15 | CrewAI | `phases/14-agent-engineering/15-crewai-role-based-crews/` |
| 16 | OpenAI Agents SDK | `phases/14-agent-engineering/16-openai-agents-sdk/` |
| 17 | Claude Agent SDK | `phases/14-agent-engineering/17-claude-agent-sdk/` |

**评估与观测（必学 4 课）**

| # | 课程 | 目录 |
|---|------|------|
| 19 | Benchmarks — SWE-bench / GAIA | `phases/14-agent-engineering/19-benchmarks-swebench-gaia/` |
| 23 | OTel GenAI | `phases/14-agent-engineering/23-otel-genai-conventions/` |
| 24 | Agent 可观测性平台 | `phases/14-agent-engineering/24-agent-observability-platforms/` |
| 30 | Eval-Driven 开发 | `phases/14-agent-engineering/30-eval-driven-agent-development/` |

**生产与安全（必学 5 课）**

| # | 课程 | 目录 |
|---|------|------|
| 26 | 🔥 Agent 失败模式 | `phases/14-agent-engineering/26-failure-modes-agentic/` |
| 27 | Prompt Injection 防御 | `phases/14-agent-engineering/27-prompt-injection-defense/` |
| 29 | 生产运行时 | `phases/14-agent-engineering/29-production-runtimes/` |
| 25 | Multi-Agent 辩论 | `phases/14-agent-engineering/25-multi-agent-debate/` |
| 21 | Computer Use Agent | `phases/14-agent-engineering/21-computer-use-agents/` |

**Agent Workbench（选学，按需 3-4 课）**

| # | 课程 | 目录 |
|---|------|------|
| 31 | 为什么模型会失败 | `phases/14-agent-engineering/31-agent-workbench-why-models-fail/` |
| 32 | Minimal Workbench | `phases/14-agent-engineering/32-minimal-agent-workbench/` |
| 38 | Verification Gates | `phases/14-agent-engineering/38-verification-gates/` |
| 39 | Reviewer Agent | `phases/14-agent-engineering/39-reviewer-agent/` |

---

## 按需补充（不纳入主线）

以下内容在需要时回到对应目录查阅，不必预先学完。

| 场景 | 去哪里 |
|------|--------|
| 底层数学概念忘了 | `phases/01-math-foundations/` 找对应课 |
| 做 CV 相关 Agent（截图理解、视频分析） | `phases/04-computer-vision/` |
| 做语音 Agent | `phases/06-speech-and-audio/` |
| 需要深入理解 RL | `phases/09-reinforcement-learning/` |
| 长时间自主运行的 Agent 安全 | `phases/15-autonomous-systems/` |
| 多 Agent 协作/Swarm | `phases/16-multi-agent-and-swarms/` |
| 生产部署、GPU 推理、K8s | `phases/17-infrastructure-and-production/` |
| AI 安全对齐、红队测试 | `phases/18-ethics-safety-alignment/` |
| 综合实战项目 | `phases/19-capstone-projects/` |

---

## 每节课的学习步骤

```
1. 做课前测     → 打开 quiz.json，做 stage=pre 的题目
2. 阅读文档     → 打开 docs/en.md 理解概念
3. 手写实现     → 打开 code/*.py，逐行理解 + 亲自跑
4. 调库验证     → 用 PyTorch/sklearn 等框架跑一遍同样的事情
5. 破坏实验     → 故意改参数、改维度，看效果变化
6. 做课后测     → 做 quiz.json 里 stage=post 的题目
7. 记录笔记     → 3 句话总结：解决了什么 + 核心思想 + 和什么有关联
8. 收集产出     → 查看 outputs/ 里的 prompt/skill/agent，安装到 Claude Code
```

## AI 辅助学习指令模板

| 做什么 | 对 Claude Code 说什么 |
|--------|---------------------|
| 讲解文档 | `打开 phases/XX/.../docs/ 给我逐段讲解` |
| 解释代码 | `逐行解释 phases/XX/.../code/ 里的代码` |
| 运行代码 | `运行 phases/XX/.../code/ 并解释输出` |
| 做测验 | `用 phases/XX/.../quiz.json 考我，一题一题来` |
| 追问概念 | `刚才讲的 XX 我没懂，换个更简单的例子再讲` |
| 关联知识 | `这节课的 XX 和之前学的 YY 有什么关系` |
| 保存笔记 | `帮我把这节课的核心要点保存到 memory` |

---

> **开始日期：** 2026-06-05
> **目标节奏：** 每天 1-2 小时，工作日推进 1-2 课
> **第一阶段预计完成：** 2-3 周
> **第二阶段预计完成：** 4-6 周
> **第三阶段预计完成：** 4-6 周
