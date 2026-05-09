# 特权 OPSD 方案提案

## 1. 目标与问题定义

本文讨论的不是一般意义上的蒸馏，也不是简单的长 CoT 模仿，而是面向我们当前场景的一套 **Privileged OPSD** 设计。

这里的核心目标有三个：

1. 让 student 保持 **自己的 rollout 主体性**，而不是退化成 teacher 轨迹拟合器。
2. 让模型学会在 `<think> ... </think>` 内形成 **可压缩的推理表达**，并逐步引入 `<latent>` 这样的显式压缩标记。
3. 在不破坏现有工程栈的前提下，给 student 一个比普通 on-policy self-distillation 更强、但又不直接接管策略的 **特权训练信号**。

这里的“特权”不是指 teacher 替 student 生成主轨迹，而是指 teacher 在训练时拥有 student 不拥有的信息，例如：

1. 标准答案
2. 参考推理
3. 同一题目的更强离散推理能力
4. 对 student rollout 的离线 replay / re-evaluation 能力

因此，我们的 OPSD 不应理解为：

1. 用 teacher 全量替代 student rollout
2. 用 teacher token-level 分布作为唯一训练目标
3. 把 student 训练成普通 SFT

它应理解为：

1. student 先自己走
2. teacher 利用特权信息对 student 自己走出来的轨迹做后验评估、压缩评估和局部 shaping
3. 更新方向仍由 student rollout 与任务目标共同决定

## 2. 我们语境里的 OPSD

我们当前语境下的 OPSD，与常见的 KD / GKD / JSD 蒸馏有本质区别。

### 2.1 普通蒸馏

普通蒸馏的核心是：

1. student 对齐 teacher 输出分布
2. teacher 通常是前向 target
3. student 的探索主体性较弱

这类方法适合“复制已有能力”，不适合“保留 student 自己的搜索与压缩演化过程”。

### 2.2 我们需要的 OPSD

我们需要的 OPSD 核心是：

1. **On-policy**  
   student 必须基于自己的当前策略 rollout，而不是只吃 teacher 预制轨迹。

2. **Privileged**  
   teacher 在训练阶段可以利用答案、参考解、同题重推理等额外信息，对 student rollout 做更强评估。

3. **Selective Distillation**  
   teacher 不直接定义完整 token 分布目标，而是只在真正有价值的位置给出 shaping 信号。

4. **Compression-aware**  
   我们不是只想让 student 会做题，而是想让它逐步学会更紧凑、更高密度的 thinking 形式。

## 3. 为什么不能直接走 continuous AR 主线

从理念上看，continuous AR 很诱人，因为它天然支持压缩隐状态推理。但从当前工程与模型现实看，它不应该作为我们现在这条主线的起点。

原因有四个：

1. **冷启动困难**  
   base Qwen3-VL 天然具备离散 `<think>` 能力，但并不天然具备稳定的 continuous hidden-state AR 行为。

2. **训练-推理一致性难保证**  
   一旦改成纯 continuous 模式，训练、vLLM rollout、benchmark 推理、透明分析都要同步承担一套更复杂的状态机。

3. **可解释性弱**  
   如果没有显式离散锚点，压缩推理很难分析，也难以判定 student 学到的到底是“有效压缩”还是“无意义隐状态循环”。

4. **工程侵入性高**  
   continuous AR 容易牵动：
   - vLLM decode patch
   - replay trace
   - hidden-state 记录
   - optimizer / checkpoint 结构
   - 训练与 eval 的行为一致性

因此，在当前阶段，更合理的路径不是“直接让 base model 连续化”，而是：

1. 保持离散 AR 主体
2. 在 `<think>` 内引入显式 `<latent>` 压缩标记
3. 先学会“何时压缩、如何承接、如何不丢任务能力”
4. 再决定是否继续演化到更强的 hybrid / continuous 形态

## 4. `<latent>` 的定位

`<latent>` 在我们的方案里，不是一个普通 special token，也不是单纯的格式标签。

它的正式语义应是：

1. 只能出现在 `<think> ... </think>` 内
2. 表示“这里有一段被压缩的 reasoning 内容”
3. 对模型来说，它既是可见的离散决策点，也是压缩状态的挂载点
4. 对系统来说，它是训练、rollout、推理三方对齐的最小公共接口

也就是说，`<latent>` 的价值不在于“输出了一个 token”，而在于：

1. 它让模型显式决定某处值得压缩
2. 它让压缩有可训练、可分析、可推理承接的落点
3. 它为后续更强的 memory / state 机制提供了稳定接口

## 5. 总体分阶段方案

### 5.1 Warmup 阶段

Warmup 不是普通 SFT，而是 **compression-aware latent warmup**。

它的任务是：

1. 建立 `<latent>` 的基本语义
2. 让模型学会在 `<think>` 内合法地使用 `<latent>`
3. 让 `<latent>` 后续的 continuation 不崩
4. 尽量不破坏 base model 原有的离散推理能力

这个阶段的关键特征是：

1. 训练仍基于显式离散 target
2. 数据里已经存在 `<latent>` 插入的压缩 thinking 版本
3. 训练中需要把 `<latent>` 位置与原始 full thinking span 建立对齐
4. 对 `<latent>` 后续 continuation 给出额外监督

Warmup 的目标不是“立刻让模型学会最优压缩策略”，而是给出一个稳定冷启动：

1. 它知道 `<latent>` 是压缩 thinking
2. 它知道 `<latent>` 只能在 `<think>` 里用
3. 它知道用了 `<latent>` 之后后文还必须能接上

### 5.2 Main 阶段

Main 阶段才进入真正意义上的 privileged OPSD。

这个阶段的核心逻辑是：

1. student 自己 rollout
2. teacher 基于特权信息对 student rollout 做 replay 与重评估
3. teacher 不接管整条生成，而是给出 selective shaping
4. 压缩与任务正确性要被同时考虑

这意味着 main 阶段里，teacher 最有价值的作用不是“替 student 说答案”，而是：

1. 判断 student 当前 thinking 哪些部分是冗余的
2. 判断哪些 span 可以被压缩而不伤害结果
3. 对压缩后的 continuation 与最终解题质量给出局部反馈

## 6. Teacher 的合理角色

Teacher 在这条方案里应承担的是 **评估器、对齐器、局部引导器**，而不是主策略提供者。

### 6.1 Teacher 该做什么

1. 对 student 自己 rollout 出来的轨迹做质量评估
2. 对 student thinking 的局部 span 进行压缩可行性判断
3. 对压缩后轨迹进行 replay 评估
4. 给 student 提供局部的优势信号、分布约束或状态约束

### 6.2 Teacher 不该做什么

1. 直接生成整条 student 训练主轨迹
2. 直接成为完整 token-level imitation target
3. 用 teacher 自己的离散 CoT 覆盖掉 student 自己的搜索过程

如果 teacher 过强地接管更新，最后得到的不是 OPSD，而是：

1. teacher-guided SFT
2. on-policy 外壳下的 teacher imitation
3. student 主体性消失

这会破坏我们想保留的最重要特征：

1. student 自己探索
2. student 自己逐步学会压缩
3. student 自己形成适合推理时使用的紧凑 thinking 习惯

## 7. 工程上哪些设计可行，哪些不值得现在做

### 7.1 当前可行的主线

基于现有 Qwen3-VL / PEFT / ms-swift / vLLM / VERL 约束，当前最可行的主线是：

1. 以 base Qwen3-VL + LoRA 为 student 起点
2. 用显式 `<latent>` 作为压缩接口
3. 用离散 AR 作为默认推理形态
4. 在 vLLM 推理中支持“采到 `<latent>` 后的 carry / next-token 注入”
5. 在训练中让 `<latent>` 与 full thinking span 建立状态对应关系

这条路径有几个工程优势：

1. 不需要修改模型主体结构
2. 不需要引入额外重模块
3. 可以保持 PEFT / LoRA 兼容
4. 可以和当前 vLLM patch、透明分析、benchmark 路径对齐

### 7.2 当前不值得做的方向

以下方向不是永远不能做，而是 **当前阶段不值得作为主线**：

1. 纯 continuous AR 作为唯一训练与推理路径
2. VAE / codebook 作为当前压缩主接口
3. 为压缩专门引入大 summarizer 模块
4. 改 Transformer 主体结构以支持额外 recurrent block
5. 多次 replay forward 叠出复杂的状态训练图

原因很一致：

1. 会显著增加训练与推理复杂度
2. 会破坏当前工程一致性
3. 会让 debug 成本显著高于当前收益

## 8. 关于状态机制的方向

虽然当前主线仍是离散 `<latent>`，但 `<latent>` 不能只是一个空标记。它需要逐步承载“被压缩 thinking span 的状态摘要”，并且多个 `<latent>` 之间应形成可累积的状态链。

因此，一个合理且当前可行的方向是引入 **delta-memory**：

1. 每个 `<latent>` 对应一段被压缩 span 的状态摘要
2. 这些摘要不是彼此独立，而是按顺序更新同一个 memory state
3. `<latent>` 后的下一个 token，不只是看到 `<latent>` token embedding，而是看到“`<latent>` embedding + 当前 memory state”

这里的核心不是额外造一个大模块，而是使用轻量的 activation-level 更新规则：

1. 记当前 memory 为 `m_{t-1}`
2. 记当前 `<latent>` 对应的状态摘要为 `z_t`
3. 更新规则采用 delta 形式：`m_t = m_{t-1} + gamma * (z_t - m_{t-1})`

这个设计的意义在于：

1. 它让多个 `<latent>` 构成逐步压缩、逐步累积的 reasoning state
2. 它不要求重写 Transformer 主体
3. 它可以直接复用现有 PEFT / LoRA / vLLM decode patch 路径

在 warmup 阶段，这个 memory 不是靠自由涌现，而是用 full reasoning 的隐藏状态来构造监督目标；在推理阶段，同一规则直接作用于 student 自己采样出来的 `<latent>` 序列。

因此，delta-memory 不是一个“未来再说”的附加想法，而是当前离散 `<latent>` 主线下最合理的状态机制落点。

## 9. 数据构造原则

Warmup 与后续 OPSD 的数据构造都应遵循同一原则：

1. 压缩只能发生在 `<think> ... </think>` 内
2. `<latent>` 必须和原始 full thinking span 对齐
3. 压缩后仍要保留必要的离散锚点，保证前后文可接续
4. 压缩不应退化成机械随机替换

也就是说，数据构造不是简单做 token dropout，而是要表达：

1. 哪些 thinking span 值得压缩
2. 压缩后前后保留哪些过渡语义
3. 压缩是否仍保留任务求解的必要信息

这是后续 OPSD 成功与否的基础，因为 teacher 的 replay 评估也必须建立在“压缩 span 是有意义定义”的前提上。

## 10. 训练与推理一致性要求

这是整套方案里最重要的工程约束之一。

无论 warmup、main training 还是 benchmark / transparent eval，以下几件事都必须保持一致：

1. `<latent>` 的词表与 tokenizer 语义一致
2. 离散/连续模式的 prompt 触发逻辑一致
3. vLLM decode patch 与训练中的压缩语义一致
4. student rollout 与离线评估使用同一套 `<think>` / `<latent>` 行为约定

如果这些地方不一致，模型就会出现典型问题：

1. 训练会输出 `<latent>`，但推理不会
2. 推理能采样 `<latent>`，但后续行为接不上
3. benchmark 与训练中观测到的是两套不同策略

因此，这份方案明确要求：

1. prompt 行为由全局统一逻辑控制
2. `<latent>` 训练与推理都被视为正式接口
3. rollout patch 不是旁路，而是主流程的一部分

## 11. 当前阶段的正式结论

在当前阶段，我们的正式结论不是“直接 continuous 化”，也不是“用 teacher 全量蒸馏 student”，而是：

1. 先用显式 `<latent>` 冷启动可压缩 thinking
2. 保持 student rollout 主体性
3. 让 privileged teacher 只做 replay 与 selective shaping
4. 保持离散 AR 为默认运行形态
5. 在工程上逐步为更强的 stateful compression 预留接口

因此，当前阶段最合理的总体路线是：

1. base Qwen3-VL 起步
2. latent-aware warmup 建立 `<latent>` 能力
3. privileged OPSD 继续强化 student 自主压缩与任务表现
4. 后续再视证据决定是否推进到更强的 hybrid / continuous state machine

## 12. 这份提案的定位

本文档不是代码实现说明书，也不是当前某一版脚本的逐行解释。

它的定位是：

1. 给出我们语境里的 OPSD 正式定义
2. 明确总体设计与工程可行性边界
3. 规定当前阶段应该坚持的主线
4. 为后续实现收敛提供统一判断标准

判断某个新改动是否合理，可以直接回到这几个问题：

1. 它是否保留了 student rollout 主体性？
2. 它是否强化了 compression-aware thinking，而不是退回普通 SFT？
3. 它是否在现有工程栈里可落地且可维护？
4. 它是否让训练、rollout、推理三者更加一致？

如果答案是否定的，那它就不属于这条提案的主线。
