# Agentic RL 学习课程

一套循序渐进的交互式 HTML 课程，讲解 Agentic Reinforcement Learning（智能体强化学习）：从基础原理到算法、系统、生态，再到客户自建路线，以及基于 AWS 的完整实验设计。

**在线阅读：[liangruibupt.github.io/agentic-rl-learning](https://liangruibupt.github.io/agentic-rl-learning/)**

纯静态页面，不依赖任何外部 CDN。也可以克隆后用浏览器直接打开 `index.html`。

## 课程目录

| 级别 | 页面 | 你会学到 | 交互组件 |
|---|---|---|---|
| — | [index.html](index.html) | 学习路线图，SFT / 单轮 RL / Agentic RL 对比 | |
| 基础班 | [01 什么是 Agentic RL](01-basics-what-is-agentic-rl.html) | 把 LLM 看成策略；单轮与多轮的本质区别；为什么它现在变得重要 | 单轮/多轮切换、小测验 |
| 基础班 | [01b RL 和后训练的关系](01b-basics-rl-and-post-training.html) | 后训练是阶段、RL 是方法；SFT / DPO / RLHF / RLVR / Agentic RL 的数据来源与在线性；InstructGPT、Llama 3、Tulu 3、DeepSeek-R1、Qwen3 的后训练流水线；RL 的收益、争论与风险 | 可点击阶段图、方法切换、在线/同策略 2×2、流水线时间线、小测验 |
| 基础班 | [02 一次训练迭代](02-basics-training-loop.html) | 从取任务到权重同步的 8 个步骤；时间花在哪里 | 逐步动画 |
| 基础班 | [03 GRPO 入门](03-basics-grpo.html) | 组内相对优势；为什么全对或全错的组没有梯度 | 奖励滑块、GRPO / Dr.GRPO 切换 |
| 中级班 | [04 奖励设计](04-intermediate-reward-design.html) | 奖励来源、结果主导 + 有界塑形的推导、LLM 评审奖励设计与校准、防作弊清单（可运行示例见 [labs/llm-judge](labs/llm-judge)） | 复现 Wordle 奖励倒挂案例、评审分跨档演示 |
| 中级班 | [05 多轮训练关键技术](05-intermediate-agentic-techniques.html) | loss mask、信用分配、训推一致、上下文管理与成本、冷启动、格式漂移、监控指标 | token mask 与奖励分配可视化 |
| 进阶班 | [06 算法家族](06-advanced-algorithms.html) | PPO、GRPO、RLOO、DAPO、Dr.GRPO、GSPO、CISPO；离策略修正 | clip 计算器 |
| 进阶班 | [07 系统架构与异步](07-advanced-systems.html) | 系统组件与资源划分；同步 vs 异步 rollout、陈旧度；用 Little 定律估并发；rollout 可靠性；映射到 AWS | 可点击架构图、GPU 空等甘特图、并发估算器 |
| 进阶班 | [08 前沿玩家与方案](08-landscape.html) | 前沿实验室、开源框架、托管服务、公开结果与基准污染、环境与评测（事实核对于 2026-11-11） | 分类筛选、框架需求筛选、选型助手、厂商宣称检查表 |
| 落地班 | [09 客户自建](09-build-your-own.html) | 选型、P0 就绪度评分卡、P0–P6 路线图、环境工程、评测协议、四条路线的成本对比、90 天计划、常见问题 | 选型向导、评分卡、成本估算器 |
| 落地班 | [10 AWS 实验设计](10-aws-experiment.html) | 用 AgentCore + SageMaker AI 多轮 RL + Bedrock 设计一个可复现实验：预注册、奖励、统计功效、rollout 规模、成本 | 可点击架构图、功效计算器、rollout 规模与成本估算器 |
| 落地班 | [11 实操：Bedrock RFT 单轮 RL](11-lab-bedrock-rft.html) | 生成并检查数据（按句式留出的测试集与 OOD 切片）、写奖励 Lambda、提交 GRPO 任务、看懂训练指标、对比基线（代码见 [labs/bedrock-rft](labs/bedrock-rft)） | 流程步进、奖励函数实验台、零方差组模拟、训练曲线诊断、成本与功效计算器 |

## AWS 实验概要（第 10 课）

- **任务**：ShopOps 退换货 / 改单客服智能体，结果以订单数据库最终状态判定。
- **智能体与环境**：Strands 智能体部署在 Amazon Bedrock AgentCore Runtime，每个 rollout 一个隔离会话。
- **训练**：Amazon SageMaker AI 多轮 RL（GRPO + LoRA），底座为 Qwen 3.6 27B（us-west-2）。GPT-OSS-20B 也受支持，但它的 serverless SFT 序列长度固定 4K，放不下约 6,300 token 的完整轨迹，所以 SFT 组和 SFT→RL 组用不了。
- **参照与教师**：Amazon Bedrock 上的前沿模型，兼任 SFT 教师和小权重 LLM 评审。
- **实验组**：基线、提示词优化、SFT、RL、SFT→RL，以及去掉违规惩罚的消融组。
- **指标**：pass^1、pass^4、违规率、完成率、回合数、每千任务成本。
- **B 线**：AgentCore RL Toolkit + verl / slime，跑在 SageMaker HyperPod 或 Amazon EKS 上完全自建。

## 本地预览

```bash
python3 -m http.server 8000 --bind 127.0.0.1
# 打开 http://127.0.0.1:8000
```

## 主要参考资料

- [Multi-Turn Reinforcement Learning on Amazon SageMaker AI: A hands-on guide](https://aws.amazon.com/blogs/machine-learning/multi-turn-reinforcement-learning-on-amazon-sagemaker-ai-a-hands-on-guide/)
- [aws-samples/sample-multi-turn-rl-wordle](https://github.com/aws-samples/sample-multi-turn-rl-wordle)
- [awslabs/agentcore-rl-toolkit](https://github.com/awslabs/agentcore-rl-toolkit)
- [Amazon Bedrock reinforcement fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html)
- [SkyRL-Agent](https://arxiv.org/abs/2511.16108) · [Agent-R1](https://arxiv.org/abs/2511.14460) · [VerlTool](https://arxiv.org/abs/2509.01055)

## 说明

内容基于截至 2026-11 的公开资料整理。产品支持的模型、区域和价格变化很快，落地前请以官方文档为准。外部实验数字（厂商和论文报告的结果）均来自所列公开资料，未经本仓库复现；第 04 课 4.5–4.8 的 LLM 评审校准是本仓库自己在 Amazon Bedrock 上跑的，数据和脚本在 [labs/llm-judge](labs/llm-judge)；[labs/bedrock-rft](labs/bedrock-rft) 的脚本还没有在真实账号上跑过。

## License

[MIT](LICENSE)
