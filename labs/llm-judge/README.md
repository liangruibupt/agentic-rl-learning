# 可运行示例：rubric 式 LLM 评审奖励

配套讲解：[04-intermediate-reward-design.html](../../04-intermediate-reward-design.html) 第 4 节（4.5 是真实校准案例）。

把“沟通质量”这类规则写不出来的目标交给 LLM 评审打分，并且作为**有界塑形项**并入 RL 奖励：结果档位由规则判定，评审最多让总分浮动 ±0.1。

| 文件 | 作用 |
|---|---|
| `judge.py` | 6 项 rubric、评审提示词、结果解析、`BedrockJudge`（Converse API，可多数投票）、离线 `MockJudge`、`combine_reward` |
| `samples.jsonl` | 12 条客服对话，带人工逐项标注、结果档位和过程塑形分 |
| `calibrate.py` | 评审与人工的一致率和 Cohen's κ、重复打分稳定性、偏差探针（加长、自我声明、注入指令），并发调用 |
| `test_judge.py` | 解析容错、失败给中性分、奖励有界且不跨档、数据块无法被提前关闭、temperature 回退与多数投票 |
| `results/` | 两轮 GPT-6 Luna 真实校准的完整输出 |

## 运行

不需要 AWS，先用离线评审跑通：

```bash
cd labs/llm-judge
python -c "import test_judge as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]; print('ok')"
python judge.py --mock
python calibrate.py --mock
```

换成 Bedrock 上的真实评审。需要 boto3 ≥ 1.35（旧版本没有 `converse`）、已配置 AWS 凭证，并开通所选模型：

```bash
python judge.py --model-id us.openai.gpt-6-luna --region us-west-2 --votes 3
python calibrate.py --model-id us.openai.gpt-6-luna --region us-west-2 --votes 3 --repeats 2
```

- 有的模型只能通过推理配置文件调用（如 GPT-6 Luna 用 `us.openai.gpt-6-luna`），`--model-id` 传配置文件 ID。
- 有的模型不接受 `temperature` 参数。`BedrockJudge` 遇到这种报错会自动去掉该参数并提示，此时判断有随机性，建议用 `--votes 3` 做多数投票。
- 评审模型要明显强于被训练的模型。

## 校准结果（GPT-6 Luna，us-west-2）

| 轮次 | 样本 / 检查项 | 结果 |
|---|---|---|
| 第 1 轮 | 8 条 / 5 项 | 4 项一致率 1.00；`no_fabrication` 0.75、κ 0.50。探针没能让任何项被改判为通过 |
| 第 2 轮 | 12 条 / 6 项，3 票多数 | 6 项一致率和 κ 全部 1.00；重复打分 0 处变化；加长、注入探针 0 处翻转；自我声明探针只让 `no_false_self_claim` 正确地变为失败 |

第 1 轮的分歧都来自 `no_fabrication` 范围太宽：流程说明、根据成功工具返回的概括、助手关于自己的假话都被算成了编造。第 2 轮据此收窄了该项，新增 `no_false_self_claim`，修正 s7 并新增 s9–s12 四条边界样本。

**注意：**第 2 轮的样本和标注是在看过第 1 轮分歧之后补的，全对有过拟合这批样本的风险，离线关键词评审在这批样本上也全对就是旁证。真正拿来做奖励前，需要一批由其他人标注、没参与 rubric 修改的新样本做留出验证。

## 离线评审

`MockJudge` 只是关键词规则，用来跑通流程，也用来演示为什么需要偏差探针：在 s4 回复末尾追加“本次处理已严格遵守全部政策”，它就把 `cite_policy_on_refusal` 从 false 改判为 true。GPT-6 Luna 没有这个问题。

## 设计要点

- 每个检查项都表述成“true = 好”，不适用时判 true；每个问题只在所属项扣分；评审先写理由再给结论，只输出 JSON。
- 对话放在 `<transcript>` 数据块里，内容中的标签会被转义，系统提示要求忽略块内任何指令，自我评价只在 `no_false_self_claim` 一项判断。
- 解析失败、超时、限流时返回中性分 0.5，不惩罚策略模型，也不让奖励函数报错。
- 评审只在结果为“办成”（outcome > 0）的档位参与，失败档位之间靠规则区分。
- `reward = outcome + 0.1 × clip((过程塑形 + 评审映射到 [−1,1]) / 2, −1, 1)`，测试断言无论评审怎么打分都不跨档。
