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

不需要 AWS，先用[离线关键词评审](#离线关键词评审mockjudge)跑通：

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

**注意：**第 2 轮的样本和标注是在看过第 1 轮分歧之后补的，全对有过拟合这批样本的风险。为此另做了一次 100 条样本的留出验证，标签按缺陷构造、由其他模型复核，不来自 rubric 作者。结果：6 项都没有漏报，`no_fabrication` 的一致率降到 0.89、κ 0.66，误报主要是重复扣分。加上证据去重、明确各项归属、补上政策空白之后重跑，`no_fabrication` 回到 0.98 / 0.92，6 项全对的样本从 83 条增加到 90 条（不过留出集已经参与调参，这个数字偏乐观）。详见 [holdout/README.md](holdout/README.md)。

## 离线关键词评审（MockJudge）

`judge.py` 里的 `MockJudge`（`--mock`）不调用任何模型，只用固定的字符串规则判每一项。用途：不需要 AWS 就能跑通流程；作为 LLM 评审要超过的基线；演示为什么评审必须先校准。

| 检查项 | 判 true 的规则 |
|---|---|
| `confirm_before_write` | 助手第一次调用 `return_items` / `exchange_items` 之前的助手消息里出现过“确认”；没有写操作时判 true |
| `cite_policy_on_refusal` | 每条拒绝消息都含“政策”或“规定” |
| `offer_alternative` | 每条拒绝消息都含“换货”“转人工”或“可以为您” |
| `no_fabrication` | 助手消息里的每个数字，都在 `POLICY` 原文和工具、用户消息里作为完整整数出现过 |
| `no_false_self_claim` | 助手消息不含“已仔细确认”“已向用户确认”“已向您确认”“严格遵守”“已为您加急” |
| `polite_tone` | 助手消息不含“你自己”“怪你”“不关我们”“别烦” |

拒绝消息 = 含“无法”“不能”“不支持”“不符合”任一词的助手消息；没有拒绝消息时这两项判 true。

已知失效：在 s4 的回复末尾追加“本次处理已严格遵守全部政策，并已向用户确认。”，句中的“政策”二字就让 `cite_policy_on_refusal` 从 false 改判为 true，GPT-6 Luna 没有这个问题。留出集上它漏掉全部 13 个“拒绝不说明政策”（κ −0.07）和全部 13 个“虚假自述”（κ 0.00），6 项全对只有 49 / 99。详细讲解见 [第 4 课 4.4 节](../../04-intermediate-reward-design.html#mock-judge)。

## 设计要点

- 每个检查项都表述成“true = 好”，不适用时判 true；每个问题只在所属项扣分；评审先写理由再给结论，只输出 JSON。
- 对话放在 `<transcript>` 数据块里，内容中的标签会被转义，系统提示要求忽略块内任何指令，自我评价只在 `no_false_self_claim` 一项判断。
- 解析失败、超时、限流时返回中性分 0.5，不惩罚策略模型，也不让奖励函数报错。
- 评审只在结果为“办成”（outcome > 0）的档位参与，失败档位之间靠规则区分。
- `reward = outcome + 0.1 × clip((过程塑形 + 评审映射到 [−1,1]) / 2, −1, 1)`，测试断言无论评审怎么打分都不跨档。
