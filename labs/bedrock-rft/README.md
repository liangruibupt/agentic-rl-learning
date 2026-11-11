# 实操：用 Amazon Bedrock RFT 做单轮 RL

配套讲解：[11-lab-bedrock-rft.html](../../11-lab-bedrock-rft.html)

任务：电商客服工单分拣。模型读一条工单，输出 `category`、`priority`、`order_id`、`amount` 四个字段的 JSON，奖励按字段给部分分，格式扣分有上限（0.15，小于最小字段权重 0.2）。

| 文件 | 作用 |
|---|---|
| `make_dataset.py` | 生成合成数据集（默认 400 条训练、150 条测试；`--dev N` 另出开发集，`--with-reply` 加 `reply` 字段） |
| `train.jsonl` / `test.jsonl` | 用 `--seed 7` 生成的数据，可直接使用 |
| `check_dataset.py` | 上传前检查：格式、100–20K 条限制、归一化后去重的泄漏检查、同模板比例、标签与难度分布 |
| `reward_lambda.py` | 奖励 Lambda，返回 `aggregate_reward_score` 和 `metrics_list`；可选的 LLM 评审路径默认关闭 |
| `evaluate.py` | 用同一个奖励函数评测基线或微调模型，`--save` 写出逐题结果 |
| `compare.py` | 两份评测结果的配对 bootstrap 置信区间和题内相关 ρ |
| `run_rft.py` | 通过 OpenAI 兼容 API 上传数据、创建 RFT 任务、分页读取训练事件、列出检查点 |
| `test_*.py` | 离线单元测试（不需要 AWS，也不需要 pytest） |

## 快速开始

```bash
pip install openai boto3 aws-bedrock-token-generator
python check_dataset.py --train train.jsonl --test test.jsonl
for t in test_reward test_check_dataset test_compare test_run_rft; do
  python -c "import $t as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]; print('$t ok')"
done

zip reward.zip reward_lambda.py
aws lambda create-function --function-name ticket-reward \
  --runtime python3.12 --handler reward_lambda.lambda_handler \
  --zip-file fileb://reward.zip --timeout 300 --memory-size 512 \
  --role arn:aws:iam::<ACCOUNT>:role/<LAMBDA_EXEC_ROLE> --region us-west-2
# 调用 RFT 的身份需要对该函数的 lambda:InvokeFunction（见 Bedrock 用户指南）

python evaluate.py --model openai.gpt-oss-20b --runs 3 --save base.jsonl          # 基线
python run_rft.py --lambda-arn arn:aws:lambda:us-west-2:<ACCOUNT>:function:ticket-reward --save-events events.json
python evaluate.py --model <FINE_TUNED_MODEL_ID> --runs 3 --save ft.jsonl         # 微调后
python compare.py base.jsonl ft.jsonl
```

## 可选：LLM 评审 `reply` 字段

设置 Lambda 环境变量 `REWARD_JUDGE_MODEL_ID=<Bedrock 模型 ID>` 打开（另有 `REWARD_JUDGE_REGION`、`REWARD_JUDGE_WEIGHT`=0.1、`REWARD_JUDGE_WORKERS`=8、`REWARD_JUDGE_TIMEOUT`=20、`REWARD_REPLY_MAX_CHARS`=80），用 `make_dataset.py --with-reply --prefix reply_` 生成数据，Lambda 执行角色加 `bedrock:InvokeModel`。评审只对四个字段全对的回答调用，最多影响 0.1 分。上线前按第 04 课 4.4–4.8 校准；离线测试只用了假客户端。

接口与契约于 2026-11-11 对照 Bedrock 用户指南（数据格式、奖励函数输入输出、超参数、事件、推理）和 [AWS 技术演练博客](https://aws.amazon.com/blogs/machine-learning/reinforcement-fine-tuning-on-amazon-bedrock-with-openai-compatible-apis-a-technical-walkthrough/) 核对。脚本还没有在真实账号上跑过，未核实的项目（每题采样数、每次 Lambda 调用的条数、步数公式等）见第 11 课第 2 节。
