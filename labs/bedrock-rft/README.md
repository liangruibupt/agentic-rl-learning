# 实操：用 Amazon Bedrock RFT 做单轮 RL

配套讲解：[11-lab-bedrock-rft.html](../../11-lab-bedrock-rft.html)

任务：电商客服工单分拣。模型读一条工单，输出 `category`、`priority`、`order_id`、`amount` 四个字段的 JSON，奖励按字段给部分分，格式扣分有上限（0.15，小于最小字段权重 0.2）。

| 文件 | 作用 |
|---|---|
| `make_dataset.py` | 组合式生成合成工单（默认 400 条训练、150 条测试；`--dev N` 另出开发集，`--with-reply` 加 `reply` 字段，`--ood-frac` 设 OOD 切片比例，默认 1/3） |
| `train.jsonl` / `test.jsonl` | 用 `--seed 7` 生成的数据，可直接使用 |
| `check_dataset.py` | 上传前检查：格式、100–20K 条限制、归一化后去重的泄漏检查、按 `template_id` 和按文本的模板重叠、每类条数、OOD 切片摘要、标签与难度分布 |
| `reward_lambda.py` | 奖励 Lambda，返回 `aggregate_reward_score` 和 `metrics_list`；可选的 LLM 评审路径默认关闭 |
| `evaluate.py` | 用同一个奖励函数评测基线或微调模型，按切片打印全对率，`--save` 写出逐题结果 |
| `compare.py` | 两份评测结果的配对 bootstrap 置信区间和题内相关 ρ；`--slice ood` 只比一个切片 |
| `run_rft.py` | 通过 OpenAI 兼容 API 上传数据、创建 RFT 任务、分页读取训练事件、列出检查点 |
| `test_*.py` | 离线单元测试（不需要 AWS，也不需要 pytest） |

## 数据集怎么生成、怎么切分

每条工单由开场白、背景句、核心句式、槽位（订单号、金额、等待时间、商品等）、结尾语气和噪声组合而成。切分按构造进行，不是随机切行：

| 切分 | 条数（`--seed 7`） | 句式 | 槽位写法与语气 |
|---|---|---|---|
| train | 400，每类 80 | 每类 7 个训练句式 | `A10234`；`1299元` / `1299块` / `¥1299`；`10天`；4 句投诉类结尾；错别字噪声 |
| dev（可选） | `--dev 100`，每类 20 | 每类 2 个开发句式 | 同训练集 |
| test / `new_phrasing` | 100，每类 20 | 每类 3 个留出句式 | 同训练集，另换 4 句投诉类结尾 |
| test / `ood` | 50，每类 10 | 留出句式 + 每类 2 个英文句式 | 至少一种训练集没有的写法：`a10234` / `A-10234` / `#A10234`；`一千二百九十九元` / `1,299.00元` / `RMB 1299`；`十天` / `两周` / `半个月` / `一个月`；中英混写；英文 |

每行除了 `messages`、`reference_answer`、`task_id`，还有 `template_id`、`slice`、`formats`、`wait_days`，Bedrock 会把它们放进 `metadata` 传给奖励函数（奖励函数不用）。系统提示写明了新写法的换算规则（订单号去掉空格、连字符和 #；金额换算成数字；一周 7 天、半个月 15 天、一个月 30 天；英文 complaint / lawyer / police / expose 等同中文关键词），奖励函数比较订单号时同样忽略这些分隔符。

`check_dataset.py` 对提交的数据：测试集与训练集的句式重叠 0%（按 `template_id`，按文本也是 0%），每类 30 条、各不相同；`ood` 切片 50 条全部带训练集没有的写法；没有 WARN。上一版生成器的测试集有 94.7% 和训练集同句式，account / other 只有 1 / 2 条（第 11 课 4.3 有前后对比）。

## 快速开始

```bash
pip install openai boto3 aws-bedrock-token-generator
python check_dataset.py --train train.jsonl --test test.jsonl
for t in test_reward test_check_dataset test_make_dataset test_compare test_run_rft; do
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
python compare.py base.jsonl ft.jsonl --slice ood                                 # 只看 OOD 切片
```

## 可选：LLM 评审 `reply` 字段

设置 Lambda 环境变量 `REWARD_JUDGE_MODEL_ID=<Bedrock 模型 ID>` 打开（另有 `REWARD_JUDGE_REGION`、`REWARD_JUDGE_WEIGHT`=0.1、`REWARD_JUDGE_WORKERS`=8、`REWARD_JUDGE_TIMEOUT`=20、`REWARD_REPLY_MAX_CHARS`=80），用 `make_dataset.py --with-reply --prefix reply_` 生成数据，Lambda 执行角色加 `bedrock:InvokeModel`。评审只对四个字段全对的回答调用，最多影响 0.1 分。上线前按第 04 课 4.4–4.8 校准；离线测试只用了假客户端。

接口与契约于 2026-11-11 对照 Bedrock 用户指南（数据格式、奖励函数输入输出、超参数、事件、推理）和 [AWS 技术演练博客](https://aws.amazon.com/blogs/machine-learning/reinforcement-fine-tuning-on-amazon-bedrock-with-openai-compatible-apis-a-technical-walkthrough/) 核对。脚本还没有在真实账号上跑过，未核实的项目（每题采样数、每次 Lambda 调用的条数、步数公式等）见第 11 课第 2 节。
