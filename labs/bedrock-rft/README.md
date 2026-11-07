# 实操：用 Amazon Bedrock RFT 做单轮 RL

配套讲解：[11-lab-bedrock-rft.html](../../11-lab-bedrock-rft.html)

任务：电商客服工单分拣。模型读一条工单，输出 `category`、`priority`、`order_id`、`amount` 四个字段的 JSON，奖励按字段给部分分。

| 文件 | 作用 |
|---|---|
| `make_dataset.py` | 生成合成数据集（默认 400 条训练、150 条测试），格式符合 Bedrock RFT 开源权重模型要求 |
| `train.jsonl` / `test.jsonl` | 用 `--seed 7` 生成的数据，可直接使用 |
| `reward_lambda.py` | 奖励 Lambda，返回 `aggregate_reward_score` 和各字段得分 |
| `test_reward.py` | 奖励函数的单元测试，部署前必跑 |
| `run_rft.py` | 通过 OpenAI 兼容 API 上传数据、创建 RFT 任务、轮询训练指标 |
| `evaluate.py` | 用同一个奖励函数评测基线或微调后的模型 |

## 快速开始

```bash
pip install openai boto3 aws-bedrock-token-generator
python -c "import test_reward as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]; print('ok')"

zip reward.zip reward_lambda.py
aws lambda create-function --function-name ticket-reward \
  --runtime python3.12 --handler reward_lambda.lambda_handler \
  --zip-file fileb://reward.zip --timeout 300 --memory-size 512 \
  --role arn:aws:iam::<ACCOUNT>:role/<LAMBDA_EXEC_ROLE> --region us-west-2
# 按 Bedrock 用户指南配置 Bedrock 调用该 Lambda 的权限

python evaluate.py --model openai.gpt-oss-20b --runs 3              # 基线
python run_rft.py --lambda-arn arn:aws:lambda:us-west-2:<ACCOUNT>:function:ticket-reward
python evaluate.py --model <FINE_TUNED_MODEL_ID> --runs 3           # 微调后
```

API 调用方式参照 AWS 博客 [Reinforcement fine-tuning on Amazon Bedrock with OpenAI-Compatible APIs](https://aws.amazon.com/blogs/machine-learning/reinforcement-fine-tuning-on-amazon-bedrock-with-openai-compatible-apis-a-technical-walkthrough/)，奖励函数的输入输出契约参照 [aws-samples 示例](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/custom-models/bedrock-reinforcement-fine-tuning)。这两个脚本还没有在真实账号上跑过，运行前请对照最新的 Bedrock 用户指南核对参数。
