# 单一案件多轮串行模拟

入口脚本：`scripts/simulate_case.py`

## 功能

该脚本实现 `PROJECT-0530.md` 中“单一案件的多轮、串行模拟”流程：

1. 基于静态案件材料，申请方和被申请方分别生成私有初始策略。
2. 每一轮依次模拟两个公开盘问环节：
   - 申请方律师盘问被申请方证人；
   - 被申请方律师盘问申请方证人。
3. 每轮公开盘问结束后，申请方和被申请方依次作本轮最后陈述。
4. 最后陈述结束后，点评 Agent 只基于当前轮公开问答、双方最后陈述和仲裁庭意见，对申请方/被申请方分别点评并评分。
5. 点评完成后，仅当前训练方基于公开问答和本轮点评进行内部复盘，并更新下一轮策略；另一方策略保持冻结。
6. 训练方按 `--strategy-block-size` 分块交替：`--position` 指定的一方先连续学习 k 轮，然后对方连续学习 k 轮，如此循环。
7. N 轮结束后，只为 `--position` 指定的一方生成庭前建议。

输出文件只保存公开问答和所选阵营自己的私有策略/复盘；对方私有策略和复盘只在运行内存中使用，不写入所选阵营产物。

注意：多轮记录是庭前训练中的多次“重开演练”，不是同一场真实庭审中连续发生、不可撤回的问答。某一轮中的不利回答会被视为训练暴露出的风险，用于修正后续策略，而不是自动成为最终庭审口径。

## 离线验证

不调用模型，只检查流程和输出格式：

```bash
python scripts/simulate_case.py ^
  --case-doc inputs/test_case/case001/main.md ^
  --position claimant ^
  --rounds 1 ^
  --qa-pairs 1 ^
  --dry-run ^
  --overwrite
```

## 调用 OpenAI 兼容模型

PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxx"
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_MODEL="deepseek-v4-falsh"

python scripts\simulate_case.py `
  --case-doc inputs\test_case\case001\main.md `
  --position claimant `
  --rounds 2 `
  --qa-pairs 2 `
  --strategy-block-size 2 `
  --overwrite
```

可选阵营：

- `--position claimant`：生成 `申请方庭前建议.md`
- `--position respondent`：生成 `被申请方庭前建议.md`

## 主要参数

- `--rounds`：多轮重生式模拟轮数，默认 3。
- `--qa-pairs`：每个盘问环节的问题-回答组数，默认 2。
- `--strategy-block-size` / `--learning-block-size`：策略交替学习的块大小 k，默认 1。比如 `--strategy-block-size 3` 表示所选阵营先学习 3 轮、对手策略冻结；然后对手学习 3 轮、所选阵营策略冻结。
- `--disable-rag`：关闭本地 RAG 检索。默认会从 `ref_rules_doc` 和 `data/processed/rag_corpus` 建立轻量关键词索引，并把检索结果注入各 Agent 的提示词。
- `--rag-top-k`：每次 Agent 调用注入的 RAG 文本块数量，默认 4。
- `--rag-max-context-chars`：每次 Agent 调用注入的 RAG 检索结果字符上限，默认 5000。
- `--rag-test-query`：只构建 RAG 索引并打印指定查询的检索结果，用于离线检查知识库是否可用。
- `--skip-tribunal`：跳过每个盘问环节后的中立仲裁庭点评，可减少模型调用次数。
- `--max-case-chars`：每次提示词中放入的案件材料字符上限，默认 60000。
- `--max-history-chars`：每次提示词中放入的公开历史字符上限，默认 30000。
- `--outputs-dir`：输出根目录，默认 `outputs/test_case`。
- `--events-path`：可选 JSONL 事件流路径。用于 Streamlit 页面实时展示律师提问、证人回答、仲裁庭点评、点评 Agent 评分、复盘和策略更新。

## RAG 工具

当前实现是“检索注入式工具调用”，不是 OpenAI function-calling 协议。脚本会在每次 Agent 生成前，根据案件材料、当前问题、策略和公开历史检索：

- 法律条文知识库：`ref_rules_doc`
- 公开案例数据库：`data/processed/rag_corpus`

如果 Agent 使用了 RAG 内容，应按项目需求在输出中写明：

```markdown
> [法律条文知识库（RAG）, chunk_id] 原文：...
```

## 输出

以 `case001` 和 `claimant` 为例：

- `outputs/test_case/case001/申请方庭前建议.md`
- `outputs/test_case/case001/训练过程总结.md`
- `outputs/test_case/case001/申请方模拟记录.json`

`训练过程总结.md` 包含：

- 每轮点评 Agent 对申请方/被申请方的评分和点评；
- 每轮训练方名称；
- 每轮训练方的总结复盘；
- 每轮训练方更新后的盘问/应答策略。

`申请方模拟记录.json` 包含公开盘问记录、双方最后陈述、点评 Agent 记录、训练方更新记录，以及所选阵营自己的策略版本和复盘。

庭前建议包含：

- 我方律师的提问策略；
- 我方证人的回答策略；
- 需要我方律师/证人关注的风险点；
- 结合前 N 轮训练的具体例子；
- 庭前行动清单。
