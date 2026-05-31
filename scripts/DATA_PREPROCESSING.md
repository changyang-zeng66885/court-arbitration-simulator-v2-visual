# 数据预处理说明

本项目的数据预处理脚本位于 `scripts/preprocess_cases.py`。默认模式不会调用 LLM，只会完成：

- 读取 `original_case_raw/jusmundi_cases.xlsx`
- 去重 Excel 中的重复案例记录
- 将 Excel 里的旧 PDF 路径映射到本项目的 `original_case_raw/pdfs/`
- 抽取 PDF 文本
- 生成严格分离的 RAG 案例集和测试案例源材料
- 写入 manifest、split 文件，并用 PDF/text 哈希检查数据泄露

## 离线预处理

```powershell
python scripts\preprocess_cases.py
```

默认输出：

- `data/processed/manifest.json`：所有可用案例、跳过原因、哈希、元信息
- `data/processed/splits.json`：RAG/test 拆分结果
- `data/processed/rag_corpus/caseXXX/`：公开案例数据库（RAG）源材料
- `data/processed/test_case_sources/caseXXX/`：测试案例的真实源材料和清洗提示词

默认拆分使用 `--seed 530 --test-size 2`。当前默认结果是：

- RAG：`case001`, `case003`, `case004`, `case005`, `case007`, `case008`, `case009`
- 测试：`case002`, `case006`

## 指定测试案例

```powershell
python scripts\preprocess_cases.py --test-cases case002 case006
```

也可以指定 RAG 案例，剩余案例自动作为测试案例：

```powershell
python scripts\preprocess_cases.py --rag-cases case001 case003 case004
```

脚本会拒绝 RAG/test 重叠，并检查相同 PDF 哈希或抽取文本哈希是否同时出现在两边。

## 生成中文模拟案例

如需直接调用 OpenAI-compatible API 生成 `inputs/test_case/caseXXX/main.md`：

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_MODEL="qwen3.6-flash"
python scripts\preprocess_cases.py --generate-test-cases --overwrite
```

脚本会读取 `原始数据清洗prompt.md`，结合测试案例 PDF 文本生成中文庭前模拟案例。默认每个案例最多向模型发送 120000 个字符，可用 `--max-source-chars` 调整。

