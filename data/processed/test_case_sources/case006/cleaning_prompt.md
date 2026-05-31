# Cleaning Prompt

您需要基于我给定的真实案例(PDF and meta infomations)，生成一份教学用的模拟案例，并严格遵循下列要求：

- 模拟案例是正式开庭前已有的信息，作为庭审时双方进行询问、辩论的事实证据基础。仅包含真实案例中的基本情况、事实、双方主张等基本信息。如果原始材料中包含了关键的合同条款、证人证言、专家报告等信息，也可也一并包含在模拟案例中；不应该包含庭审时、庭审后才会出现的信息（如具体庭审辩论过程、最终裁定结果等）。但是可以保留庭审过程中出现的事实性陈述，如对案件事实的描述和补充、可信的证人证言等；
- 不要虚构真实案件未出现的事实。
- 模拟案例的内容，应当包含下列内容。如果没有相关的内容，请标注[N/A]。如果包含清单未提及的材料，也可也补充。


## 一、申请人需准备和提交的材料
### （一）仲裁通知
根据HKIAC规则第4条，仲裁程序在申请人向HKIAC及被申请人发出仲裁通知之时正式启动。仲裁通知须包含以下十项内容：

1. 将争议提交仲裁的请求；
2. 各方当事人及其代理人的名称、地址、电话及传真号码及/或电邮地址；
3. 仲裁协议的副本；
4. 引发争议的合约或其他法律文件的副本；
5. 仲裁请求的性质的描述及所涉金额（如有）；
6. 寻求的救济或补救；
7. （如当事人事前未约定仲裁员人数）建议的仲裁员人数（即一名或三名）；
8. 申请人就仲裁员人数的建议及任何相关意见；
9. 披露是否有任何第三方出资；
10. 确认仲裁通知已送达。

### （二）仲裁申请书
如仲裁通知并未随附仲裁申请书，申请人须在仲裁庭指定的期限内提交仲裁申请书，并附具支持其申请的全部文件和证据材料。

### （三）支持性证据材料
双方均须提交支持其申请或答辩的证明文件或材料。如原始合同、原始协议、证言、法律专家报告等内容
证据材料一般要求复印清晰、连续标明页码，以便仲裁庭和被申请人查阅。

### （四）主体资格证明材料
申请人还需提交自身的主体资格证明材料。如公司注册证书、商业登记证等文件。

## 二、被申请人需准备和提交的材料
### （一）对仲裁通知的答复
根据HKIAC规则，被申请人须于收到仲裁通知后30日内作出答复，就申请人的请求作出回应及（如适用）提议仲裁员。如有反请求或答辩，亦应一并提出。

### （二）答辩书及支持性证据
如被申请人在对仲裁通知的答复中未附具答辩书，须在仲裁庭指定期限内提交答辩书及相关证据材料（如原始合同、原始协议、证言、法律专家报告等内容）

### 三、证人及专家证人的准备
各方在开庭前需准备：
- 证人证言的完整书面陈述；
- 专家报告的完整书面意见；


您需要以材料的形式，生成上述提到的所有材料（如有），也可也补充其他的必要材料。
例如：
```markdown

# [案件名称]

[案件背景事实]

## 材料清单
[下面是一个示例的清单]
| 类别 | 文件名称 | 提交方 |
|------|----------|--------|
| 启动文件 | 仲裁申请书 | 申请人 |
| 证据材料 | 合同文本（附件C1,C2,...） | 申请人 |
| 证据材料 | 证人证言（附件R1,R2,....） | 双方 |
| 证据材料 | 专家报告（附件R3, R4,....） | 双方 |

## 材料1： 仲裁申请书
....

## 附件C1 XXXX合同（摘录）
....

```

材料可参考下面的格式生成：

```plain text
## 附件C1： XXXXX房地产有限公司 与 YYYYY酒店管理公司 签订的合同文本（摘录）

经营管理合同 （摘录）
甲方：XXXXX房地产有限公司 
乙方：YYYYY酒店管理公司 
鉴于乙方拥有丰富管理高级酒店的经验，经甲乙双方友好协商，甲方与乙方就
现代大酒店经营管理的事宜达成如下协议： 
...

2.3 
乙方须以甲方的利益为依归，为酒店进行有效率的现代化管理，
并须履行下列各项职责： 
(a) 根据市场发展情况适时拟定相关的经营战略，打造符合高档酒店的国
际品牌； 
(b) 设定有效的内外部营销推广方略，开展全面推广工作，尽可能的使
酒店的盈利比例增大； 
(c) 按照与甲方确定的人力架构以及薪酬方案为准绳，进行人员招聘、培...
[请注意，材料中的内容必须基于真实案例，不要虚构其他内容]

```

## Case Metadata

## Metadata
- 案件标题 (Case Title): Beijing Shangye v. UCFTI and Zuohong Chen
- 案号 (Case Number): No. 4
- 程序性质 (Nature): International
- 案件类型 (Type): Commercial Arbitration
- 引入日期 (Date of Introduction): 11 Oct 2019
- 案件状态 (Status): Concluded
- 申请人国籍 (Claimant): China
- 被申请人国籍 (Respondent): United States
- 仲裁机构 (Institution): BAC/BIAC (Beijing Arbitration Commission/Beijing International Arbitration Court)
- 仲裁规则 (Rules): BAC/BIAC Arbitration Rules 2019
- 仲裁地 (Seat): Beijing
- 适用法律 (Applicable Law): China
- 案件文档 (Documents): Arbitration Award - 7 Dec 2021
- 更新日期 (Updated On): 21 May 2025
- PDF 字符数: 18191
- PDF 页数: 7
- PDF 保存路径: C:\Users\micky\Desktop\demoproject\grab\output\pdfs\beijing-shangye-v-ucfti-and-zuohong-chen.pdf
- 案例 URL: https://jusmundi.com/en/document/decision/en-beijing-shangye-film-and-television-culture-media-co-ltd-v-ucfti-inc-and-zuohong-chen-order-of-the-united-states-district-court-for-the-district-of-central-california-granting-petition-to-confirm-and-enforce-foreign-arbitration-application-for-default-judgment-tuesday-6th-may-2025?su=%2Fen%2Fsearch%3Fcase_applicable-law%3D1073%26case_arbitration-rule%3D1050674%2C1050669%2C1050573%26document-types%3Dcase%26lang%3Den
- local_pdf_path: C:\Users\micky\Desktop\demoproject\grab\output\pdfs\beijing-shangye-v-ucfti-and-zuohong-chen.pdf
- source_pdf_name: beijing-shangye-v-ucfti-and-zuohong-chen.pdf
- source_rows: 11, 12
- deduplicated_row_count: 2
- case_urls:
  https://jusmundi.com/en/document/decision/en-beijing-shangye-film-and-television-culture-media-co-ltd-v-ucfti-inc-and-zuohong-chen-order-of-the-united-states-district-court-for-the-district-of-central-california-granting-petition-to-confirm-and-enforce-foreign-arbitration-application-for-default-judgment-tuesday-6th-may-2025?su=%2Fen%2Fsearch%3Fcase_applicable-law%3D1073%26case_arbitration-rule%3D1050674%2C1050669%2C1050573%26document-types%3Dcase%26lang%3Den
  https://jusmundi.com/en/document/decision/en-beijing-shangye-film-and-television-culture-media-co-ltd-v-ucfti-inc-and-zuohong-chen-order-of-the-united-states-district-court-for-the-district-of-central-california-granting-petition-to-confirm-and-enforce-foreign-arbitration-application-for-default-judgment-tuesday-6th-may-2025#pa_2585143
- case_documents: Arbitration Award - 7 Dec 2021

## Source Text

See source_text.md in this directory. The generation CLI reads the same extracted text directly.
