# ResumeParser

## 介绍
C# 课程设计 - 智能简历解析系统，解析和人岗匹配部分。

## 拉取 bert-base-chinese

```bash
git clone https://huggingface.co/google-bert/bert-base-chinese
```

### 简历解析 & 岗位JD解析

利用合适的 `prompt` 使用 `Grok-beta 大模型` 完成解析任务，得到 `json` 结构化数据。

### 人岗匹配

采用语义相似度和结构化数据相似度结合的方式实现，使用 `google-bert/bert-base-chinese` 预训练模型。

语义相似度(TextSimilarity)：通过预处理文本(去除无用信息(如姓名等)，去除停用词)后计算得出其语义向量，之后计算其余弦相似度得到一个 $[-1, 1]$ ，对其归一化处理得到 $[0, 1]$ 的相似度。

结构化相似度(StructuredScore)：通过解析后的结构化数据进行对教育背景、技术栈、工作经验的条件匹配，每个部分 1 分，总分 3 分，加权处理得到一个 $[0, 1]$ 的相似度。

最终结果为 $ 0.6 \times TextSimilarity + 0.4 \times StructuredScore $ 。

