import json
import jieba
from httpx import Client
from jsonrpcserver.response import ErrorResponse
from jsonrpcserver.result import SuccessResult
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from openai import OpenAI
import logging

from jsonrpcserver import method, serve, Result, Success, Error


class JobMatcher:
    def __init__(self, job_json, resume_json):
        self.job_data = json.loads(job_json)
        self.resume_data = json.loads(resume_json)
        self.stopwords = self.load_stopwords()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', )
        self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)

    @staticmethod
    def load_stopwords():
        # 加载中文停用词表
        stopwords = set()
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    stopwords.add(line.strip())
            return stopwords
        except Exception as e:
            logging.error("停用词表加载失败，请检查文件路径。")
            raise e

    def preprocess_text(self, text):
        # 分词
        words = jieba.lcut(text)
        # 去停用词
        words = [word for word in words if word not in self.stopwords and word.strip() != '']
        return ''.join(words)  # Bert 不需要空格分隔

    def get_sentence_vector(self, text):
        # 获取句子的 BERT 向量表示
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            outputs = self.bert_model(**inputs)
            # 使用 [CLS] token 的向量
            sentence_vector = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
            return sentence_vector
        except Exception as e:
            logging.error("BERT 向量计算失败。")
            raise e

    @staticmethod
    def cosine_similarity(vec1, vec2):
        # 计算余弦相似度
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity

    def extract_job_info(self):
        # 提取并合并岗位信息
        try:
            job_title = self.job_data.get('job_title', '')
            job_description = self.job_data.get('job_description', '')
            responsibilities = '，'.join(self.job_data.get('responsibilities', []))
            requirements = self.job_data.get('requirements', {})
            requirements_education = requirements.get('education', '')
            requirements_skills = '，'.join([skill.get('skill_name', '') + skill.get('proficiency', '') for skill in requirements.get('skills', [])])
            requirements_plus = '，'.join(requirements.get('plus', []))
            job_text = '。'.join([job_title, job_description, responsibilities, requirements_education, requirements_skills, requirements_plus])
            return job_text
        except Exception as e:
            logging.error("岗位信息提取失败。")
            raise e

    def extract_resume_info(self):
        # 提取并合并求职者信息
        try:
            education_list = self.resume_data.get('education', [])
            education = '，'.join([edu.get('school', '') + edu.get('degree', '') + edu.get('major', '') for edu in education_list])
            competitions_list = self.resume_data.get('competitions', [])
            competitions = '，'.join([comp.get('name', '') + comp.get('award', '') for comp in competitions_list])
            projects_list = self.resume_data.get('projects', [])
            projects = '，'.join([proj.get('name', '') + proj.get('description', '') + proj.get('highlights', '') for proj in projects_list])
            work_experience_list = self.resume_data.get('work_experience', [])
            work_experience = '，'.join([exp.get('company', '') + exp.get('position', '') + exp.get('responsibilities', '') for exp in work_experience_list])
            skills_list = self.resume_data.get('skills', [])
            skills = '，'.join([skill.get('skill', '') + skill.get('proficiency', '') for skill in skills_list])
            certificates = '，'.join(self.resume_data.get('certificates', []))
            tech_tags = '，'.join(self.resume_data.get('tech_tags', []))
            resume_text = '。'.join([education, competitions, projects, work_experience, skills, certificates, tech_tags])
            return resume_text
        except Exception as e:
            logging.error("简历信息提取失败。")
            raise e

    def match(self):
        # 主匹配函数
        try:
            # 提取信息
            job_text = self.extract_job_info()
            resume_text = self.extract_resume_info()

            # 文本预处理
            job_processed_text = self.preprocess_text(job_text)
            resume_processed_text = self.preprocess_text(resume_text)

            # 计算文本相似度
            job_vec = self.get_sentence_vector(job_processed_text)
            resume_vec = self.get_sentence_vector(resume_processed_text)
            text_similarity = self.cosine_similarity(job_vec, resume_vec)

            # 归一化相似度得分在 0 到 1 之间
            text_similarity = float((text_similarity + 1) / 2)

            # 结构化数据匹配（学历、技能、经验等）
            structured_score = self.match_structured_data()

            # 总评分，确保在 0 到 1 之间
            total_score = float(0.6 * text_similarity + 0.4 * structured_score)

            return {
                'text_similarity': round(text_similarity, 4),
                'structured_score': round(structured_score, 4),
                'total_score': round(total_score, 4),
            }
        except Exception as e:
            logging.error("匹配过程中发生错误。")
            raise e

    def match_structured_data(self):
        # 结构化数据匹配
        score = 0.0
        max_score = 3.0  # 学历、技能、经验各占 1 分

        # 学历匹配
        education_score = 0.0
        job_education = self.job_data.get('requirements', {}).get('education', '')
        resume_education_list = self.resume_data.get('education', [])
        if resume_education_list:
            resume_education = resume_education_list[0].get('degree', '')
            if job_education and resume_education:
                if job_education == resume_education:
                    education_score = 1.0
                else:
                    education_score = 0.5  # 学历不完全匹配，但有相关学历
        score += education_score

        # 技能匹配
        skill_score = 0.0
        job_skills = [skill.get('skill_name', '') for skill in self.job_data.get('requirements', {}).get('skills', [])]
        resume_skills = [skill.get('skill', '') for skill in self.resume_data.get('skills', [])]
        if job_skills and resume_skills:
            matched_skills = set(job_skills) & set(resume_skills)
            skill_match_ratio = len(matched_skills) / len(job_skills)
            skill_score = skill_match_ratio  # 比例得分，最多 1 分
        score += skill_score

        # 工作经验匹配
        experience_score = 0.0
        job_experience_required = self.job_data.get('requirements', {}).get('experience', 0)
        work_experience_list = self.resume_data.get('work_experience', [])
        if work_experience_list:
            # 计算工作经验年数
            total_months = 0
            for exp in work_experience_list:
                start_date = exp.get('start_date', '')
                end_date = exp.get('end_date', '')
                if start_date and end_date:
                    start_year, start_month = map(int, start_date.split('-'))
                    end_year, end_month = map(int, end_date.split('-'))
                    months = (end_year - start_year) * 12 + (end_month - start_month)
                    total_months += months
            total_years = total_months / 12
            if total_years >= job_experience_required:
                experience_score = 1.0
            elif total_years > 0:
                experience_score = total_years / job_experience_required  # 比例得分
            else:
                experience_score = 0.0
        score += experience_score

        # 归一化得分在 0 到 1 之间
        structured_score = score / max_score
        return structured_score


class Grok:
    def __init__(self, base_url, api_key, http_client=None):
        if http_client is None:
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        else:
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=http_client
            )

    def comp(self, duty, prompt):
        completion = self._client.chat.completions.create(
            model="grok-beta",
            messages=[
                {
                    "role": "system",
                    "content": duty,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        return completion.choices[0].message.content

http_client = Client(proxy="http://127.0.0.1:12346")
grok = Grok(base_url="https://api.x.ai/v1", api_key="xai-j2zmXPhm8WjAJ1duNse81HTp7Fp1BaN2sgY9z0JXpiZWgRiOMb31ljDpQxNRGvp0nek37fZEdqS2Tr5s", http_client=http_client)

@method()
def job_parse(job_content) -> Result:
    prompt = """
你是一位专业的简历和职位描述解析助手，擅长将复杂的职位描述（JD）解析为结构化的数据格式。以下是一个职位描述（JD），请将其解析为以下 JSON 格式：

### 输出 JSON 格式：
```json
{
  "job_title": "职位名称",
  "job_description": "职位概述",
  "responsibilities": [
    "职责1",
    "职责2",
    "职责3"
  ],
  "requirements": {
    "education": "学历要求",
    "skills": [
      {
        "skill_name": "技能名称",
        "proficiency": "熟练程度"
      }
    ],
    "plus": [
      "额外的加分项1",
      "额外的加分项2"
    ]
  }
}
```

### 解析要求：
1. **`job_title`**：提取 JD 中的职位名称。
2. **`job_description`**：提取 JD 的整体描述或概述。
3. **`responsibilities`**：提取 JD 中列出的主要职责项，作为一个列表。
4. **`requirements`**：
   - **`education`**：提取 JD 中的学历要求，例如“本科及以上”或“计算机相关专业”。
   - **`skills`**：提取 JD 中列出的技能要求，每个技能包括：
     - `skill_name`：技能名称，例如“Java”或“Python”。
     - `proficiency`：熟练程度，例如“熟练使用”或“掌握”。
   - **`plus`**：提取 JD 中的加分项或优先条件。

### 示例：
输入 JD：
```
我们正在寻找一位软件开发工程师，负责开发和优化系统功能。主要职责包括：
1. 负责软件系统的设计和开发；
2. 排查系统问题并进行优化；
3. 解决技术难题，确保系统稳定性。

要求：
- 本科及以上学历，计算机相关专业；
- 熟悉 Java 和 Python 编程语言；
- 有分布式系统设计经验者优先；
- 熟悉 Docker 和 Kubernetes（加分项）。
```

输出 JSON：
```json
{
  "job_title": "软件开发工程师",
  "job_description": "负责开发和优化系统功能",
  "responsibilities": [
    "负责软件系统的设计和开发",
    "排查系统问题并进行优化",
    "解决技术难题，确保系统稳定性"
  ],
  "requirements": {
    "education": "本科及以上学历，计算机相关专业",
    "skills": [
      {
        "skill_name": "Java",
        "proficiency": "熟悉"
      },
      {
        "skill_name": "Python",
        "proficiency": "熟悉"
      }
    ],
    "plus": [
      "有分布式系统设计经验者优先",
      "熟悉 Docker 和 Kubernetes"
    ]
  }
}
```

### 注意：
- 确保提取的信息完整且准确。
- 如果某些字段在 JD 中未明确提及（例如没有加分项或技能要求），将其设置为空或忽略。
- 保持 JSON 的结构和字段名称完全一致。

**现在，请按照上述要求解析以下 JD：**

    """ + job_content

    duty = "You are a professional resume and job description (JD) parsing assistant, specializing in converting complex job descriptions (JDs) into structured data formats."

    json_content = grok.comp(duty, prompt).strip().replace("```json", "").replace("```", "")
    return Success(json_content)

@method()
def resume_parse(resume_content) -> Result:
    prompt = """
你是一位专业的简历解析助手，擅长将复杂的简历内容解析为结构化的数据格式。以下是目标输出的 JSON 格式以及解析要求，请根据输入的简历内容生成符合格式的 JSON。

### 输出 JSON 格式
```json
{
    "personal_info": {
    "name": "姓名",
    "phone": "电话号码",
    "email": "电子邮箱",
    "wechat": "微信号",
    "blog": "个人博客链接",
    "github": "GitHub 链接"
  },
  "education": [
    {
    "school": "学校名称",
      "degree": "学位",
      "major": "专业",
      "start_date": "开始时间（格式：YYYY-MM）",
      "end_date": "结束时间（格式：YYYY-MM）"
    }
  ],
  "competitions": [
    {
    "name": "竞赛名称",
      "date": "获奖时间（格式：YYYY-MM）",
      "award": "奖项"
    }
  ],
  "projects": [
    {
    "name": "项目名称",
      "start_date": "开始时间（格式：YYYY-MM）",
      "end_date": "结束时间（格式：YYYY-MM）",
      "description": "项目描述",
      "highlights": "项目亮点"
    }
  ],
  "work_experience": [
    {
    "company": "公司名称",
      "position": "职位名称",
      "responsibilities": "工作职责",
      "start_date": "入职时间（格式：YYYY-MM）",
      "end_date": "离职时间（格式：YYYY-MM）"
    }
  ],
  "skills": [
    {
    "skill": "技能名称",
      "proficiency": "熟练程度"
    }
  ],
  "certificates": ["证书名称1", "证书名称2"],
  "tech_tags": ["技术标签1", "技术标签2"]
}
```

### 解析要求
1. **`personal_info`**：提取个人信息，包括姓名、手机号、邮箱、微信号、博客链接和 GitHub 链接。如果某项信息缺失，则跳过该字段。
2. **`education`**：提取教育背景，包括学校名称、学位、专业、开始时间和结束时间。
3. **`competitions`**：提取参加的竞赛信息，包括竞赛名称、获奖时间和奖项。
4. **`projects`**：提取项目经历，包括项目名称、开始时间、结束时间、项目描述和项目亮点。
5. **`work_experience`**：提取工作经历，包括公司名称、职位名称、工作职责、入职时间和离职时间。
6. **`skills`**：提取技能列表，每项技能包括技能名称和熟练程度。
7. **`certificates`**：提取证书信息，输出为一个字符串列表。
8. **`tech_tags`**：提取技术标签，输出为一个列表。

### 示例
输入简历内容：
```
姓名：张三  
电话：13800000000  
邮箱：zhangsan@example.com  
微信：zhangsan_wechat  
博客：https://zhangsan.blog  
GitHub：https://github.com/zhangsan  

教育背景：  
- 清华大学，本科，计算机科学，2015年9月 - 2019年7月  

竞赛经历：  
- ACM 国际大学生程序设计竞赛，2018年5月，银牌  

项目经历：  
- 智能推荐系统，2019年8月 - 2020年6月  
  项目描述：开发了一款基于机器学习的智能推荐系统。  
  项目亮点：提高了用户点击率20%。  

工作经历：  
- 百度，高级研发工程师，2020年7月 - 2023年10月  
  职责：负责搜索算法的优化和维护。  

技能：  
- Python 编程（精通）  

证书：  
- CET-4  
- CET-6  

技术标签：  
- Python  
- 机器学习
```

输出 JSON：
```json
{
    "personal_info": {
    "name": "张三",
    "phone": "13800000000",
    "email": "zhangsan@example.com",
    "wechat": "zhangsan_wechat",
    "blog": "https://zhangsan.blog",
    "github": "https://github.com/zhangsan"
  },
  "education": [
    {
    "school": "清华大学",
      "degree": "本科",
      "major": "计算机科学",
      "start_date": "2015-09",
      "end_date": "2019-07"
    }
  ],
  "competitions": [
    {
    "name": "ACM 国际大学生程序设计竞赛",
      "date": "2018-05",
      "award": "银牌"
    }
  ],
  "projects": [
    {
    "name": "智能推荐系统",
      "start_date": "2019-08",
      "end_date": "2020-06",
      "description": "开发了一款基于机器学习的智能推荐系统。",
      "highlights": "提高了用户点击率20%"
    }
  ],
  "work_experience": [
    {
    "company": "百度",
      "position": "高级研发工程师",
      "responsibilities": "负责搜索算法的优化和维护。",
      "start_date": "2020-07",
      "end_date": "2023-10"
    }
  ],
  "skills": [
    {
    "skill": "Python 编程",
      "proficiency": "精通"
    }
  ],
  "certificates": ["CET-4", "CET-6"],
  "tech_tags": ["Python", "机器学习"]
}
```

### 注意
- 确保提取的信息完整且准确。
- 如果简历中某些字段未提及（如微信号或技术标签），可以省略对应字段。
- 将时间格式统一为 `YYYY-MM`。
- 保持 JSON 格式的字段名称和层级严格一致。

**现在，请根据以下简历内容生成对应的 JSON：**

    """ + resume_content

    duty = "You are a professional resume parsing assistant, specializing in converting complex resume content into structured data formats."

    json_content = grok.comp(duty, prompt).strip().replace("```json", "").replace("```", "")

    return Success(json_content)

@method()
def match(job_json, resume_json) -> Result:
    print(Success(JobMatcher(job_json, resume_json).match()))
    return Success(JobMatcher(job_json, resume_json).match())

if __name__ == '__main__':
    serve(
        port=8899,
    )