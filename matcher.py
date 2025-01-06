import json
import logging

import jieba
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class Matcher:
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