import sys
import os
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

# 添加G-Eval路径
sys.path.append('./geval')

class CustomGEvalRunner:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化G-Eval运行器"""
        self.config = config or self.get_default_config()
        self.evaluation_results = {}
        
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'api_type': 'openai',  # 或 'huggingface', 'local'
            'model_name': 'gpt-3.5-turbo',
            'evaluation_aspects': [
                'coherence',      # 话题连贯性
                'consistency',    # 话题一致性  
                'fluency',        # 流畅性
                'relevance',      # 相关性
                'diversity'       # 多样性
            ],
            'scoring_scale': 5,   # 评分范围1-5
            'model_config': {
                'openai': {
                    'api_key': os.getenv('OPENAI_API_KEY', ''),
                    'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                },
                'huggingface': {
                    'model_name': 'microsoft/DialoGPT-medium',
                    'api_key': os.getenv('HUGGINGFACE_API_KEY', '')
                }
            }
        }
    
    def prepare_geval_input(self, topicgpt_results_path: str, 
                           original_docs_path: str) -> Dict[str, Any]:
        """将TopicGPT结果转换为G-Eval输入格式"""
        
        print("📋 准备G-Eval输入数据...")
        
        # 读取TopicGPT结果
        with open(topicgpt_results_path, 'r', encoding='utf-8') as f:
            topic_results = json.load(f)
        
        # 读取原始文档
        with open(original_docs_path, 'r', encoding='utf-8') as f:
            original_docs = [json.loads(line) for line in f]
        
        # 构建G-Eval输入格式
        geval_input = {
            "task_description": "评估话题建模结果的质量",
            "topics": topic_results.get('topics', {}),
            "assignments": topic_results.get('assignments', {}),
            "documents": original_docs,
            "evaluation_aspects": self.config['evaluation_aspects'],
            "metadata": topic_results.get('metadata', {})
        }
        
        print(f"✅ G-Eval输入准备完成，包含 {len(geval_input['topics'])} 个话题和 {len(original_docs)} 个文档")
        return geval_input
    
    def run_evaluation(self, geval_input: Dict[str, Any]) -> Dict[str, Any]:
        """运行G-Eval评估"""
        print("📊 开始G-Eval评估...")
        
        evaluation_results = {}
        
        for aspect in geval_input['evaluation_aspects']:
            print(f"评估维度: {aspect}")
            
            # 构建评估prompt
            eval_prompt = self.build_evaluation_prompt(
                aspect, 
                geval_input['topics'], 
                geval_input['assignments'],
                geval_input['documents']
            )
            
            # 调用模型进行评估
            score, reasoning = self.evaluate_aspect(eval_prompt, aspect)
            
            evaluation_results[aspect] = {
                "score": score,
                "reasoning": reasoning,
                "prompt": eval_prompt
            }
        
        # 计算总分和加权分数
        overall_score = self.calculate_overall_score(evaluation_results)
        evaluation_results['overall_score'] = overall_score
        
        # 添加评估元数据
        evaluation_results['metadata'] = {
            'evaluation_date': datetime.now().isoformat(),
            'model_used': self.config['model_name'],
            'scoring_scale': self.config['scoring_scale'],
            'total_aspects': len(geval_input['evaluation_aspects'])
        }
        
        print(f"✅ 评估完成，总分: {overall_score:.2f}")
        return evaluation_results
    
    def evaluate_aspect(self, prompt: str, aspect: str) -> tuple:
        """评估特定维度"""
        try:
            response = self.call_model(prompt)
            score, reasoning = self.parse_evaluation_response(response, aspect)
            return score, reasoning
        except Exception as e:
            print(f"评估维度 {aspect} 时出错: {e}")
            return 3.0, f"评估失败: {str(e)}"
    
    def call_model(self, prompt: str) -> str:
        """调用模型"""
        api_type = self.config['api_type']
        
        if api_type == 'openai':
            return self.call_openai_model(prompt)
        elif api_type == 'huggingface':
            return self.call_huggingface_model(prompt)
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
    
    def call_openai_model(self, prompt: str) -> str:
        """调用OpenAI模型"""
        try:
            import openai
            openai.api_key = self.config['model_config']['openai']['api_key']
            openai.base_url = self.config['model_config']['openai']['base_url']
            
            response = openai.ChatCompletion.create(
                model=self.config['model_name'],
                messages=[
                    {"role": "system", "content": "你是一个专业的话题建模质量评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return self.get_fallback_response()
    
    def call_huggingface_model(self, prompt: str) -> str:
        """调用HuggingFace模型"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config['model_config']['huggingface']['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.3
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.config['model_config']['huggingface']['model_name']}",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()[0]['generated_text']
            else:
                print(f"HuggingFace API调用失败: {response.status_code}")
                return self.get_fallback_response()
                
        except Exception as e:
            print(f"HuggingFace API调用失败: {e}")
            return self.get_fallback_response()
    
    def get_fallback_response(self) -> str:
        """获取备用响应"""
        return "评分: 3.0\n理由: 由于模型调用失败，给出中等评分。"
    
    def build_evaluation_prompt(self, aspect: str, topics: Dict[str, Any], 
                               assignments: Dict[str, Any], documents: List[Dict[str, Any]]) -> str:
        """构建特定维度的评估prompt"""
        
        aspect_prompts = {
            "coherence": """
            评估以下话题建模结果中每个话题的内部连贯性。
            
            评估标准：
            1. 话题内的关键词是否语义相关
            2. 话题描述是否与关键词一致
            3. 分配到该话题的文档是否符合话题主题
            4. 话题内部逻辑是否清晰
            
            请对每个话题的连贯性进行1-5分评分，并给出详细理由。
            """,
            
            "consistency": """
            评估话题建模结果的整体一致性。
            
            评估标准：
            1. 不同话题之间是否有明确区分
            2. 话题标签和描述的命名是否一致
            3. 话题分配是否合理无重叠
            4. 整体话题体系是否协调
            
            请给出1-5分的整体一致性评分和详细理由。
            """,
            
            "fluency": """
            评估话题建模结果的流畅性。
            
            评估标准：
            1. 话题描述是否自然流畅
            2. 关键词表达是否准确
            3. 话题标题是否简洁明了
            4. 整体表达是否易于理解
            
            请给出1-5分的流畅性评分和详细理由。
            """,
            
            "relevance": """
            评估话题建模结果与原始文档的相关性。
            
            评估标准：
            1. 话题是否准确反映了文档内容
            2. 话题分配是否合理
            3. 话题覆盖面是否全面
            4. 是否遗漏了重要主题
            
            请给出1-5分的相关性评分和详细理由。
            """,
            
            "diversity": """
            评估话题建模结果的多样性。
            
            评估标准：
            1. 话题是否涵盖了不同的主题领域
            2. 话题之间是否有足够的差异性
            3. 是否避免了话题重复
            4. 话题分布是否均衡
            
            请给出1-5分的多样性评分和详细理由。
            """
        }
        
        base_prompt = aspect_prompts.get(aspect, "请评估话题建模结果的质量")
        
        # 准备话题信息
        topics_info = []
        for topic_id, topic_data in topics.items():
            if isinstance(topic_data, dict):
                topic_info = {
                    "id": topic_id,
                    "title": topic_data.get('title', f'Topic {topic_id}'),
                    "description": topic_data.get('description', ''),
                    "keywords": topic_data.get('keywords', [])
                }
                topics_info.append(topic_info)
        
        # 准备文档样本
        sample_docs = documents[:5] if len(documents) > 5 else documents
        
        prompt = f"""
        {base_prompt}
        
        话题建模结果：
        {json.dumps(topics_info, ensure_ascii=False, indent=2)}
        
        话题分配统计：
        {self.get_assignment_statistics(assignments)}
        
        原始文档样本：
        {json.dumps([{"id": doc['id'], "text": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']} for doc in sample_docs], ensure_ascii=False, indent=2)}
        
        请按照标准进行评估并给出分数(1-5)和详细理由。
        格式：
        评分: [分数]
        理由: [详细理由]
        """
        
        return prompt
    
    def get_assignment_statistics(self, assignments: Dict[str, Any]) -> Dict[str, int]:
        """获取话题分配统计"""
        stats = {}
        for doc_id, assignment in assignments.items():
            topic_id = assignment.get('topic_id', 'unknown')
            stats[topic_id] = stats.get(topic_id, 0) + 1
        return stats
    
    def parse_evaluation_response(self, response: str, aspect: str) -> tuple:
        """解析评估响应"""
        try:
            # 尝试提取分数
            score = 3.0  # 默认分数
            reasoning = response
            
            # 查找分数
            if "评分:" in response:
                score_part = response.split("评分:")[1].split("\n")[0].strip()
                try:
                    score = float(score_part)
                    # 确保分数在1-5范围内
                    score = max(1.0, min(5.0, score))
                except:
                    pass
            
            # 查找理由
            if "理由:" in response:
                reasoning = response.split("理由:")[1].strip()
            
            return score, reasoning
            
        except Exception as e:
            print(f"解析评估响应失败: {e}")
            return 3.0, f"解析失败: {response}"
    
    def calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        """计算总分"""
        scores = []
        weights = {
            'coherence': 0.25,
            'consistency': 0.25,
            'fluency': 0.15,
            'relevance': 0.25,
            'diversity': 0.10
        }
        
        for aspect, result in evaluation_results.items():
            if aspect != 'overall_score' and aspect != 'metadata':
                score = result.get('score', 3.0)
                weight = weights.get(aspect, 1.0 / len(evaluation_results))
                scores.append(score * weight)
        
        if scores:
            return sum(scores)
        else:
            return 3.0
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {output_path}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        report = []
        report.append("# 话题建模质量评估报告")
        report.append(f"\n评估时间: {results.get('metadata', {}).get('evaluation_date', 'Unknown')}")
        report.append(f"使用模型: {results.get('metadata', {}).get('model_used', 'Unknown')}")
        report.append(f"评分范围: 1-{results.get('metadata', {}).get('scoring_scale', 5)}")
        
        report.append(f"\n## 总体评分: {results.get('overall_score', 0):.2f}/5.00")
        
        report.append("\n## 各维度详细评分")
        for aspect, result in results.items():
            if aspect not in ['overall_score', 'metadata']:
                score = result.get('score', 0)
                reasoning = result.get('reasoning', '')
                report.append(f"\n### {aspect.title()}: {score:.2f}/5.00")
                report.append(f"**理由**: {reasoning}")
        
        report.append("\n## 改进建议")
        suggestions = self.generate_improvement_suggestions(results)
        for suggestion in suggestions:
            report.append(f"- {suggestion}")
        
        return "\n".join(report)
    
    def generate_improvement_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        for aspect, result in results.items():
            if aspect not in ['overall_score', 'metadata']:
                score = result.get('score', 3.0)
                if score < 3.0:
                    if aspect == 'coherence':
                        suggestions.append("提高话题内部连贯性：确保关键词语义相关，话题描述准确")
                    elif aspect == 'consistency':
                        suggestions.append("改善话题一致性：明确话题边界，避免重叠")
                    elif aspect == 'fluency':
                        suggestions.append("提升表达流畅性：优化话题描述和关键词表达")
                    elif aspect == 'relevance':
                        suggestions.append("增强相关性：确保话题准确反映文档内容")
                    elif aspect == 'diversity':
                        suggestions.append("增加话题多样性：涵盖更多不同主题领域")
        
        if not suggestions:
            suggestions.append("当前话题建模结果质量良好，可以保持现有设置")
        
        return suggestions

def main():
    """测试G-Eval运行器"""
    # 创建运行器
    geval_runner = CustomGEvalRunner()
    
    # 准备输入数据（假设已有TopicGPT结果）
    if os.path.exists('data/output/topicgpt_results/topicgpt_results.json'):
        geval_input = geval_runner.prepare_geval_input(
            'data/output/topicgpt_results/topicgpt_results.json',
            'data/input/dataset.jsonl'
        )
        
        # 运行评估
        results = geval_runner.run_evaluation(geval_input)
        
        # 保存结果
        geval_runner.save_evaluation_results(results, 'results/geval_results.json')
        
        # 生成报告
        report = geval_runner.generate_evaluation_report(results)
        with open('results/evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("G-Eval评估完成！")
    else:
        print("未找到TopicGPT结果文件，请先运行话题建模")

if __name__ == "__main__":
    main() 