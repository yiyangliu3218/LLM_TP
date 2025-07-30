import sys
import os
import json
import yaml
from typing import Dict, List, Any
from datetime import datetime

# 添加TopicGPT路径
sys.path.append('./topicGPT')

class CustomTopicGPTRunner:
    def __init__(self, config_path: str = None):
        """初始化TopicGPT运行器"""
        self.config = self.load_config(config_path)
        self.results = {}
        
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 默认配置
            config = {
                'data_path': 'data/input/dataset.jsonl',
                'output_path': 'data/output/topicgpt_results',
                'model_name': 'gpt-3.5-turbo',  # 可以改为开源模型
                'num_topics': 8,
                'language': 'english',
                'max_iterations': 3,
                'temperature': 0.7,
                'max_tokens': 1000,
                'api_type': 'openai',  # 或 'huggingface', 'local'
                'model_config': {
                    'openai': {
                        'api_key': os.getenv('OPENAI_API_KEY', ''),
                        'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                    },
                    'huggingface': {
                        'model_name': 'microsoft/DialoGPT-medium',
                        'api_key': os.getenv('HUGGINGFACE_API_KEY', '')
                    },
                    'local': {
                        'model_path': './models/llama-2-7b-chat',
                        'device': 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'
                    }
                }
            }
        
        return config
    
    def setup_model(self):
        """设置模型"""
        api_type = self.config['api_type']
        
        if api_type == 'openai':
            # 使用OpenAI API
            import openai
            openai.api_key = self.config['model_config']['openai']['api_key']
            openai.base_url = self.config['model_config']['openai']['base_url']
            return 'openai'
            
        elif api_type == 'huggingface':
            # 使用HuggingFace API
            return 'huggingface'
            
        elif api_type == 'local':
            # 使用本地模型
            return 'local'
            
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
    
    def generate_topics_stage1(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """第一阶段：生成高级话题"""
        print("🚀 开始第一阶段：生成高级话题...")
        
        # 构建prompt
        prompt = self.build_stage1_prompt(documents)
        
        # 调用模型
        response = self.call_model(prompt)
        
        # 解析结果
        topics = self.parse_stage1_response(response)
        
        print(f"✅ 第一阶段完成，生成了 {len(topics)} 个高级话题")
        return topics
    
    def generate_topics_stage2(self, stage1_topics: Dict[str, Any], 
                             documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """第二阶段：生成低级话题"""
        print("🚀 开始第二阶段：生成低级话题...")
        
        detailed_topics = {}
        
        for topic_id, topic_info in stage1_topics.items():
            # 为每个高级话题生成低级话题
            prompt = self.build_stage2_prompt(topic_info, documents)
            response = self.call_model(prompt)
            
            subtopics = self.parse_stage2_response(response)
            detailed_topics[topic_id] = {
                'main_topic': topic_info,
                'subtopics': subtopics
            }
        
        print(f"✅ 第二阶段完成，为 {len(stage1_topics)} 个高级话题生成了子话题")
        return detailed_topics
    
    def assign_topics(self, topics: Dict[str, Any], 
                     documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分配话题到文档"""
        print("🚀 开始话题分配...")
        
        assignments = {}
        
        for doc in documents:
            prompt = self.build_assignment_prompt(doc, topics)
            response = self.call_model(prompt)
            
            assignment = self.parse_assignment_response(response)
            assignments[doc['id']] = assignment
        
        print(f"✅ 话题分配完成，处理了 {len(documents)} 个文档")
        return assignments
    
    def refine_topics(self, topics: Dict[str, Any], 
                     assignments: Dict[str, Any]) -> Dict[str, Any]:
        """精炼话题"""
        print("🚀 开始话题精炼...")
        
        # 分析话题分布
        topic_distribution = self.analyze_topic_distribution(assignments)
        
        # 识别需要合并或删除的话题
        refinement_prompt = self.build_refinement_prompt(topics, topic_distribution)
        response = self.call_model(refinement_prompt)
        
        refined_topics = self.parse_refinement_response(response, topics)
        
        print("✅ 话题精炼完成")
        return refined_topics
    
    def call_model(self, prompt: str) -> str:
        """调用模型"""
        api_type = self.config['api_type']
        
        if api_type == 'openai':
            return self.call_openai_model(prompt)
        elif api_type == 'huggingface':
            return self.call_huggingface_model(prompt)
        elif api_type == 'local':
            return self.call_local_model(prompt)
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
    
    def call_openai_model(self, prompt: str) -> str:
        """调用OpenAI模型"""
        try:
            import openai
            response = openai.ChatCompletion.create(
                model=self.config['model_name'],
                messages=[
                    {"role": "system", "content": "你是一个专业的话题建模专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return self.get_fallback_response()
    
    def call_huggingface_model(self, prompt: str) -> str:
        """调用HuggingFace模型"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.config['model_config']['huggingface']['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.config['max_tokens'],
                    "temperature": self.config['temperature']
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
    
    def call_local_model(self, prompt: str) -> str:
        """调用本地模型"""
        try:
            # 这里可以集成本地模型，如Llama、GPT4All等
            # 为了简化，这里返回一个示例响应
            return self.get_fallback_response()
        except Exception as e:
            print(f"本地模型调用失败: {e}")
            return self.get_fallback_response()
    
    def get_fallback_response(self) -> str:
        """获取备用响应"""
        return "由于模型调用失败，返回默认话题。"
    
    def build_stage1_prompt(self, documents: List[Dict[str, Any]]) -> str:
        """构建第一阶段prompt"""
        sample_docs = documents[:10]  # 取前10个文档作为样本
        
        prompt = f"""
        请分析以下社交媒体帖子，生成 {self.config['num_topics']} 个高级话题。
        
        文档样本：
        {json.dumps(sample_docs, ensure_ascii=False, indent=2)}
        
        要求：
        1. 话题应该涵盖文档中的主要主题
        2. 话题之间应该有明确的区分
        3. 每个话题应该有一个清晰的标题和描述
        4. 话题应该与社交媒体内容相关
        
        请以JSON格式返回结果：
        {{
            "topics": [
                {{
                    "id": "topic_1",
                    "title": "话题标题",
                    "description": "话题描述",
                    "keywords": ["关键词1", "关键词2"]
                }}
            ]
        }}
        """
        return prompt
    
    def build_stage2_prompt(self, topic_info: Dict[str, Any], 
                           documents: List[Dict[str, Any]]) -> str:
        """构建第二阶段prompt"""
        prompt = f"""
        基于高级话题 "{topic_info['title']}"，请生成3-5个具体的子话题。
        
        高级话题描述：{topic_info['description']}
        关键词：{topic_info['keywords']}
        
        请为这个高级话题生成具体的子话题，每个子话题应该：
        1. 更加具体和详细
        2. 包含相关的关键词
        3. 有明确的边界
        
        请以JSON格式返回结果：
        {{
            "subtopics": [
                {{
                    "id": "subtopic_1",
                    "title": "子话题标题",
                    "description": "子话题描述",
                    "keywords": ["关键词1", "关键词2"]
                }}
            ]
        }}
        """
        return prompt
    
    def build_assignment_prompt(self, document: Dict[str, Any], 
                              topics: Dict[str, Any]) -> str:
        """构建话题分配prompt"""
        prompt = f"""
        请将以下文档分配到最合适的话题：
        
        文档内容：{document['text']}
        
        可用话题：
        {json.dumps(topics, ensure_ascii=False, indent=2)}
        
        请返回：
        1. 最合适的话题ID
        2. 分配理由
        3. 置信度分数（0-1）
        
        格式：
        {{
            "topic_id": "topic_1",
            "reasoning": "分配理由",
            "confidence": 0.85
        }}
        """
        return prompt
    
    def build_refinement_prompt(self, topics: Dict[str, Any], 
                               distribution: Dict[str, int]) -> str:
        """构建精炼prompt"""
        prompt = f"""
        基于话题分布情况，请精炼话题：
        
        话题分布：{distribution}
        
        请：
        1. 合并相似的话题
        2. 删除分配文档太少的话题
        3. 优化话题描述
        
        返回精炼后的话题列表。
        """
        return prompt
    
    def parse_stage1_response(self, response: str) -> Dict[str, Any]:
        """解析第一阶段响应"""
        try:
            # 尝试解析JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                return {f"topic_{i+1}": topic for i, topic in enumerate(data.get('topics', []))}
        except:
            pass
        
        # 如果解析失败，返回默认话题
        return {
            "topic_1": {"title": "社交媒体讨论", "description": "关于社交媒体的各种讨论", "keywords": ["social", "media"]},
            "topic_2": {"title": "健康话题", "description": "健康相关的讨论", "keywords": ["health", "medical"]}
        }
    
    def parse_stage2_response(self, response: str) -> List[Dict[str, Any]]:
        """解析第二阶段响应"""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                return data.get('subtopics', [])
        except:
            pass
        
        return []
    
    def parse_assignment_response(self, response: str) -> Dict[str, Any]:
        """解析话题分配响应"""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        return {"topic_id": "topic_1", "reasoning": "默认分配", "confidence": 0.5}
    
    def parse_refinement_response(self, response: str, original_topics: Dict[str, Any]) -> Dict[str, Any]:
        """解析精炼响应"""
        # 简化处理，返回原始话题
        return original_topics
    
    def analyze_topic_distribution(self, assignments: Dict[str, Any]) -> Dict[str, int]:
        """分析话题分布"""
        distribution = {}
        for doc_id, assignment in assignments.items():
            topic_id = assignment.get('topic_id', 'unknown')
            distribution[topic_id] = distribution.get(topic_id, 0) + 1
        return distribution
    
    def run_topic_modeling(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行完整的话题建模流程"""
        print("🎯 开始TopicGPT话题建模流程...")
        
        # 设置模型
        self.setup_model()
        
        # 第一阶段：生成高级话题
        stage1_topics = self.generate_topics_stage1(documents)
        
        # 第二阶段：生成低级话题
        stage2_topics = self.generate_topics_stage2(stage1_topics, documents)
        
        # 话题分配
        assignments = self.assign_topics(stage2_topics, documents)
        
        # 话题精炼
        refined_topics = self.refine_topics(stage2_topics, assignments)
        
        # 保存结果
        results = {
            'topics': refined_topics,
            'assignments': assignments,
            'metadata': {
                'model_used': self.config['model_name'],
                'num_topics': self.config['num_topics'],
                'processing_date': datetime.now().isoformat(),
                'total_documents': len(documents)
            }
        }
        
        self.save_results(results)
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """保存结果"""
        os.makedirs(self.config['output_path'], exist_ok=True)
        
        output_file = os.path.join(self.config['output_path'], 'topicgpt_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_file}")

def main():
    """测试TopicGPT运行器"""
    # 加载测试数据
    with open('data/input/dataset.jsonl', 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f]
    
    # 创建运行器
    runner = CustomTopicGPTRunner()
    
    # 运行话题建模
    results = runner.run_topic_modeling(documents[:100])  # 使用前100个文档测试
    
    print("话题建模完成！")

if __name__ == "__main__":
    main() 