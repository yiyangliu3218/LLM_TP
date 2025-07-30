import json
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime

class FeedbackOptimizer:
    def __init__(self, config: Dict[str, Any] = None):
        """初始化反馈优化器"""
        self.config = config or self.get_default_config()
        self.optimization_history = []
        self.improvement_strategies = {
            "coherence": self.improve_coherence,
            "consistency": self.improve_consistency,
            "fluency": self.improve_fluency,
            "relevance": self.improve_relevance,
            "diversity": self.improve_diversity
        }
        
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'optimization_threshold': 3.0,  # 低于此分数认为需要优化
            'max_iterations': 5,            # 最大优化轮数
            'improvement_threshold': 0.1,   # 改进阈值
            'parameter_bounds': {
                'num_topics': (3, 15),
                'temperature': (0.1, 0.9),
                'max_tokens': (500, 2000)
            }
        }
    
    def analyze_evaluation_results(self, eval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析评估结果，识别需要改进的方面"""
        weak_aspects = []
        threshold = self.config['optimization_threshold']
        
        for aspect, result in eval_results.items():
            if aspect not in ['overall_score', 'metadata']:
                score = result.get('score', 3.0)
                if score < threshold:
                    weak_aspects.append({
                        'aspect': aspect,
                        'score': score,
                        'reasoning': result.get('reasoning', ''),
                        'improvement_potential': threshold - score
                    })
        
        # 按改进潜力排序
        weak_aspects.sort(key=lambda x: x['improvement_potential'], reverse=True)
        
        return weak_aspects
    
    def generate_optimization_strategy(self, weak_aspects: List[Dict[str, Any]], 
                                     current_config: Dict[str, Any]) -> Dict[str, Any]:
        """基于弱点生成优化策略"""
        optimizations = {}
        
        for weak_aspect in weak_aspects:
            aspect_name = weak_aspect['aspect']
            if aspect_name in self.improvement_strategies:
                optimization = self.improvement_strategies[aspect_name](
                    weak_aspect, current_config
                )
                optimizations[aspect_name] = optimization
        
        return optimizations
    
    def improve_coherence(self, weak_aspect: Dict[str, Any], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """改进话题连贯性的策略"""
        return {
            "prompt_modification": """
            特别注意：确保每个话题的关键词在语义上高度相关。
            在生成话题时，请验证关键词是否都围绕同一个核心概念。
            建议使用更严格的语义一致性检查。
            """,
            "parameter_adjustment": {
                "temperature": max(0.1, config.get('temperature', 0.7) - 0.2),  # 降低随机性
                "top_p": min(0.8, config.get('top_p', 0.9) - 0.1),
                "frequency_penalty": min(0.5, config.get('frequency_penalty', 0.0) + 0.1)
            },
            "additional_instructions": [
                "在生成话题时，确保关键词之间有明确的语义关联",
                "避免在同一个话题中包含语义不相关的词汇",
                "使用更精确的话题描述来增强连贯性"
            ]
        }
    
    def improve_consistency(self, weak_aspect: Dict[str, Any], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """改进话题一致性的策略"""
        return {
            "prompt_modification": """
            在生成多个话题时，请确保：
            1. 话题之间有明确的区分界限
            2. 避免话题内容重叠
            3. 使用一致的命名和描述格式
            4. 建立清晰的话题层次结构
            """,
            "parameter_adjustment": {
                "num_topics": max(3, config.get('num_topics', 8) - 1),  # 减少话题数量提高区分度
                "temperature": max(0.1, config.get('temperature', 0.7) - 0.1),
                "presence_penalty": min(0.5, config.get('presence_penalty', 0.0) + 0.1)
            },
            "additional_instructions": [
                "在生成话题前，先定义明确的话题边界",
                "使用对比分析确保话题间的差异性",
                "建立统一的话题命名规范"
            ]
        }
    
    def improve_fluency(self, weak_aspect: Dict[str, Any], 
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """改进流畅性的策略"""
        return {
            "prompt_modification": """
            请确保话题描述和关键词表达自然流畅：
            1. 使用清晰、简洁的语言
            2. 避免冗余和重复表达
            3. 保持语言风格的一致性
            4. 使用准确的术语和表达
            """,
            "parameter_adjustment": {
                "temperature": min(0.9, config.get('temperature', 0.7) + 0.1),  # 增加创造性
                "max_tokens": min(2000, config.get('max_tokens', 1000) + 200)
            },
            "additional_instructions": [
                "在生成话题描述时，使用自然流畅的表达",
                "避免使用过于技术化或晦涩的词汇",
                "确保话题标题简洁明了"
            ]
        }
    
    def improve_relevance(self, weak_aspect: Dict[str, Any], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """改进相关性的策略"""
        return {
            "prompt_modification": """
            请确保话题与原始文档高度相关：
            1. 仔细分析文档内容的主要主题
            2. 确保话题准确反映文档的核心内容
            3. 避免生成与文档无关的话题
            4. 考虑文档的上下文和背景
            """,
            "parameter_adjustment": {
                "temperature": max(0.1, config.get('temperature', 0.7) - 0.15),
                "top_p": min(0.9, config.get('top_p', 0.9) - 0.05)
            },
            "additional_instructions": [
                "在生成话题前，深入分析文档内容",
                "确保每个话题都有充分的文档支持",
                "避免生成过于宽泛或过于狭窄的话题"
            ]
        }
    
    def improve_diversity(self, weak_aspect: Dict[str, Any], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """改进多样性的策略"""
        return {
            "prompt_modification": """
            请确保话题具有足够的多样性：
            1. 涵盖不同的主题领域和角度
            2. 避免话题重复和相似性
            3. 确保话题分布均衡
            4. 考虑不同维度的主题划分
            """,
            "parameter_adjustment": {
                "num_topics": min(15, config.get('num_topics', 8) + 1),  # 增加话题数量
                "temperature": min(0.9, config.get('temperature', 0.7) + 0.2),  # 增加多样性
                "frequency_penalty": max(0.0, config.get('frequency_penalty', 0.0) - 0.1)
            },
            "additional_instructions": [
                "在生成话题时，考虑不同的主题维度",
                "确保话题涵盖文档中的各种主题",
                "避免生成过于相似的话题"
            ]
        }
    
    def apply_optimizations(self, config: Dict[str, Any], 
                           optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """应用优化策略到配置中"""
        new_config = config.copy()
        
        # 合并所有优化策略
        combined_optimization = {
            "parameter_adjustment": {},
            "prompt_modification": "",
            "additional_instructions": []
        }
        
        for aspect, optimization in optimizations.items():
            # 合并参数调整
            if 'parameter_adjustment' in optimization:
                for param, value in optimization['parameter_adjustment'].items():
                    if param in combined_optimization["parameter_adjustment"]:
                        # 如果参数已存在，取平均值或最保守的值
                        current_val = combined_optimization["parameter_adjustment"][param]
                        if isinstance(value, (int, float)) and isinstance(current_val, (int, float)):
                            combined_optimization["parameter_adjustment"][param] = (current_val + value) / 2
                    else:
                        combined_optimization["parameter_adjustment"][param] = value
            
            # 合并prompt修改
            if 'prompt_modification' in optimization:
                combined_optimization["prompt_modification"] += "\n" + optimization['prompt_modification']
            
            # 合并额外指令
            if 'additional_instructions' in optimization:
                combined_optimization["additional_instructions"].extend(optimization['additional_instructions'])
        
        # 应用参数调整
        new_config.update(combined_optimization["parameter_adjustment"])
        
        # 应用prompt修改
        if combined_optimization["prompt_modification"]:
            new_config['additional_instructions'] = new_config.get('additional_instructions', '') + '\n' + combined_optimization["prompt_modification"]
        
        # 应用额外指令
        if combined_optimization["additional_instructions"]:
            new_config['optimization_instructions'] = combined_optimization["additional_instructions"]
        
        return new_config
    
    def validate_optimization(self, old_score: float, new_score: float) -> bool:
        """验证优化是否有效"""
        improvement = new_score - old_score
        return improvement >= self.config['improvement_threshold']
    
    def record_optimization(self, iteration: int, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any], old_score: float, 
                          new_score: float, weak_aspects: List[Dict[str, Any]]):
        """记录优化历史"""
        optimization_record = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'old_score': old_score,
            'new_score': new_score,
            'improvement': new_score - old_score,
            'weak_aspects': weak_aspects,
            'config_changes': {
                'old': old_config,
                'new': new_config
            }
        }
        
        self.optimization_history.append(optimization_record)
    
    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        if not self.optimization_history:
            return "暂无优化历史记录"
        
        report = []
        report.append("# 话题建模优化报告")
        report.append(f"\n优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总优化轮数: {len(self.optimization_history)}")
        
        # 计算总体改进
        initial_score = self.optimization_history[0]['old_score']
        final_score = self.optimization_history[-1]['new_score']
        total_improvement = final_score - initial_score
        
        report.append(f"\n## 优化效果")
        report.append(f"- 初始分数: {initial_score:.2f}")
        report.append(f"- 最终分数: {final_score:.2f}")
        report.append(f"- 总体改进: {total_improvement:.2f}")
        report.append(f"- 改进百分比: {(total_improvement/initial_score)*100:.1f}%")
        
        report.append(f"\n## 详细优化历史")
        for i, record in enumerate(self.optimization_history):
            report.append(f"\n### 第 {record['iteration']} 轮优化")
            report.append(f"- 时间: {record['timestamp']}")
            report.append(f"- 分数变化: {record['old_score']:.2f} → {record['new_score']:.2f}")
            report.append(f"- 改进: {record['improvement']:.2f}")
            
            if record['weak_aspects']:
                report.append(f"- 识别的问题:")
                for aspect in record['weak_aspects']:
                    report.append(f"  - {aspect['aspect']}: {aspect['score']:.2f}")
        
        report.append(f"\n## 配置变化总结")
        if len(self.optimization_history) > 1:
            initial_config = self.optimization_history[0]['config_changes']['old']
            final_config = self.optimization_history[-1]['config_changes']['new']
            
            for key in set(initial_config.keys()) | set(final_config.keys()):
                old_val = initial_config.get(key, 'N/A')
                new_val = final_config.get(key, 'N/A')
                if old_val != new_val:
                    report.append(f"- {key}: {old_val} → {new_val}")
        
        return "\n".join(report)
    
    def save_optimization_history(self, output_path: str):
        """保存优化历史"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, ensure_ascii=False, indent=2)
        
        print(f"优化历史已保存到: {output_path}")

class ClosedLoopTopicModeling:
    def __init__(self, excel_path: str, config: Dict[str, Any] = None):
        """初始化闭环话题建模系统"""
        self.excel_path = excel_path
        self.config = config or {}
        
        # 导入其他模块
        from .data_processor import DatasetProcessor
        from .topicgpt_runner import CustomTopicGPTRunner
        from .geval_runner import CustomGEvalRunner
        
        self.data_processor = DatasetProcessor(excel_path)
        self.topicgpt_runner = CustomTopicGPTRunner()
        self.geval_runner = CustomGEvalRunner()
        self.optimizer = FeedbackOptimizer()
        
        self.best_results = None
        self.best_score = 0
        
    def run_complete_pipeline(self, max_iterations: int = 5, 
                            sample_size: int = 500) -> Dict[str, Any]:
        """运行完整的闭环优化流程"""
        
        print("🎯 开始闭环话题建模系统...")
        
        # 数据预处理
        print("📋 预处理数据...")
        documents = self.data_processor.prepare_for_topicgpt(
            text_column="Translated Post Description",
            max_docs=1000,
            sample_size=sample_size
        )
        
        # 保存预处理数据
        self.data_processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        
        current_config = self.topicgpt_runner.config.copy()
        
        for iteration in range(max_iterations):
            print(f"\n🔄 第 {iteration + 1} 轮迭代")
            
            # 1. 运行TopicGPT
            print("🚀 运行话题建模...")
            topic_results = self.topicgpt_runner.run_topic_modeling(documents)
            
            # 2. 运行G-Eval评估
            print("📊 运行质量评估...")
            geval_input = self.geval_runner.prepare_geval_input(
                'data/output/topicgpt_results/topicgpt_results.json',
                'data/input/dataset.jsonl'
            )
            eval_results = self.geval_runner.run_evaluation(geval_input)
            
            current_score = eval_results['overall_score']
            print(f"本轮总分: {current_score:.2f}")
            
            # 3. 检查是否需要优化
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_results = {
                    'topics': topic_results,
                    'evaluation': eval_results,
                    'iteration': iteration + 1,
                    'config': current_config.copy()
                }
            
            # 如果分数足够高，提前结束
            if current_score >= 4.0:
                print("✅ 达到满意分数，提前结束优化")
                break
            
            # 4. 生成优化策略
            weak_aspects = self.optimizer.analyze_evaluation_results(eval_results)
            if weak_aspects:
                optimizations = self.optimizer.generate_optimization_strategy(
                    weak_aspects, current_config
                )
                
                # 5. 应用优化策略
                old_config = current_config.copy()
                current_config = self.optimizer.apply_optimizations(current_config, optimizations)
                self.topicgpt_runner.config = current_config
                
                # 记录优化历史
                self.optimizer.record_optimization(
                    iteration + 1, old_config, current_config,
                    current_score, current_score, weak_aspects
                )
                
                print(f"应用优化策略: {list(optimizations.keys())}")
            else:
                print("未发现明显弱点，结束优化")
                break
        
        # 保存最终结果
        self.save_final_results()
        
        return self.best_results
    
    def save_final_results(self):
        """保存最终结果"""
        if self.best_results:
            # 保存最佳结果
            with open('results/best_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.best_results, f, ensure_ascii=False, indent=2)
            
            # 保存优化历史
            self.optimizer.save_optimization_history('results/optimization_history.json')
            
            # 生成优化报告
            report = self.optimizer.generate_optimization_report()
            with open('results/optimization_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ 最终结果已保存，最佳分数: {self.best_score:.2f}")

def main():
    """测试闭环系统"""
    # 创建闭环系统
    pipeline = ClosedLoopTopicModeling("Dataset.xlsx")
    
    # 运行完整流程
    results = pipeline.run_complete_pipeline(max_iterations=3, sample_size=300)
    
    print("闭环话题建模系统运行完成！")

if __name__ == "__main__":
    main() 