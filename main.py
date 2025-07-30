#!/usr/bin/env python3
"""
基于LLM的自动话题建模+评估+闭环优化系统
主控制脚本
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any

# 添加自定义模块路径
sys.path.append('./custom_pipeline')

def setup_environment():
    """设置环境变量和依赖"""
    print("🔧 设置环境...")
    
    # 检查必要的目录
    directories = [
        'data/input',
        'data/output',
        'results',
        'prompts',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 检查环境变量
    required_env_vars = [
        'OPENAI_API_KEY',
        'HUGGINGFACE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  警告: 以下环境变量未设置: {missing_vars}")
        print("系统将使用备用模式运行")
    
    print("✅ 环境设置完成")

def run_data_preprocessing(data_path: str, sample_size: int = 500):
    """运行数据预处理"""
    print("\n📋 开始数据预处理...")
    
    try:
        from custom_pipeline.data_processor import DatasetProcessor
        
        processor = DatasetProcessor(data_path)
        
        # 准备数据
        if data_path.endswith('.jsonl'):
            documents = processor.prepare_for_topicgpt(
                max_docs=1000,
                sample_size=sample_size
            )
        else:
            documents = processor.prepare_for_topicgpt(
                text_column="Translated Post Description",
                max_docs=1000,
                sample_size=sample_size
            )
        
        # 保存数据
        processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        
        # 获取统计信息
        stats = processor.get_data_statistics()
        
        print("✅ 数据预处理完成")
        print(f"处理文档数: {stats['total_documents']}")
        print(f"平均文本长度: {stats['avg_text_length']:.1f}")
        print(f"数据源类型: {stats['source_type']}")
        
        return documents, stats
        
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        raise

def run_topic_modeling(documents, config: Dict[str, Any] = None):
    """运行话题建模"""
    print("\n🚀 开始话题建模...")
    
    try:
        from custom_pipeline.topicgpt_runner import CustomTopicGPTRunner
        
        runner = CustomTopicGPTRunner(config)
        results = runner.run_topic_modeling(documents)
        
        print("✅ 话题建模完成")
        return results
        
    except Exception as e:
        print(f"❌ 话题建模失败: {e}")
        raise

def run_evaluation(topicgpt_results_path: str, original_docs_path: str, config: Dict[str, Any] = None):
    """运行质量评估"""
    print("\n📊 开始质量评估...")
    
    try:
        from custom_pipeline.geval_runner import CustomGEvalRunner
        
        geval_runner = CustomGEvalRunner(config)
        
        # 准备输入数据
        geval_input = geval_runner.prepare_geval_input(topicgpt_results_path, original_docs_path)
        
        # 运行评估
        eval_results = geval_runner.run_evaluation(geval_input)
        
        # 保存结果
        geval_runner.save_evaluation_results(eval_results, 'results/geval_results.json')
        
        # 生成报告
        report = geval_runner.generate_evaluation_report(eval_results)
        with open('results/evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ 质量评估完成")
        print(f"总体评分: {eval_results['overall_score']:.2f}/5.00")
        
        return eval_results
        
    except Exception as e:
        print(f"❌ 质量评估失败: {e}")
        raise

def run_closed_loop_system(data_path: str, config: Dict[str, Any] = None):
    """运行完整的闭环系统"""
    print("\n🎯 开始闭环话题建模系统...")
    
    try:
        from custom_pipeline.feedback_loop import ClosedLoopTopicModeling
        
        pipeline = ClosedLoopTopicModeling(data_path, config)
        
        # 运行完整流程
        results = pipeline.run_complete_pipeline(
            max_iterations=config.get('max_iterations', 3),
            sample_size=config.get('sample_size', 500)
        )
        
        print("✅ 闭环系统运行完成")
        return results
        
    except Exception as e:
        print(f"❌ 闭环系统运行失败: {e}")
        raise

def create_config(args) -> Dict[str, Any]:
    """创建配置"""
    config = {
        'api_type': args.api_type,
        'model_name': args.model_name,
        'num_topics': args.num_topics,
        'max_iterations': args.max_iterations,
        'sample_size': args.sample_size,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'optimization_threshold': args.optimization_threshold,
        'evaluation_aspects': [
            'coherence',
            'consistency', 
            'fluency',
            'relevance',
            'diversity'
        ]
    }
    
    return config

def save_config(config: Dict[str, Any], output_path: str):
    """保存配置"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"配置已保存到: {output_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于LLM的自动话题建模+评估+闭环优化系统')
    
    parser.add_argument('--data_path', type=str, default='mydata.jsonl',
                       help='数据文件路径（支持.jsonl或.xlsx格式）')
    parser.add_argument('--mode', type=str, default='closed_loop',
                       choices=['preprocess', 'topic_modeling', 'evaluation', 'closed_loop'],
                       help='运行模式')
    parser.add_argument('--api_type', type=str, default='openai',
                       choices=['openai', 'huggingface', 'local'],
                       help='API类型')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo',
                       help='模型名称')
    parser.add_argument('--num_topics', type=int, default=8,
                       help='话题数量')
    parser.add_argument('--max_iterations', type=int, default=3,
                       help='最大优化轮数')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='采样大小')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='温度参数')
    parser.add_argument('--max_tokens', type=int, default=1000,
                       help='最大token数')
    parser.add_argument('--optimization_threshold', type=float, default=3.0,
                       help='优化阈值')
    parser.add_argument('--config_path', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: 数据文件 '{args.data_path}' 不存在")
        print("请检查文件路径，或使用 --data_path 参数指定正确的文件路径")
        return 1
    
    # 设置环境
    setup_environment()
    
    # 创建配置
    config = create_config(args)
    
    # 保存配置
    save_config(config, 'config.json')
    
    try:
        if args.mode == 'preprocess':
            # 仅运行数据预处理
            documents, stats = run_data_preprocessing(args.data_path, args.sample_size)
            print(f"数据预处理完成，统计信息: {stats}")
            
        elif args.mode == 'topic_modeling':
            # 仅运行话题建模
            documents, _ = run_data_preprocessing(args.data_path, args.sample_size)
            results = run_topic_modeling(documents, config)
            print(f"话题建模完成，生成了 {len(results.get('topics', {}))} 个话题")
            
        elif args.mode == 'evaluation':
            # 仅运行评估
            if not os.path.exists('data/output/topicgpt_results/topicgpt_results.json'):
                print("❌ 未找到话题建模结果，请先运行话题建模")
                return 1
            
            eval_results = run_evaluation(
                'data/output/topicgpt_results/topicgpt_results.json',
                'data/input/dataset.jsonl',
                config
            )
            print(f"评估完成，总分: {eval_results['overall_score']:.2f}")
            
        elif args.mode == 'closed_loop':
            # 运行完整闭环系统
            results = run_closed_loop_system(args.data_path, config)
            print(f"闭环系统完成，最佳分数: {results.get('evaluation', {}).get('overall_score', 0):.2f}")
        
        print("\n🎉 所有任务完成！")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 