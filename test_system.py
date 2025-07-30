#!/usr/bin/env python3
"""
系统测试脚本
用于验证各个组件是否正常工作
"""

import os
import sys
import json
import tempfile
import pandas as pd
from datetime import datetime

# 添加自定义模块路径
sys.path.append('./custom_pipeline')

def test_data_processor():
    """测试数据处理器"""
    print("🧪 测试数据处理器...")
    
    try:
        from custom_pipeline.data_processor import DatasetProcessor
        
        # 创建测试数据
        test_data = {
            'Post ID': ['test_1', 'test_2', 'test_3'],
            'Translated Post Description': [
                'This is a test post about technology and AI.',
                'Another post discussing health and wellness.',
                'A post about social media and communication.'
            ],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Language': ['English', 'English', 'English'],
            'Sentiment': ['Positive', 'Neutral', 'Positive'],
            'Hate': ['Not Hate', 'Not Hate', 'Not Hate'],
            'Stress or Anxiety': ['No Stress', 'Stress Detected', 'No Stress']
        }
        
        # 创建临时Excel文件
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df = pd.DataFrame(test_data)
            df.to_excel(f.name, index=False)
            temp_path = f.name
        
        # 测试数据处理器
        processor = DatasetProcessor(temp_path)
        documents = processor.prepare_for_topicgpt(
            text_column="Translated Post Description",
            max_docs=10,
            sample_size=3
        )
        
        # 验证结果
        assert len(documents) == 3, f"期望3个文档，实际得到{len(documents)}个"
        assert 'id' in documents[0], "文档缺少id字段"
        assert 'text' in documents[0], "文档缺少text字段"
        assert 'metadata' in documents[0], "文档缺少metadata字段"
        
        print("✅ 数据处理器测试通过")
        
        # 清理临时文件
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 数据处理器测试失败: {e}")
        return False

def test_topicgpt_runner():
    """测试TopicGPT运行器"""
    print("🧪 测试TopicGPT运行器...")
    
    try:
        from custom_pipeline.topicgpt_runner import CustomTopicGPTRunner
        
        # 创建测试数据
        test_documents = [
            {
                "id": "test_1",
                "text": "This is a test post about technology and AI.",
                "metadata": {"original_index": 0}
            },
            {
                "id": "test_2", 
                "text": "Another post discussing health and wellness.",
                "metadata": {"original_index": 1}
            }
        ]
        
        # 测试运行器
        runner = CustomTopicGPTRunner()
        
        # 由于需要API调用，这里只测试配置加载
        assert 'api_type' in runner.config, "配置缺少api_type"
        assert 'model_name' in runner.config, "配置缺少model_name"
        assert 'num_topics' in runner.config, "配置缺少num_topics"
        
        print("✅ TopicGPT运行器配置测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ TopicGPT运行器测试失败: {e}")
        return False

def test_geval_runner():
    """测试G-Eval运行器"""
    print("🧪 测试G-Eval运行器...")
    
    try:
        from custom_pipeline.geval_runner import CustomGEvalRunner
        
        # 测试运行器
        runner = CustomGEvalRunner()
        
        # 验证配置
        assert 'evaluation_aspects' in runner.config, "配置缺少evaluation_aspects"
        assert 'scoring_scale' in runner.config, "配置缺少scoring_scale"
        
        # 测试评估维度
        expected_aspects = ['coherence', 'consistency', 'fluency', 'relevance', 'diversity']
        for aspect in expected_aspects:
            assert aspect in runner.config['evaluation_aspects'], f"缺少评估维度: {aspect}"
        
        print("✅ G-Eval运行器测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ G-Eval运行器测试失败: {e}")
        return False

def test_feedback_loop():
    """测试反馈优化器"""
    print("🧪 测试反馈优化器...")
    
    try:
        from custom_pipeline.feedback_loop import FeedbackOptimizer
        
        # 测试优化器
        optimizer = FeedbackOptimizer()
        
        # 测试配置
        assert 'optimization_threshold' in optimizer.config, "配置缺少optimization_threshold"
        assert 'max_iterations' in optimizer.config, "配置缺少max_iterations"
        
        # 测试评估结果分析
        test_eval_results = {
            'coherence': {'score': 2.5, 'reasoning': 'Test'},
            'consistency': {'score': 3.5, 'reasoning': 'Test'},
            'fluency': {'score': 4.0, 'reasoning': 'Test'},
            'overall_score': 3.33
        }
        
        weak_aspects = optimizer.analyze_evaluation_results(test_eval_results)
        assert len(weak_aspects) == 1, f"期望1个弱点，实际得到{len(weak_aspects)}个"
        assert weak_aspects[0]['aspect'] == 'coherence', "弱点分析错误"
        
        print("✅ 反馈优化器测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 反馈优化器测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("🧪 测试文件结构...")
    
    try:
        # 检查必要的目录
        required_dirs = ['data/input', 'data/output', 'results', 'custom_pipeline']
        for directory in required_dirs:
            assert os.path.exists(directory), f"缺少目录: {directory}"
        
        # 检查必要的文件
        required_files = [
            'custom_pipeline/data_processor.py',
            'custom_pipeline/topicgpt_runner.py', 
            'custom_pipeline/geval_runner.py',
            'custom_pipeline/feedback_loop.py',
            'main.py',
            'app.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"缺少文件: {file_path}"
        
        print("✅ 文件结构测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 文件结构测试失败: {e}")
        return False

def test_dependencies():
    """测试依赖包"""
    print("🧪 测试依赖包...")
    
    try:
        import pandas
        import numpy
        import gradio
        import openai
        import requests
        import yaml
        
        print("✅ 核心依赖包测试通过")
        
        # 测试可选依赖
        try:
            import transformers
            import torch
            print("✅ 可选依赖包测试通过")
        except ImportError:
            print("⚠️  可选依赖包未安装，但不影响基本功能")
        
        return True
        
    except ImportError as e:
        print(f"❌ 依赖包测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始系统测试...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("文件结构", test_file_structure),
        ("依赖包", test_dependencies),
        ("数据处理器", test_data_processor),
        ("TopicGPT运行器", test_topicgpt_runner),
        ("G-Eval运行器", test_geval_runner),
        ("反馈优化器", test_feedback_loop)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}测试")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}测试通过")
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        print("\n📝 下一步:")
        print("1. 设置API密钥: export OPENAI_API_KEY='your-key'")
        print("2. 运行Web界面: python app.py")
        print("3. 或运行命令行: python main.py --mode closed_loop")
    else:
        print("⚠️  部分测试失败，请检查错误信息并修复问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 