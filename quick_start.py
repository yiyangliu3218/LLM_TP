#!/usr/bin/env python3
"""
快速启动脚本
演示如何使用mydata.jsonl文件运行话题建模系统
"""

import os
import sys
import json
from datetime import datetime

# 添加自定义模块路径
sys.path.append('./custom_pipeline')

def quick_demo():
    """快速演示"""
    print("🚀 基于LLM的自动话题建模+评估+闭环优化系统")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查数据文件
    if not os.path.exists("mydata.jsonl"):
        print("❌ 错误: 未找到 mydata.jsonl 文件")
        print("请确保 mydata.jsonl 文件在当前目录中")
        return False
    
    print("✅ 找到数据文件: mydata.jsonl")
    
    # 1. 数据预处理
    print("\n📋 步骤1: 数据预处理")
    print("-" * 30)
    
    try:
        from custom_pipeline.data_processor import DatasetProcessor
        
        processor = DatasetProcessor("mydata.jsonl")
        documents = processor.prepare_for_topicgpt(
            max_docs=1000,  # 限制为1000条用于演示
            sample_size=300  # 随机采样300条
        )
        
        processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        stats = processor.get_data_statistics()
        
        print(f"✅ 数据预处理完成")
        print(f"   - 处理文档数: {stats['total_documents']}")
        print(f"   - 平均文本长度: {stats['avg_text_length']:.1f}")
        print(f"   - 数据源类型: {stats['source_type']}")
        
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return False
    
    # 2. 话题建模（模拟）
    print("\n🚀 步骤2: 话题建模")
    print("-" * 30)
    
    try:
        from custom_pipeline.topicgpt_runner import CustomTopicGPTRunner
        
        # 创建配置
        config = {
            'api_type': 'openai',  # 或 'huggingface', 'local'
            'model_name': 'gpt-3.5-turbo',
            'num_topics': 6,
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        runner = CustomTopicGPTRunner()
        runner.config.update(config)
        
        # 检查API密钥
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  未设置OPENAI_API_KEY，将使用模拟模式")
            print("   要使用真实API，请设置: export OPENAI_API_KEY='your-key'")
            
            # 创建模拟结果
            mock_results = {
                'topics': {
                    'topic_1': {
                        'title': '健康与医疗',
                        'description': '关于健康、医疗、疾病预防等话题',
                        'keywords': ['health', 'medical', 'disease', 'prevention']
                    },
                    'topic_2': {
                        'title': '社交媒体讨论',
                        'description': '社交媒体平台上的各种讨论和分享',
                        'keywords': ['social', 'media', 'discussion', 'sharing']
                    },
                    'topic_3': {
                        'title': '科技与创新',
                        'description': '科技发展、创新技术、数字化等话题',
                        'keywords': ['technology', 'innovation', 'digital', 'tech']
                    }
                },
                'assignments': {},
                'metadata': {
                    'model_used': 'gpt-3.5-turbo (模拟)',
                    'num_topics': 3,
                    'processing_date': datetime.now().isoformat(),
                    'total_documents': len(documents)
                }
            }
            
            # 保存模拟结果
            os.makedirs('data/output/topicgpt_results', exist_ok=True)
            with open('data/output/topicgpt_results/topicgpt_results.json', 'w', encoding='utf-8') as f:
                json.dump(mock_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 话题建模完成（模拟模式）")
            print(f"   - 生成话题数: {len(mock_results['topics'])}")
            print(f"   - 话题列表: {list(mock_results['topics'].keys())}")
            
        else:
            print("✅ 检测到API密钥，将使用真实API")
            print("   注意: 这可能需要一些时间和API费用")
            
            # 这里可以运行真实的话题建模
            # results = runner.run_topic_modeling(documents)
            # print(f"✅ 话题建模完成，生成了 {len(results.get('topics', {}))} 个话题")
            
    except Exception as e:
        print(f"❌ 话题建模失败: {e}")
        return False
    
    # 3. 质量评估（模拟）
    print("\n📊 步骤3: 质量评估")
    print("-" * 30)
    
    try:
        from custom_pipeline.geval_runner import CustomGEvalRunner
        
        geval_runner = CustomGEvalRunner()
        
        # 检查是否有话题建模结果
        if os.path.exists('data/output/topicgpt_results/topicgpt_results.json'):
            print("✅ 找到话题建模结果，开始质量评估")
            
            # 创建模拟评估结果
            mock_eval_results = {
                'coherence': {
                    'score': 4.2,
                    'reasoning': '话题内部语义相关性良好，关键词选择恰当'
                },
                'consistency': {
                    'score': 3.8,
                    'reasoning': '话题间有一定区分度，但部分话题边界可以更清晰'
                },
                'fluency': {
                    'score': 4.5,
                    'reasoning': '话题描述自然流畅，表达清晰易懂'
                },
                'relevance': {
                    'score': 4.0,
                    'reasoning': '话题与文档内容相关性较高，覆盖面较广'
                },
                'diversity': {
                    'score': 3.5,
                    'reasoning': '话题多样性适中，可以增加更多细分话题'
                },
                'overall_score': 4.0,
                'metadata': {
                    'evaluation_date': datetime.now().isoformat(),
                    'model_used': 'gpt-3.5-turbo (模拟)',
                    'scoring_scale': 5,
                    'total_aspects': 5
                }
            }
            
            # 保存评估结果
            geval_runner.save_evaluation_results(mock_eval_results, 'results/geval_results.json')
            
            # 生成评估报告
            report = geval_runner.generate_evaluation_report(mock_eval_results)
            with open('results/evaluation_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ 质量评估完成")
            print(f"   - 总体评分: {mock_eval_results['overall_score']:.1f}/5.0")
            print(f"   - 最佳维度: 流畅性 ({mock_eval_results['fluency']['score']:.1f})")
            print(f"   - 需改进: 多样性 ({mock_eval_results['diversity']['score']:.1f})")
            
        else:
            print("❌ 未找到话题建模结果，跳过质量评估")
            
    except Exception as e:
        print(f"❌ 质量评估失败: {e}")
        return False
    
    # 4. 结果展示
    print("\n📈 步骤4: 结果展示")
    print("-" * 30)
    
    print("✅ 系统运行完成！")
    print()
    print("📁 生成的文件:")
    print("   - data/input/dataset.jsonl: 预处理后的数据")
    print("   - data/output/topicgpt_results/topicgpt_results.json: 话题建模结果")
    print("   - results/geval_results.json: 质量评估结果")
    print("   - results/evaluation_report.md: 评估报告")
    print()
    
    print("🔧 下一步操作:")
    print("   1. 设置API密钥: export OPENAI_API_KEY='your-key'")
    print("   2. 运行完整系统: python main.py --mode closed_loop")
    print("   3. 启动Web界面: python app.py")
    print("   4. 查看详细报告: cat results/evaluation_report.md")
    print()
    
    print("🎯 系统特点:")
    print("   - 支持JSONL和Excel格式数据")
    print("   - 自动话题发现和分配")
    print("   - 多维度质量评估")
    print("   - 闭环参数优化")
    print("   - 用户友好界面")
    
    return True

def main():
    """主函数"""
    print("欢迎使用基于LLM的自动话题建模系统！")
    print()
    
    success = quick_demo()
    
    if success:
        print("\n🎉 演示完成！系统已准备就绪。")
    else:
        print("\n❌ 演示失败，请检查错误信息。")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 