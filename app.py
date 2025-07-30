#!/usr/bin/env python3
"""
基于LLM的自动话题建模+评估+闭环优化系统
Gradio Web界面
"""

import gradio as gr
import os
import sys
import json
import tempfile
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

# 添加自定义模块路径
sys.path.append('./custom_pipeline')

def setup_environment():
    """设置环境"""
    directories = ['data/input', 'data/output', 'results', 'prompts', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def analyze_excel_file(excel_file) -> Dict[str, Any]:
    """分析上传的Excel文件"""
    if excel_file is None:
        return {"error": "请上传Excel文件"}
    
    try:
        # 保存临时文件
        temp_path = tempfile.mktemp(suffix='.xlsx')
        with open(temp_path, 'wb') as f:
            f.write(excel_file)
        
        # 读取Excel文件
        df = pd.read_excel(temp_path)
        
        # 分析数据
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }
        
        # 清理临时文件
        os.unlink(temp_path)
        
        return analysis
        
    except Exception as e:
        return {"error": f"文件分析失败: {str(e)}"}

def process_excel_file(excel_file, num_topics, max_iterations, sample_size, 
                      api_type, model_name, temperature, optimization_threshold) -> Dict[str, Any]:
    """处理上传的Excel文件"""
    
    if excel_file is None:
        return {"error": "请上传Excel文件"}
    
    try:
        # 保存临时文件
        temp_path = tempfile.mktemp(suffix='.xlsx')
        with open(temp_path, 'wb') as f:
            f.write(excel_file)
        
        # 创建配置
        config = {
            'api_type': api_type,
            'model_name': model_name,
            'num_topics': num_topics,
            'max_iterations': max_iterations,
            'sample_size': sample_size,
            'temperature': temperature,
            'optimization_threshold': optimization_threshold,
            'evaluation_aspects': [
                'coherence', 'consistency', 'fluency', 'relevance', 'diversity'
            ]
        }
        
        # 运行闭环系统
        from custom_pipeline.feedback_loop import ClosedLoopTopicModeling
        
        pipeline = ClosedLoopTopicModeling(temp_path, config)
        results = pipeline.run_complete_pipeline(
            max_iterations=max_iterations,
            sample_size=sample_size
        )
        
        # 清理临时文件
        os.unlink(temp_path)
        
        # 格式化结果
        formatted_results = {
            "最佳分数": f"{results.get('evaluation', {}).get('overall_score', 0):.2f}/5.00",
            "优化轮数": results.get('iteration', 0),
            "话题数量": len(results.get('topics', {})),
            "处理文档数": results.get('topics', {}).get('metadata', {}).get('total_documents', 0),
            "各维度评分": {}
        }
        
        # 添加各维度评分
        evaluation = results.get('evaluation', {})
        for aspect, result in evaluation.items():
            if aspect not in ['overall_score', 'metadata']:
                formatted_results["各维度评分"][aspect] = f"{result.get('score', 0):.2f}/5.00"
        
        return formatted_results
        
    except Exception as e:
        return {"error": f"处理失败: {str(e)}"}

def run_data_preprocessing_only(excel_file, sample_size) -> Dict[str, Any]:
    """仅运行数据预处理"""
    if excel_file is None:
        return {"error": "请上传Excel文件"}
    
    try:
        # 保存临时文件
        temp_path = tempfile.mktemp(suffix='.xlsx')
        with open(temp_path, 'wb') as f:
            f.write(excel_file)
        
        from custom_pipeline.data_processor import DatasetProcessor
        
        processor = DatasetProcessor(temp_path)
        documents = processor.prepare_for_topicgpt(
            text_column="Translated Post Description",
            max_docs=1000,
            sample_size=sample_size
        )
        processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        stats = processor.get_data_statistics()
        
        # 清理临时文件
        os.unlink(temp_path)
        
        return {
            "处理文档数": stats['total_documents'],
            "平均文本长度": f"{stats['avg_text_length']:.1f}",
            "最短文本长度": stats['min_text_length'],
            "最长文本长度": stats['max_text_length'],
            "语言种类": len(stats['languages']),
            "情感类别": len(stats['sentiments'])
        }
        
    except Exception as e:
        return {"error": f"数据预处理失败: {str(e)}"}

def run_topic_modeling_only(excel_file, num_topics, sample_size, api_type, model_name, temperature) -> Dict[str, Any]:
    """仅运行话题建模"""
    if excel_file is None:
        return {"error": "请上传Excel文件"}
    
    try:
        # 保存临时文件
        temp_path = tempfile.mktemp(suffix='.xlsx')
        with open(temp_path, 'wb') as f:
            f.write(excel_file)
        
        # 数据预处理
        from custom_pipeline.data_processor import DatasetProcessor
        processor = DatasetProcessor(temp_path)
        documents = processor.prepare_for_topicgpt(
            text_column="Translated Post Description",
            max_docs=1000,
            sample_size=sample_size
        )
        processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        
        # 话题建模
        config = {
            'api_type': api_type,
            'model_name': model_name,
            'num_topics': num_topics,
            'temperature': temperature
        }
        
        from custom_pipeline.topicgpt_runner import CustomTopicGPTRunner
        runner = CustomTopicGPTRunner(config)
        results = runner.run_topic_modeling(documents)
        
        # 清理临时文件
        os.unlink(temp_path)
        
        return {
            "生成话题数": len(results.get('topics', {})),
            "处理文档数": len(documents),
            "话题列表": list(results.get('topics', {}).keys())
        }
        
    except Exception as e:
        return {"error": f"话题建模失败: {str(e)}"}

def run_evaluation_only() -> Dict[str, Any]:
    """仅运行评估"""
    try:
        if not os.path.exists('data/output/topicgpt_results/topicgpt_results.json'):
            return {"error": "未找到话题建模结果，请先运行话题建模"}
        
        from custom_pipeline.geval_runner import CustomGEvalRunner
        geval_runner = CustomGEvalRunner()
        
        geval_input = geval_runner.prepare_geval_input(
            'data/output/topicgpt_results/topicgpt_results.json',
            'data/input/dataset.jsonl'
        )
        eval_results = geval_runner.run_evaluation(geval_input)
        
        return {
            "总体评分": f"{eval_results['overall_score']:.2f}/5.00",
            "各维度评分": {
                aspect: f"{result.get('score', 0):.2f}/5.00"
                for aspect, result in eval_results.items()
                if aspect not in ['overall_score', 'metadata']
            }
        }
        
    except Exception as e:
        return {"error": f"评估失败: {str(e)}"}

def get_results_summary() -> str:
    """获取结果摘要"""
    try:
        summary = []
        
        # 检查话题建模结果
        if os.path.exists('data/output/topicgpt_results/topicgpt_results.json'):
            with open('data/output/topicgpt_results/topicgpt_results.json', 'r', encoding='utf-8') as f:
                topic_results = json.load(f)
            summary.append(f"✅ 话题建模完成，生成了 {len(topic_results.get('topics', {}))} 个话题")
        
        # 检查评估结果
        if os.path.exists('results/geval_results.json'):
            with open('results/geval_results.json', 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            summary.append(f"✅ 质量评估完成，总分: {eval_results.get('overall_score', 0):.2f}/5.00")
        
        # 检查优化报告
        if os.path.exists('results/optimization_report.md'):
            summary.append("✅ 优化报告已生成")
        
        if not summary:
            summary.append("暂无结果")
        
        return "\n".join(summary)
        
    except Exception as e:
        return f"获取结果摘要失败: {str(e)}"

# 创建Gradio界面
def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="基于LLM的自动话题建模+评估+闭环优化系统", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🎯 基于LLM的自动话题建模+评估+闭环优化系统
        
        本系统集成了TopicGPT和G-Eval，提供完整的话题建模、质量评估和自动优化功能。
        
        ## 功能特点
        - 📊 **智能话题建模**: 基于LLM的自动话题发现和分配
        - 📈 **质量评估**: 多维度自动评估话题建模质量
        - 🔄 **闭环优化**: 基于评估结果的自动参数优化
        - 🎨 **用户友好**: 简洁的Web界面操作
        
        ## 使用说明
        1. 上传Excel数据集文件
        2. 配置参数（可选）
        3. 选择运行模式
        4. 查看结果和报告
        """)
        
        with gr.Tab("📋 数据分析"):
            gr.Markdown("### 数据集分析")
            with gr.Row():
                excel_file_analysis = gr.File(label="上传Excel文件", file_types=['.xlsx', '.xls'])
                analyze_btn = gr.Button("🔍 分析数据", variant="primary")
            
            analysis_output = gr.JSON(label="数据分析结果")
            analyze_btn.click(analyze_excel_file, inputs=[excel_file_analysis], outputs=[analysis_output])
        
        with gr.Tab("🚀 完整流程"):
            gr.Markdown("### 闭环话题建模系统")
            
            with gr.Row():
                with gr.Column():
                    excel_file = gr.File(label="上传Excel数据集", file_types=['.xlsx', '.xls'])
                    
                    gr.Markdown("#### 基本参数")
                    with gr.Row():
                        num_topics = gr.Slider(3, 15, value=8, step=1, label="话题数量")
                        max_iterations = gr.Slider(1, 5, value=3, step=1, label="最大优化轮数")
                    
                    sample_size = gr.Slider(100, 1000, value=500, step=50, label="采样大小")
                    
                    gr.Markdown("#### 模型参数")
                    with gr.Row():
                        api_type = gr.Dropdown(
                            choices=['openai', 'huggingface', 'local'],
                            value='openai',
                            label="API类型"
                        )
                        model_name = gr.Textbox(
                            value='gpt-3.5-turbo',
                            label="模型名称"
                        )
                    
                    with gr.Row():
                        temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="温度参数")
                        optimization_threshold = gr.Slider(1.0, 5.0, value=3.0, step=0.1, label="优化阈值")
                
                with gr.Column():
                    run_btn = gr.Button("🎯 运行完整流程", variant="primary", size="lg")
                    results_output = gr.JSON(label="运行结果")
        
            run_btn.click(
                process_excel_file,
                inputs=[excel_file, num_topics, max_iterations, sample_size, 
                       api_type, model_name, temperature, optimization_threshold],
                outputs=[results_output]
            )
        
        with gr.Tab("🔧 分步执行"):
            gr.Markdown("### 分步骤执行")
            
            with gr.Accordion("📋 数据预处理", open=False):
                with gr.Row():
                    excel_file_preprocess = gr.File(label="上传Excel文件", file_types=['.xlsx', '.xls'])
                    sample_size_preprocess = gr.Slider(100, 1000, value=500, step=50, label="采样大小")
                    preprocess_btn = gr.Button("📋 数据预处理", variant="secondary")
                
                preprocess_output = gr.JSON(label="预处理结果")
                preprocess_btn.click(
                    run_data_preprocessing_only,
                    inputs=[excel_file_preprocess, sample_size_preprocess],
                    outputs=[preprocess_output]
                )
            
            with gr.Accordion("🚀 话题建模", open=False):
                with gr.Row():
                    excel_file_topic = gr.File(label="上传Excel文件", file_types=['.xlsx', '.xls'])
                    num_topics_topic = gr.Slider(3, 15, value=8, step=1, label="话题数量")
                    sample_size_topic = gr.Slider(100, 1000, value=500, step=50, label="采样大小")
                
                with gr.Row():
                    api_type_topic = gr.Dropdown(choices=['openai', 'huggingface', 'local'], value='openai', label="API类型")
                    model_name_topic = gr.Textbox(value='gpt-3.5-turbo', label="模型名称")
                    temperature_topic = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="温度参数")
                
                topic_btn = gr.Button("🚀 话题建模", variant="secondary")
                topic_output = gr.JSON(label="话题建模结果")
                
                topic_btn.click(
                    run_topic_modeling_only,
                    inputs=[excel_file_topic, num_topics_topic, sample_size_topic, 
                           api_type_topic, model_name_topic, temperature_topic],
                    outputs=[topic_output]
                )
            
            with gr.Accordion("📊 质量评估", open=False):
                eval_btn = gr.Button("📊 质量评估", variant="secondary")
                eval_output = gr.JSON(label="评估结果")
                eval_btn.click(run_evaluation_only, outputs=[eval_output])
        
        with gr.Tab("📈 结果查看"):
            gr.Markdown("### 结果摘要")
            
            refresh_btn = gr.Button("🔄 刷新结果", variant="secondary")
            summary_output = gr.Textbox(label="结果摘要", lines=10, interactive=False)
            
            refresh_btn.click(get_results_summary, outputs=[summary_output])
            
            gr.Markdown("### 生成的文件")
            gr.Markdown("""
            - `data/input/dataset.jsonl`: 预处理后的数据
            - `data/output/topicgpt_results/topicgpt_results.json`: 话题建模结果
            - `results/geval_results.json`: 质量评估结果
            - `results/evaluation_report.md`: 评估报告
            - `results/optimization_report.md`: 优化报告
            - `results/best_results.json`: 最佳结果
            """)
        
        with gr.Tab("⚙️ 配置说明"):
            gr.Markdown("""
            ## 参数说明
            
            ### 基本参数
            - **话题数量**: 要生成的话题数量，建议3-15个
            - **最大优化轮数**: 闭环优化的最大迭代次数
            - **采样大小**: 从数据集中采样的文档数量
            
            ### 模型参数
            - **API类型**: 
              - `openai`: 使用OpenAI API（需要OPENAI_API_KEY）
              - `huggingface`: 使用HuggingFace API（需要HUGGINGFACE_API_KEY）
              - `local`: 使用本地模型
            - **模型名称**: 具体的模型名称
            - **温度参数**: 控制输出的随机性，0.1-1.0
            - **优化阈值**: 低于此分数的话题将被优化
            
            ### 评估维度
            - **连贯性 (Coherence)**: 话题内部语义一致性
            - **一致性 (Consistency)**: 话题间的区分度
            - **流畅性 (Fluency)**: 话题描述的自然程度
            - **相关性 (Relevance)**: 话题与文档的匹配度
            - **多样性 (Diversity)**: 话题覆盖的广度
            """)
    
    return demo

def main():
    """主函数"""
    # 设置环境
    setup_environment()
    
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main() 