#!/bin/bash

# 🚀 LLM Topic Modeling System Setup Script
# 自动安装依赖和设置项目

echo "🚀 开始设置LLM话题建模系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python版本: $python_version"

# 安装依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p data/input data/output results logs prompts

# 克隆子项目（如果需要）
if [ ! -d "topicGPT" ]; then
    echo "📥 克隆TopicGPT项目..."
    git clone https://github.com/chtmp223/topicGPT.git
fi

if [ ! -d "geval" ]; then
    echo "📥 克隆G-Eval项目..."
    git clone https://github.com/nlpyang/geval.git
fi

# 设置权限
chmod +x main.py
chmod +x app.py
chmod +x quick_start.py
chmod +x free_model_demo.py

echo "✅ 项目设置完成！"
echo ""
echo "🎯 下一步操作:"
echo "   1. 运行快速演示: python3 quick_start.py"
echo "   2. 启动Web界面: python3 app.py"
echo "   3. 运行完整系统: python3 main.py --mode closed_loop"
echo "   4. 查看Colab指南: cat COLAB_GUIDE.md"
echo ""
echo "📚 项目文档:"
echo "   - README.md: 项目概述"
echo "   - PROJECT_SUMMARY.md: 详细总结"
echo "   - COLAB_GUIDE.md: Colab使用指南"
echo ""
echo "🌐 GitHub仓库: https://github.com/yiyangliu3218/LLM_TP" 