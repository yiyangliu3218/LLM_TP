@echo off
REM 🚀 LLM Topic Modeling System Setup Script (Windows)
REM 自动安装依赖和设置项目

echo 🚀 开始设置LLM话题建模系统...

REM 检查Python版本
python --version
if %errorlevel% neq 0 (
    echo ❌ Python未安装，请先安装Python 3.7+
    pause
    exit /b 1
)

REM 安装依赖
echo 📦 安装Python依赖...
pip install -r requirements.txt

REM 创建必要的目录
echo 📁 创建项目目录...
if not exist "data\input" mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "results" mkdir results
if not exist "logs" mkdir logs
if not exist "prompts" mkdir prompts

REM 克隆子项目（如果需要）
if not exist "topicGPT" (
    echo 📥 克隆TopicGPT项目...
    git clone https://github.com/chtmp223/topicGPT.git
)

if not exist "geval" (
    echo 📥 克隆G-Eval项目...
    git clone https://github.com/nlpyang/geval.git
)

echo ✅ 项目设置完成！
echo.
echo 🎯 下一步操作:
echo    1. 运行快速演示: python quick_start.py
echo    2. 启动Web界面: python app.py
echo    3. 运行完整系统: python main.py --mode closed_loop
echo    4. 查看Colab指南: type COLAB_GUIDE.md
echo.
echo 📚 项目文档:
echo    - README.md: 项目概述
echo    - PROJECT_SUMMARY.md: 详细总结
echo    - COLAB_GUIDE.md: Colab使用指南
echo.
echo 🌐 GitHub仓库: https://github.com/yiyangliu3218/LLM_TP
pause 