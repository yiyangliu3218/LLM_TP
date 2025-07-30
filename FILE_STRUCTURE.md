# 📁 项目文件结构说明

## 🎯 核心文件

### 主要控制脚本
- **`main.py`** - 完整功能的主控制脚本，支持所有模式（话题建模、评估、闭环优化）
- **`colab_main.py`** - Colab专用的简化版本，专注于话题建模功能

### 核心功能模块
- **`llama3_topic_modeling.py`** - Llama 3话题建模核心实现
- **`custom_pipeline/`** - 自定义管道模块（数据处理、TopicGPT、G-Eval、反馈循环）

### 配置和依赖
- **`config.json`** - 主配置文件
- **`requirements.txt`** - Python依赖列表

## 📚 文档文件

### 主要文档
- **`README.md`** - 项目主文档，包含安装和使用说明
- **`COLAB_USAGE_EXAMPLES.md`** - Colab使用示例和参数控制指南
- **`PROJECT_SUMMARY.md`** - 项目功能总结

### 安装和部署
- **`setup.sh`** - Linux/Mac安装脚本
- **`setup.bat`** - Windows安装脚本
- **`app.py`** - Web界面（Gradio）

## 🗂️ 目录结构

```
LLM_TP/
├── main.py                    # 主控制脚本（完整功能）
├── colab_main.py             # Colab专用简化版本
├── llama3_topic_modeling.py  # Llama 3话题建模核心
├── config.json               # 配置文件
├── requirements.txt          # 依赖列表
├── README.md                 # 项目主文档
├── COLAB_USAGE_EXAMPLES.md   # Colab使用指南
├── PROJECT_SUMMARY.md        # 项目总结
├── setup.sh                  # Linux/Mac安装脚本
├── setup.bat                 # Windows安装脚本
├── app.py                    # Web界面
├── LICENSE                   # 许可证
├── .gitignore               # Git忽略文件
├── mydata.jsonl             # 示例数据文件
├── custom_pipeline/         # 自定义管道模块
├── topicGPT/                # TopicGPT项目（子模块）
├── geval/                   # G-Eval项目（子模块）
├── data/                    # 数据目录
├── results/                 # 结果目录
└── prompts/                 # 提示词目录
```

## 🚀 使用方式

### 1. 本地运行
```bash
# 完整功能
python main.py --mode closed_loop --api_type llama3

# 仅话题建模
python main.py --mode topic_modeling --api_type llama3

# 仅评估
python main.py --mode evaluation
```

### 2. Colab运行（推荐）
```python
# 克隆项目
!git clone https://github.com/yiyangliu3218/LLM_TP.git
%cd LLM_TP

# 安装依赖
!pip install -r requirements.txt
!pip install transformers torch accelerate bitsandbytes sentence-transformers scikit-learn matplotlib seaborn wordcloud

# 运行话题建模
!python colab_main.py --mode llama3 --num_topics 8 --sample_size 500
```

### 3. Web界面
```bash
python app.py
```

## 🎯 文件选择建议

### 新手用户
- 使用 `colab_main.py` - 简单易用，专注于话题建模

### 高级用户
- 使用 `main.py` - 完整功能，支持所有模式

### 开发者
- 查看 `custom_pipeline/` - 了解核心实现
- 参考 `llama3_topic_modeling.py` - 了解Llama 3集成

## 📖 文档阅读顺序

1. **`README.md`** - 了解项目概览
2. **`COLAB_USAGE_EXAMPLES.md`** - 学习具体使用方法
3. **`PROJECT_SUMMARY.md`** - 了解详细功能

---

**🎉 现在项目结构非常清晰，易于使用和维护！** 