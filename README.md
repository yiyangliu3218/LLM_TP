# 🎯 基于LLM的自动话题建模+评估+闭环优化系统

一个集成了TopicGPT和G-Eval的完整话题建模系统，提供智能话题发现、质量评估和自动优化功能。

## 🌟 功能特点

- **📊 智能话题建模**: 基于LLM的自动话题发现和分配
- **📈 质量评估**: 多维度自动评估话题建模质量
- **🔄 闭环优化**: 基于评估结果的自动参数优化
- **🎨 用户友好**: 简洁的Web界面操作
- **🔧 灵活配置**: 支持多种LLM API和本地模型
- **📋 完整报告**: 自动生成详细的分析报告

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据预处理     │ -> │   话题建模       │ -> │   质量评估       │
│  (Excel -> JSON) │    │  (TopicGPT)     │    │   (G-Eval)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   结果展示       │ <- │   闭环优化       │ <────────┘
│  (Web界面)      │    │  (自动调参)      │
└─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd LLM_TP

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export HUGGINGFACE_API_KEY="your-huggingface-api-key"

# 或创建.env文件
echo "OPENAI_API_KEY=your-openai-api-key" > .env
echo "HUGGINGFACE_API_KEY=your-huggingface-api-key" >> .env
```

### 3. 运行系统

#### 方式一：Web界面（推荐）

```bash
python app.py
```

然后在浏览器中访问 `http://localhost:7860`

#### 方式二：命令行

```bash
# 运行完整闭环系统
python main.py --mode closed_loop --excel_path Dataset.xlsx

# 仅数据预处理
python main.py --mode preprocess --excel_path Dataset.xlsx

# 仅话题建模
python main.py --mode topic_modeling --excel_path Dataset.xlsx

# 仅质量评估
python main.py --mode evaluation
```

## 📋 数据格式

系统支持Excel格式的输入数据，需要包含以下列：

- `Post ID`: 帖子ID（可选）
- `Translated Post Description`: 翻译后的帖子描述（主要文本内容）
- `Date`: 日期（可选）
- `Language`: 语言（可选）
- `Sentiment`: 情感标签（可选）
- `Hate`: 仇恨言论标签（可选）
- `Stress or Anxiety`: 压力/焦虑标签（可选）

## ⚙️ 配置参数

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_topics` | 8 | 要生成的话题数量 |
| `max_iterations` | 3 | 最大优化轮数 |
| `sample_size` | 500 | 采样大小 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_type` | openai | API类型（openai/huggingface/local） |
| `model_name` | gpt-3.5-turbo | 模型名称 |
| `temperature` | 0.7 | 温度参数 |
| `optimization_threshold` | 3.0 | 优化阈值 |

### 评估维度

- **连贯性 (Coherence)**: 话题内部语义一致性
- **一致性 (Consistency)**: 话题间的区分度
- **流畅性 (Fluency)**: 话题描述的自然程度
- **相关性 (Relevance)**: 话题与文档的匹配度
- **多样性 (Diversity)**: 话题覆盖的广度

## 📊 输出结果

系统会生成以下文件：

```
results/
├── best_results.json          # 最佳结果
├── geval_results.json         # 质量评估结果
├── evaluation_report.md       # 评估报告
├── optimization_report.md     # 优化报告
└── optimization_history.json  # 优化历史

data/
├── input/
│   └── dataset.jsonl         # 预处理后的数据
└── output/
    └── topicgpt_results/
        └── topicgpt_results.json  # 话题建模结果
```

## 🔧 高级配置

### 自定义模型配置

```python
config = {
    'api_type': 'huggingface',
    'model_name': 'microsoft/DialoGPT-medium',
    'num_topics': 10,
    'temperature': 0.8,
    'max_tokens': 1500,
    'evaluation_aspects': ['coherence', 'relevance', 'diversity']
}
```

### 本地模型支持

```python
config = {
    'api_type': 'local',
    'model_path': './models/llama-2-7b-chat',
    'device': 'cuda'  # 或 'cpu'
}
```

## 🛠️ 开发指南

### 项目结构

```
LLM_TP/
├── custom_pipeline/           # 自定义模块
│   ├── data_processor.py     # 数据处理器
│   ├── topicgpt_runner.py    # TopicGPT运行器
│   ├── geval_runner.py       # G-Eval运行器
│   └── feedback_loop.py      # 反馈优化器
├── topicGPT/                 # TopicGPT项目
├── geval/                    # G-Eval项目
├── data/                     # 数据目录
├── results/                  # 结果目录
├── main.py                   # 主控制脚本
├── app.py                    # Web界面
├── requirements.txt          # 依赖文件
└── README.md                 # 项目文档
```

### 扩展功能

1. **添加新的评估维度**:
   - 在 `geval_runner.py` 中添加新的评估函数
   - 更新 `evaluation_aspects` 配置

2. **支持新的模型**:
   - 在 `topicgpt_runner.py` 中添加新的模型调用函数
   - 更新配置选项

3. **自定义优化策略**:
   - 在 `feedback_loop.py` 中添加新的优化函数
   - 更新 `improvement_strategies` 字典

## 🐛 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确设置
   - 确认网络连接正常
   - 检查API配额是否充足

2. **内存不足**
   - 减少 `sample_size` 参数
   - 使用更小的模型
   - 分批处理数据

3. **话题质量不佳**
   - 调整 `temperature` 参数
   - 增加 `num_topics` 数量
   - 优化prompt模板

### 日志和调试

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 查看系统日志
tail -f logs/system.log
```

## 📈 性能优化

### 提高处理速度

1. **并行处理**: 使用多进程处理大量文档
2. **缓存机制**: 缓存中间结果避免重复计算
3. **模型优化**: 使用量化模型减少内存占用

### 提高话题质量

1. **Prompt工程**: 优化prompt模板
2. **参数调优**: 根据数据特点调整参数
3. **后处理**: 添加话题合并和过滤逻辑

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [TopicGPT](https://github.com/chtmp223/topicGPT) - 话题建模框架
- [G-Eval](https://github.com/nlpyang/geval) - 质量评估框架
- [Gradio](https://gradio.app/) - Web界面框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 本系统需要有效的LLM API密钥才能正常运行。请确保在使用前正确配置相关API密钥。 