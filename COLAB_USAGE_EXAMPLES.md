# 🚀 Colab使用示例 - 灵活的参数控制

## 📋 基本用法

### 1. 克隆代码并安装依赖

```python
# 克隆你的GitHub仓库
!git clone https://github.com/yiyangliu3218/LLM_TP.git
%cd LLM_TP

# 安装依赖
!pip install -r requirements.txt
!pip install transformers torch accelerate bitsandbytes
!pip install sentence-transformers scikit-learn
!pip install matplotlib seaborn plotly umap-learn hdbscan wordcloud
```

### 2. 上传数据文件

```python
from google.colab import files
uploaded = files.upload()  # 上传mydata.jsonl文件
```

## 🎯 使用main.py（完整功能）

### 示例1：运行Llama 3话题建模

```python
# 使用Llama 3开源模型
!python main.py --mode topic_modeling --api_type llama3 --num_topics 10 --sample_size 300
```

### 示例2：运行完整闭环系统

```python
# 运行完整的闭环优化系统
!python main.py --mode closed_loop --api_type llama3 --num_topics 8 --max_iterations 3
```

### 示例3：仅运行数据预处理

```python
# 只进行数据预处理
!python main.py --mode preprocess --data_path mydata.jsonl --sample_size 500
```

### 示例4：仅运行评估

```python
# 只进行质量评估
!python main.py --mode evaluation
```

## 🦙 使用colab_main.py（简化版本）

### 示例1：Llama 3话题建模

```python
# 使用Llama 3进行话题建模
!python colab_main.py --mode llama3 --num_topics 8 --sample_size 500
```

### 示例2：Sentence Transformers话题建模

```python
# 使用sentence-transformers进行话题建模
!python colab_main.py --mode sentence_transformers --num_topics 6 --sample_size 300
```

### 示例3：比较两种方法

```python
# 同时运行两种方法并比较结果
!python colab_main.py --mode both --num_topics 8 --sample_size 400
```

## 🔧 参数说明

### main.py 参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--mode` | 运行模式 | `closed_loop` | `preprocess`, `topic_modeling`, `evaluation`, `closed_loop` |
| `--api_type` | API类型 | `llama3` | `openai`, `huggingface`, `local`, `llama3` |
| `--model_name` | 模型名称 | `meta-llama/Meta-Llama-3-8B-Instruct` | 任意模型名称 |
| `--num_topics` | 话题数量 | `8` | 1-20 |
| `--sample_size` | 采样大小 | `500` | 100-1000 |
| `--max_iterations` | 最大迭代次数 | `3` | 1-10 |
| `--temperature` | 温度参数 | `0.7` | 0.1-1.0 |

### colab_main.py 参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--mode` | 运行模式 | `llama3` | `llama3`, `sentence_transformers`, `both` |
| `--num_topics` | 话题数量 | `8` | 1-20 |
| `--sample_size` | 采样大小 | `500` | 100-1000 |

## 🎯 实际使用场景

### 场景1：快速探索（小数据集）

```python
# 快速测试，使用小数据集
!python colab_main.py --mode both --num_topics 5 --sample_size 200
```

### 场景2：详细分析（大数据集）

```python
# 详细分析，使用大数据集
!python main.py --mode closed_loop --api_type llama3 --num_topics 12 --sample_size 800 --max_iterations 5
```

### 场景3：对比实验

```python
# 对比不同参数的效果
!python colab_main.py --mode llama3 --num_topics 6 --sample_size 300
!python colab_main.py --mode llama3 --num_topics 8 --sample_size 300
!python colab_main.py --mode llama3 --num_topics 10 --sample_size 300
```

### 场景4：资源优化

```python
# 内存不足时使用较小的参数
!python colab_main.py --mode sentence_transformers --num_topics 6 --sample_size 200
```

## 📊 结果查看

### 查看生成的话题

```python
import json

# 读取Llama 3结果
with open('llama3_topic_modeling_results.json', 'r', encoding='utf-8') as f:
    llama_results = json.load(f)

print("Llama 3话题:")
for topic_id, topic in llama_results['topics'].items():
    print(f"{topic_id}: {topic['title']}")
    print(f"  关键词: {', '.join(topic['keywords'][:5])}")
    print(f"  文档数: {topic['size']}")
```

### 查看评估结果

```python
# 查看质量评估
print(f"总体评分: {llama_results['overall_score']:.2f}/5.0")
for aspect, score in llama_results['evaluation'].items():
    print(f"{aspect}: {score:.1f}/5.0")
```

### 下载结果文件

```python
from google.colab import files

# 下载结果文件
files.download('llama3_topic_modeling_results.json')
files.download('sentence_transformers_results.json')
```

## 🆘 常见问题解决

### 内存不足

```python
# 减少采样大小
!python colab_main.py --mode llama3 --sample_size 200

# 或使用更轻量的方法
!python colab_main.py --mode sentence_transformers --sample_size 200
```

### 运行时间太长

```python
# 减少话题数量
!python colab_main.py --mode llama3 --num_topics 5

# 或减少采样大小
!python colab_main.py --mode llama3 --sample_size 300
```

### 模型下载失败

```python
# 使用备用模型
!python main.py --mode topic_modeling --api_type sentence_transformers
```

## 🎉 一键运行示例

```python
# 完整的一键运行示例
import os
from google.colab import files

print("🚀 开始话题建模...")

# 1. 克隆仓库
!git clone https://github.com/yiyangliu3218/LLM_TP.git
%cd LLM_TP

# 2. 安装依赖
!pip install -r requirements.txt
!pip install transformers torch accelerate bitsandbytes
!pip install sentence-transformers scikit-learn
!pip install matplotlib seaborn plotly umap-learn hdbscan wordcloud

# 3. 上传数据
uploaded = files.upload()

if 'mydata.jsonl' in uploaded:
    print("✅ 数据上传成功！")
    
    # 4. 运行话题建模（选择一种方法）
    # 方法A: Llama 3
    !python colab_main.py --mode llama3 --num_topics 8 --sample_size 500
    
    # 方法B: Sentence Transformers
    # !python colab_main.py --mode sentence_transformers --num_topics 8 --sample_size 500
    
    # 方法C: 比较两种方法
    # !python colab_main.py --mode both --num_topics 8 --sample_size 500
    
    # 5. 下载结果
    files.download('llama3_topic_modeling_results.json')
    print("🎉 完成！")
else:
    print("❌ 请上传mydata.jsonl文件")
```

---

**🎯 现在你可以灵活地使用不同的参数来控制话题建模的功能了！** 