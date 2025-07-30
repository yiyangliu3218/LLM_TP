import pandas as pd
import json
import os
from typing import List, Dict, Any

class DatasetProcessor:
    def __init__(self, data_path: str):
        """初始化数据处理器"""
        self.data_path = data_path
        self.df = None
        self.processed_data = None
        
    def load_data(self):
        """加载数据"""
        print(f"正在加载数据集: {self.data_path}")
        
        if self.data_path.endswith('.jsonl'):
            # 直接加载JSONL文件
            documents = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        documents.append(json.loads(line))
            print(f"JSONL数据集加载完成，共 {len(documents)} 条记录")
            return documents
        else:
            # 加载Excel文件
            self.df = pd.read_excel(self.data_path)
            print(f"Excel数据集加载完成，共 {len(self.df)} 条记录")
            return self.df
    
    def prepare_for_topicgpt(self, text_column: str = "text", 
                           max_docs: int = 1000, sample_size: int = None) -> List[Dict[str, Any]]:
        """将数据转换为TopicGPT需要的格式"""
        
        if self.data_path.endswith('.jsonl'):
            # 直接处理JSONL文件
            return self.prepare_jsonl_for_topicgpt(max_docs, sample_size)
        else:
            # 处理Excel文件
            return self.prepare_excel_for_topicgpt(text_column, max_docs, sample_size)
    
    def prepare_jsonl_for_topicgpt(self, max_docs: int = 1000, sample_size: int = None) -> List[Dict[str, Any]]:
        """处理JSONL文件"""
        print("正在处理JSONL数据...")
        
        # 读取JSONL文件
        documents = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    documents.append(doc)
        
        # 数据清洗
        print("正在清洗数据...")
        # 移除空值或过短的文本
        documents = [doc for doc in documents if doc.get('text', '').strip() and len(doc.get('text', '').strip()) > 10]
        
        # 采样数据（如果指定了sample_size）
        if sample_size and sample_size < len(documents):
            import random
            random.seed(42)
            documents = random.sample(documents, sample_size)
            print(f"采样 {sample_size} 条记录")
        
        # 限制最大文档数
        if len(documents) > max_docs:
            documents = documents[:max_docs]
            print(f"限制为前 {max_docs} 条记录")
        
        # 确保格式正确
        processed_documents = []
        for doc in documents:
            processed_doc = {
                "id": str(doc.get('id', '')),
                "text": str(doc.get('text', '')),
                "metadata": {
                    "original_index": len(processed_documents),
                    "source": "jsonl"
                }
            }
            processed_documents.append(processed_doc)
        
        self.processed_data = processed_documents
        print(f"JSONL数据预处理完成，共 {len(processed_documents)} 条文档")
        
        return processed_documents
    
    def prepare_excel_for_topicgpt(self, text_column: str = "Translated Post Description", 
                                 max_docs: int = 1000, sample_size: int = None) -> List[Dict[str, Any]]:
        """处理Excel文件（原有功能）"""
        
        if self.df is None:
            self.load_data()
        
        # 选择文本列
        if text_column not in self.df.columns:
            available_columns = list(self.df.columns)
            print(f"错误: 列 '{text_column}' 不存在")
            print(f"可用列: {available_columns}")
            # 尝试找到合适的文本列
            for col in ['Translated Post Description', 'Post description', 'description', 'text', 'content']:
                if col in self.df.columns:
                    text_column = col
                    print(f"使用列: {text_column}")
                    break
            else:
                raise ValueError(f"未找到合适的文本列")
        
        # 数据清洗
        print("正在清洗数据...")
        # 移除空值
        df_clean = self.df.dropna(subset=[text_column])
        # 移除过短的文本
        df_clean = df_clean[df_clean[text_column].str.len() > 10]
        
        # 采样数据（如果指定了sample_size）
        if sample_size and sample_size < len(df_clean):
            df_clean = df_clean.sample(n=sample_size, random_state=42)
            print(f"采样 {sample_size} 条记录")
        
        # 限制最大文档数
        if len(df_clean) > max_docs:
            df_clean = df_clean.head(max_docs)
            print(f"限制为前 {max_docs} 条记录")
        
        # 转换为TopicGPT格式
        documents = []
        for idx, row in df_clean.iterrows():
            doc = {
                "id": str(row.get('Post ID', idx)),
                "text": str(row[text_column]),
                "metadata": {
                    "original_index": idx,
                    "date": str(row.get('Date', '')),
                    "language": str(row.get('Language', '')),
                    "sentiment": str(row.get('Sentiment', '')),
                    "hate": str(row.get('Hate', '')),
                    "stress_anxiety": str(row.get('Stress or Anxiety', ''))
                }
            }
            documents.append(doc)
        
        self.processed_data = documents
        print(f"Excel数据预处理完成，共 {len(documents)} 条文档")
        
        return documents
    
    def save_to_jsonl(self, documents: List[Dict[str, Any]], output_path: str):
        """保存为JSONL格式"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"数据已保存到: {output_path}")
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if self.processed_data is None:
            raise ValueError("请先调用 prepare_for_topicgpt() 方法")
        
        text_lengths = [len(doc['text']) for doc in self.processed_data]
        
        stats = {
            "total_documents": len(self.processed_data),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "source_type": "jsonl" if self.data_path.endswith('.jsonl') else "excel"
        }
        
        # 如果是Excel数据，添加额外统计
        if not self.data_path.endswith('.jsonl') and self.processed_data:
            metadata_keys = set()
            for doc in self.processed_data:
                if 'metadata' in doc:
                    metadata_keys.update(doc['metadata'].keys())
            
            for key in metadata_keys:
                if key in ['language', 'sentiment', 'hate', 'stress_anxiety']:
                    values = list(set([doc['metadata'].get(key, '') for doc in self.processed_data]))
                    stats[f"{key}_categories"] = values
        
        return stats

def main():
    """测试数据处理器"""
    # 测试JSONL文件
    if os.path.exists("mydata.jsonl"):
        print("测试JSONL文件处理...")
        processor = DatasetProcessor("mydata.jsonl")
        
        # 准备数据
        documents = processor.prepare_for_topicgpt(
            max_docs=1000,  # 限制为1000条用于测试
            sample_size=500  # 随机采样500条
        )
        
        # 保存数据
        processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        
        # 获取统计信息
        stats = processor.get_data_statistics()
        print("\n数据统计信息:")
        for key, value in stats.items():
            print(f"- {key}: {value}")
    else:
        print("未找到mydata.jsonl文件，测试Excel文件处理...")
        processor = DatasetProcessor("Dataset.xlsx")
        
        # 准备数据
        documents = processor.prepare_for_topicgpt(
            text_column="Translated Post Description",
            max_docs=1000,  # 限制为1000条用于测试
            sample_size=500  # 随机采样500条
        )
        
        # 保存数据
        processor.save_to_jsonl(documents, "data/input/dataset.jsonl")
        
        # 获取统计信息
        stats = processor.get_data_statistics()
        print("\n数据统计信息:")
        for key, value in stats.items():
            print(f"- {key}: {value}")

if __name__ == "__main__":
    main() 