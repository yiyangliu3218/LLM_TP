#!/usr/bin/env python3
"""
数据解压缩脚本
用于解压压缩的数据文件
"""

import os
import gzip
import json
import argparse

def extract_gz_file(gz_path, output_path=None):
    """解压.gz文件"""
    if output_path is None:
        output_path = gz_path.replace('.gz', '')
    
    print(f"正在解压 {gz_path}...")
    
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f_in.read())
    
    print(f"✅ 解压完成: {output_path}")
    return output_path

def verify_jsonl_file(file_path):
    """验证JSONL文件格式"""
    print(f"验证文件格式: {file_path}")
    
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    count += 1
                    if count >= 5:  # 只检查前5行
                        break
        
        print(f"✅ 文件格式正确，至少包含 {count} 条记录")
        return True
    except Exception as e:
        print(f"❌ 文件格式错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='数据解压缩工具')
    parser.add_argument('--input', type=str, default='data/input/mydata.jsonl.gz',
                       help='压缩文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（可选）')
    parser.add_argument('--verify', action='store_true',
                       help='验证解压后的文件格式')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 文件 '{args.input}' 不存在")
        return 1
    
    try:
        # 解压文件
        output_file = extract_gz_file(args.input, args.output)
        
        # 验证文件格式
        if args.verify:
            verify_jsonl_file(output_file)
        
        print(f"\n🎉 数据解压完成！")
        print(f"📁 输出文件: {output_file}")
        print(f"💡 现在可以使用这个文件进行话题建模了")
        
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
