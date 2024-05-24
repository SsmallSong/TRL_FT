import json
import os
import gzip

# 读取JSON文件
input_file = '/home/wxt/hautong/TRL_FT/rag/article.json.gz'
output_dir = '/home/wxt/hautong/TRL_FT/rag/renmin_docs'

# # 如果输出目录不存在，则创建
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# 读取JSON文件内容到一个列表
with gzip.open(input_file, 'rt', encoding='utf-8') as f:
    data = json.load(f)


# 将每个JSON对象写入单独的TXT文件
for i, article in enumerate(data, start=1):
    file_name = os.path.join(output_dir, f"{i}.txt")
    with open(file_name, 'w', encoding='utf-8') as f:
        title = article['title'].replace('\r\n', ' ')
        content = article['content'].replace('\r\n', ' ').replace('             ', ' ')
        
        # 写入URL
        f.write(f"URL: {article['url']}\n")
        # 写入标题
        f.write(f"Title: {title}\n")
        # 写入内容
        f.write(f"Content:\n{content}\n")

print("所有文章已成功保存为TXT文件。")
