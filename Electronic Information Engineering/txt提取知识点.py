import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import chardet
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取单个课本txt文件并自动检测编码
file_path = r'清洗output\信息论与编码理论.txt'

# 读取停止词词典
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    return stopwords

# 分词并过滤特定词性和停止词
def filter_words(text, exclude_pos, stopwords):
    words = pseg.cut(text)
    filtered_words = [word for word, flag in words if flag not in exclude_pos and word not in stopwords]
    logging.debug(f"Filtered words: {filtered_words}")
    return filtered_words

# 读取单个文件内容并检测编码
def read_file(file_path):
    # 尝试自动检测编码
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        detected_encoding = result.get('encoding')
        confidence = result.get('confidence')

        if detected_encoding is None or confidence < 0.7:
            logging.warning(f"Encoding detection failed or low confidence for {file_path}. Trying default encodings.")
            encodings_to_try = ['utf-8', 'gbk', 'gb2312']
        else:
            logging.info(f"Detected encoding for {file_path}: {detected_encoding} (Confidence: {confidence:.2f})")
            encodings_to_try = [detected_encoding]

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                logging.info(f"Successfully read {file_path} using {encoding} encoding.")
                return text
            except UnicodeDecodeError:
                logging.warning(f"Failed to read {file_path} using {encoding} encoding. Trying next encoding.")
                continue

        # 如果所有尝试都失败，抛出异常
        raise UnicodeDecodeError(f"Unable to decode {file_path} with any of the tried encodings.")

# 定义需要排除的词性和停止词
exclude_pos = {'v', 'nr', 'ns', 'm', 'd'}  # 动词, 人名, 地名, 数字, 副词
stopwords_file = r'stopwords.txt'  # 停止词词典路径
stopwords = load_stopwords(stopwords_file)

# 加载自定义词典
custom_dict_path = r'custom_dict.txt'
jieba.load_userdict(custom_dict_path)

# 读取文件内容
try:
    text = read_file(file_path)
except Exception as e:
    logging.error(f"Error reading file {file_path}: {e}")
    exit(1)

# 分词并过滤特定词性和停止词
filtered_words = filter_words(text, exclude_pos, stopwords)

# 将过滤后的词语重新组合成字符串
filtered_text = " ".join(filtered_words)

# 计算TF-IDF
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split(),  # 使用空格分割的词语列表作为输入
    token_pattern=None,             # 禁用默认的正则表达式分词
    stop_words=None,                # 不使用内置的停止词
    lowercase=False                 # 不转换为小写
)
tfidf_matrix = vectorizer.fit_transform([filtered_text])
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]

# 创建DataFrame并将TF-IDF值大于一定阈值的词汇筛选出来
df = pd.DataFrame({'word': feature_names, 'tfidf': tfidf_scores})

# 添加课程名称列
course_name = os.path.basename(file_path)  # 使用文件名作为课程名称
df['course'] = course_name

# 调整列顺序以符合“课程-知识点-TF-IDF”格式
df = df[['course', 'word', 'tfidf']]

# 输出结果到CSV文件，确保使用 utf-8-sig 编码
output_file = 'single_course_knowledgepoint_tfidf.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"单个课程-知识点-TF-IDF矩阵已保存到 {output_file}")