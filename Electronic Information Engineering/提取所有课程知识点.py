import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# 读取停止词词典
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    return stopwords

# 分词并过滤特定词性和停止词
def filter_words(text, exclude_pos, stopwords):
    words = pseg.cut(text)
    filtered_words = [word for word, flag in words if flag not in exclude_pos and word not in stopwords]
    logging.debug(f"过滤后的词汇: {filtered_words}")
    return filtered_words

# 读取文件内容并检测编码
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logging.info(f"成功读取文件: {file_path} 使用编码: utf-8")
        return text
    except UnicodeDecodeError:
        logging.error(f"无法读取文件: {file_path}，请检查文件编码或内容。")
        raise

# 定义需要排除的词性和停止词
exclude_pos = {'v', 'nr', 'ns', 'm', 'd'}  # 动词, 人名, 地名, 数字, 副词
stopwords_file = 'stopwords.txt'  # 停止词词典路径
stopwords = load_stopwords(stopwords_file)

# 加载自定义词典
custom_dict_path = 'custom_dict.txt'
jieba.load_userdict(custom_dict_path)

# 读取文件夹内所有txt文件
folder_path = 'pdf_output'
all_data = []
knowledge_point_lines = []  # 用于存储知识点的字符串行

# 用于存储知识点和课程的关系
links = []
knowledge_point_id_map = {}  # 存储知识点及其对应的ID
knowledge_point_counter = 1  # 知识点计数器从1开始
course_id_map = {}  # 存储课程及其对应的ID
course_counter = 1  # 课程计数器从1开始

# 初始化 .points 文件
points_file = 'knowledge_points.points'
points_data = []  # 用于存储每行数据
all_feature_names = set()  # 用于存储所有知识点的集合

# 首先遍历所有文件，收集所有知识点
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        try:
            text = read_file(file_path)
            filtered_words = filter_words(text, exclude_pos, stopwords)
            all_feature_names.update(filtered_words)  # 更新所有知识点集合
        except Exception as e:
            logging.error(f"读取文件时出错: {file_path} - 错误信息: {e}")

# 将所有知识点转换为列表
all_feature_names = list(all_feature_names)

# 重新遍历文件，计算 TF-IDF
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        try:
            text = read_file(file_path)
            filtered_words = filter_words(text, exclude_pos, stopwords)
            filtered_text = " ".join(filtered_words)

            # 计算TF-IDF
            vectorizer = TfidfVectorizer(
                vocabulary=all_feature_names,  # 使用统一的特征集
                tokenizer=lambda x: x.split(),
                token_pattern=None,
                stop_words=None,
                lowercase=False
            )
            tfidf_matrix = vectorizer.fit_transform([filtered_text])
            tfidf_scores = tfidf_matrix.toarray()[0]

            # 创建DataFrame并将TF-IDF值大于一定阈值的词汇筛选出来
            df = pd.DataFrame({'word': all_feature_names, 'tfidf': tfidf_scores})

            # 添加课程名称列
            course_name = os.path.basename(file_path).replace('.txt', '')  # 使用文件名作为课程名称

            # 为课程分配 ID
            if course_name not in course_id_map:
                course_id_map[course_name] = course_counter
                course_counter += 1

            course_id = course_id_map[course_name]
            df['course'] = course_name
            df['course_id'] = course_id  # 添加课程 ID 列

            all_data.append(df)

            # 生成知识点和课程的链接
            for index, row in df.iterrows():
                knowledge_point = row['word']
                tfidf_value = row['tfidf']

                # 为知识点分配 ID
                if knowledge_point not in knowledge_point_id_map:
                    knowledge_point_id_map[knowledge_point] = knowledge_point_counter
                    knowledge_point_counter += 1

                knowledge_point_id = knowledge_point_id_map[knowledge_point]
                # 仅当TF-IDF值大于某个阈值时才创建链接
                if tfidf_value > 0.1:  # 设定一个合理的阈值
                    links.append(f"{knowledge_point_id} {course_id}")  # 使用课程 ID

            # 生成 .points 文件的 TF-IDF 值表示
            tfidf_values = [0] * (knowledge_point_counter + 1)  # 初始化为 0，长度为知识点计数器 + 1
            for index, row in df.iterrows():
                knowledge_point = row['word']
                if knowledge_point in knowledge_point_id_map:
                    knowledge_point_id = knowledge_point_id_map[knowledge_point]
                    tfidf_values[knowledge_point_id] = row['tfidf']  # 存在则用 TF-IDF 值替代

            points_data.append(f"{course_id} " + " ".join(map(str, tfidf_values)))  # 添加到数据列表

        except Exception as e:
            logging.error(f"读取文件时出错: {file_path} - 错误信息: {e}")

# 写入 .points 文件
with open(points_file, 'w', encoding='utf-8') as kp_file:
    for line in points_data:
        kp_file.write(line + '\n')
print(f"知识点文件已保存为 {points_file}")

# 输出每行的特征值个数
for i, line in enumerate(points_data):
    feature_count = len(line.strip().split()) - 1  # 减去课程 ID
    print(f"第 {i + 1} 行特征值个数: {feature_count}")

# 合并所有DataFrame
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    # 在 DataFrame 中添加知识点 ID
    final_df['knowledge_point_id'] = final_df['word'].map(knowledge_point_id_map)

    # 按照知识点的字母顺序排序
    final_df = final_df.sort_values(by='word')

    # 创建透视表，课程名为列，知识点为行，TF-IDF值为值
    pivot_df = final_df.pivot_table(index=['knowledge_point_id', 'word'], columns='course_id', values='tfidf', fill_value=0)

    # 修改课程 ID 和知识点 ID 从 1 开始
    pivot_df.index = pd.MultiIndex.from_tuples([(kp_id, word) for kp_id, word in zip(range(1, len(pivot_df.index) + 1), pivot_df.index.get_level_values(1))])
    pivot_df.columns = range(1, len(pivot_df.columns) + 1)

    # 输出结果到CSV文件，确保使用 utf-8-sig 编码
    output_file = 'knowledgepoint_tfidf_matrix.csv'
    pivot_df.to_csv(output_file, encoding='utf-8-sig')

    print(f"知识点-TF-IDF矩阵已保存到 {output_file}")

    # 生成 course.txt 文件
    with open('course.txt', 'w', encoding='utf-8') as course_file:
        for course_name, course_id in course_id_map.items():
            course_file.write(f"{course_id} {course_name}\n")
    print("课程文件已保存为 course.txt")

    # 生成 .links 文件
    with open('knowledge_points.links', 'w', encoding='utf-8') as links_file:
        for link in links:
            links_file.write(link + '\n')
    print("链接文件已保存为 knowledge_points.links")
else:
    print("没有读取到任何有效的课程数据。")