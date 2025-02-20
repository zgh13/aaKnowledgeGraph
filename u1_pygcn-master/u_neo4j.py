from neo4j import GraphDatabase

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # 清空数据库

    def create_graph(self, points_file, links_file):
        with self.driver.session() as session:
            # 读取知识点TF-IDF矩阵
            tfidf_data = self.read_points(points_file)
            # 读取知识点链接
            links_data = self.read_links(links_file)

            # 创建课程和知识点节点
            for course_id, knowledge_points in links_data.items():
                for kp_id in knowledge_points:
                    tfidf_value = tfidf_data.get((course_id, kp_id), 0)
                    # 这里不再限制 TF-IDF 值，只要有连接就创建
                    session.write_transaction(self.create_relationship, course_id, kp_id, tfidf_value)

    def read_points(self, points_file):
        tfidf_data = {}
        with open(points_file, 'r') as f:
            for line in f:
                values = list(map(float, line.split()))
                course_id = int(values[0])
                for kp_id, tfidf_value in enumerate(values[2:], start=1):
                    tfidf_data[(course_id, kp_id)] = tfidf_value
        return tfidf_data

    def read_links(self, links_file):
        links_data = {}
        with open(links_file, 'r') as f:
            for line in f:
                kp_id, course_id = map(int, line.split())
                if course_id not in links_data:
                    links_data[course_id] = []
                links_data[course_id].append(kp_id)
        return links_data

    @staticmethod
    def create_relationship(tx, course_id, kp_id, tfidf_value):
        tx.run("MERGE (c:Course {id: $course_id}) "
               "MERGE (k:KnowledgePoint {id: $kp_id}) "
               "MERGE (c)-[r:HAS_TFIDF {value: $tfidf_value}]->(k)",
               course_id=course_id, kp_id=kp_id, tfidf_value=tfidf_value)

if __name__ == "__main__":
    graph = Neo4jGraph("bolt://localhost:7687", "neo4j", "12345678")
    graph.clear_database()  # 清空数据库
    graph.create_graph("data/eng/knowledge_points.points", "data/eng/knowledge_points.links")
    graph.close()