import json


# 1. 读取 JSON 文件
def json_to_jsonl(json_file_path, jsonl_file_path):
    """
    将 JSON 文件转换为 JSONL 文件。

    参数:
        json_file_path (str): JSON 文件的路径。
        jsonl_file_path (str): 输出的 JSONL 文件路径。
    """
    # 打开 JSON 文件并加载数据
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)  # 加载 JSON 数据

    # 2. 写入 JSONL 文件
    with open(jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
        if isinstance(data, list):  # 如果 JSON 是一个数组
            for item in data:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")  # 每行写入一个 JSON 对象
        elif isinstance(data, dict):  # 如果 JSON 是一个对象
            jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")  # 写入单个 JSON 对象
        else:
            raise ValueError("JSON 文件必须是数组或对象")


# 3. 调用函数进行转换
json_file_path = "processed_data/processed_data_all.json"  # 输入的 JSON 文件路径
jsonl_file_path = "processed_data/processed_data_all.jsonl"  # 输出的 JSONL 文件路径
json_to_jsonl(json_file_path, jsonl_file_path)

print(f"转换完成：{jsonl_file_path}")