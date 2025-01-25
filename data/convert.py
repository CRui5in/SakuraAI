import json

# 示例输入文件路径
input_file = "processed_data/processed_data_all.jsonl"
# 示例输出文件路径
output_file = "processed_data/converted_data.jsonl"

# 定义转换函数
def convert_to_role_format(prompt, completion):
    """
    将 prompt 和 completion 转换为基于 role 的对话格式。

    参数:
        prompt (str): 原始 prompt 文本。
        completion (str): 原始 completion 文本。

    返回:
        messages (list): 转换后的对话格式，包含 role 和 content。
    """
    # 删除 prompt 和 completion 中的 #、*、"用户："、"助手："、- 和 \n
    prompt = (
        prompt.replace("#", "")
        .replace("*", "")
        .replace("用户：", "")
        .replace("-", "")
        .replace("\n", " ")
    )
    completion = (
        completion.replace("#", "")
        .replace("*", "")
        .replace("助手：", "")
        .replace("-", "")
        .replace("\n", " ")
    )

    # 转换为基于 role 的对话格式
    messages = [
        {"role": "user", "content": prompt.strip()},  # 用户输入
        {"role": "assistant", "content": completion.strip()},  # 助手回复
    ]
    return messages

# 读取原始数据并转换
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析 JSONL 文件中的每一行
        data = json.loads(line.strip())
        prompt = data["prompt"]
        completion = data["completion"]

        # 转换为基于 role 的对话格式
        messages = convert_to_role_format(prompt, completion)

        # 将转换后的数据写入新的 JSONL 文件
        outfile.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

print(f"数据转换完成！转换后的文件已保存到 {output_file}")