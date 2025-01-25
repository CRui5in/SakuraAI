import json
import requests
import logging
from paddleocr import PaddleOCR
from openai import OpenAI
from tqdm import tqdm

# 设置 PaddleOCR 日志级别为 ERROR，忽略 DEBUG 信息
logging.getLogger("ppocr").setLevel(logging.ERROR)

# 初始化 PaddleOCR
ocr = PaddleOCR(lang="ch")

# 初始化 OpenAI 客户端
client = OpenAI(api_key="sk-a7c57896b5ac417a961e0561f6dc9c87", base_url="https://api.deepseek.com")

# 下载图片
def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)

# OCR 识别图片中的文字
def extract_text_from_image(image_url):
    try:
        # 下载图片
        image_path = "temp_image.jpg"
        download_image(image_url, image_path)

        # 提取文字
        result = ocr.ocr(image_path, cls=True)
        extracted_text = " ".join([line[1][0] for line in result[0]])  # 拼接识别结果

        # 如果识别出的字符数小于 10，则忽略
        if len(extracted_text) < 10:
            return None
        return extracted_text
    except Exception as e:
        return None

# 屏蔽 desc 中第一个 # 及后面的内容
def clean_desc(desc):
    if "#" in desc:
        return desc.split("#")[0].strip()
    return desc

# 调用 OpenAI API 生成 prompt 和 completion
def generate_prompt_completion(text):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "你是一个专业的旅游助手，根据用户提供的信息生成详细的旅游交通攻略，信息里面分别有标题、描述、图片文字，你需要通过这些信息生成语言自然并且合适的prompt和completion，其中图片文字中可能有些旅游交通无关信息，请你自行分辨有没有必要加入问答中，并严格确保返回如下JSON格式，"
                    "如：{"
                    "\"prompt\": \"用户：我想从关西国际机场到京都。\", "
                    "\"completion\": \"助手：你可以在携程购买haruka快车，从关西机场直接到达JR京都站。\""
                    "}，"
                    "包含'prompt'和'completion'两个字段。"},
                {"role": "user", "content": (
                    f"根据以下获取的信息生成一段详细的旅游交通推荐攻略，紧扣旅游交通推荐主题，忽略无关内容，需要包含'prompt'和'completion'两个字段：{text}"
                )}
            ],
            stream=False
        )
        # 解析返回的JSON数据
        result = json.loads(response.choices[0].message.content)
        if "prompt" in result and "completion" in result:
            return result["prompt"], result["completion"]
        else:
            print("返回的JSON格式不正确")
            return None, None
    except Exception as e:
        print(f"API 调用失败：{e}")
        return None, None

# 处理 JSON 数据
def process_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")  # 开始写入 JSON 数组

        # 使用 tqdm 显示进度条，遍历整个 JSON 数据
        for i, item in enumerate(tqdm(data, desc="处理数据")):
            # 跳过 type 为 video 的数据
            if item["type"] == "video":
                continue

            # 提取 title 和 desc，并清理 desc
            title = item["title"]
            desc = clean_desc(item["desc"])

            # 提取图片中的文字
            image_text = ""
            if item["image_list"]:
                image_urls = item["image_list"].split(",")
                for url in image_urls:
                    text = extract_text_from_image(url)
                    if text:
                        image_text += text + " "

            # 合并 title、desc 和 OCR 识别的文字
            combined_text = f"标题：{title}\n描述：{desc}\n图片文字：{image_text.strip()}"

            # 调用 OpenAI API 生成 prompt 和 completion
            prompt, completion = generate_prompt_completion(combined_text)
            if prompt and completion:
                # 写入 JSON 文件
                result = {
                    "prompt": prompt,
                    "completion": completion
                }
                if i > 1:
                    f.write(",\n")  # 添加逗号分隔符
                json.dump(result, f, ensure_ascii=False, indent=2)

        f.write("\n]")  # 结束 JSON 数组

# 主函数
def main():
    # 加载 JSON 数据
    with open("data_traffic.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据并实时写入 JSON 文件
    process_data(data, "processed_data_7.json")

    print("数据处理完成，结果已保存到 processed_data_7.json")

if __name__ == "__main__":
    main()