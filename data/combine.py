import json

all_data = []

for i in range(1, 8):
    filename = f'processed_data/processed_data_{i}.json'
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
        all_data.extend(data)

with open('processed_data/processed_data_all.json', 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)

print("合并完成，数据已保存到 processed_data_all.json")