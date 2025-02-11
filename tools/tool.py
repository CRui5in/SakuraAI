import json

def extract_json_block(text):
    start = text.find('```json')
    if start == -1:
        return text

    start = start + 7  # 跳过 ```json
    end = text.find('```', start)
    if end == -1:
        return text

    return text[start:end].strip()


def merge_dicts(dict1, dict2):
    merged = dict1.copy()

    for key, value in dict2.items():
        if key in merged:
            if isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, int):
                merged[key] = value

    return merged
