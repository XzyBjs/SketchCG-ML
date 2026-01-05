# make_categories.py
from pathlib import Path

txt_path = Path('tiny_test_set.txt')
out_path = Path('categories.txt')

first = {}          # 类名 -> 第一次出现的 label
with txt_path.open(encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 用空格切最后两段：路径  label
        path_part, label_part = line.rsplit(' ', 1)
        cls_name = path_part.split('/')[0]
        label    = int(label_part)

        # 只记录第一次
        if cls_name not in first:
            first[cls_name] = label

if not first:
    raise SystemExit('没有解析到任何类别，请检查文件格式或路径')

# 按 label 升序排列
sorted_items = sorted(first.items(), key=lambda x: x[1])
max_label = sorted_items[-1][1]

# 构造 categories 列表
categories = [''] * (max_label + 1)
for name, label in sorted_items:
    categories[label] = name

with out_path.open('w', encoding='utf-8') as f:
    for name in categories:
        f.write(name + '\n')

print(f'categories.txt 已生成，共 {len(categories)} 行。')