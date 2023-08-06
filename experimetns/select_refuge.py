import csv

# 读取preds_metrics.csv文件
input_file = 'preds_metrics.csv'
output_file = 'positive.csv'
selected_data = []

with open(input_file, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # 获取表头
    for row in reader:
        ID, sum_value = row[0].split()[-1], float(row[-1])
        if sum_value >= 0 and not ('/g' in ID or '/n' in ID):
            print(ID)
            selected_data.append(row)

# 按照sum降序排序
selected_data.sort(key=lambda x: float(x[-1]), reverse=True)

# 选取前400条数据
selected_data = selected_data[:400]

# 保存到positive.csv文件
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(selected_data)
