import csv
import os
import numpy as np

min_id = 0
file_path = "datasets/yelp.csv"

base_name = file_path.split('/')[1].split('.')[0]

data_list = []

with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        user_id = row[0]
        item_id = row[1]
        rating = row[2]
        timestamp = row[3]
        data_list.append([user_id, item_id, rating, timestamp])
        line_count += 1

sorted_data = sorted(data_list, key=lambda x:x[3])

with open(os.path.join('datasets', base_name + '.txt'), 'w') as f:
    for row in sorted_data:
        user_id = int(row[0])
        item_id = int(row[1])
        if min_id == 0:
            user_id += 1
            item_id += 1
        f.write("%s %s\n" % (str(user_id), str(item_id)))