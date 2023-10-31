from collections import defaultdict

K = 20  # k core
data = "datasets/ml-1m"

def get_user_item_dict(sentiment_data):
    """
    build user & item dictionary
    :param sentiment_data: [user, item, [feature1, opinion1, sentiment1], [feature2, opinion2, sentiment2] ...]
    :return: user dictionary {u1:[i, i, i...], u2:[i, i, i...]}, similarly, item dictionary
    """
    user_dict = {}
    item_dict = {}
    for row in sentiment_data:
        user = row[0]
        item = row[1]
        if user not in user_dict:
            user_dict[user] = [item]
        else:
            user_dict[user].append(item)
        if item not in item_dict:
            item_dict[item] = [user]
        else:
            item_dict[item].append(user)
    return user_dict, item_dict



user_set = set()
item_set = set()
inters = []
user_seq_dict = defaultdict(list)

with open(data + ".txt") as f:
    for line in f:
        u_i = line.rstrip().split(' ')
        user = int(u_i[0])
        item = int(u_i[1])
        user_set.add(user)
        item_set.add(item)
        inters.append([user, item])
        user_seq_dict[user].append(item)


# user filtering
print('======================= filtering review data =======================')
last_length = len(inters)
un_change_count = 0  # iteratively filtering users and features, if the data stay unchanged twice, stop
user_dict, item_dict = get_user_item_dict(inters)


print("original review length: ", last_length)
print("original user length: ", len(user_dict))
print("original item length: ", len(item_dict))
while True:
    user_dict, item_dict = get_user_item_dict(inters)
    valid_user = set()  # the valid users
    for key, value in user_dict.items():
        if len(value) > (K - 1):
            valid_user.add(key)
    inters = [x for x in inters if x[0] in valid_user]  # remove user with small interactions
    user_dict, item_dict = get_user_item_dict(inters)
    valid_item = set()  # the valid items
    for key, value in item_dict.items():
        if len(value) > (K - 1):
            valid_item.add(key)
    inters = [x for x in inters if x[1] in valid_item]  # remove item with small interactions
    length = len(inters)
    if length != last_length:
        last_length = length
    else:
        break
print('valid review length: ', len(inters))
print("valid user: ", len(user_dict))
print('valid item : ', len(item_dict))

# print(user_dict)

# rename users, items, and features to integer names
user_name_dict = {}
item_name_dict = {}

count = 1
for user in user_dict:
    if user not in user_name_dict:
        user_name_dict[user] = count
        count += 1

count = 1
for item in item_dict:
    if item not in item_name_dict:
        item_name_dict[item] = count
        count += 1

for i in range(len(inters)):
    inters[i][0] = user_name_dict[inters[i][0]]
    inters[i][1] = item_name_dict[inters[i][1]]

with open(data+'_%d_core.txt' % K, 'w') as f:
    for d in inters:
        f.write("%s %s\n" % (d[0], d[1]))