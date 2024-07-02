import random
import copy
import json

with open("..\\disease_names", encoding="utf-8") as f:
    disease_names = [s.strip() for s in f.readlines()]

with open("..\\knowledge", encoding="utf-8") as f:
    lines = f.readlines()


def get_disease(desc):
    for name in disease_names:
        if name in desc:
            return name
    return None


disease_desc = {}
all_desc = []
data = [json.loads(line.strip()) for line in lines]
for s in data:
    desc = s["病情描述"]
    disease = get_disease(desc)

    if disease not in disease_desc:
        disease_desc[disease] = []

    disease_desc[disease].append(desc)
    all_desc.append(desc)


pos_data = []
neg_data = []
count = 0
# 制作正样本
for key, value_list in disease_desc.items():
    value_list = list(set(value_list))
    num = 3000
    if len(value_list) < num:
        value_list1 = value_list
        value_list2 = copy.deepcopy(value_list)
        random.shuffle(value_list2)

    else:
        value_list1 = random.sample(value_list, num)
        value_list2 = random.sample(value_list, num)

    count += 1
    print(key, f"{count} / {len(disease_desc)}", len(value_list))

    for s1, s2 in zip(value_list1, value_list2):
        if s1 == s2:
            continue

        pos_data.append(str([s1, s2, 1]))

# 制作负样本
# 注意：由于不相同的负样本的情况，可能千奇百怪，故让负样本数量多余正样本
# 这里的2是一个超参数
neg_num = 2 * len(pos_data)
neg_desc = (neg_num // len(all_desc)) * all_desc + random.sample(all_desc, neg_num % len(all_desc))
neg_desc2 = copy.deepcopy(neg_desc)
random.shuffle(neg_desc2)
neg_data = [str([s1, s2, 0]) for s1, s2 in zip(neg_desc, neg_desc2)]

with open("train_data", "w", encoding="utf-8") as f:
    f.writelines("\n".join(pos_data+neg_data))
