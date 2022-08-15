import pandas as pd

f = open('../data/all.txt', 'r', encoding='utf8')
lines = f.readlines()
f.close()

data = {'candidates': [], 'label': []}

for index, line in enumerate(lines):
    data['candidates'].append(line.strip())
    data['label'].append(0)
    if index == 10:
        break

df = pd.DataFrame(data=data, columns=['candidates', 'label'])
print(df)
