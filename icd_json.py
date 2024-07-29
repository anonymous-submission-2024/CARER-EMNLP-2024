import json 
import pandas as pd
from tqdm import tqdm
with open('data/codes.json') as f:
    icd_hierarchy  = json.load(f)
print(len(icd_hierarchy))
# print(icd_hierarchy[0])
# print(icd_hierarchy[1])
# print(icd_hierarchy[2])
# print(icd_hierarchy[3])
all_icd_lists = []
for i in tqdm(range(len(icd_hierarchy))):
    icd_path = icd_hierarchy[i]

    for icd_node in icd_path:
        # print(icd_node)
        if 'descr' in icd_node.keys():
            code, name,depth = icd_node['code'], icd_node['descr'].lower(), icd_node['depth']
            icd_node_dict = [code,name,depth]
            if icd_node_dict not in all_icd_lists:
                all_icd_lists.append(icd_node_dict)
                # print(code, name, depth)
df = pd.DataFrame(all_icd_lists,columns=['ICD9_CODE','LONG_TITLE','DEPTH'])
df.to_csv('all_icd.csv',index=False)
print(df)