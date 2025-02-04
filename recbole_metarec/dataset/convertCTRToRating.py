import os
import sys

import numpy as np
import os.path as path

import pandas as pd

from tqdm import tqdm


def negative_sampling(df_inter, num_negatives=2):
    """  
    对用户-物品交互记录进行负采样处理  
    :param df_inter: 包含用户-物品交互记录的DataFrame  
    :param num_negatives: 每个正样本对应的负样本数量  
    :return: 处理后的DataFrame  
    """
    # 只保留需要的列  
    df_inter = df_inter[['HADM_ID:token', 'NDC:token', 'TIMESTEP:float']]

    # 获取全物品集合  
    all_items = set(df_inter['NDC:token'].unique())

    # 初始化结果列表  
    results = []

    # 按HADM_ID和TIMESTEP分组  
    grouped = df_inter.groupby(['HADM_ID:token', 'TIMESTEP:float'])

    for (hadm_id, timestep), group in tqdm(grouped, ncols=100, ascii=True, total=grouped.ngroups, desc=f"negative sampling"):
        positive_items = set(group['NDC:token'])
        negative_items = list(all_items - positive_items)

        group_results = []  # 初始化当前组的结果列表
        for item in positive_items:
            group_results.append({'HADM_ID:token': hadm_id, 'NDC:token': item, 'rating:float': 5})
        if negative_items:  # 确保有负样本可采样  
            sampled_negatives = np.random.choice(negative_items,
                                                 size=min(num_negatives * len(positive_items), len(negative_items)),
                                                 replace=False)
            for neg_item in sampled_negatives:
                insert_pos = np.random.randint(0, len(group_results) + 1)
                group_results.insert(insert_pos, {'HADM_ID:token': hadm_id, 'NDC:token': neg_item, 'rating:float': 1})

        results.extend(group_results)

    result_df = pd.DataFrame(results)

    return result_df


def convert(input_data, selected_fields, output_file):
    output_data = pd.DataFrame()
    for column in selected_fields:
        output_data[column] = input_data.iloc[:, column]
    with open(output_file, 'w') as fp:
        fp.write('\t'.join([selected_fields[column] for column in output_data.columns]) + '\n')
        for i in tqdm(range(output_data.shape[0]), leave=False, ascii=True, ncols=100):
            fp.write('\t'.join([str(output_data.iloc[i, j])
                                for j in range(output_data.shape[1])]) + '\n')

if __name__ == "__main__":
    srcDatasetPath = 'mimic-iii-v1.4-drug-rec'
    desDatasetPath = 'mimic-iii-v1.4-drug-rec-Rating'
    os.makedirs(desDatasetPath, exist_ok=True)
    selected_fields = {
        0: 'HADM_ID:token',
        1: 'NDC:token',
        2: 'rating:float'
    }

    def ctr_to_rating(df_name):
        df_inter = pd.read_csv(path.join(srcDatasetPath, df_name), sep='\t', header=0)
        df_inter_rating = negative_sampling(df_inter)
        convert(df_inter_rating, selected_fields,
                path.join(desDatasetPath,
                          desDatasetPath + '.' + '.'.join(df_name.split('.')[-2:])))

    ctr_to_rating(f"{srcDatasetPath}.valid.inter")
    ctr_to_rating(f"{srcDatasetPath}.train.inter")
    ctr_to_rating(f"{srcDatasetPath}.test.inter")

    # df_inter = pd.read_csv(path.join(desDatasetPath, f"{desDatasetPath}.train.inter"), sep='\t', header=0)
    # print(df_inter.describe())

    # df_item = pd.read_csv(path.join(desDatasetPath, f"{desDatasetPath}.item"), sep='\t', header=0)
    # df_user = pd.read_csv(path.join(desDatasetPath, f"{desDatasetPath}.user"), sep='\t', header=0)

    # print(df_item.describe())
    # print(df_user.describe())

