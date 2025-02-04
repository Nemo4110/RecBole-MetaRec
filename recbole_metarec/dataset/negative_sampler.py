import os
import sys
import numpy as np
import os.path as path
import pandas as pd

from tqdm import tqdm
from convertCTRToRating import convert



def negative_sampling(df_inter, num_negatives=2):
    """
    对用户-物品交互记录进行负采样处理
    :param df_inter: 包含用户-物品交互记录的DataFrame
    :param num_negatives: 每个正样本对应的负样本数量
    :return: 处理后的DataFrame
    """
    df_inter = df_inter[['HADM_ID:token', 'NDC:token', 'TIMESTEP:float']]
    all_items = set(df_inter['NDC:token'].unique())

    results = []
    grouped = df_inter.groupby(['HADM_ID:token', 'TIMESTEP:float'])
    for (hadm_id, timestep), group in tqdm(grouped, ncols=100, ascii=True, total=grouped.ngroups, desc=f"negative sampling"):
        positive_items = set(group['NDC:token'])
        negative_items = list(all_items - positive_items)

        group_results = []  # 初始化当前组的结果列表
        for item in positive_items:
            group_results.append({'HADM_ID:token': hadm_id, 'NDC:token': item, 'label:token': 1})
        if negative_items:  # 确保有负样本可采样
            sampled_negatives = np.random.choice(negative_items,
                                                 size=min(num_negatives * len(positive_items), len(negative_items)),
                                                 replace=False)
            for neg_item in sampled_negatives:
                insert_pos = np.random.randint(0, len(group_results) + 1)
                group_results.insert(insert_pos, {'HADM_ID:token': hadm_id, 'NDC:token': neg_item, 'label:token': 0})

        results.extend(group_results)

    result_df = pd.DataFrame(results)
    return result_df


if __name__ == "__main__":
    srcDatasetPath = 'mimic-iii-v1.4-drug-rec'
    desDatasetPath = 'mimic-iii-v1.4-drug-rec-CTR'
    os.makedirs(desDatasetPath, exist_ok=True)
    selected_fields = {
        0: 'HADM_ID:token',
        1: 'NDC:token',
        2: 'label:token'
    }
    def exe(df_name):
        df_inter = pd.read_csv(path.join(srcDatasetPath, df_name), sep='\t', header=0)
        df_inter_rating = negative_sampling(df_inter)
        convert(df_inter_rating, selected_fields,
                path.join(desDatasetPath,
                          desDatasetPath + '.' + '.'.join(df_name.split('.')[-2:])))

    exe(f"{srcDatasetPath}.valid.inter")
    exe(f"{srcDatasetPath}.train.inter")
    exe(f"{srcDatasetPath}.test.inter")




