import pandas as pd


data_path = 'input/incidence_strains_spb_4_age_groups.csv'
epid_data = pd.read_csv(data_path, index_col=0)

data_columns = epid_data.columns
strains = ['A(H1N1)pdm09', 'A(H3N2)', 'B']

default_age_groups = ['0-2', '3-6', '7-14']
new_df = epid_data.loc[:, ['Год', 'Неделя', 'A(H1N1)pdm09_15 и ст.',
                       'A(H3N2)_15 и ст.', 'B_15 и ст.', 'Население 15 и ст.',
                       'A(H1N1)pdm09_15 и ст._rel', 'A(H3N2)_15 и ст._rel', 'B_15 и ст._rel']]
for strain in strains:
    default_age_group_acum = 0
    default_age_group_rel = 0
    pop_size = 0
    for default_age_group in default_age_groups:
        default_age_group_acum += epid_data[strain + "_" + default_age_group]
        default_age_group_rel += epid_data[strain + "_" + default_age_group + '_rel']
        pop_size += epid_data["Население " + default_age_group]
    new_df[strain + "_" + '0-14'] = default_age_group_acum
    new_df[strain + "_" + '0-14_rel'] = default_age_group_rel
    new_df["Население 0-14"] = pop_size

print(' ')
