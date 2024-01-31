import pandas as pd
import matplotlib.pyplot as plt


def divide_A_non_defined_to_H1(row):
    if row['A(H1)pdm09'] + row['A(H3)'] == 0:
        return 0
    else:
        val = row['A(H1)pdm09'] + row['A_non_defined'] *\
            row['A(H1)pdm09']/(row['A(H1)pdm09'] + row['A(H3)'])
        return val


def divide_A_non_defined_to_H3(row):
    if row['A(H1)pdm09'] + row['A(H3)'] == 0:
        return 0
    else:
        val = row['A(H3)'] + row['A_non_defined'] *\
            row['A(H3)']/(row['A(H1)pdm09'] + row['A(H3)'])
        return val


# preprocess source epidemic data to input data for application
def preprocess_excel_source(data: pd.DataFrame):
    # actual_data = pd.read_excel(data, skiprows=[0])
    print("Preprocessing started...")
    actual_data = data
    actual_data = actual_data.rename(columns={'Unnamed: 0': 'date', 'ОРВИ': 'ari',
                                              'Население': 'population',
                                              'Число образцов тестированных на грипп': 'tested',
                                              'A (субтип не определен)': 'A_non_defined',
                                              })
    actual_data_preprocessed = actual_data[['date', 'ari', 'population', 'tested',
                                            'A_non_defined', 'A(H1)pdm09', 'A(H3)', 'B']]
    actual_data_preprocessed['date'] = actual_data_preprocessed['date'].str.replace(
        u'\xa0', u' ')
    dates = list(actual_data_preprocessed['date'])
    dates = [elem.replace(u'\xa0', u' ') for elem in dates]
    dates = [elem.split('(')[0] for elem in dates]
    years = [int(elem.split('.')[0]) for elem in dates]
    weeks = [int(elem.split('.')[1]) for elem in dates]
    actual_data_preprocessed.insert(loc=0, column='num_of_week', value=weeks)
    actual_data_preprocessed.insert(loc=0, column='year', value=years)
    actual_data_preprocessed['date']
    actual_data_final = actual_data_preprocessed.drop(columns=['date'])

    actual_data_final['A(H1)pdm09_abs'] = actual_data_final.apply(
        divide_A_non_defined_to_H1, axis=1)
    actual_data_final['A(H3)_abs'] = actual_data_final.apply(
        divide_A_non_defined_to_H3, axis=1)
    actual_data_final['B_abs'] = actual_data_final['B']
    actual_data_final['A(H1)pdm09_rel'] = actual_data_final['A(H1)pdm09_abs'] / \
        actual_data_final['tested']
    actual_data_final['A(H3)_rel'] = actual_data_final['A(H3)_abs'] / \
        actual_data_final['tested']
    actual_data_final['B_rel'] = actual_data_final['B_abs'] / \
        actual_data_final['tested']
    actual_data_final['A(H1)pdm09_total'] = (
        actual_data_final['A(H1)pdm09_rel']*actual_data_final['ari']).astype(int)
    actual_data_final['A(H3)_total'] = (
        actual_data_final['A(H3)_rel']*actual_data_final['ari']).astype(int)
    actual_data_final['B_total'] = (
        actual_data_final['B_rel']*actual_data_final['ari']).astype(int)

    year = 2024
    print("Year of data", year)
    actual_data_final = actual_data_final[actual_data_final['year'] == year]
    real_input = pd.read_csv(
        'data/input/incidence_strains_spb_2_age_groups.csv')
    real_input = real_input[real_input['Год'] != year]

    add_df = pd.DataFrame(columns=['index', 'Год', 'Неделя', 'A(H1N1)pdm09_15 и ст.',
                                   'A(H3N2)_15 и ст.',
                                   'B_15 и ст.', 'Население 15 и ст.', 'A(H1N1)pdm09_15 и ст._rel',
                                   'A(H3N2)_15 и ст._rel', 'B_15 и ст._rel', 'A(H1N1)pdm09_0-14',
                                   'A(H1N1)pdm09_0-14_rel', 'Население 0-14', 'A(H3N2)_0-14',
                                   'A(H3N2)_0-14_rel', 'B_0-14', 'B_0-14_rel'])

    add_df['Год'] = actual_data_final['year']
    add_df['Население 15 и ст.'] = actual_data_final['population']
    add_df['Неделя'] = actual_data_final['num_of_week']
    add_df['A(H1N1)pdm09_15 и ст.'] = actual_data_final['A(H1)pdm09_total']
    add_df['A(H3N2)_15 и ст.'] = actual_data_final['A(H3)_total']
    add_df['B_15 и ст.'] = actual_data_final['B_total']
    add_df['A(H1N1)pdm09_15 и ст._rel'] = actual_data_final['A(H1)pdm09_rel']
    add_df['A(H3N2)_15 и ст._rel'] = actual_data_final['A(H3)_rel']
    add_df['B_15 и ст._rel'] = actual_data_final['B_rel']
    add_df = add_df.fillna(0)

    real_input = pd.concat([real_input, add_df], ignore_index=True)
    real_input['index'] = real_input.index
    real_input = real_input.sort_values(by=['Год', 'Неделя'])
    real_input.to_csv(
        'data/input/incidence_strains_spb_2_age_groups.csv', index=False)
    return
