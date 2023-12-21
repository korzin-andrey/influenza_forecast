import pandas as pd


def getDays2Weeks(df_simul_daily, model_group):
    '''
    model_detail, df_data_weekly, df_data_detail_weekly,
    if model_detail:
        df_simul_weekly = pd.DataFrame(columns=df_data_detail_weekly.columns)
    else:
        df_simul_weekly = pd.DataFrame(columns=df_data_weekly.columns)'''

    df_simul_weekly = pd.DataFrame(columns=model_group)
    for subgroup in model_group:
        days_num = len(df_simul_daily[subgroup])
        wks_num = int(days_num / 7.0)

        simul_weekly = []
        simul_daily = list(df_simul_daily[subgroup])

        for i in range(wks_num):
            simul_weekly.append(sum([simul_daily[j] for j in range(i * 7, (i + 1) * 7)]))

        df_simul_weekly[subgroup] = simul_weekly

    return df_simul_weekly
