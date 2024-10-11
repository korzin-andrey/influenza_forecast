
WEEK_SEASON_START = 26
WEEK_SEASON_END = 25

WEEK_EPI_START = 40  # 1 oct
WEEK_EPI_END = 24  # 1 may


def int0(num):
    if num == '':
        return 0
    else:
        return int(num)


def isSorted(arr):
    return len(arr) == arr[-1] - arr[0] + 1


def strainName(strain_num):

    if strain_num == 0:
        return "A(H1N1)pdm09"
    if strain_num == 1:
        return "A(H3N2)"
    if strain_num == 2:
        return "B"
    else:
        return strain_num


def extractPopSize(df, year, incidence, age_groups):
    # Takes raw incidence data and returns population size

    df_year = df[(df["Год"] == year)]

    if incidence == "strain" or incidence == "total":
        return list(df_year["Население"])[0]
    elif incidence == "strain_age-group" or incidence == "age-group":
        return [df_year["Население " + age_group].iloc[0] for age_group in age_groups]
        # sum(list(df_year["Население " + age_group])[0] for age_group in age_groups)


def extractARIForSeason(df, first_year):

    df_season = df[((df["Год"] == first_year) & (df["Неделя"] >= WEEK_EPI_START)) |
                   ((df["Год"] == first_year+1) & (df["Неделя"] <= WEEK_EPI_END))]
    df_season = df_season.drop(labels=[df_season.index[0]], axis=0) if first_year == 2010 else df_season
    return df_season


def prevalenceType(incidence, raw_data, age_groups, strains):
    new_df = raw_data
    if incidence == "strain":
        # Strains
        for strain in strains:
            strain_acum = 0
            for age_group in age_groups:
                strain_acum += new_df[strain + "_" + age_group]
                new_df.drop(strain + "_" + age_group, axis=1, inplace=True)
                new_df.drop(strain + "_" + age_group + "_rel", axis=1, inplace=True)
            new_df[strain] = strain_acum

        # Population
        strain_population = 0
        for age_group in age_groups:
            strain_population += new_df["Население " + age_group]
            new_df.drop("Население " + age_group, axis=1, inplace=True)
        new_df["Население"] = strain_population

        # Rel incidence
        for strain in strains:
            new_df[strain + "_rel"] = new_df[strain] * 1000 / new_df["Население"]

    elif incidence == "age-group":
        # Age groups
        for age_group in age_groups:
            age_group_acum = 0
            for strain in strains:
                age_group_acum += new_df[strain + "_" + age_group]
                new_df.drop(strain + "_" + age_group, axis=1, inplace=True)
                new_df.drop(strain + "_" + age_group + "_rel", axis=1, inplace=True)
            new_df[age_group] = age_group_acum

        # Rel incidence
        for age_group in age_groups:
            new_df[age_group + "_rel"] = new_df[age_group] * 1000 / new_df["Население " + age_group]

    elif incidence == "total":
        # Cumulative incidence
        strain_age_group_acum = 0
        for strain in strains:
            for age_group in age_groups:
                strain_age_group_acum += new_df[strain + "_" + age_group]
                new_df.drop(strain + "_" + age_group, axis=1, inplace=True)
                new_df.drop(strain + "_" + age_group + "_rel", axis=1, inplace=True)
        new_df["Все"] = strain_age_group_acum

        # Population
        total_population = 0
        for age_group in age_groups:
            total_population += new_df["Население " + age_group]
            new_df.drop("Население " + age_group, axis=1, inplace=True)
        new_df["Население"] = total_population

        # Rel incidence
        new_df["Все_rel"] = new_df["Все"] * 1000 / new_df["Население"]

    elif incidence == "temp":
        # Age groups
        default_age_groups = ['0-2', '3-6', '7-14']
        for strain in strains:
            default_age_group_acum = 0
            for default_age_group in default_age_groups:
                default_age_group_acum += new_df[strain + "_" + default_age_group]
                new_df.drop(strain + "_" + default_age_group, axis=1, inplace=True)
                new_df.drop(strain + "_" + default_age_group + "_rel", axis=1, inplace=True)
            new_df[strain + "_" + age_groups[0]] = default_age_group_acum

        # Population
        age_population = 0
        for default_age_group in default_age_groups:
            age_population += new_df["Население " + default_age_group]
            new_df.drop("Население " + default_age_group, axis=1, inplace=True)
        new_df["Население " + age_groups[0]] = age_population

        # Rel incidence
        for strain in strains:
            new_df[strain + "_" + age_groups[0] + "_rel"] = new_df[strain + "_" + age_groups[0]] * 1000 / new_df["Население " + age_groups[0]]

    return new_df

