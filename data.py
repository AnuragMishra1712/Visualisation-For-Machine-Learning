import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)


def app():
    st.title('Census Income Dataset- KDD')

    with st.beta_expander("Click to know about Details about data"):
        st.write(' This data was extracted from the census bureau database found at\n'
                '| http://www.census.gov/ftp/pub/DES/www/welcome.html\n'
                '| Donor: Terran Lane and Ronny Kohavi\n'
                '|        Data Mining and Visualization\n'
                '|        Silicon Graphics.\n'
                '|        e-mail: terran@ecn.purdue.edu, ronnyk@sgi.com for questions.\n'
                '|\n'
                '| The data was split into train/test in approximately 2/3, 1/3\n'
                '| proportions using MineSet\'s MIndUtil mineset-to-mlc.\n'
                '|\n'
                '| Prediction task is to determine the income level for the person\n'
                '| represented by the record.  Incomes have been binned at the $50K\n'
                '| level to present a binary classification problem, much like the\n'
                '| original UCI/ADULT database.  The goal field of this data, however,\n'
                '| was drawn from the "total person income" field rather than the\n'
                '| "adjusted gross income" and may, therefore, behave differently than the\n'
                '| orginal ADULT goal field.\n'
                '|\n'
                '| More information detailing the meaning of the attributes can be\n'
                '| found in http://www.bls.census.gov/cps/cpsmain.htm\n'
                '| To make use of the data descriptions at this site, the following mappings\n'
                '| to the Census Bureau\'s internal database column names will be needed:')
    with st.beta_expander('MetaData'):
        st.write('91 distinct values for attribute #0 (age) continuous\n'
                 '|    9 distinct values for attribute #1 (class of worker) nominal\n'
                 '|   52 distinct values for attribute #2 (detailed industry recode) nominal\n'
                 '|   47 distinct values for attribute #3 (detailed occupation recode) nominal\n'
                 '|   17 distinct values for attribute #4 (education) nominal\n'
                 '| 1240 distinct values for attribute #5 (wage per hour) continuous\n'
                 '|    3 distinct values for attribute #6 (enroll in edu inst last wk) nominal\n'
                 '|    7 distinct values for attribute #7 (marital stat) nominal\n'
                 '|   24 distinct values for attribute #8 (major industry code) nominal\n'
                 '|   15 distinct values for attribute #9 (major occupation code) nominal\n'
                 '|    5 distinct values for attribute #10 (race) nominal\n'
                 '|   10 distinct values for attribute #11 (hispanic origin) nominal\n'
                 '|    2 distinct values for attribute #12 (sex) nominal\n'
                 '|    3 distinct values for attribute #13 (member of a labor union) nominal\n'
                 '|    6 distinct values for attribute #14 (reason for unemployment) nominal\n'
                 '|    8 distinct values for attribute #15 (full or part time employment stat) nominal\n'
                 '|  132 distinct values for attribute #16 (capital gains) continuous\n'
                 '|  113 distinct values for attribute #17 (capital losses) continuous\n'
                 '| 1478 distinct values for attribute #18 (dividends from stocks) continuous\n'
                 '|    6 distinct values for attribute #19 (tax filer stat) nominal\n'
                 '|    6 distinct values for attribute #20 (region of previous residence) nominal\n'
                 '|   51 distinct values for attribute #21 (state of previous residence) nominal\n'
                 '|   38 distinct values for attribute #22 (detailed household and family stat) nominal\n'
                 '|    8 distinct values for attribute #23 (detailed household summary in household) nominal\n'
                 '|   10 distinct values for attribute #24 (migration code-change in msa) nominal\n'
                 '|    9 distinct values for attribute #25 (migration code-change in reg) nominal\n'
                 '|   10 distinct values for attribute #26 (migration code-move within reg) nominal\n'
                 '|    3 distinct values for attribute #27 (live in this house 1 year ago) nominal\n'
                 '|    4 distinct values for attribute #28 (migration prev res in sunbelt) nominal\n'
                 '|    7 distinct values for attribute #29 (num persons worked for employer) continuous\n'
                 '|    5 distinct values for attribute #30 (family members under 18) nominal\n'
                 '|   43 distinct values for attribute #31 (country of birth father) nominal\n'
                 '|   43 distinct values for attribute #32 (country of birth mother) nominal\n'
                 '|   43 distinct values for attribute #33 (country of birth self) nominal\n'
                 '|    5 distinct values for attribute #34 (citizenship) nominal\n'
                 '|    3 distinct values for attribute #35 (own business or self employed) nominal\n'
                 '|    3 distinct values for attribute #36 (fill inc questionnaire for veteran\'s admin) nominal\n'
                 '|    3 distinct values for attribute #37 (veterans benefits) nominal\n'
                 '|   53 distinct values for attribute #38 (weeks worked in year) continuous\n'
                 '|    2 distinct values for attribute #39 (year) nominal\n'
                 '| ')

    st.write('This Page shows the data used for the Project')
    data = pd.read_csv('subset.csv')

    # PEMNTVTY','PENATVdata = data.drop(data.columns[24], axis=1)
    #     # data.columns = ['AAGE','ACLSWKR','ADTIND','ADTOCC','AHGA','AHRSPAY','AHSCOL','AMARITL','AMJIND','AMJOCC','ARACE','AREORGN',
    #     #        'ASEX','AUNMEN','AUNTYPE','AWKSTAT','CAPGAIN','CAPLOSS','DIVVAL','FILESTAT','GRINREG','GRINST',
    #     #        'HHDFMX','HHDREL','MIGMTR1','MIGMTR3','MIGMTR4','MIGSAME','MIGSUN','NOEMP','PARENT',
    #     #        'PEFNTVTY','TY','PRCITSHP','SEOTR','VETQVA','VETYN',' WKSWORK','YEAR','LABEL']

    st.write(data)
    # Creating a barplot for 'Income'
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write('EDA to check how the data is?')


    income = data['LABEL'].value_counts(normalize=True)

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(7, 5))
    sns.barplot(income.index, income.values, palette='bright')
    plt.title('Distribution of Income', fontdict={
        'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
        'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=10)
    st.pyplot(plt.show())
    st.write('Shows Class is imbalance')

    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write('Show Age distribution on data')

    age = data['AAGE'].value_counts()

    plt.figure(figsize=(10, 5))
    plt.style.use('fivethirtyeight')
    sns.distplot(data['AAGE'], bins=20)
    plt.title('Distribution of Age', fontdict={
        'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
        'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=10)
    st.pyplot(plt.show())

    st.write(' ')
    st.write(' ')
    st.write(' ')



    columns_with_nan = ['GRINST', 'MIGMTR1', 'MIGMTR3','MIGMTR4','MIGSUN','PEFNTVTY','PEMNTVTY','PENATVTY']
    for col in columns_with_nan:
        data[col].fillna(data[col].mode()[0], inplace=True)


    for col in data.columns:
        if data[col].dtypes == 'object':
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    X = data.drop('LABEL', axis=1)
    Y = data['LABEL']
    for col in X.columns:
        scaler = StandardScaler()
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))


    ros = RandomOverSampler(random_state=42)
    ros.fit(X, Y)
    X_resampled, Y_resampled = ros.fit_resample(X, Y)


    X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)
    st.write('')
    st.write('')
    st.write('')
    st.write('Final shape of the data used by the model')
    st.write("X_train shape:", X_train.shape)
    st.write("X_test shape:", X_test.shape)
    st.write("Y_train shape:", Y_train.shape)
    st.write("Y_test shape:", Y_test.shape)
    #st.write('This is what the scaled test data looks like:', X_test)

    # Creating a barplot for 'Sex'
    sex = data['ASEX'].value_counts()

    plt.style.use('default')
    plt.figure(figsize=(7, 5))
    sns.barplot(sex.index, sex.values)
    plt.title('Distribution of Sex', fontdict={
        'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Sex', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
        'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=10)
    plt.grid()
    st.pyplot(plt.show())


    # Creating a countplot of income across age
    plt.style.use('default')
    plt.figure(figsize=(20, 7))
    sns.countplot(data['AAGE'], hue=data['LABEL'])
    plt.title('Distribution of Income Class across Age', fontdict={
        'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
    plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
    plt.ylabel('Number of people', fontdict={
        'fontname': 'Monospace', 'fontsize': 15})
    plt.tick_params(labelsize=12)
    plt.legend(loc=1, prop={'size': 15})
    st.pyplot(plt.show())










