import streamlit as st
from catboost import CatBoostClassifier,Pool
import data
import numpy as np
import plotly
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def app():
    st.title('This Page Provides Performance of the Model')
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    @st.cache
    def model_cache(modelname):
        cached_model = CatBoostClassifier().load_model(f'data/{modelname}')
        return cached_model

    model = model_cache('catboostmodel')
    df = pd.read_csv('census-incomedata.csv', header=None)
    #from sklearn.model_selection import train_test_split
    # Import packages


    # Set random seed
    #np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.2,
                                                        random_state=42)
    # Set categorical features for training
    cat = np.where(df.dtypes != np.float64)[0]
    cat_features = cat[:-1]

    # Create Pool training set
    train_set = Pool(data=X_train,
                     label=y_train,
                     cat_features=cat_features)

    # Create Pool evaluation set
    eval_set = Pool(data=X_test,
                    label=y_test,
                    cat_features=cat_features)

    # model = CatBoostClassifier(loss_function='MultiClass',
    #                            iterations=50,
    #                            eval_metric='Accuracy',
    #                            custom_metric='F1',
    #                            depth=6,
    #                            verbose=False,
    #                            learning_rate=0.125,
    #                            random_seed=42)

    #  Fit model
    trained_model = model.fit(train_set,
                              eval_set=eval_set,
                              plot=True)

    st.write(model.best_score_)

    # trained_model.save_model("/Users/anurag/Desktop/Fooling-Lime-and-Shap/Streamlit-SHAP-Explorer/data/catboostmodel",
    #                          format="cbm",
    #                          export_parameters=None,
    #                          pool=train_set)

    # data = data.drop(data.columns[24], axis=1)
    # data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY', 'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC',
    #                 'ARACE', 'AREORGN',
    #                 'ASEX', 'AUNMEN', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS', 'DIVVAL', 'FILESTAT', 'GRINREG',
    #                 'GRINST',
    #                 'HHDFMX', 'HHDREL', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP', 'PARENT',
    #                 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN', ' WKSWORK', 'YEAR',
    #                 'LABEL']
    # columns_with_nan = ['GRINST', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSUN', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY']
    # for col in columns_with_nan:
    #     data[col].fillna(data[col].mode()[0], inplace=True)
    #
    # for col in data.columns:
    #     if data[col].dtypes == 'object':
    #         encoder = LabelEncoder()
    #         data[col] = encoder.fit_transform(data[col])
    #
    # X = data.drop('LABEL', axis=1)
    # Y = data['LABEL']
    # for col in X.columns:
    #     scaler = StandardScaler()
    #     X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
    #
    # ros = RandomOverSampler(random_state=42)
    # ros.fit(X, Y)
    # X_resampled, Y_resampled = ros.fit_resample(X, Y)
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)
    # st.write('')
    # st.title('Model Prediction')
    # ran_for = RandomForestClassifier(random_state=42)
    # ran_for.fit(X_train, Y_train)
    #
    #
    #
    # Y_pred_ran_for = ran_for.predict(X_test)
    #
    # st.write('Random Forest Classifier:')
    # st.write('Predictions on Training Testing Split Data')
    # st.write('Accuracy score:', round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
    # st.write('F1 score:', round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))
    #
    # st.write('')
    # st.write('')
    # st.write('Confusion Matrix Plot')
    #
    #
    # cm = confusion_matrix(Y_test, Y_pred_ran_for)
    # plt.style.use('default')
    # sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    # plt.savefig('heatmap.png')
    # st.pyplot(plt.show())
    # st.write('')
    # st.write('')
    # st.write('')
    #
    # census_df_test = pd.read_csv('census-income.test', header=None)
    # census_df_test = census_df_test.drop(census_df_test.columns[24], axis=1)
    # census_df_test.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY', 'AHSCOL', 'AMARITL', 'AMJIND',
    #                           'AMJOCC', 'ARACE', 'AREORGN',
    #                           'ASEX', 'AUNMEN', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS', 'DIVVAL', 'FILESTAT',
    #                           'GRINREG', 'GRINST',
    #                           'HHDFMX', 'HHDREL', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP',
    #                           'PARENT',
    #                           'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN', ' WKSWORK',
    #                           'YEAR', 'LABEL']
    #
    # let = LabelEncoder()
    # census_df_test['LABEL'] = let.fit_transform(census_df_test['LABEL'])
    # census_df_test = census_df_test.replace(' ?', np.nan)
    # columns_with_nan = ['GRINST', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSUN', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY']
    #
    # for col in columns_with_nan:
    #     census_df_test[col].fillna(census_df_test[col].mode()[0], inplace=True)
    #
    #
    # for col in census_df_test.columns:
    #     if census_df_test[col].dtypes == 'object':
    #         encoder = LabelEncoder()
    #         census_df_test[col] = encoder.fit_transform(census_df_test[col])
    # Xt = census_df_test.drop('LABEL', axis=1)
    # Yt = census_df_test['LABEL']
    #
    # for col in Xt.columns:
    #     scaler = StandardScaler()
    #     Xt[col] = scaler.fit_transform(Xt[col].values.reshape(-1, 1))
    #
    # ros2 = RandomOverSampler(random_state=42)
    # ros2.fit(Xt, Yt)


    # loaded_model = pickle.load(open('census-kdd.pkl', 'rb'))
    # result = loaded_model.score(Xt, Yt)
    # st.write('Accuracy Score on The test data :',round(result*100,2))




