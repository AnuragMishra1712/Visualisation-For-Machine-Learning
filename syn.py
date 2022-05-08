import numpy as np
import sklearn.model_selection
import streamlit.components.v1 as components

import matplotlib.pyplot as plt
# import pickle
import seaborn as sns
import streamlit as st
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, plot_confusion_matrix,plot_roc_curve
from lime.lime_tabular import LimeTabularExplainer
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)
def app():
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    st.title('Exploring Shap and lime on Synthetic Data')

    st.title('My Hypothesis are based  on this Adversarial Function')
    from PIL import Image

    image = Image.open('formula.jpg')

    st.image(image, caption='Adversarial Function')

    st.write('A function that performs in a way described in the equation below where is e(x) is our adversarial classifier, f(x) is the biased classifier and ψ(x) is our unbiased classifier (e.g., makes predictions based on innocuous features that are correlated based on situations and uncorrelated with sensitive attributes).')
    st.write('1) Lime and Shap will not be able to distinguish correlated columns')
    st.write('2) Lime and Shap will is not reliable under biased conditions')

    A = np.random.randint(10, size=(1000, 4))
    A[:, 2] = 2 * A[:, 1]
    X_a = pd.DataFrame(A, columns=['loan_id', 'Race', 'Sex', 'LoanYorN'], dtype=float)
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, [1000, 4])
    X = X_a + noise
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_orignal = X.copy()
    X = scaler.fit_transform(X)

    st.write('Intentionally Created correlated columns for Fooling Lime and Shap')
    hm = sns.heatmap(X_a.corr(), annot=True)

    hm.set(xlabel='\nAtrributes', ylabel='Target Class\t', title="Correlation matrix biased synthetic Dataset\n")

    st.pyplot(plt.show())

    st.header(
        'Here the classed have been intentionally created in a way it is biased on race which means if race = African_american then loand is denied')
    y = np.matmul(X_a, np.array([3, 2, 4, 1]))
    print(y)
    mean_y = np.mean(y)
    print(mean_y)
    y[y > mean_y] = 1
    y[y != 1] = 0
    print(y)

    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2)
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(X_tr, y_tr)
    y_pred = rfc.predict(X_test)
    acc = np.sum(y_test == y_pred) / y_test.shape[0]
    st.write("Accuracy score= {%.2f}" % acc)

    def model_auc(model):
        train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
        val_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        st.write(f'Train AUC: {train_auc}, Val Auc: {val_auc}')

    # model performance
    model_auc(rfc)

    st.header('Lets Look at Lime Explanations')
    st.write('It very Clear that column Race and Sex which were taken as correlated fails the Lime prediction as')
    class_names = [0,1]
    X_test = pd.DataFrame(X_test, columns=['loan_id', 'Race', 'Sex', 'LoanYorN'], dtype=float)
    # instantiate the explanations for the data set
    explainer = LimeTabularExplainer(
        training_data=np.array(X_tr),
        feature_names=X_a.columns,
        class_names=['Loan Denied', 'Loan Approved'],
        mode='classification')
    exp = explainer.explain_instance(num_features=15, num_samples=50,
                                     data_row=X_test.iloc[2],
                                     predict_fn=rfc.predict_proba)
    st.write('Lime Explanation based on inputs')


    exp.show_in_notebook(show_table=True)
    components.html(exp.as_html(), height=300)

    st.title('Lets Look at Shap Explanations Now')

    explainer = shap.TreeExplainer(rfc)
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
    print(f"Explainer Expected Value: {expected_value}")
    idx = 100  # row selected for fast runtime
    select = range(idx)
    features = X_test.iloc[select]
    feature_display = X_orignal.loc[features.index]
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shap_values = explainer.shap_values(features)[1]
        shap_interaction_values = explainer.shap_interaction_values(features)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]

    st.pyplot(shap.summary_plot(shap_values, feature_display, plot_type='bar'))



    st.write('Here also we see that shap gives almost equal weight to these correlated attributes')

    st.pyplot(shap.summary_plot(shap_values, features))












