from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import streamlit as st
import streamlit.components.v1 as components

#%matplotlib inline
st.set_page_config(page_title='SHAP Values Explorer')
st.set_option('deprecation.showPyplotGlobalUse', False)


def app():
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)


    # Load CatBoost model
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    @st.cache
    def model_cache(modelname):
        cached_model = CatBoostClassifier().load_model(f'data/{modelname}')
        return cached_model

    model = model_cache('catboost_model')


    # Load pickle files
    @st.cache
    def st_cache(filename):
        cached_data = pickle.load(open(f'data/{filename}', 'rb'))
        return cached_data

    expected_value = st_cache('expected_value')
    X_test = st_cache('X_test')
    eval_set_features = st_cache('eval_set_features')
    train_set_features = st_cache('train_set_features')
    shap_values = st_cache('shap_values')


    st.title('Explore the SHAP values of a CatBoost Binary Income Classfier')

        # Input sidebar subheader
    st.sidebar.subheader('Input the features of Income classifer here :')
    col1, col2 = st.sidebar.beta_columns(2)

    AAGE = col1.slider(label='Age', min_value=0, max_value=100, format='%i', step=1, value=0)
    ACLSWKR = col2.selectbox('WorkerClass',('Not in universe', 'Federal government', 'Local government', 'Never worked', 'Private', 'Self-employed-incorporated', 'Self-employed-not incorporated', 'State government','Without pay'))
    ADTIND = col1.slider(label='IndustryCode', min_value=0, max_value=60, format='%i', value=0)
    ADTOCC = col2.slider(label='OccCode', min_value=0, max_value=50, format='%i', value=0)
    AHGA= col1.selectbox('Education',('Children', '7th and 8th grade', '9th grade', '10th grade', 'High school graduate', '11th grade', '12th grade no diploma', '5th or 6th grade', 'Less than 1st grade', 'Bachelors degree(BA AB BS)', '1st 2nd 3rd or 4th grade', 'Some college but no degree', 'Masters degree(MA MS MEng MEd MSW MBA)', 'Associates degree-occup /vocational', 'Associates degree-academic program', 'Doctorate degree(PhD EdD)', 'Prof school degree (MD DDS DVM LLB JD)'))
    AHRSPAY = col2.slider(label='Wageph', min_value=0, max_value=10000, format='%i', value=0)
    AHSCOL = col1.selectbox('LastEnroledin',('Not in universe', 'High school', 'College or university'))
    AMARITL = col2.selectbox('MaritalStat',('Never married', 'Married-civilian spouse present', 'Married-spouse absent', 'Separated', 'Divorced', 'Widowed', 'Married-A F spouse present'))
    AMJIND = col1.selectbox('MajorIC',( 'Not in universe or children', 'Entertainment', 'Social services', 'Agriculture', 'Education', 'Public administration', 'Manufacturing-durable goods', 'Manufacturing-nondurable goods', 'Wholesale trade', 'Retail trade', 'Finance insurance and real estate', 'Private household services', 'Business and repair services', 'Personal services except private HH', 'Construction', 'Medical except hospital',' Other professional services', 'Transportation', 'Utilities and sanitary services, Mining, Communications', 'Hospital services, Forestry and fisheries', 'Armed Forces'))
    AMJOCC = col2.selectbox('MajorOC',('Not in universe, Professional specialty', 'Other service', 'Farming forestry and fishing, Sales', 'Adm support including clerical', 'Protective services', 'Handlers equip cleaners etc' , 'Precision production craft & repair', 'Technicians and related support', 'Machine operators assmblrs & inspctrs', 'Transportation and material moving', 'Executive admin and managerial', 'Private household services',' Armed Forces'))
    ARACE =col1.selectbox('Race',('White', 'Black', 'Other', 'Amer Indian Aleut or Eskimo',' Asian or Pacific Islander'))
    AREORGN =col2.selectbox('HispanicOrg',('Mexican (Mexicano)', 'Mexican-American',' Puerto Rican', 'Central or South American', 'All other', 'Other Spanish', 'Chicano, Cuban', 'Do not know', 'NA'))
    ASEX = col1.selectbox('Sex',('Male','Female'))
    AUNMEN = col2.selectbox('MemofLabUni',('Not in universe', 'No', 'Yes'))
    AUNTYPE = col1.selectbox('UnmpReason',( 'Not in universe', 'Re-entrant', 'Job loser - on layoff', 'New entrant', 'Job leaver', 'Other job loser'))
    AWKSTAT = col2.selectbox('FullorPtimeStat',('Children or Armed Forces', 'Full-time schedules', 'Unemployed part- time', 'Not in labor force', 'Unemployed full-time', 'PT for non-econ reasons usually FT', 'PT for econ reasons usually PT', 'PT for econ reasons usually FT'))
    CAPGAIN = col1.slider(label='Capgain', min_value=0, max_value=100000, format='%i', value=0)
    CAPLOSS =col2.slider(label='CapLoss', min_value=0, max_value=4700, format='%i', value=0)
    DIVVAL = col1.slider(label='Dividends', min_value=0, max_value=100000, format='%i', value=0)
    FILESTAT = col2.selectbox('TaxFilerStatus',('Nonfiler', 'Joint one under 65 & one 65+', 'Joint both under 65', 'Single', 'Head of household', 'Joint both 65+'))

    GRINREG = col1.selectbox('prevregionres',('Not in universe', 'South', 'Northeast', 'West', 'Midwest', 'Abroad'))
    GRINST = col2.selectbox('prevstate',( 'Not in universe', 'Utah',' Michigan', 'North Carolina',' North Dakota',' Virginia', 'Vermont', 'Wyoming', 'West Virginia', 'Pennsylvania', 'Abroad',' Oregon', 'California', 'Iowa',' Florida', 'Arkansas',' Texas', 'South Carolina', 'Arizona', 'Indiana', 'Tennessee',' Maine',' Alaska',' Ohio',' Montana', 'Nebraska', 'Mississippi', 'District of Columbia', 'Minnesota', 'Illinois',' Kentucky', 'Delaware', 'Colorado', 'Maryland', 'Wisconsin', 'New Hampshire', 'Nevada', 'New York', 'Georgia', 'Oklahoma', 'New Mexico', 'South Dakota',' Missouri',' Kansas', 'Connecticut',' Louisiana', 'Alabama',' Massachusetts', 'Idaho', 'New Jersey'))
    HHDFMX = col1.selectbox('famstatus',( 'Child <18 never marr not in subfamily', 'Other Rel <18 never marr child of subfamily RP', 'Other Rel <18 never marr not in subfamily', 'Grandchild <18 never marr child of subfamily RP', 'Grandchild <18 never marr not in subfamily', 'Secondary individual', 'In group quarters', 'Child under 18 of RP of unrel subfamily', 'RP of unrelated subfamily',' Spouse of householder', 'Householder', 'Other Rel <18 never married RP of subfamily',' Grandchild <18 never marr RP of subfamily', 'Child <18 never marr RP of subfamily', 'Child <18 ever marr not in subfamily', 'Other Rel <18 ever marr RP of subfamily',' Child <18 ever marr RP of subfamily',' Nonfamily householder', 'Child <18 spouse of subfamily RP', 'Other Rel <18 spouse of subfamily RP', 'Other Rel <18 ever marr not in subfamily',' Grandchild <18 ever marr not in subfamily', 'Child 18+ never marr Not in a subfamily', 'Grandchild 18+ never marr not in subfamily', 'Child 18+ ever marr RP of subfamily', 'Other Rel 18+ never marr not in subfamily', 'Child 18+ never marr RP of subfamily',' Other Rel 18+ ever marr RP of subfamily', 'Other Rel 18+ never marr RP of subfamily', 'Other Rel 18+ spouse of subfamily RP', 'Other Rel 18+ ever marr not in subfamily',' Child 18+ ever marr Not in a subfamily', 'Grandchild 18+ ever marr not in subfamily', 'Child 18+ spouse of subfamily RP', 'Spouse of RP of unrelated subfamily', 'Grandchild 18+ ever marr RP of subfamily', 'Grandchild 18+ never marr RP of subfamily', 'Grandchild 18+ spouse of subfamily RP'))
    HHDREL = col2.selectbox('homesummary',('Child under 18 never married', 'Other relative of householder', 'Nonrelative of householder',' Spouse of householder',' Householder', 'Child under 18 ever married', 'Group Quarters- Secondary individual', 'Child 18 or older'))
    MIGMTR1 = col1.selectbox('mig1',('Not in universe', 'Nonmover',' MSA to MSA',' NonMSA to nonMSA',' MSA to nonMSA', 'NonMSA to MSA', 'Abroad to MSA', 'Not identifiable',' Abroad to nonMSA'))
    MIGMTR3 = col2.selectbox('mig3',('Not in universe',' Nonmover',' Same county', 'Different county same state',' Different state same division',' Abroad',' Different region', 'Different division same region'))
    MIGMTR4 = col1.selectbox('mig4',('Not in universe', 'Nonmover', 'Same county', 'Different county same state', 'Different state in West',' Abroad',' Different state in Midwest', 'Different state in South', 'Different state in Northeast'))
    MIGSAME = col2.selectbox('samehoue',('Not in universe under 1 year old', 'Yes', 'No.'))
    MIGSUN = col1.selectbox('migsun',('Not in universe', 'Yes',' No.'))
    NOEMP = col2.slider(label = 'NofEmpl',min_value=0, max_value=6, format='%i', value=0)
    PARENT =col1.selectbox('Under18parents',('Both parents present',' Neither parent present', 'Mother only present',' Father only present',' Not in universe'))
    PEFNTVTY = col2.selectbox('fbirthc',('Mexico',' United-States', 'Puerto-Rico', 'Dominican-Republic', 'Jamaica', 'Cuba', 'Portugal', 'Nicaragua',' Peru', 'Ecuador', 'Guatemala', 'Philippines', 'Canada', 'Columbia', 'El-Salvador',' Japan', 'England', 'Trinadad&Tobago', 'Honduras', 'Germany', 'Taiwan', 'Outlying-U S (Guam USVI etc)', 'India', 'Vietnam', 'China',' Hong Kong',' Cambodia',' France', 'Laos', 'Haiti',' South Korea', 'Iran', 'Greece', 'Italy', 'Poland',' Thailand',' Yugoslavia', 'Holand-Netherlands', 'Ireland', 'Scotland', 'Hungary', 'Panama'))
    PEMNTVTY = col1.selectbox('mbirthc', ('Mexico', ' United-States', 'Puerto-Rico', 'Dominican-Republic', 'Jamaica', 'Cuba', 'Portugal', 'Nicaragua',' Peru', 'Ecuador', 'Guatemala', 'Philippines', 'Canada', 'Columbia', 'El-Salvador', ' Japan', 'England','Trinadad&Tobago', 'Honduras', 'Germany', 'Taiwan', 'Outlying-U S (Guam USVI etc)', 'India', 'Vietnam', 'China','Hong Kong', ' Cambodia', ' France', 'Laos', 'Haiti', ' South Korea', 'Iran', 'Greece', 'Italy', 'Poland','Thailand', ' Yugoslavia', 'Holand-Netherlands', 'Ireland', 'Scotland', 'Hungary', 'Panama'))
    PENATVTY = col2.selectbox('sbirthc', (
    'Mexico', ' United-States', 'Puerto-Rico', 'Dominican-Republic', 'Jamaica', 'Cuba', 'Portugal', 'Nicaragua', ' Peru',
    'Ecuador', 'Guatemala', 'Philippines', 'Canada', 'Columbia', 'El-Salvador', ' Japan', 'England', 'Trinadad&Tobago',
    'Honduras', 'Germany', 'Taiwan', 'Outlying-U S (Guam USVI etc)', 'India', 'Vietnam', 'China', ' Hong Kong', ' Cambodia',
    ' France', 'Laos', 'Haiti', ' South Korea', 'Iran', 'Greece', 'Italy', 'Poland', ' Thailand', ' Yugoslavia',
    'Holand-Netherlands', 'Ireland', 'Scotland', 'Hungary', 'Panama'))

    PRCITSHP =col1.selectbox('citizenship',('Native- Born in the United States', 'Foreign born- Not a citizen of U S' , 'Native- Born in Puerto Rico or U S Outlying',' Native- Born abroad of American Parent(s)',' Foreign born- U S citizen by naturalization'))


    SEOTR =col2.selectbox('ownerornot',( '0', '2', '1'))
    VETQVA =col1.selectbox('fillqorno',('Not in universe', 'Yes', 'No'))
    VETYN =col2.selectbox('vetbenifists',('0', '2', '1'))
    WKSWORK =col1.slider(label='nofweeksworked',min_value=0, max_value=52, format='%i', value=0)
    YEAR =col2.selectbox('Year',('94','95'))

    input_list = [AAGE,ACLSWKR,ADTIND ,ADTOCC, AHGA, AHRSPAY, AHSCOL,
       AMARITL, AMJIND, AMJOCC, ARACE, AREORGN, ASEX, AUNMEN,
       AUNTYPE, AWKSTAT, CAPGAIN, CAPLOSS, DIVVAL, FILESTAT,
       GRINREG, GRINST, HHDFMX, HHDREL, MIGMTR1, MIGMTR3,
       MIGMTR4, MIGSAME, MIGSUN,NOEMP, PARENT, PEFNTVTY,
       PEMNTVTY, PENATVTY, PRCITSHP, SEOTR, VETQVA, WKSWORK,
       VETYN, YEAR]

    input_preds_class = model.predict(input_list)
    input_preds_proba = model.predict_proba(input_list)

    submit = st.sidebar.button('Get predictions')
    if submit:
    # Write predictions on Streamlit app
        st.sidebar.write('Class predicted :', input_preds_class[0])
        st.sidebar.write(pd.DataFrame({'Genre': model.classes_, 'Probability': input_preds_proba}))

    df_input = pd.DataFrame([input_list], columns=['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY', 'AHSCOL',
                                                       'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 'AUNMEN',
                                                       'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS', 'DIVVAL', 'FILESTAT',
                                                       'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 'MIGMTR1', 'MIGMTR3',
                                                       'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP', 'PARENT', 'PEFNTVTY',
                                                       'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN',
                                                       ' WKSWORK', 'YEAR'])
        # Calculate shap values of inputed instance
    explainer = shap.TreeExplainer(model)
    input_shap_values = explainer.shap_values(df_input)



    # SHAP force plot for inputed instance predicted class
    st.subheader('Force plot')

    force_plot = shap.force_plot(explainer.expected_value[np.argmax(input_preds_proba)],
                        input_shap_values[np.argmax(input_preds_proba)],
                        eval_set_features,
                        matplotlib=True,
                        show=False)

    plt.suptitle(f'Class predicted : {model.classes_[np.argmax(input_preds_proba)]}',
                 fontsize=20,
                 y=1.35)

    st.pyplot(force_plot)
    #
    # Force plot expander explanations
    with st.beta_expander("More on force plots"):
         st.markdown("""
            The Force plot shows how each feature has contributed in moving away or towards the base value (average class output of the evaluation dataset) in to the predicted value of the specific instance (inputed on the left side bar) for the predicted class.
    
            Those values are **log odds**: SHAP doesn't support output probabilities for Multiclassification as of now.
    
            The SHAP values displayed are additive. Once the negative values (blue) are substracted from the positive values (pink), the distance from the base value to the output remains.
    
         """)

    #
    #
    # SHAP decision plot for inputed instance
    st.subheader('Decision plot')

    def class_labels(row_index):
        return [f'{model.classes_[i]} (pred: {input_preds_proba[i].round(2)})' for i in range(len(expected_value))]

    decision_plot, ax = plt.subplots()
    ax = shap.multioutput_decision_plot(expected_value,
                                   input_shap_values,
                                   row_index=0,
                                   feature_names=eval_set_features,
                                   legend_labels=class_labels(0),
                                   legend_location='lower right',
                                   link='logit',
                                   highlight=np.argmax(input_preds_proba)) # Highlight the predicted class

    st.pyplot(decision_plot)
    #
    # Decision plot expander explanations
    with st.beta_expander("More on decision plots"):
         st.markdown("""
         Just like the force plot, the [**decision plot**](https://slundberg.github.io/shap/notebooks/plots) shows how each feature has contributed in moving away or towards the base value (the grey line, aka. the average model output on the evaluation dataset) to the predicted value of the specific instance (inputed on the left side bar), but allows us to visualize those effects **for each class**.
    It also show the impact of less influencial features more clearly.
    
    From SHAP documentation:
    - *The x-axis represents the model's output. In this case, the units are log odds. (SHAP doesn't support probability output for multiclass)*
    - *The plot is centered on the x-axis at explainer.expected_value (the base value). All SHAP values are relative to the model's expected value like a linear model's effects are relative to the intercept.*
    - *The y-axis lists the model's features. By default, the features are ordered by descending importance. The importance is calculated over the observations plotted. _This is usually different than the importance ordering for the entire dataset._ In addition to feature importance ordering, the decision plot also supports hierarchical cluster feature ordering and user-defined feature ordering.*
    - *Each observation's prediction is represented by a colored line. At the top of the plot, each line strikes the x-axis at its corresponding observation's predicted value. This value determines the color of the line on a spectrum.*
    - *Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model's base value. This shows how each feature contributes to the overall prediction.*
    - *At the bottom of the plot, the observations converge at explainer.expected_value (the base value)*""")

    # Set up 2 columns to display in the body of the app
    st.subheader('Dependence plot: SHAP values of the evaluation dataset')
    colbis1, colbis2, colbis3 = st.beta_columns(3)



    # Selectors for dependence plot
    class_selector = colbis1.selectbox('Income Class:', model.classes_, index=1)
    feature_selector = colbis2.selectbox('Main feature :', X_test.columns, index=2)
    interaction_selector = colbis3.selectbox('Interaction feature :', X_test.columns, index=3)

    # SHAP dependence plot
    shap.initjs()
    dependence_plot= shap.dependence_plot(feature_selector,
                            shap_values[model.classes_.tolist().index(class_selector)],
                            X_test,
                            interaction_index=interaction_selector,
                            x_jitter=0.95,
                            alpha=0.4,
                            dot_size=6,
                            show=True)

    plt.title(f'Income Class : {model.classes_[model.classes_.tolist().index(class_selector)]}', fontsize=10)

    st.pyplot(dependence_plot)


    explainer1 = shap.KernelExplainer(model=model.predict_proba, data=shap.sample(X_test,50))

    shap_values = explainer1.shap_values(X_test.iloc[0:50, :], nsamples=50)
    st.pyplot(shap.summary_plot(shap_values, X_test, class_names=['<50K', '>=50k']))


    shap.initjs()

    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values[1], X_test.iloc[0:50,:]))
















