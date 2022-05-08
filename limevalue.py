import pickle
import pandas as pd
import lime
from lime import lime_tabular
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Import packages
import numpy as np
import streamlit.components.v1 as components
import lime
from IPython.display import HTML
from lime import lime_tabular
import pickle
import shap
from sklearn.preprocessing import StandardScaler




from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import streamlit as st
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer

import numpy as np


#st.set_page_config(page_title='LIME Values Explorer')
st.set_option('deprecation.showPyplotGlobalUse', False)


def app():# Load CatBoost model
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

    # expected_value = st_cache('expected_value')
    # X_test = st_cache('X_test')
    # eval_set_features = st_cache('eval_set_features')
    # train_set_features = st_cache('train_set_features')
    # shap_values = st_cache('shap_values')

    # Set Streamlit app body title
    st.title('Explore the LIME values of a CatBoost Binary Income Classfier')

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
        #st.sidebar.write(pd.DataFrame({'Genre': model.classes_, 'Probability': input_preds_proba}))



    df_input = pd.DataFrame([input_list],columns=['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY', 'AHSCOL',
     'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 'AUNMEN',
     'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS', 'DIVVAL', 'FILESTAT',
     'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 'MIGMTR1', 'MIGMTR3',
     'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP', 'PARENT', 'PEFNTVTY',
     'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYN',
     ' WKSWORK', 'YEAR'])

    census_df_test = pd.read_csv('limedata.csv')

    X_train, X_test, y_train, y_test = train_test_split(census_df_test.iloc[:,:-1],census_df_test.iloc[:,-1:], test_size=0.2, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    RandomForestClassifier(random_state=42)
    y_pred_ran_for = clf.predict(X_test)


    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['<50k', '>=50k'],
        mode='classification')
    exp = explainer.explain_instance(num_features=15,num_samples=50,
        data_row=X_test.iloc[5],
        predict_fn=clf.predict_proba)
    st.write('Lime Explanation based on inputs')
    exp.show_in_notebook(show_table=True)
    components.html(exp.as_html(), height=800)













