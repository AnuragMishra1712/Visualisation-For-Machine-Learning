import streamlit as st
import pandas as pd
from multiapp import MultiApp
import data,home,prediction,limevalue,shapcopy,syn
import numpy as np


app = MultiApp()

st.title("\n"
         "Explanations Tool\n")

app.add_app('Home',home.app)
app.add_app('Data',data.app)
app.add_app('Prediction',prediction.app)
app.add_app('Lime',limevalue.app)
app.add_app('Shap',shapcopy.app)
app.add_app('Synthetic',syn.app)
# dashboardurl = 'http://192.168.1.33:8501/'
#st.components.v1.iframe(dashboardurl, width=None, height=900, scrolling=True)


app.run()
