import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_risk_timeline(history_data):
    fig = px.line(history_data, x='timestamp', y='risk_score',
                  color='disease', title='Risk Score Timeline')
    return fig

def create_disease_distribution(history_data):
    fig = px.pie(history_data, names='disease',
                 title='Disease Prediction Distribution')
    return fig