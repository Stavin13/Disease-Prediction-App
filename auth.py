import streamlit as st
import hashlib
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, is_admin INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT,
                  timestamp TEXT,
                  age INTEGER,
                  gender TEXT,
                  symptoms TEXT,
                  prediction TEXT,
                  risk_score INTEGER)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_user():
    st.sidebar.title("👤 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        hashed_pwd = hash_password(password)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", 
                 (username, hashed_pwd))
        user = c.fetchone()
        conn.close()
        
        if user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['is_admin'] = bool(user[2])
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

def show_admin_dashboard():
    st.title("👑 Admin Dashboard")
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    st.subheader("📊 Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        st.metric("Unique Users", df['username'].nunique())
    with col3:
        st.metric("High Risk Cases", len(df[df['risk_score'] > 75]))
    
    st.subheader("📈 Predictions Over Time")
    fig = px.line(df, x='timestamp', y='risk_score', color='username')
    st.plotly_chart(fig)