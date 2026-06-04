# file: app.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide")

st.title("📊 Writing Intelligence Dashboard")


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("word_history.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

df["daily"] = df["words"].diff().fillna(0)
df["rolling"] = df["words"].rolling(7).mean()


# -------------------------
# METRICS
# -------------------------
latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Words", int(latest["words"]))
col2.metric("Daily Change", int(latest["daily"]))
col3.metric("7-day Avg", round(latest["rolling"], 1))
col4.metric("Streak", int((df["daily"][::-1] > 0).cumprod().sum()))


st.markdown("---")


# =========================
# SMALL MULTI-PANEL LAYOUT
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Total Growth")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(df["date"], df["words"], marker="o")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with c2:
    st.subheader("Daily Writing")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(df["date"], df["daily"])
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# =========================
# HEATMAP (GITHUB STYLE)
# =========================
st.subheader("🔥 Writing Heatmap (Daily Activity)")

df["day"] = df["date"].dt.date

heat = df.groupby("day")["daily"].sum().reset_index()
heat["day"] = pd.to_datetime(heat["day"])
heat["dow"] = heat["day"].dt.weekday
heat["week"] = heat["day"].dt.isocalendar().week

pivot = heat.pivot(index="dow", columns="week", values="daily").fillna(0)

fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(pivot, cmap="Greens", ax=ax)
ax.set_title("Writing Intensity")
st.pyplot(fig)


# =========================
# CHAPTER BREAKDOWN
# =========================
st.subheader("📚 Chapter Activity")

try:
    ch = pd.read_csv("chapter_history.csv")
    ch_group = ch.groupby("chapter")["count"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ch_group.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mentions")
    st.pyplot(fig)

except:
    st.info("No chapter data yet")


# =========================
# GIT ANALYSIS
# =========================
st.subheader("🧠 Words per Git Commit")

try:
    g = pd.read_csv("git_words.csv")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(g["date"], g["words"])
    ax.set_ylabel("Words")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

except:
    st.info("No git data yet")
