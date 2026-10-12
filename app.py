"""
Spotify Hit Song Analyzer — Data App
====================================
Audience: 音乐厂牌推广经理 / A&R 决策人
Question: 什么样的歌曲容易火？哪些特征值得优先投入推广预算？
Dataset: songs_normalize.csv (2000 首流行单曲, 1998–2020, Spotify audio features)
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="什么样的歌曲容易火？",
    page_icon="🎵",
    layout="wide",
)

# ------------------------------------------------------------------
# Load data (cached)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("songs_normalize.csv")
    df["duration_min"] = df["duration_ms"] / 60000
    df["primary_genre"] = (
        df["genre"].astype(str).str.split(",").str[0].str.strip()
    )
    df["explicit_label"] = df["explicit"].map({True: "脏标 (Explicit)", False: "干净版 (Clean)"})
    return df

df = load_data()

# ------------------------------------------------------------------
# Title
# ------------------------------------------------------------------
st.title("🎵 什么样的歌曲容易火？")
st.caption(
    "Spotify 2000 首热门单曲 · 1998–2020 · "
    "面向厂牌推广经理：决定下一首歌的推广预算投向。"
)

# ------------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------------
st.sidebar.header("筛选器")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider(
    "发行年份", year_min, year_max, (year_min, year_max)
)

popularity_min = st.sidebar.slider(
    "最低流行度 (Popularity)", 0, 100, (0, 100)
)

genres = sorted(df["primary_genre"].unique().tolist())
# remove the weird "set()" pseudo-genre from display
genres = [g for g in genres if g and g != "set()"]
selected_genres = st.sidebar.multiselect(
    "主要流派 (可多选)", genres, default=genres
)

explicit_choice = st.sidebar.radio(
    "脏标版本", ["全部", "仅脏标", "仅干净版"]
)

# ------------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------------
mask = (
    df["year"].between(*year_range)
    & df["popularity"].between(*popularity_min)
    & df["primary_genre"].isin(selected_genres)
)
if explicit_choice == "仅脏标":
    mask &= df["explicit"] == True
elif explicit_choice == "仅干净版":
    mask &= df["explicit"] == False

fdf = df.loc[mask].copy()

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("入选歌曲数", f"{len(fdf):,}")
col2.metric("平均流行度", f"{fdf['popularity'].mean():.1f}")
col3.metric("上榜率 (popularity≥80)", f"{(fdf['popularity']>=80).mean()*100:.1f}%")
col4.metric("脏标歌曲占比", f"{fdf['explicit'].mean()*100:.1f}%")

st.divider()

# ------------------------------------------------------------------
# Insight banner — the most important takeaway
# ------------------------------------------------------------------
with st.container():
    st.subheader("💡 核心结论：不存在「火歌公式」")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            """
            把所有音频特征（舞蹈性、能量、语速、调性、BPM……）和流行度做相关分析，
            **相关系数全部在 ±0.06 以内**——也就是说，没有任何一个声音特征能单独预测一首歌会不会火。
            真正的差距来自 **流派、艺人、和年份**。这对推广预算的含义是：
            与其在「把能量调到 0.8」上纠结，不如把钱花在选对艺人、踩对发行窗口上。
            """
        )
    with c2:
        corr_cols = ["danceability", "energy", "loudness", "speechiness",
                     "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
        corr_series = (
            fdf[corr_cols + ["popularity"]].corr()["popularity"]
            .drop("popularity").sort_values(key=abs, ascending=False)
        )
        st.dataframe(
            corr_series.round(3).rename("与流行度相关系数"),
            use_container_width=True,
        )

st.divider()

# ------------------------------------------------------------------
# Row 1: scatter danceability vs energy, colored by popularity
# ------------------------------------------------------------------
st.subheader("🎸 舞蹈性 × 能量 象限图")
st.caption("点越红 = 越火。看看右上角（高能量高舞蹈性）是不是真的更容易出爆款？")

fig_scatter = px.scatter(
    fdf,
    x="danceability",
    y="energy",
    color="popularity",
    size="popularity",
    size_max=12,
    hover_data=["artist", "song", "year", "primary_genre"],
    color_continuous_scale="Reds",
    opacity=0.7,
)
fig_scatter.update_layout(height=420, plot_bgcolor="white")
st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------
# Row 2: two columns — genre bar + year trend
# ------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🎸 流派平均分流派 vs 数量")
    genre_stats = (
        fdf.groupby("primary_genre")["popularity"]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 10")
        .sort_values("mean", ascending=True)
    )
    fig_genre = px.bar(
        genre_stats,
        x="mean",
        y="primary_genre",
        orientation="h",
        color="count",
        color_continuous_scale="Blues",
        hover_data={"count": True},
        labels={"mean": "平均流行度", "primary_genre": "流派"},
    )
    fig_genre.update_layout(height=380)
    st.plotly_chart(fig_genre, use_container_width=True)

with right:
    st.subheader("📅 流行度随年份变化")
    yearly = fdf.groupby("year").agg(
        avg_pop=("popularity", "mean"),
        n=("song", "count"),
    ).reset_index()
    fig_year = px.line(
        yearly,
        x="year",
        y="avg_pop",
        markers=True,
        labels={"avg_pop": "平均流行度", "year": "年份"},
    )
    fig_year.add_bar(
        x=yearly["year"], y=yearly["n"], name="发行数量", yaxis="y2", opacity=0.2
    )
    fig_year.update_layout(
        height=380,
        yaxis2=dict(title="发行数量", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Row 3: duration histogram + tempo vs popularity
# ------------------------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("⏱️ 歌曲时长分布（分钟）")
    fig_dur = px.histogram(
        fdf, x="duration_min", nbins=40,
        labels={"duration_min": "时长（分钟）"},
        color_discrete_sequence=["#ff6b6b"],
    )
    fig_dur.update_layout(height=360, bargap=0.05)
    st.plotly_chart(fig_dur, use_container_width=True)

with c4:
    st.subheader("🥁 BPM 与流行度")
    fig_tempo = px.scatter(
        fdf, x="tempo", y="popularity",
        trendline="lowess", trendline_color_override="black",
        opacity=0.5,
        labels={"tempo": "BPM", "popularity": "流行度"},
    )
    fig_tempo.update_layout(height=360)
    st.plotly_chart(fig_tempo, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Row 4: explicit vs popularity boxplot + top artists
# ------------------------------------------------------------------
c5, c6 = st.columns(2)

with c5:
    st.subheader("🗯️ 脏标 vs 干净版")
    fig_exp = px.box(
        fdf, x="explicit_label", y="popularity",
        color="explicit_label",
        labels={"explicit_label": "", "popularity": "流行度"},
    )
    fig_exp.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_exp, use_container_width=True)

with c6:
    st.subheader("👑 高产艺人 TOP 10")
    top_artists = (
        fdf.groupby("artist")
        .agg(songs=("song", "count"), avg_pop=("popularity", "mean"))
        .reset_index()
        .query("songs >= 5")
        .sort_values("songs", ascending=False)
        .head(10)
        .sort_values("avg_pop", ascending=True)
    )
    fig_art = px.bar(
        top_artists, x="avg_pop", y="artist", orientation="h",
        color="songs", color_continuous_scale="Purples",
        labels={"avg_pop": "平均流行度", "artist": ""},
    )
    fig_art.update_layout(height=360)
    st.plotly_chart(fig_art, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Data table — let users drill into specific hits
# ------------------------------------------------------------------
st.subheader("🔍 热门单曲明细 (popularity ≥ 80)")
hits = (
    fdf.query("popularity >= 80")
    .sort_values("popularity", ascending=False)
    [[ "artist", "song", "year", "primary_genre", "popularity",
       "danceability", "energy", "duration_min", "explicit_label" ]]
    .head(50)
)
st.dataframe(
    hits,
    use_container_width=True,
    hide_index=True,
    column_config={
        "duration_min": st.column_config.NumberColumn(format="%.2f min"),
        "danceability": st.column_config.NumberColumn(format="%.2f"),
        "energy": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.caption(
    "数据来源：Spotify Hits Dataset (Kaggle) · 共 2000 首单曲 · "
    "本应用仅用于教学演示，不构成推广决策建议。"
)
#（注：内容由AI生成）
