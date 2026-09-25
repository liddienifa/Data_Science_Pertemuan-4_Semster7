import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spotify Track Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for Modern, Clean & Minimalist Aesthetics
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #2D3142;
        background-color: #FAFAFA;
    }

    /* Main background override */
    .stApp {
        background-color: #FAFAFA;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F3EF;
        border-right: 1px solid #E5E3DC;
    }

    /* Card Containers */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #EAE8E1;
        border-radius: 4px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    .metric-title {
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8C8A84;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 500;
        color: #1A1A1A;
        letter-spacing: -0.5px;
    }

    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 500;
        color: #1A1A1A;
        letter-spacing: -0.3px;
        margin-top: 16px;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #EAE8E1;
    }

    .section-subheader {
        font-size: 13px;
        color: #6B6964;
        margin-bottom: 24px;
    }

    /* Table & Dataframe Clean Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #EAE8E1;
        border-radius: 4px;
    }

    /* Streamlit Button Styling */
    .stButton>button {
        background-color: #1A1A1A;
        color: #FFFFFF;
        border-radius: 2px;
        border: none;
        padding: 8px 20px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #333333;
        color: #FFFFFF;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Synthetic Data Generator (Fallback / Demo Data)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000
    genres = ['acoustic', 'pop', 'classical', 'hip-hop', 'rock', 'ambient']
    
    df = pd.DataFrame({
        'track_id': [f"track_{i:04d}" for i in range(n)],
        'track_name': [f"Composition {i+1}" for i in range(n)],
        'artists': [f"Artist {np.random.randint(1, 50)}" for _ in range(n)],
        'track_genre': np.random.choice(genres, n),
        'popularity': np.random.randint(10, 95, n),
        'danceability': np.random.uniform(0.2, 0.9, n),
        'energy': np.random.uniform(0.1, 0.95, n),
        'loudness': np.random.uniform(-20, -3, n),
        'speechiness': np.random.uniform(0.02, 0.3, n),
        'acousticness': np.random.uniform(0.0, 0.9, n),
        'instrumentalness': np.random.uniform(0.0, 0.8, n),
        'liveness': np.random.uniform(0.05, 0.4, n),
        'valence': np.random.uniform(0.1, 0.9, n),
        'tempo': np.random.uniform(70, 170, n)
    })
    return df

df = load_data()

# Feature Preprocessing & KMeans Clustering
feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 
    'tempo', 'popularity'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {
    0: "Acoustic & Minimal",
    1: "High Energy Rhythm",
    2: "Ambient & Instrumental",
    3: "Speech & Vocal Focus",
    4: "Melancholic & Slow"
}
df['cluster_name'] = df['cluster'].map(cluster_names)

# ---------------------------------------------------------
# Sidebar Filtering
# ---------------------------------------------------------
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("---")

selected_genres = st.sidebar.multiselect(
    "Filter Genre",
    options=sorted(df['track_genre'].unique()),
    default=df['track_genre'].unique()[:3]
)

selected_clusters = st.sidebar.multiselect(
    "Filter Cluster",
    options=sorted(df['cluster_name'].unique()),
    default=df['cluster_name'].unique()
)

popularity_range = st.sidebar.slider(
    "Popularity Range",
    min_value=int(df['popularity'].min()),
    max_value=int(df['popularity'].max()),
    value=(20, 90)
)

# Apply Filter
filtered_df = df[
    (df['track_genre'].isin(selected_genres)) &
    (df['cluster_name'].isin(selected_clusters)) &
    (df['popularity'].between(popularity_range[0], popularity_range[1]))
]

# ---------------------------------------------------------
# Main Page Layout
# ---------------------------------------------------------
st.title("Spotify Track Intelligence")
st.markdown("<div class='section-subheader'>Data Mining and Audio Feature Clustering Dashboard</div>", unsafe_allow_html=True)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Filtered Tracks</div>
            <div class="metric-value">{len(filtered_df):,}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Popularity</div>
            <div class="metric-value">{filtered_df['popularity'].mean():.1f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Energy</div>
            <div class="metric-value">{filtered_df['energy'].mean():.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Clusters</div>
            <div class="metric-value">{filtered_df['cluster'].nunique()}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1])

# Color Palette for Charts (Sophisticated, Neutral, Muted)
color_palette = ['#2D3142', '#8C8A84', '#B8B5AD', '#E5E3DC', '#4A4E69']

with col_left:
    st.markdown("<div class='section-header'>Cluster Feature Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Average audio attributes grouped by cluster personas</div>", unsafe_allow_html=True)
    
    cluster_profile = filtered_df.groupby('cluster_name')[['danceability', 'energy', 'acousticness', 'valence']].mean().reset_index()
    
    fig_radar = px.line_polar(
        cluster_profile.melt(id_vars='cluster_name'),
        r='value',
        theta='variable',
        color='cluster_name',
        line_close=True,
        color_discrete_sequence=color_palette,
        template='plotly_white'
    )
    fig_radar.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(title=None, orientation="h", y=-0.2),
        font=dict(family="Plus Jakarta Sans", size=12)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_right:
    st.markdown("<div class='section-header'>Audio Feature Correlation</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Energy versus Valence interaction across clusters</div>", unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        filtered_df,
        x='energy',
        y='valence',
        color='cluster_name',
        size='popularity',
        color_discrete_sequence=color_palette,
        template='plotly_white',
        hover_data=['track_name', 'artists']
    )
    fig_scatter.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(title=None, orientation="h", y=-0.2),
        font=dict(family="Plus Jakarta Sans", size=12),
        xaxis=dict(showgrid=True, gridcolor='#EAE8E1'),
        yaxis=dict(showgrid=True, gridcolor='#EAE8E1')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Table Section
# ---------------------------------------------------------
st.markdown("<div class='section-header'>Track Directory</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subheader'>Detailed dataset view with segment labels</div>", unsafe_allow_html=True)

display_cols = ['track_id', 'track_name', 'artists', 'track_genre', 'cluster_name', 'popularity', 'danceability', 'energy', 'tempo']

st.dataframe(
    filtered_df[display_cols].sort_values('popularity', ascending=False),
    use_container_width=True,
    height=320
)