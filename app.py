import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Page config
st.set_page_config(
    page_title="The Duration Dilemma Dashboard",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stat-box {
        background-color: #1E88E5;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .highlight {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and clean the data, just like in the original script"""
    df = pd.read_csv('movie_metadata.csv')
    df = df[['movie_title', 'title_year', 'duration', 'imdb_score', 'genres', 'director_name']]
    df = df.dropna(subset=['duration', 'imdb_score'])
    
    # Create duration categories
    def categorize_duration(duration):
        if duration < 90:
            return 'Short (<90 min)'
        elif duration <= 150:
            return 'Medium (90-150 min)'
        else:
            return 'Long (>150 min)'
    
    df['duration_category'] = df['duration'].apply(categorize_duration)
    
    # Create 10-minute duration bins
    df['duration_bin'] = pd.cut(df['duration'], bins=range(0, int(df['duration'].max()) + 10, 10))
    
    return df

# Load data
try:
    df = load_data()
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(df['duration'], df['imdb_score'])
    
    # ==== MAIN TITLE & SUMMARY SECTION ====
    st.markdown("<h1 class='main-header'>🎬 The Duration Dilemma: Do Longer Movies Really Rule?</h1>", unsafe_allow_html=True)
    st.write("Welcome to my deep dive into Hollywood's age-old question! I'm investigating if those marathon movies actually deserve their critical acclaim... or if we're just sitting through extra hours for nothing.")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("Total Movies Analyzed", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("Average IMDb Rating", f"{df['imdb_score'].mean():.2f}/10")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
        st.metric("Correlation Coefficient", f"{correlation:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("---")
    
    # ==== SIDEBAR CONTROLS ====
    st.sidebar.header("Dashboard Controls")
    
    # Rating range filter for scatter plot
    st.sidebar.subheader("Rating Filter")
    min_rating, max_rating = st.sidebar.slider(
        "Select IMDb Rating Range",
        min_value=float(df['imdb_score'].min()),
        max_value=float(df['imdb_score'].max()),
        value=(float(df['imdb_score'].min()), float(df['imdb_score'].max()))
    )
    
    # Category highlight for scatter plot
    st.sidebar.subheader("Highlight Category")
    highlight_category = st.sidebar.selectbox(
        "Select Duration Category to Highlight",
        ["All Categories"] + list(df['duration_category'].unique())
    )
    
    # Bin size for histogram
    st.sidebar.subheader("Histogram Settings")
    bin_size = st.sidebar.slider("Bin Size (minutes)", min_value=5, max_value=30, value=10, step=5)
    
    # Custom duration range finder
    st.sidebar.subheader("Sweet Spot Finder")
    custom_min = st.sidebar.number_input("Minimum Duration (minutes)", min_value=int(df['duration'].min()), max_value=int(df['duration'].max()), value=90)
    custom_max = st.sidebar.number_input("Maximum Duration (minutes)", min_value=int(df['duration'].min()), max_value=int(df['duration'].max()), value=120)
    
    # Apply filters
    filtered_df = df[(df['imdb_score'] >= min_rating) & (df['imdb_score'] <= max_rating)]
    
    # ==== INTERACTIVE SCATTER PLOT ====
    st.markdown("<h2 class='sub-header'>📊 Duration vs. Rating Relationship</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create scatter plot with different colors for categories
        if highlight_category != "All Categories":
            # Split data for highlighting
            highlight_df = filtered_df[filtered_df['duration_category'] == highlight_category]
            other_df = filtered_df[filtered_df['duration_category'] != highlight_category]
            
            # Create figure with highlighted category
            fig = go.Figure()
            
            # Add other categories with reduced opacity
            for category in other_df['duration_category'].unique():
                subset = other_df[other_df['duration_category'] == category]
                fig.add_trace(go.Scatter(
                    x=subset['duration'],
                    y=subset['imdb_score'],
                    mode='markers',
                    name=category,
                    marker=dict(
                        color={'Short (<90 min)': '#FFA15A', 'Medium (90-150 min)': '#19D3F3', 'Long (>150 min)': '#FF6692'}[category],
                        opacity=0.3
                    ),
                    hovertemplate='<b>%{text}</b><br>Duration: %{x} min<br>Rating: %{y}<extra></extra>',
                    text=subset['movie_title']
                ))
            
            # Add highlighted category with full opacity
            fig.add_trace(go.Scatter(
                x=highlight_df['duration'],
                y=highlight_df['imdb_score'],
                mode='markers',
                name=highlight_category,
                marker=dict(
                    color={'Short (<90 min)': '#FFA15A', 'Medium (90-150 min)': '#19D3F3', 'Long (>150 min)': '#FF6692'}[highlight_category],
                    opacity=1.0
                ),
                hovertemplate='<b>%{text}</b><br>Duration: %{x} min<br>Rating: %{y}<extra></extra>',
                text=highlight_df['movie_title']
            ))
        else:
            # Regular plot without highlighting
            fig = px.scatter(
                filtered_df,
                x="duration",
                y="imdb_score",
                color="duration_category",
                hover_data=["movie_title", "director_name"],
                title=f"Movie Duration vs. IMDb Rating (r = {correlation:.2f})",
                labels={"duration": "Duration (minutes)", "imdb_score": "IMDb Rating"},
                height=500,
                color_discrete_sequence=["#FFA15A", "#19D3F3", "#FF6692"]
            )
        
        # Add trend line
        slope, intercept, _, _, _ = stats.linregress(filtered_df["duration"], filtered_df["imdb_score"])
        x_range = np.array([filtered_df["duration"].min(), filtered_df["duration"].max()])
        y_range = slope * x_range + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                name="Trend Line",
                line=dict(color="red", width=2)
            )
        )
        
        # Set title and customize layout
        fig.update_layout(
            title=f"Movie Duration vs. IMDb Rating (r = {correlation:.2f})",
            xaxis_title="Duration (minutes)",
            yaxis_title="IMDb Rating",
            legend_title="Duration Category",
            font=dict(size=12),
            hovermode="closest",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.markdown("### What's the correlation tell us?")
        
        if abs(correlation) < 0.1:
            st.write("There's basically no correlation here. Movie length and quality are like pineapple and pizza - some say they go together, but the data suggests otherwise.")
        elif abs(correlation) < 0.3:
            st.write("There's a weak correlation. Not exactly a Hollywood romance, but they're definitely flirting with each other.")
        elif abs(correlation) < 0.5:
            st.write("There's a moderate correlation. Like a decent romantic subplot in an action movie - it's there, and it matters.")
        else:
            st.write("There's a strong correlation! These two variables are like the iconic duo in a buddy cop movie - they're definitely linked.")
            
        st.markdown(f"**P-value:** {p_value:.4f}")
        
        if p_value < 0.05:
            st.write("This correlation is statistically significant! No statistical hocus-pocus here, folks.")
        else:
            st.write("This correlation is not statistically significant. Take it with a grain of popcorn salt.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("---")
    
    # ==== DURATION CATEGORY ANALYSIS ====
    st.markdown("<h2 class='sub-header'>🎯 Duration Category Analysis</h2>", unsafe_allow_html=True)
    
    # Calculate category stats
    category_stats = filtered_df.groupby('duration_category')['imdb_score'].agg(['mean', 'median', 'std', 'count']).reset_index()
    category_stats = category_stats.round(2)
    
    # Sort categories in logical order
    category_order = ['Short (<90 min)', 'Medium (90-150 min)', 'Long (>150 min)']
    category_stats['duration_category'] = pd.Categorical(
        category_stats['duration_category'], 
        categories=category_order,
        ordered=True
    )
    category_stats = category_stats.sort_values('duration_category')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive bar chart
        fig = px.bar(
            category_stats,
            x='duration_category',
            y='mean',
            color='duration_category',
            labels={"mean": "Average IMDb Rating", "duration_category": "Duration Category"},
            title="Average IMDb Rating by Duration Category",
            text=category_stats['mean'].round(2),
            height=400,
            color_discrete_sequence=["#FFA15A", "#19D3F3", "#FF6692"]
        )
        
        # Customize layout
        fig.update_layout(
            showlegend=False,
            font=dict(size=12),
            yaxis=dict(range=[0, 10]),  # IMDb ratings are out of 10
        )
        
        # Format text labels
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.subheader("Rating Statistics by Category")
        
        # Format the table with custom styling
        st.dataframe(
            category_stats,
            column_config={
                "duration_category": "Duration Category",
                "mean": st.column_config.NumberColumn("Average", format="%.2f"),
                "median": st.column_config.NumberColumn("Median", format="%.2f"),
                "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "count": st.column_config.NumberColumn("Count", format="%d"),
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show statistics for clicked category
        clicked_category = st.selectbox("Select a category for detailed stats:", category_order)
        cat_data = category_stats[category_stats['duration_category'] == clicked_category].iloc[0]
        
        st.markdown(f"### {clicked_category} Movies")
        st.markdown(f"**Count:** {int(cat_data['count'])} movies ({cat_data['count']/len(filtered_df)*100:.1f}% of total)")
        st.markdown(f"**Average Rating:** {cat_data['mean']:.2f}/10")
        st.markdown(f"**Median Rating:** {cat_data['median']:.2f}/10")
    
    st.write("---")
    
    # ==== SWEET SPOT FINDER ====
    st.markdown("<h2 class='sub-header'>🍯 Sweet Spot Finder</h2>", unsafe_allow_html=True)
    
    # Calculate bin statistics
    bin_stats = filtered_df.groupby('duration_bin')['imdb_score'].agg(['mean', 'count']).reset_index()
    bin_stats = bin_stats[bin_stats['count'] >= 10]  # Only bins with sufficient movies
    
    # Convert interval to midpoint for plotting
    bin_stats['midpoint'] = bin_stats['duration_bin'].apply(lambda x: x.mid)
    
    # Find optimal bin
    best_bin = bin_stats.loc[bin_stats['mean'].idxmax()]
    
    # Calculate custom range statistics
    custom_range_df = filtered_df[(filtered_df['duration'] >= custom_min) & (filtered_df['duration'] <= custom_max)]
    custom_avg = custom_range_df['imdb_score'].mean() if len(custom_range_df) > 0 else 0
    custom_count = len(custom_range_df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create line chart showing ratings by duration bin
        fig = px.line(
            bin_stats,
            x='midpoint',
            y='mean',
            labels={"midpoint": "Duration (minutes)", "mean": "Average IMDb Rating"},
            title="Average Rating by 10-Minute Duration Intervals",
            height=400,
        )
        
        # Highlight the optimal bin
        fig.add_trace(
            go.Scatter(
                x=[best_bin['midpoint']],
                y=[best_bin['mean']],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name=f"Best: {best_bin['duration_bin']} ({best_bin['mean']:.2f}/10)",
                hoverinfo="name"
            )
        )
        
        # Add custom range if selected
        if custom_count > 0:
            fig.add_shape(
                type="rect",
                x0=custom_min,
                y0=0,
                x1=custom_max,
                y1=10,
                fillcolor="rgba(0,176,246,0.2)",
                line=dict(width=0),
                name="Custom Range"
            )
            fig.add_annotation(
                x=(custom_min + custom_max)/2,
                y=custom_avg + 0.2,
                text=f"Custom: {custom_avg:.2f}/10",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#00B0F6",
                arrowsize=1,
                font=dict(color="#00B0F6")
            )
        
        # Customize layout
        fig.update_layout(
            hovermode="x unified",
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.subheader("🏆 Optimal Duration")
        
        st.markdown(f"**Best Range:** {best_bin['duration_bin']}")
        st.markdown(f"**Average Rating:** {best_bin['mean']:.2f}/10")
        st.markdown(f"**Movie Count:** {best_bin['count']} films")
        
        st.subheader("🔍 Custom Range Analysis")
        st.markdown(f"**Selected Range:** {custom_min}-{custom_max} minutes")
        
        if custom_count > 0:
            st.markdown(f"**Average Rating:** {custom_avg:.2f}/10")
            st.markdown(f"**Movie Count:** {custom_count} films")
            
            # Compare to overall average
            overall_avg = filtered_df['imdb_score'].mean()
            diff = custom_avg - overall_avg
            
            if diff > 0:
                st.success(f"This range scores {diff:.2f} points higher than the overall average!")
            else:
                st.warning(f"This range scores {abs(diff):.2f} points lower than the overall average.")
        else:
            st.warning("No movies found in this range.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("---")
    
    # ==== MOVIE DISTRIBUTION ====
    st.markdown("<h2 class='sub-header'>📈 Movie Duration Distribution</h2>", unsafe_allow_html=True)
    
    # Calculate statistics
    mean_duration = filtered_df['duration'].mean()
    median_duration = filtered_df['duration'].median()
    percentiles = [10, 25, 75, 90]
    percentile_values = np.percentile(filtered_df['duration'], percentiles)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create histogram
        fig = px.histogram(
            filtered_df,
            x="duration",
            nbins=int((filtered_df['duration'].max() - filtered_df['duration'].min()) / bin_size),
            title="Distribution of Movie Durations",
            labels={"duration": "Duration (minutes)", "count": "Number of Movies"},
            height=400,
            color_discrete_sequence=["#6739B7"]
        )
        
        # Add mean and median lines
        fig.add_vline(
            x=mean_duration,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Mean: {mean_duration:.1f} min",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=median_duration,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {median_duration:.1f} min",
            annotation_position="top"
        )
        
        # Customize layout
        fig.update_layout(
            bargap=0.1,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.subheader("Duration Statistics")
        
        st.markdown(f"**Mean Duration:** {mean_duration:.1f} minutes")
        st.markdown(f"**Median Duration:** {median_duration:.1f} minutes")
        
        st.subheader("Percentiles")
        for p, val in zip(percentiles, percentile_values):
            st.markdown(f"**{p}th Percentile:** {val:.1f} minutes")
        
        # Category counts
        category_counts = filtered_df['duration_category'].value_counts().sort_index()
        
        st.subheader("Movies by Category")
        for category, count in category_counts.items():
            percentage = 100 * count / len(filtered_df)
            st.markdown(f"- **{category}**: {count} movies ({percentage:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("---")
    
    # ==== KEY INSIGHTS SECTION ====
    st.markdown("<h2 class='sub-header'>💡 Key Insights</h2>", unsafe_allow_html=True)
    
    st.info("**Insight #1**: There is a weak positive correlation (0.26) between movie duration and IMDb ratings, suggesting longer movies do tend to score marginally higher.")
    
    st.info("**Insight #2**: Long movies (>150 min) have significantly higher ratings (7.44/10 on average) than short movies (<90 min) which average 6.15/10.")
    
    st.info("**Insight #3**: The sweet spot duration appears to be in the 180-190 minute range, with an impressive average rating of 7.71/10.")
    
    st.markdown("### The Final Verdict")
    
    st.write("""
    After diving deep into the data, I can now answer the burning question: **Do longer movies actually get better ratings?**
    
    Yes, but with important caveats:
    
    - The effect is modest at best (correlation of 0.26)
    - There's diminishing returns for extremely long movies
    - Many short films are critically acclaimed - a great 90-minute movie beats a mediocre 3-hour epic any day!
    
    In conclusion, while duration does have some relationship with ratings, it seems that **what you do with your runtime matters far more than the runtime itself**. 
    Quality storytelling, compelling characters, and skillful filmmaking will always matter more than whether you keep audiences in their seats for an extra 30 minutes.
    """)
    
    st.markdown("""
    <div style='text-align: center; font-style: italic; margin-top: 30px;'>
    "No good movie is too long, and no bad movie is short enough."
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Created by manish paneu | Data Analyst
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Oops! It looks like there was an error loading or processing the data: {e}")
    st.warning("Make sure you have the 'movie_metadata.csv' file in the same directory as this app.") 