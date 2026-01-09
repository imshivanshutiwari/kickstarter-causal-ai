"""
Kickstarter Counterfactual Demand Simulator
Interactive Streamlit Dashboard for predicting campaign outcomes.

Features:
- Input hypothetical campaign configuration
- Get counterfactual predictions with uncertainty
- See similar historical campaigns
- Visualize demand curves
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Page config
st.set_page_config(
    page_title="Kickstarter Counterfactual Simulator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E2E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3D3D5C;
    }
    .stMetric label {
        color: #A0A0C0 !important;
    }
    .stMetric .css-1wivap2 {
        color: #00D4AA !important;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00D4AA;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #2e1a1a 0%, #3e2121 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #EF4444;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load historical campaign data."""
    data_path = Path(__file__).parent / "data" / "processed" / "kickstarter_causal_features.csv"
    if not data_path.exists():
        data_path = Path(__file__).parent / "data" / "raw" / "kickstarter_raw.csv"
    if not data_path.exists():
        data_path = Path(__file__).parent / "data" / "raw" / "kickstarter_raw_data.csv"
    
    df = pd.read_csv(data_path)
    return df


@st.cache_resource
def load_models():
    """Load trained models."""
    model_path = Path(__file__).parent / "src" / "models" / "causal_models.pkl"
    
    if model_path.exists():
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        return models
    else:
        # Build simple model on the fly
        df = load_data()
        duration_col = 'campaign_duration_days' if 'campaign_duration_days' in df.columns else 'duration_days'
        feature_cols = ['avg_reward_price', 'goal_ambition', duration_col, 'trend_index', 'concurrent_campaigns']
        
        df_model = df[feature_cols + ['funding_ratio']].dropna()
        X = df_model[feature_cols]
        y = df_model['funding_ratio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        
        # Quantile models
        quantile_models = {}
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            qm = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, max_depth=4, random_state=42)
            qm.fit(X, y)
            quantile_models[q] = qm
        
        return {
            'ols': model,
            'scaler': scaler,
            'quantile_models': quantile_models,
            'feature_cols': feature_cols,
            'tsls_coef': 0.00002
        }


def predict_outcome(features, models):
    """Make prediction using loaded models."""
    scaler = models['scaler']
    model = models['ols']
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    return max(0, prediction)


def predict_quantiles(features, models):
    """Get quantile predictions for uncertainty."""
    quantile_models = models.get('quantile_models', {})
    
    if not quantile_models:
        # Fallback
        base_pred = predict_outcome(features, models)
        return {
            0.1: base_pred * 0.6,
            0.25: base_pred * 0.8,
            0.5: base_pred,
            0.75: base_pred * 1.2,
            0.9: base_pred * 1.5
        }
    
    predictions = {}
    for q, model in quantile_models.items():
        predictions[q] = max(0, model.predict(features)[0])
    
    return predictions


# ... (previous imports)
from src.ai_consultant import AIConsultant

# ... (omitted functions)

def main():
    """Main dashboard application."""
    
    # Load data and models
    df = load_data()
    models = load_models()
    
    # Initialize AI
    ai_consultant = AIConsultant() 
    # NOTE: You could add st.sidebar.text_input("OpenAI Key") and pass it here
    
    # ... (omitted setup code until sidebar end) ...
    
    # Get column names
    duration_col = 'campaign_duration_days' if 'campaign_duration_days' in df.columns else 'duration_days'
    goal_col = 'funding_goal' if 'funding_goal' in df.columns else 'goal'
    
    # Header
    st.title("🚀 Kickstarter Counterfactual Simulator")
    st.markdown("*Estimate what would happen if you launched with different pricing strategies*")
    st.markdown("---")
    
    # SIDEBAR: User inputs
    st.sidebar.header("📋 Campaign Configuration")
    
    # Category selection
    categories = sorted(df['category'].dropna().unique())
    category = st.sidebar.selectbox("Category", categories, index=0)
    
    # Get category-specific stats
    cat_df = df[df['category'] == category]
    cat_median_goal = cat_df[goal_col].median() if len(cat_df) > 0 else 50000
    cat_median_price = cat_df['avg_reward_price'].median() if len(cat_df) > 0 else 50
    
    # Funding goal
    funding_goal = st.sidebar.number_input(
        "Funding Goal ($)", 
        min_value=1000, 
        max_value=1000000, 
        value=int(cat_median_goal),
        step=1000,
        help="Your campaign's funding target"
    )
    
    # Campaign duration
    campaign_duration = st.sidebar.slider(
        "Campaign Duration (days)", 
        15, 60, 30,
        help="How long your campaign will run"
    )
    
    # Average reward price
    avg_reward_price = st.sidebar.slider(
        "Average Reward Price ($)", 
        10, 500, int(cat_median_price),
        help="Average price across your reward tiers"
    )
    
    # Additional inputs
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Advanced Settings")
    
    num_tiers = st.sidebar.slider("Number of Reward Tiers", 1, 10, 5)
    trend_index = st.sidebar.slider("Market Trend Index (0-100)", 0, 100, 50, 
                                    help="Current interest level in your category")
    
    # Calculate derived features
    goal_ambition = funding_goal / cat_median_goal if cat_median_goal > 0 else 1.0
    
    # Feature vector
    feature_cols = models.get('feature_cols', ['avg_reward_price', 'goal_ambition', duration_col, 'trend_index', 'concurrent_campaigns'])
    
    features = pd.DataFrame({
        'avg_reward_price': [avg_reward_price],
        'goal_ambition': [goal_ambition],
        duration_col: [campaign_duration],
        'trend_index': [trend_index],
        'concurrent_campaigns': [df['concurrent_campaigns'].median() if 'concurrent_campaigns' in df.columns else 10]
    })
    
    # Ensure column order matches training and fill missing (including NLP) with 0
    # 0 for NLP embeddings implies "Average Text Quality" in PCA space
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0
    features = features[feature_cols]
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Prediction & Dynamics", "🤖 AI Consultant", "📚 Similar Campaigns"])
    
    # ==========================
    # TAB 1: PREDICTION
    # ==========================
    with tab1:
        st.header("🔮 Predicted Outcomes")
        
        # Get predictions
        funding_ratio = predict_outcome(features, models)
        expected_funding = funding_ratio * funding_goal
        quantile_preds = predict_quantiles(features, models)
        success_prob = min(1.0, max(0.0, (funding_ratio / 1.0)*0.7)) # Simplified logic
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta = f"{(funding_ratio-1)*100:.1f}% over goal" if funding_ratio > 1 else f"{(1-funding_ratio)*100:.1f}% short"
            st.metric("Expected Funding", f"${expected_funding:,.0f}", delta)
        
        with col2:
            st.metric("Funding Ratio", f"{funding_ratio:.2f}x")
        
        with col3:
            status = "✅ Likely Success" if funding_ratio >= 1.0 else "⚠️ May Fall Short"
            st.metric("Outlook", status)
        
        st.markdown("---")
        
        # COUNTERFACTUAL ANALYSIS
        st.subheader("🔄 Counterfactual: What If You Changed Price?")
        
        # Generate counterfactual predictions across price range
        min_price = max(10, int(avg_reward_price * 0.3))
        max_price = int(avg_reward_price * 2.5)
        price_range = range(min_price, max_price, max(1, (max_price - min_price) // 30))
        
        counterfactual_outcomes = []
        for price in price_range:
            features_cf = features.copy()
            features_cf['avg_reward_price'] = price
            cf_funding_ratio = predict_outcome(features_cf, models)
            counterfactual_outcomes.append({
                'price': price,
                'funding_ratio': cf_funding_ratio,
                'expected_funding': cf_funding_ratio * funding_goal
            })
        
        cf_df = pd.DataFrame(counterfactual_outcomes)
        optimal_idx = cf_df['expected_funding'].idxmax()
        optimal_price = cf_df.loc[optimal_idx, 'price']
        optimal_funding = cf_df.loc[optimal_idx, 'expected_funding']

        # Plot demand curve
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cf_df['price'],
            y=cf_df['funding_ratio'],
            mode='lines+markers',
            name='Predicted Funding Ratio',
            line=dict(color='#00D4AA', width=3),
            marker=dict(size=6)
        ))
        
        # Success threshold line
        fig.add_hline(y=1.0, line_dash="dash", line_color="#22C55E", 
                      annotation_text="Success Threshold (100%)")
        
        # Current price marker
        fig.add_vline(x=avg_reward_price, line_dash="dot", line_color="#EF4444",
                      annotation_text=f"Your Price (${avg_reward_price})")
        
        fig.update_layout(
            title="📈 Counterfactual Demand Curve",
            xaxis_title="Average Reward Price ($)",
            yaxis_title="Funding Ratio",
            hovermode='x unified',
            template='plotly_dark',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- NEW VISUALIZATION: GOAL SENSITIVITY ---
        st.markdown("---")
        st.subheader("🎯 Goal Sensitivity: How Ambitious Should You Be?")
        
        # Generate Goal Sensitivity Data
        # Range from 50% to 200% of current goal
        goal_multipliers = np.linspace(0.5, 2.0, 20)
        goal_sensitivity = []
        
        for mult in goal_multipliers:
            test_goal = funding_goal * mult
            features_goal = features.copy()
            # Recalculate ambition based on this new goal
            features_goal['goal_ambition'] = test_goal / cat_median_goal if cat_median_goal > 0 else 1.0
            
            # Predict
            pred_ratio = predict_outcome(features_goal, models)
            
            goal_sensitivity.append({
                'Goal': test_goal,
                'Funding Ratio': pred_ratio,
                'Probability': min(1.0, max(0.0, (pred_ratio / 1.0)*0.7)) * 100
            })
            
        df_goal_sens = pd.DataFrame(goal_sensitivity)
        
        fig_goal = px.line(df_goal_sens, x='Goal', y='Probability', 
                          title='Success Probability vs Funding Goal',
                          markers=True)
        fig_goal.add_vline(x=funding_goal, line_dash="dot", line_color="#EF4444", 
                          annotation_text="Current Goal")
        fig_goal.update_traces(line_color='#F59E0B', line_width=3)
        fig_goal.update_layout(yaxis_title="Probability of Success (%)", template='plotly_dark')
        
        st.plotly_chart(fig_goal, use_container_width=True)
        # -------------------------------------------
        
        
        # --- NEW VISUALIZATION: BENCHMARK RADAR ---
        with st.sidebar:
            st.markdown("---")
            st.subheader("🕸️ Category Benchmark")
            
            # Prepare Radar Data
            categories_radar = ['Price', 'Duration', 'Goal Ambition', 'Trend Score']
            
            # Normalize user values (0-1 scale approx for viz)
            # We compare against category medians we calculated earlier
            
            # 1. Price Score (Higher is more expensive relative to market)
            price_score = min(2.0, avg_reward_price / cat_median_price) if cat_median_price > 0 else 1.0
            
            # 2. Duration Score (Higher is longer)
            cat_median_duration = cat_df[duration_col].median() if len(cat_df) > 0 else 30
            dur_score = min(2.0, campaign_duration / cat_median_duration) if cat_median_duration > 0 else 1.0
            
            # 3. Ambition Score
            amb_score = min(2.0, goal_ambition)
            
            # 4. Trend Score (Normalized 0-1 from 0-100 input)
            tr_score = trend_index / 50.0 # 50 is avg
            
            data_radar = pd.DataFrame(dict(
                r=[price_score, dur_score, amb_score, tr_score],
                theta=categories_radar
            ))
            
            fig_radar = px.line_polar(data_radar, r='r', theta='theta', line_close=True)
            fig_radar.update_traces(fill='toself', line_color='#A855F7')
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 2.5]
                    )),
                template='plotly_dark',
                title="Vs. Market Average (1.0 = Avg)",
                height=300,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        # -------------------------------------------
        
        # Uncertainty chart (omitted for brevity, assume similar code)
        # For full implementation, keep existing code here.
        
    # ==========================
    # TAB 2: AI CONSULTANT
    # ==========================
    with tab2:
        st.header("🤖 Causal AI Strategy Consultant")
        st.markdown("This agent interprets the causal models to provide actionable strategy.")
        
        # Prepare data for AI
        campaign_data = {
            'avg_reward_price': avg_reward_price,
            'campaign_duration': campaign_duration,
            'goal_ambition': goal_ambition,
            'category': category
        }
        
        prediction_data = {
            'funding_ratio': funding_ratio,
            'success_probability': success_prob,
            'confidence_high': quantile_preds.get(0.9, 0) * funding_goal,
            'confidence_low': quantile_preds.get(0.1, 0) * funding_goal
        }
        
        cf_data = {
            'optimal_price': optimal_price,
            'optimal_funding': optimal_funding
        }
        
        if st.button("Generate Strategy Report", type="primary"):
            with st.spinner("Analyzing causal inference models..."):
                advice = ai_consultant.generate_advice(campaign_data, prediction_data, cf_data)
                
                st.markdown(f"## {advice['headline']}")
                
                # Summary Box
                st.info(advice['summary'])
                
                col_ai_1, col_ai_2 = st.columns(2)
                
                with col_ai_1:
                    st.subheader("🎯 Recommended Actions")
                    for i, strategy in enumerate(advice['strategies'], 1):
                        st.markdown(f"**{i}.** {strategy}")
                
                with col_ai_2:
                    st.subheader("⚠️ Risk Assessment")
                    risk_score = advice['risk_score']
                    st.slider("Risk Level", 0, 10, risk_score, disabled=True)
                    if risk_score > 7:
                        st.error("High Risk Detected")
                    elif risk_score > 4:
                        st.warning("Moderate Risk")
                    else:
                        st.success("Low Risk")

        # --- NEW VISUALIZATION: FEATURE IMPORTANCE (Always Visible) ---
        st.markdown("---")
        with st.expander("🧠 Model Logic: What drives success globally?"):
            st.write("Relative importance of factors based on our Causal Model:")
            
            # Extract coefficients
            if 'ols' in models:
                coeffs = models['ols'].coef_
                feat_names = models['feature_cols']
                
                # Create DataFrame
                fi_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Impact': coeffs
                }).sort_values('Impact', key=abs, ascending=True)
                
                # Make names prettier
                name_map = {
                    'avg_reward_price': 'Price ($)',
                    'goal_ambition': 'Goal Ambition',
                    'campaign_duration_days': 'Duration',
                    'duration_days': 'Duration',
                    'trend_index': 'Trendiness',
                    'concurrent_campaigns': 'Competition',
                    'nlp_dim_0': 'Keyword Quality'
                }
                fi_df['Feature'] = fi_df['Feature'].map(lambda x: name_map.get(x, x))
                
                # Color by positive/negative
                fi_df['Type'] = fi_df['Impact'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                color_map = {'Positive': '#22C55E', 'Negative': '#EF4444'}
                
                fig_imp = px.bar(fi_df, x='Impact', y='Feature', orientation='h',
                                title='Factor Impact on Funding Ratio',
                                color='Type', color_discrete_map=color_map)
                fig_imp.update_layout(showlegend=False, template='plotly_dark')
                
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Model coefficients not available.")
        # -------------------------------------------

    # ==========================
    # TAB 3: SIMILAR CAMPAIGNS
    # ==========================
    with tab3:
        st.header("📚 Similar Historical Campaigns")
        
        # Find similar campaigns
        feature_subset = ['goal_ambition', 'avg_reward_price', duration_col, 'trend_index']
        available_features = [f for f in feature_subset if f in df.columns and f in features.columns]
        
        if available_features:
            user_features = features[available_features].values
            hist_features = df[available_features].fillna(0).values
            
            distances = euclidean_distances(user_features, hist_features)[0]
            similar_indices = distances.argsort()[:10]
            
            display_cols = ['category', goal_col, 'avg_reward_price', 'funding_ratio', 'status']
            available_display = [c for c in display_cols if c in df.columns]
            
            similar_campaigns = df.iloc[similar_indices][available_display].copy()
            similar_campaigns.columns = ['Category', 'Goal', 'Avg Price', 'Funding Ratio', 'Status'][:len(available_display)]
            
            st.dataframe(similar_campaigns, use_container_width=True)
            
        # --- NEW: Similar Campaigns Comparison Chart ---
        st.markdown("---")
        st.subheader("📊 How You Compare to Similar Campaigns")
        
        if len(similar_campaigns) > 0:
            fig_compare = go.Figure()
            
            # Similar campaigns funding ratios
            fig_compare.add_trace(go.Bar(
                x=list(range(1, len(similar_campaigns)+1)),
                y=similar_campaigns['Funding Ratio'].values if 'Funding Ratio' in similar_campaigns.columns else [1]*len(similar_campaigns),
                name='Similar Campaigns',
                marker_color='#6366F1'
            ))
            
            # Your predicted ratio
            fig_compare.add_hline(y=funding_ratio, line_dash="dash", line_color="#00D4AA",
                                 annotation_text=f"Your Prediction: {funding_ratio:.2f}x")
            
            fig_compare.update_layout(
                title="Your Funding Ratio vs Similar Campaigns",
                xaxis_title="Campaign #",
                yaxis_title="Funding Ratio",
                template='plotly_dark',
                height=350
            )
            st.plotly_chart(fig_compare, use_container_width=True)

    # ==========================
    # TAB 4: ANALYTICS (NEW!)
    # ==========================
    # Add new tab for analytics
    tab4 = st.tabs(["📈 Market Analytics"])[0] if False else None
    
    # Since we can't dynamically add tabs, let's add analytics section in the sidebar and main area
    st.markdown("---")
    st.header("📈 Market Analytics Dashboard")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Chart 1: Category Success Rates
    with col_chart1:
        st.subheader("🏆 Success Rate by Category")
        if 'is_successful' in df.columns:
            cat_success = df.groupby('category')['is_successful'].mean().sort_values(ascending=True).tail(10) * 100
            fig_cat = px.bar(
                x=cat_success.values,
                y=cat_success.index,
                orientation='h',
                labels={'x': 'Success Rate (%)', 'y': 'Category'},
                color=cat_success.values,
                color_continuous_scale='Viridis'
            )
            fig_cat.update_layout(
                showlegend=False,
                template='plotly_dark',
                height=400,
                coloraxis_showscale=False
            )
            # Highlight user's category
            if category in cat_success.index:
                fig_cat.add_vline(x=cat_success[category], line_dash="dot", line_color="#EF4444",
                                 annotation_text=f"Your Category: {cat_success[category]:.1f}%")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Category success data not available")
    
    # Chart 2: Funding Distribution
    with col_chart2:
        st.subheader("💰 Funding Distribution")
        if 'funding_ratio' in df.columns:
            fig_dist = px.histogram(
                df[df['funding_ratio'] < 5],  # Filter outliers
                x='funding_ratio',
                nbins=50,
                labels={'funding_ratio': 'Funding Ratio'},
                color_discrete_sequence=['#10B981']
            )
            fig_dist.add_vline(x=funding_ratio, line_dash="dash", line_color="#EF4444",
                              annotation_text=f"Your Prediction: {funding_ratio:.2f}x")
            fig_dist.add_vline(x=1.0, line_dash="dot", line_color="#F59E0B",
                              annotation_text="Success Line")
            fig_dist.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Funding distribution data not available")
    
    col_chart3, col_chart4 = st.columns(2)
    
    # Chart 3: Duration vs Success Scatter
    with col_chart3:
        st.subheader("⏱️ Duration vs Success")
        if duration_col in df.columns and 'funding_ratio' in df.columns:
            sample_df = df.sample(n=min(500, len(df)), random_state=42)
            fig_scatter = px.scatter(
                sample_df,
                x=duration_col,
                y='funding_ratio',
                color='is_successful' if 'is_successful' in sample_df.columns else None,
                opacity=0.6,
                labels={duration_col: 'Campaign Duration (days)', 'funding_ratio': 'Funding Ratio'},
                color_discrete_map={0: '#EF4444', 1: '#22C55E'}
            )
            # Add user's position
            fig_scatter.add_trace(go.Scatter(
                x=[campaign_duration],
                y=[funding_ratio],
                mode='markers',
                marker=dict(size=15, color='#A855F7', symbol='star'),
                name='Your Campaign'
            ))
            fig_scatter.update_layout(
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Duration data not available")
    
    # Chart 4: Goal vs Backers Trend
    with col_chart4:
        st.subheader("👥 Goal vs Backers Correlation")
        if goal_col in df.columns and 'backers_count' in df.columns:
            sample_df = df.sample(n=min(500, len(df)), random_state=42)
            fig_backers = px.scatter(
                sample_df,
                x=goal_col,
                y='backers_count',
                color='funding_ratio',
                opacity=0.6,
                labels={goal_col: 'Funding Goal ($)', 'backers_count': 'Number of Backers'},
                color_continuous_scale='RdYlGn'
            )
            fig_backers.update_layout(
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_backers, use_container_width=True)
        else:
            st.info("Backers data not available")

if __name__ == "__main__":
    main()
