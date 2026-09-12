import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# ==========================================
# PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Restaurant Tipping Behavior — Statistical Dashboard",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Restaurant Tipping Behavior — Applied Statistical Modeling")
st.caption("M.Sc. Data Science — Applied Statistical Modeling & Web Dashboard (Lab-4)")

@st.cache_data
def load_data():
    df = sns.load_dataset("tips")
    df['tip_pct'] = (df['tip'] / df['total_bill']) * 100
    return df

df = load_data()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Tab 1: Data Exploration", 
    "🧪 Tab 2: Hypothesis Testing Lab", 
    "📈 Tab 3: Live Prediction & Diagnostics"
])

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS
# ==========================================
with tab1:
    st.header("Exploratory Data Analysis & Descriptive Metrics")
    
    # Sidebar Filters
    st.sidebar.header("Filter Data (Tab 1)")
    bill_range = st.sidebar.slider(
        "Total Bill Range ($)", 
        float(df['total_bill'].min()), 
        float(df['total_bill'].max()), 
        (float(df['total_bill'].min()), float(df['total_bill'].max()))
    )
    selected_days = st.sidebar.multiselect(
        "Day of Week", 
        df['day'].unique(), 
        default=list(df['day'].unique())
    )
    selected_time = st.sidebar.multiselect(
        "Time of Day", 
        df['time'].unique(), 
        default=list(df['time'].unique())
    )

    # Apply filters
    filtered_df = df[
        (df['total_bill'] >= bill_range[0]) & 
        (df['total_bill'] <= bill_range[1]) &
        (df['day'].isin(selected_days)) &
        (df['time'].isin(selected_time))
    ]

    st.subheader("Summary Statistics")
    
    num_cols = ['total_bill', 'tip', 'size', 'tip_pct']
    stats_summary = []
    
    for col in num_cols:
        series = filtered_df[col]
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        stats_summary.append({
            "Feature": col,
            "Mean": series.mean(),
            "Median": series.median(),
            "Std Dev": series.std(),
            "IQR": q75 - q25,
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis()
        })
        
    stats_df = pd.DataFrame(stats_summary).set_index("Feature")
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Feature Distribution")
        selected_dist_col = st.selectbox("Select Numerical Feature", num_cols, index=1)
        fig_dist = px.histogram(
            filtered_df, 
            x=selected_dist_col, 
            color="smoker", 
            marginal="box", 
            title=f"Distribution of {selected_dist_col.title()}",
            template="plotly_white"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.subheader("Bivariate Correlation Matrix")
        corr_matrix = filtered_df[num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale="Blues", 
            title="Pearson Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 2: HYPOTHESIS TESTING LAB
# ==========================================
with tab2:
    st.header("Automated Hypothesis Testing Lab")
    
    st.subheader("Test 1: Compare 2 Independent Groups")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        group_cat = st.selectbox("Categorical Grouping Factor", ["smoker", "sex", "time"], index=0)
    with col2:
        num_metric = st.selectbox("Continuous Target Metric", ["tip", "total_bill", "tip_pct"], index=0)
    with col3:
        alpha = st.number_input("Significance Level (α)", value=0.05, step=0.01)

    unique_groups = df[group_cat].unique()
    if len(unique_groups) >= 2:
        g1_val, g2_val = unique_groups[0], unique_groups[1]
        g1_data = df[df[group_cat] == g1_val][num_metric]
        g2_data = df[df[group_cat] == g2_val][num_metric]

        st.markdown(f"**Comparing:** `{group_cat}` = **{g1_val}** vs. **{g2_val}** on **{num_metric}**")

        # Normality Check (Shapiro-Wilk)
        shapiro_g1 = stats.shapiro(g1_data)
        shapiro_g2 = stats.shapiro(g2_data)
        
        # Equal Variance Check (Levene's Test)
        levene_test = stats.levene(g1_data, g2_data)

        # Decision Logic
        is_normal = (shapiro_g1.pvalue > alpha) and (shapiro_g2.pvalue > alpha)
        
        if is_normal:
            equal_var = levene_test.pvalue > alpha
            test_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=equal_var)
            test_name = f"Two-Sample t-Test (equal_var={equal_var})"
        else:
            test_stat, p_val = stats.mannwhitneyu(g1_data, g2_data)
            test_name = "Mann-Whitney U Test (Non-Parametric)"

        m1, m2, m3 = st.columns(3)
        m1.metric("Shapiro Normality (p-val)", f"{shapiro_g1.pvalue:.4f} / {shapiro_g2.pvalue:.4f}")
        m2.metric("Levene Variance p-val", f"{levene_test.pvalue:.4f}")
        m3.metric(f"Final Test: {test_name}", f"p = {p_val:.4e}")

        if p_val < alpha:
            st.error(f"**Conclusion:** Reject Null Hypothesis ($H_0$). There is a statistically significant difference in {num_metric} between {g1_val} and {g2_val} at α = {alpha}.")
        else:
            st.warning(f"**Conclusion:** Fail to Reject Null Hypothesis ($H_0$). No statistically significant difference detected between groups at α = {alpha}.")

    st.divider()

    st.subheader("Test 2: Categorical Association or Multi-Group ANOVA")
    test_choice = st.radio("Select Test Type", ["Option A: Chi-Square Test of Independence", "Option B: One-Way ANOVA"])

    if test_choice == "Option A: Chi-Square Test of Independence":
        cat_col1 = st.selectbox("Categorical Feature 1", ["smoker", "day", "sex", "time"], index=0)
        cat_col2 = st.selectbox("Categorical Feature 2", ["day", "sex", "time", "smoker"], index=1)
        
        contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency_table)
        
        st.write("Contingency Table:")
        st.dataframe(contingency_table)
        st.info(f"**Chi-Square Statistic:** {chi2:.4f} | **Degrees of Freedom:** {dof} | **p-value:** {p_chi2:.4e}")
        
        if p_chi2 < alpha:
            st.error(f"**Conclusion:** Reject $H_0$. Significant relationship exists between {cat_col1} and {cat_col2}.")
        else:
            st.warning(f"**Conclusion:** Fail to Reject $H_0$. {cat_col1} and {cat_col2} appear to be independent.")

    else:
        group_col = st.selectbox("Multi-Group Categorical Variable (≥3 groups)", ["day", "size"], index=0)
        metric_col = st.selectbox("Metric to Compare Across Groups", ["tip", "total_bill", "tip_pct"], index=0)

        groups = [group[metric_col].values for _, group in df.groupby(group_col)]
        f_stat, p_anova = stats.f_oneway(*groups)

        st.info(f"**One-Way ANOVA F-Statistic:** {f_stat:.4f} | **p-value:** {p_anova:.4e}")
        if p_anova < alpha:
            st.error(f"**Conclusion:** Reject $H_0$. At least one group mean significantly differs from the others.")
        else:
            st.warning(f"**Conclusion:** Fail to Reject $H_0$. No significant differences found across group means.")

# ==========================================
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ==========================================
with tab3:
    st.header("OLS Statistical Model & Residual Diagnostics")

    # Fit statsmodels OLS Model predicting Tip Amount
    model_formula = "tip ~ total_bill + size + C(sex) + C(smoker) + C(day) + C(time)"
    model = smf.ols(formula=model_formula, data=df).fit()

    col_mod1, col_mod2 = st.columns([1, 1])

    with col_mod1:
        st.subheader("Model Parameter Interpretation")
        summary_df = pd.DataFrame({
            "Coefficient (β)": model.params,
            "Std Error": model.bse,
            "t-statistic": model.tvalues,
            "p-value": model.pvalues,
            "Conf. Int Lower": model.conf_int()[0],
            "Conf. Int Upper": model.conf_int()[1]
        })
        st.dataframe(summary_df.style.format("{:.3f}"), use_container_width=True)
        st.metric("Model R-squared", f"{model.rsquared:.4f}", delta=f"Adj R²: {model.rsquared_adj:.4f}")

    with col_mod2:
        st.subheader("Multicollinearity Diagnostics (VIF)")
        X_vif = df[['total_bill', 'size']].copy()
        X_vif['Intercept'] = 1.0
        vif_data = pd.DataFrame({
            "Variable": X_vif.columns,
            "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        })
        st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), use_container_width=True)
        st.caption("Note: VIF values < 5 indicate low, acceptable levels of multicollinearity.")

    st.divider()

    # Interactive Live Prediction Section
    st.subheader("Live Tip Amount Prediction")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        input_bill = st.slider("Total Bill ($)", 3.0, 60.0, 20.0)
    with c2:
        input_size = st.slider("Party Size", 1, 6, 2)
    with c3:
        input_sex = st.selectbox("Sex", ["Male", "Female"], index=0)
    with c4:
        input_smoker = st.selectbox("Smoker", ["Yes", "No"], index=1)
    with c5:
        input_day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"], index=2)
    with c6:
        input_time = st.selectbox("Time", ["Dinner", "Lunch"], index=0)

    input_data = pd.DataFrame({
        "total_bill": [input_bill],
        "size": [input_size],
        "sex": [input_sex],
        "smoker": [input_smoker],
        "day": [input_day],
        "time": [input_time]
    })

    prediction_results = model.get_prediction(input_data).summary_frame(alpha=0.05)
    pred_val = prediction_results['mean'].values[0]
    ci_lower = prediction_results['mean_ci_lower'].values[0]
    ci_upper = prediction_results['mean_ci_upper'].values[0]

    st.success(f"**Predicted Tip Amount:** `${pred_val:,.2f}`")
    st.info(f"**95% Confidence Interval:** `${ci_lower:,.2f}` to `${ci_upper:,.2f}`")

    st.divider()

    # Residual Diagnostics (Gauss-Markov Assumptions)
    st.subheader("Gauss-Markov Model Diagnostics")
    
    fitted_vals = model.fittedvalues
    residuals = model.resid

    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("**1. Linearity & Homoscedasticity Check**")
        fig_res = px.scatter(
            x=fitted_vals, 
            y=residuals, 
            labels={"x": "Fitted Values", "y": "Residuals"},
            title="Residuals vs. Fitted Values",
            opacity=0.6
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)

    with d_col2:
        st.markdown("**2. Residual Normality Check (Q-Q Plot)**")
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, 
            y=osr, 
            mode='markers', 
            name='Residuals',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        line_x = np.array([np.min(osm), np.max(osm)])
        line_y = intercept + slope * line_x
        
        fig_qq.add_trace(go.Scatter(
            x=line_x, 
            y=line_y, 
            mode='lines', 
            name='Theoretical Normal', 
            line=dict(color='red', dash='dash')
        ))
        
        fig_qq.update_layout(
            title="Normal Q-Q Plot", 
            xaxis_title="Theoretical Quantiles", 
            yaxis_title="Sample Quantiles",
            template="plotly_white"
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # Jarque-Bera Test & Residual Moments
    jb_res = stats.jarque_bera(residuals)
    jb_stat, jb_pval = jb_res.statistic, jb_res.pvalue
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)

    st.caption(f"**Jarque-Bera Test:** Statistic = {jb_stat:.2f}, p-value = {jb_pval:.4e} | **Residual Skewness:** {skew:.2f} | **Kurtosis:** {kurt:.2f}")