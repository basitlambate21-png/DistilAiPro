import importlib
import sys

# Force reload of all modules
for mod in ['simulation.mccabe_thiele', 'simulation.thermodynamics', 
            'simulation.distillation_model', 'optimization.surrogate_model',
            'optimization.optimizer', 'utils.pdf_generator']:
    if mod in sys.modules:
        del sys.modules[mod]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import joblib
import os
import sys
import io
import base64
import time
from datetime import datetime
from utils.pdf_generator import PDFReportGenerator, pdf_download_button

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.distillation_model import DistillationColumn
from simulation.mccabe_thiele import McCabeThiele
from simulation.thermodynamics import Thermodynamics, SYSTEMS, get_system_list
from optimization.surrogate_model import DistillationSurrogate, train_model
from optimization.optimizer import DistillationOptimizer
try:
    import torch
    import optuna
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.warning("⚠️ PyTorch/Optuna not available. Running in lightweight mode.")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="DistilAI Pro - Industrial Distillation Suite",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem; }
    .section-header { font-size: 1.4rem; font-weight: 700; color: #1e1e2e; margin: 1.5rem 0 1rem 0; padding-left: 1rem; border-left: 4px solid #667eea; }
    .card { background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; margin-bottom: 1rem; }
    .card-dark { background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); color: white; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
    .metric-box { text-align: center; padding: 1.2rem; border-radius: 12px; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-top: 4px solid; }
    .system-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-right: 8px; }
    .badge-ideal { background: #d1fae5; color: #065f46; }
    .badge-azeotrope { background: #fee2e2; color: #991b1b; }
    .badge-nonideal { background: #fef3c7; color: #92400e; }
    .badge-empirical { background: #dbeafe; color: #1e40af; }
    .badge-industrial { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
    .info-pill { background: #dbeafe; color: #1e40af; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 4px; }
    .warning-pill { background: #fef3c7; color: #92400e; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 4px; }
    .empirical-banner { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 0 12px 12px 0; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
defaults = {
    'trained_model': False, 'current_system': 'benzene-toluene',
    'simulation_results': None, 'optimization_results': None,
    'mccabe_data': None, 'uploaded_data': None, 'monitor_data': [],
    'monitoring': False, 'industrial_data': None,
    'use_industrial': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 1rem;">
        <h1 style="color: white; margin: 0; font-size: 1.8rem;">⚗️ DistilAI</h1>
        <p style="color: #a5b4fc; margin: 0.3rem 0 0 0; font-size: 0.85rem;">Industrial Distillation Suite</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation",
        ["🏠 Home", "📚 System Database", "🔬 Thermodynamics", "📐 McCabe-Thiele", 
         "🔧 Process Simulation", "🤖 AI Optimization", "📤 Industrial Data Upload", "📡 Live Monitor"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Active System")

    # Build system options with industrial data at top if available
    physical_systems = list(SYSTEMS.keys())
    system_labels = {}
    for name in physical_systems:
        info = SYSTEMS[name]
        badge = "AZEOTROPE" if "azeotrope" in info['description'].lower() else "IDEAL" if info['model'] == 'ideal' else "NON-IDEAL"
        system_labels[name] = f"{name.replace('-', ' ').title()} [{badge}]"

    industrial_key = 'industrial-custom'
    has_industrial = st.session_state.industrial_data is not None

    if has_industrial:
        df = st.session_state.industrial_data
        n_points = len(df)
        system_labels[industrial_key] = f"📊 Plant Data ({n_points} points) [EMPIRICAL]"
        all_options = [industrial_key] + physical_systems
    else:
        all_options = physical_systems

    # Determine current selection
    current = st.session_state.current_system
    if current not in all_options:
        current = all_options[0]

    selected_label = st.selectbox(
        "Select System",
        [system_labels[k] for k in all_options],
        index=all_options.index(current),
        label_visibility="collapsed"
    )

    # Map back to key
    selected_key = None
    for k, v in system_labels.items():
        if v == selected_label:
            selected_key = k
            break

    # Update state
    if selected_key == industrial_key:
        st.session_state.use_industrial = True
        st.session_state.current_system = industrial_key
        df = st.session_state.industrial_data
        st.markdown(f"""
        <div class="card-dark" style="margin-top: 0.5rem;">
            <span class="system-badge badge-industrial">📊 PLANT DATA</span>
            <p style="margin: 0.5rem 0 0 0; color: #a5b4fc; font-size: 0.8rem;">
                {len(df)} operating points loaded<br>
                KNN surrogate model active
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Data Ranges:**")
        st.markdown(f"<div class='info-pill'>R: {df['reflux_ratio'].min():.1f} – {df['reflux_ratio'].max():.1f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-pill'>F: {df['feed_rate'].min():.0f} – {df['feed_rate'].max():.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-pill'>zF: {df['feed_composition'].min():.3f} – {df['feed_composition'].max():.3f}</div>", unsafe_allow_html=True)
    else:
        st.session_state.use_industrial = False
        st.session_state.current_system = selected_key
        info = SYSTEMS[selected_key]
        badge_class = "badge-azeotrope" if "azeotrope" in info['description'].lower() else "badge-ideal" if info['model'] == 'ideal' else "badge-nonideal"
        st.markdown(f"""
        <div class="card-dark" style="margin-top: 0.5rem;">
            <span class="system-badge {badge_class}">{info['model'].upper()}</span>
            <p style="margin: 0.5rem 0 0 0; color: #a5b4fc; font-size: 0.8rem;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    if not has_industrial:
        st.markdown("""
        <div style="margin-top: 0.5rem; padding: 0.5rem; background: #f3f4f6; border-radius: 8px; font-size: 0.75rem; color: #6b7280;">
            💡 Upload plant data in <strong>"📤 Industrial Data Upload"</strong> to enable empirical mode
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.75rem;'><p>DistilAI Pro v3.0</p><p>Built for Industry</p></div>")

# ==================== HELPERS ====================
def get_active_column():
    if st.session_state.use_industrial and st.session_state.industrial_data is not None:
        return DistillationColumn(system='industrial-custom', industrial_data=st.session_state.industrial_data)
    return DistillationColumn(system=st.session_state.current_system)

def show_empirical_banner():
    if st.session_state.use_industrial:
        st.markdown("""
        <div class="empirical-banner">
            <strong>📊 Empirical Mode Active</strong><br>
            Using KNN surrogate trained from plant data. Physical properties are approximate.
        </div>
        """, unsafe_allow_html=True)

# ==================== HOME ====================
if page == "🏠 Home":
    st.markdown('<div class="main-title">⚗️ DistilAI Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Complete Industrial Distillation Analysis & Optimization Platform</div>', unsafe_allow_html=True)
    if st.session_state.industrial_data is not None:
        st.markdown(f"""
        <div class="empirical-banner" style="margin-bottom: 2rem;">
            <strong>✅ Plant Data Loaded</strong> — {len(st.session_state.industrial_data)} operating points available for empirical simulation and AI optimization
        </div>
        """, unsafe_allow_html=True)
    cols = st.columns(4)
    stats = [("8+", "Chemical Systems", "#667eea"), ("3", "Thermo Models", "#764ba2"), ("∞", "AI Optimizations", "#f093fb"), ("Real-time", "Process Monitor", "#10b981")]
    for col, (val, lbl, clr) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-box" style="border-color: {clr};">
                <h2 style="color: {clr}; margin: 0; font-size: 2rem;">{val}</h2>
                <p style="color: #6b7280; margin: 0.3rem 0 0 0; font-size: 0.85rem;">{lbl}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Platform Capabilities")
    feat_cols = st.columns(3)
    features = [
        ("📚 System Database", "8+ pre-loaded chemical systems + your uploaded plant data with real thermodynamic data", "667eea"),
        ("📐 McCabe-Thiele", "Interactive graphical stage design with empirical VLE from plant data or rigorous thermodynamics", "764ba2"),
        ("🤖 AI Optimization", "Neural network surrogate models with Bayesian optimization for minimum energy consumption", "f093fb"),
        ("🔬 Thermodynamics", "Real VLE calculations using Antoine equation, Wilson/NRTL models, and rigorous enthalpy balances", "10b981"),
        ("📤 Industrial Upload", "Upload your plant CSV data, build empirical models, and optimize your actual process", "f59e0b"),
        ("📡 Live Monitor", "Real-time trending with configurable alarm limits for purity, duty, and temperature", "ef4444")
    ]
    for i, (title, desc, color) in enumerate(features):
        with feat_cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="border-top: 3px solid #{color};">
                <h4 style="color: #{color}; margin-bottom: 0.5rem;">{title}</h4>
                <p style="color: #4b5563; font-size: 0.9rem; line-height: 1.5; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== SYSTEM DATABASE ====================
elif page == "📚 System Database":
    st.markdown('<div class="section-header">📚 Chemical System Database</div>', unsafe_allow_html=True)
    if st.session_state.industrial_data is not None:
        st.markdown("### 📊 Your Uploaded Plant Data")
        with st.expander("**📊 Plant Data (Custom)**  <span style='background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;'>EMPIRICAL</span>", expanded=True):
            cols = st.columns([2, 1])
            with cols[0]:
                df = st.session_state.industrial_data
                st.markdown(f"**{len(df)}** operating points loaded")
                st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")
                st.markdown("**Statistics:**")
                st.dataframe(df.describe().round(3), use_container_width=True)
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(10), use_container_width=True, height=250)
            with cols[1]:
                st.markdown("### 📈 Distributions")
                fig = make_subplots(rows=2, cols=2, subplot_titles=('Reflux Ratio', 'Feed Rate', 'Feed Composition', 'Reboiler Duty'))
                for idx, col in enumerate(['reflux_ratio', 'feed_rate', 'feed_composition', 'reboiler_duty']):
                    row, col_idx = idx // 2 + 1, idx % 2 + 1
                    fig.add_trace(go.Histogram(x=df[col], name=col, showlegend=False, marker_color='#667eea'), row=row, col=col_idx)
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                if 'top_purity' in df.columns and 'bottom_purity' in df.columns:
                    alphas = []
                    for _, row in df.iterrows():
                        xd, xb = row['top_purity'], row['bottom_purity']
                        if xd > xb and xb > 0.01 and xd < 0.99:
                            try:
                                alpha = (xd/(1-xd) / (xb/(1-xb))) ** (1/20)
                                if 1 < alpha < 50: alphas.append(alpha)
                            except: pass
                    if alphas: st.metric("Est. Relative Volatility", f"{np.median(alphas):.2f}")
        st.markdown("---")
    st.markdown("### Pre-loaded Physical Systems")
    for name, info in SYSTEMS.items():
        is_azeo = "azeotrope" in info['description'].lower()
        badge = "AZEOTROPE" if is_azeo else "IDEAL" if info['model'] == 'ideal' else "NON-IDEAL"
        badge_color = "#ef4444" if is_azeo else "#10b981" if info['model'] == 'ideal' else "#f59e0b"
        with st.expander(f"**{name.replace('-', ' ').title()}**  <span style='background:{badge_color};color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;'>{badge}</span>", expanded=False):
            cols = st.columns([2, 1])
            with cols[0]:
                components = info['components']()
                st.markdown("**Components:**")
                for comp in components:
                    st.markdown(f"<div class='info-pill'>{comp['name']} (MW: {comp['mw']} g/mol, BP: {comp.get('boiling_point', 'N/A')}°C)</div>", unsafe_allow_html=True)
                st.markdown(f"<br>**Model:** {info['model'].upper()}", unsafe_allow_html=True)
                st.markdown(f"**Description:** {info['description']}")
                st.markdown("**Antoine Constants:**")
                antoine_data = []
                for comp in components:
                    A, B, C = comp['antoine']
                    antoine_data.append({'Component': comp['name'], 'A': A, 'B': B, 'C': C, 'Cp_liquid': comp['cp_liquid'], 'Cp_vapor': comp['cp_vapor'], 'Hvap': comp['hvap']})
                st.dataframe(pd.DataFrame(antoine_data), use_container_width=True, hide_index=True)
            with cols[1]:
                thermo = Thermodynamics(components, model=info['model'])
                x_vals = np.linspace(0.01, 0.99, 50)
                y_vals = []
                for x in x_vals:
                    x_arr = np.array([x, 1-x])
                    _, y = thermo.bubble_point(x_arr, 101.325)
                    y_vals.append(y[0])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray'), name='y=x'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=badge_color, width=3), name='VLE'))
                fig.update_layout(title="VLE Preview", xaxis_title="x", yaxis_title="y", height=300, showlegend=False, template='plotly_white', margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

# ==================== THERMODYNAMICS ====================
elif page == "🔬 Thermodynamics":
    st.markdown('<div class="section-header">🔬 Thermodynamic Properties</div>', unsafe_allow_html=True)
    show_empirical_banner()
    if st.session_state.use_industrial:
        st.warning("⚠️ Thermodynamics page shows physical properties. Switch to a physical system for full VLE analysis, or use the empirical data overview below.")
        if st.session_state.industrial_data is not None:
            df = st.session_state.industrial_data
            st.markdown("### 📊 Plant Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Operating Range:**")
                st.dataframe(pd.DataFrame({
                    'Parameter': ['Reflux Ratio', 'Feed Rate (mol/hr)', 'Feed Composition', 'Top Purity', 'Bottom Purity'],
                    'Min': [df['reflux_ratio'].min(), df['feed_rate'].min(), df['feed_composition'].min(), 
                           df['top_purity'].min() if 'top_purity' in df.columns else 'N/A',
                           df['bottom_purity'].min() if 'bottom_purity' in df.columns else 'N/A'],
                    'Max': [df['reflux_ratio'].max(), df['feed_rate'].max(), df['feed_composition'].max(),
                           df['top_purity'].max() if 'top_purity' in df.columns else 'N/A',
                           df['bottom_purity'].max() if 'bottom_purity' in df.columns else 'N/A']
                }), use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Correlations:**")
                if 'top_purity' in df.columns and 'reflux_ratio' in df.columns:
                    fig = px.scatter(df, x='reflux_ratio', y='top_purity', trendline='lowess', title='Purity vs Reflux Ratio')
                    fig.update_layout(template='plotly_white', height=300)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        system = st.session_state.current_system
        components = SYSTEMS[system]['components']()
        model = SYSTEMS[system]['model']
        thermo = Thermodynamics(components, model=model)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### ⚙️ Conditions")
            T = st.slider("Temperature (°C)", 20, 200, 80)
            x1 = st.slider("Component 1 Mole Fraction", 0.0, 1.0, 0.5)
            x = np.array([x1, 1-x1])
            P_sat = [thermo.antoine(T, i) for i in range(thermo.n_comp)]
            T_bp, y = thermo.bubble_point(x, 101.325)
            T_dp, x_dew = thermo.dew_point(y, 101.325)
            H_l = thermo.liquid_enthalpy(T, x)
            H_v = thermo.vapor_enthalpy(T, y)
            st.markdown("### 📊 Results")
            st.markdown(f"""
            <div class="card">
                <p><strong>Bubble Point:</strong> {T_bp:.2f} °C</p>
                <p><strong>Dew Point:</strong> {T_dp:.2f} °C</p>
                <p><strong>Vapor Comp (y):</strong> {y[0]:.4f}</p>
                <p><strong>Liquid Enthalpy:</strong> {H_l:.2f} kJ/kmol</p>
                <p><strong>Vapor Enthalpy:</strong> {H_v:.2f} kJ/kmol</p>
                <p><strong>Heat of Vaporization:</strong> {H_v - H_l:.2f} kJ/kmol</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### 💨 Vapor Pressures")
            for i, comp in enumerate(components):
                st.markdown(f"<div class='info-pill'>{comp['name']}: {P_sat[i]:.2f} kPa</div>", unsafe_allow_html=True)
        with col2:
            x_range = np.linspace(0.01, 0.99, 100)
            T_bubble_vals, T_dew_vals, y_vals = [], [], []
            for xi in x_range:
                x_arr = np.array([xi, 1-xi])
                Tb, yv = thermo.bubble_point(x_arr, 101.325)
                Td, xd = thermo.dew_point(yv, 101.325)
                T_bubble_vals.append(Tb); T_dew_vals.append(Td); y_vals.append(yv[0])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=T_bubble_vals, mode='lines', name='Bubble Point', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=y_vals, y=T_dew_vals, mode='lines', name='Dew Point', line=dict(color='red', width=2)))
            fig.add_vline(x=x1, line_dash="dash", line_color="green", annotation_text=f"x={x1:.2f}")
            fig.update_layout(title=f"T-x-y Diagram: {system.replace('-', ' ').title()}", xaxis_title="Mole Fraction", yaxis_title="Temperature (°C)", template='plotly_white', height=500, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

# ==================== MCCABE-THIELE ====================
elif page == "📐 McCabe-Thiele":
    st.markdown('<div class="section-header">📐 McCabe-Thiele Graphical Method</div>', unsafe_allow_html=True)
    show_empirical_banner()
    import importlib
    import simulation.mccabe_thiele
    importlib.reload(simulation.mccabe_thiele)
    from simulation.mccabe_thiele import McCabeThiele
    system = st.session_state.current_system
    if st.session_state.use_industrial:
        thermo = None
        mt = McCabeThiele(thermo, empirical_data=st.session_state.industrial_data)
        st.info("📊 Using empirical VLE curve estimated from plant data")
    else:
        components = SYSTEMS[system]['components']()
        model = SYSTEMS[system]['model']
        thermo = Thermodynamics(components, model=model)
        mt = McCabeThiele(thermo)
    x_test = np.linspace(0.01, 0.99, 100)
    y_test = []
    for xt in x_test:
        if st.session_state.use_industrial:
            y_test.append(mt.empirical_alpha * xt / (1 + (mt.empirical_alpha - 1) * xt))
        else:
            _, yt = thermo.bubble_point(np.array([xt, 1-xt]), 101.325)
            y_test.append(yt[0])
    y_test = np.array(y_test)
    diff = y_test - x_test
    azeo_idx = np.where(np.diff(np.sign(diff)))[0]
    col_ctrl, col_graph = st.columns([1, 3])
    with col_ctrl:
        st.markdown("### 🎛️ Design Parameters")
        if len(azeo_idx) > 0 and not st.session_state.use_industrial:
            x_azeo = x_test[azeo_idx[0]]
            st.markdown(f"<div class='warning-pill' style='display: block; margin-bottom: 1rem;'>⚠️ Azeotrope at x = {x_azeo:.3f}</div>", unsafe_allow_html=True)
            max_xd = min(0.95, x_azeo - 0.02)
        else:
            max_xd = 0.999
        x_d = st.slider("xD (Distillate)", 0.5, max_xd, min(0.95, max_xd), 0.01, help="Desired top product purity")
        x_b = st.slider("xB (Bottoms)", 0.001, 0.5, 0.05, 0.01, help="Desired bottom product purity")
        z_f = st.slider("zF (Feed)", x_b + 0.05, x_d - 0.05, 0.5, 0.01, help="Feed composition")
        R = st.slider("R (Reflux Ratio)", 1.0, 15.0, 3.0, 0.1, help="Reflux ratio L/D")
        q = st.slider("q (Feed Condition)", 0.0, 1.5, 1.0, 0.1, help="1.0=saturated liquid, 0.0=saturated vapor")
        if st.button("📐 Step Off Stages", type="primary", use_container_width=True):
            with st.spinner("Calculating stages..."):
                mt = McCabeThiele(thermo if not st.session_state.use_industrial else None, 
                                 empirical_data=st.session_state.industrial_data if st.session_state.use_industrial else None)
                stages = mt.step_off_stages(R, x_d, x_b, z_f, q)
                R_min = mt.minimum_reflux(z_f, x_d, x_b, q)
                st.session_state.mccabe_data = {'mt': mt, 'stages': stages, 'R_min': R_min, 'params': {'R': R, 'x_d': x_d, 'x_b': x_b, 'z_f': z_f, 'q': q}}
                st.success(f"✅ Calculated: {stages['n_stages']} stages, Feed at stage {stages['feed_stage']}")
        if st.button("🔄 Clear & Recalculate", type="secondary", use_container_width=True):
            st.session_state.mccabe_data = None
            st.rerun()
        if st.session_state.mccabe_data:
            data = st.session_state.mccabe_data
            st.markdown("---")
            st.markdown("### 📊 Results")
            res_cols = st.columns(2)
            with res_cols[0]:
                st.metric("Total Stages", data['stages']['n_stages'])
                st.metric("Feed Stage", data['stages']['feed_stage'])
            with res_cols[1]:
                st.metric("R_min", f"{data['R_min']:.2f}")
                ratio = R / data['R_min'] if data['R_min'] > 0 else 999
                st.metric("R/R_min", f"{ratio:.2f}")
            if data['stages'].get('empirical_mode'):
                st.markdown("<div class='warning-pill'>Empirical VLE used</div>", unsafe_allow_html=True)
            st.markdown("### 📋 Stage Data")
            stage_df = []
            for i in range(0, len(data['stages']['stages_x']), 2):
                if i < len(data['stages']['stages_x']):
                    stage_df.append({'Stage': i//2 + 1, 'x (Liquid)': f"{data['stages']['stages_x'][i]:.4f}",
                        'y (Vapor)': f"{data['stages']['stages_y'][i]:.4f}" if i < len(data['stages']['stages_y']) else "-",
                        'Type': 'FEED' if (i//2 + 1) == data['stages']['feed_stage'] else 'Stage'})
            st.dataframe(pd.DataFrame(stage_df), use_container_width=True, height=250)
            st.markdown("---")
            if st.button("📄 Generate PDF Report", key="mccabe_pdf", type="secondary", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        from utils.pdf_generator import PDFReportGenerator
                        pdf_gen = PDFReportGenerator()
                        data = st.session_state.mccabe_data
                        buffer, filename = pdf_gen.generate_mccabe_report(system, data['params'], data['stages'], data['R_min'])
                        pdf_download_button(buffer, filename, "Download McCabe-Thiele Report")
                    except Exception as e:
                        st.error(f"PDF Error: {str(e)}")
    with col_graph:
        if st.session_state.mccabe_data:
            data = st.session_state.mccabe_data
            mt = data['mt']; stages = data['stages']; p = data['params']
            x_eq, y_eq, _ = mt.equilibrium_curve()
            op = mt.operating_lines(p['R'], p['x_d'], p['x_b'], p['z_f'], p['q'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray', width=1), name='y = x'))
            fig.add_trace(go.Scatter(x=x_eq, y=y_eq, mode='lines', line=dict(color='#2563eb', width=3), name='Equilibrium Curve'))
            x_rect = np.linspace(p['x_b'], op['intersection'][0], 100)
            y_rect = op['rectifying']['slope'] * x_rect + op['rectifying']['intercept']
            fig.add_trace(go.Scatter(x=x_rect, y=y_rect, mode='lines', line=dict(color='#059669', width=2), name='Rectifying Line'))
            x_strip = np.linspace(op['intersection'][0], p['x_d'], 100)
            y_strip = op['stripping']['slope'] * x_strip + op['stripping']['intercept']
            fig.add_trace(go.Scatter(x=x_strip, y=y_strip, mode='lines', line=dict(color='#dc2626', width=2), name='Stripping Line'))
            if p['q'] == 1:
                fig.add_vline(x=p['z_f'], line_dash="dash", line_color="#7c3aed", annotation_text="q-line (q=1)")
            else:
                x_q = np.linspace(0, 1, 100)
                y_q = op['q_line']['slope'] * x_q + op['q_line']['intercept']
                fig.add_trace(go.Scatter(x=x_q, y=y_q, mode='lines', line=dict(dash='dash', color='#7c3aed', width=2), name=f'q-line (q={p["q"]:.2f})'))
            stages_x = stages['stages_x']; stages_y = stages['stages_y']
            stair_x, stair_y = [], []
            for i in range(len(stages_x)):
                stair_x.append(stages_x[i]); stair_y.append(stages_y[i])
                if i < len(stages_x) - 1 and i % 2 == 0:
                    stair_x.append(stages_x[i+1]); stair_y.append(stages_y[i])
            fig.add_trace(go.Scatter(x=stair_x, y=stair_y, mode='lines+markers', line=dict(color='black', width=1.5), marker=dict(size=5, color='black'), name=f'Stages ({stages["n_stages"]})'))
            fig.add_trace(go.Scatter(x=[p['x_d']], y=[p['x_d']], mode='markers+text', marker=dict(size=14, color='#059669', symbol='diamond'), text=[f'xD={p["x_d"]:.3f}'], textposition='top right', textfont=dict(size=11, color='#059669'), name='xD'))
            fig.add_trace(go.Scatter(x=[p['x_b']], y=[p['x_b']], mode='markers+text', marker=dict(size=14, color='#dc2626', symbol='diamond'), text=[f'xB={p["x_b"]:.3f}'], textposition='bottom left', textfont=dict(size=11, color='#dc2626'), name='xB'))
            fig.add_trace(go.Scatter(x=[p['z_f']], y=[p['z_f']], mode='markers+text', marker=dict(size=14, color='#7c3aed', symbol='diamond'), text=[f'zF={p["z_f"]:.3f}'], textposition='bottom right', textfont=dict(size=11, color='#7c3aed'), name='zF'))
            feed_stage = stages['feed_stage']
            if feed_stage > 0 and feed_stage <= len(stages_x)//2:
                feed_idx = (feed_stage - 1) * 2
                if feed_idx < len(stages_x):
                    fig.add_trace(go.Scatter(x=[stages_x[feed_idx]], y=[stages_y[feed_idx]], mode='markers', marker=dict(size=14, color='red', symbol='square'), name=f'Feed Stage ({feed_stage})'))
            mode_text = "EMPIRICAL" if stages.get('empirical_mode') else "RIGOROUS"
            fig.update_layout(title=dict(text=f'McCabe-Thiele: {system.replace("-", " ").title()} [{mode_text}]<br><sup>R={p["R"]:.2f}, xD={p["x_d"]:.3f}, xB={p["x_b"]:.3f}, zF={p["z_f"]:.3f}, q={p["q"]:.2f} | Total Stages: {stages["n_stages"]}, Feed Stage: {stages["feed_stage"]}</sup>', font=dict(size=14)), xaxis_title='Liquid Mole Fraction, x', yaxis_title='Vapor Mole Fraction, y', template='plotly_white', height=700, xaxis=dict(range=[0, 1], dtick=0.1), yaxis=dict(range=[0, 1], dtick=0.1), legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<div style='text-align: center; padding: 5rem; color: #9ca3af;'><h2>📐 Set parameters and click \"Step Off Stages\"</h2><p>The complete McCabe-Thiele diagram will appear here</p></div>")

# ==================== PROCESS SIMULATION ====================
elif page == "🔧 Process Simulation":
    st.markdown('<div class="section-header">🔧 Process Simulation</div>', unsafe_allow_html=True)
    show_empirical_banner()
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("### ⚙️ Column Configuration")
        c1, c2 = st.columns(2)
        with c1:
            n_stages = st.number_input("Stages", min_value=5, max_value=100, value=20, step=1)
            feed_stage = st.number_input("Feed Stage", min_value=1, max_value=99, value=10, step=1)
        with c2:
            pressure = st.number_input("Pressure (kPa)", min_value=50.0, max_value=500.0, value=101.325, step=0.1)
        st.markdown("---")
        st.markdown("### 🎛️ Operating Conditions")
        system = st.session_state.current_system
        if st.session_state.use_industrial and st.session_state.industrial_data is not None:
            df = st.session_state.industrial_data
            rr_min, rr_max = float(df['reflux_ratio'].min()), float(df['reflux_ratio'].max())
            fr_min, fr_max = float(df['feed_rate'].min()), float(df['feed_rate'].max())
            fc_min, fc_max = float(df['feed_composition'].min()), float(df['feed_composition'].max())
            rr_default = (rr_min + rr_max) / 2
            fr_default = (fr_min + fr_max) / 2
            fc_default = (fc_min + fc_max) / 2
            st.markdown("<div class='info-pill'>Ranges from plant data</div>", unsafe_allow_html=True)
        else:
            rr_min, rr_max = 1.0, 10.0
            fr_min, fr_max = 20.0, 200.0
            fc_min, fc_max = 0.1, 0.9
            rr_default, fr_default, fc_default = 3.0, 100.0, 0.5
        rr = st.slider("Reflux Ratio", rr_min, rr_max, rr_default, 0.1)
        fr = st.slider("Feed Rate (mol/hr)", fr_min, fr_max, fr_default, 1.0)
        fc = st.slider("Feed Composition", fc_min, fc_max, fc_default, 0.01)
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                try:
                    column = DistillationColumn(n_stages=int(n_stages), feed_stage=int(feed_stage), pressure=float(pressure), system=system,
                        industrial_data=st.session_state.industrial_data if st.session_state.use_industrial else None)
                    result = column.simulate(rr, fr, fc)
                    st.session_state.simulation_results = result
                    st.session_state.simulation_inputs = {'rr': rr, 'fr': fr, 'fc': fc, 'n_stages': n_stages, 'feed_stage': feed_stage, 'pressure': pressure}
                    st.success("✅ Simulation complete!")
                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
    with col_out:
        if st.session_state.simulation_results is not None:
            r = st.session_state.simulation_results
            if r.get('empirical_mode') and r.get('empirical_warnings'):
                for warning in r['empirical_warnings']:
                    st.warning(f"⚠️ {warning}")
            if r.get('empirical_mode'):
                st.markdown("<div class='info-pill'>Empirical prediction from plant data</div>", unsafe_allow_html=True)
            st.markdown("### 📊 Results")
            mcols = st.columns(4)
            metrics = [("Top Purity", f"{r['top_purity']:.4f}", "#10b981"), ("Bottom Purity", f"{r['bottom_purity']:.4f}", "#f59e0b"), ("Reboiler Duty", f"{r['reboiler_duty']:.1f} kW", "#ef4444"), ("Condenser Duty", f"{r['condenser_duty']:.1f} kW", "#3b82f6")]
            for mc, (lbl, val, clr) in zip(mcols, metrics):
                with mc:
                    st.markdown(f"<div class='metric-box' style='border-color: {clr};'><h3 style='color: {clr}; margin: 0; font-size: 1.3rem;'>{val}</h3><p style='color: #6b7280; margin: 0.3rem 0 0 0; font-size: 0.8rem;'>{lbl}</p></div>", unsafe_allow_html=True)
            st.markdown("### 🔍 Additional Data")
            acols = st.columns(3)
            with acols[0]: st.metric("Distillate", f"{r['distillate_rate']:.1f} mol/hr")
            with acols[1]: st.metric("Bottoms", f"{r['bottoms_rate']:.1f} mol/hr")
            with acols[2]: st.metric("α (Rel. Vol.)", f"{r['relative_volatility']:.2f}")
            if not r.get('empirical_mode') and 'gilliland_efficiency' in r:
                with st.expander("📐 Design Details"):
                    st.markdown(f"**Gilliland Efficiency:** {r['gilliland_efficiency']:.3f}")
                    st.markdown(f"**Stage Efficiency:** {r['stage_efficiency']:.2f}")
                    st.markdown(f"**Actual Stages (eff. adjusted):** {r['n_stages'] * r['stage_efficiency']:.1f}")
            st.markdown("### 🌡️ Temperature Profile")
            t_df = pd.DataFrame({'Location': ['Top', 'Feed', 'Bottom'], 'Temperature (°C)': [r['top_temperature'], r['feed_temperature'], r['bottom_temperature']]})
            fig = px.bar(t_df, x='Location', y='Temperature (°C)', color='Temperature (°C)', color_continuous_scale='RdYlBu_r', text='Temperature (°C)')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.markdown("### 📄 Export Report")
            if st.button("📄 Generate PDF Report", key="sim_pdf", type="secondary", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        from utils.pdf_generator import PDFReportGenerator
                        pdf_gen = PDFReportGenerator()
                        inputs = st.session_state.get('simulation_inputs', {'n_stages': 20, 'feed_stage': 10, 'pressure': 101.325, 'rr': rr, 'fr': fr, 'fc': fc})
                        buffer, filename = pdf_gen.generate_simulation_report(system, inputs, r)
                        pdf_download_button(buffer, filename, "Download Simulation Report")
                    except Exception as e:
                        st.error(f"PDF Error: {str(e)}")
        else:
            st.info("Click 'Run Simulation' to see results")

# ==================== AI OPTIMIZATION ====================
elif page == "🤖 AI Optimization":
    st.markdown('<div class="section-header">🤖 AI-Powered Optimization</div>', unsafe_allow_html=True)
    show_empirical_banner()

    # ==================== KEY FIX: Check if industrial data is active ====================
    if st.session_state.use_industrial and st.session_state.industrial_data is not None:
        # EMPRICAL MODE: Use plant data directly for optimization
        st.markdown("""
        <div class="info-pill" style="display: block; margin-bottom: 1rem;">
            📊 Using plant data for optimization (no neural network needed)
        </div>
        """, unsafe_allow_html=True)

        system = st.session_state.current_system
        col_setup, col_res = st.columns([1, 2])

        with col_setup:
            st.markdown("### 🎯 Setup")
            target = st.slider("Target Purity", 0.80, 0.99, 0.95, 0.01)

            st.markdown("### ⚙️ Current Operation")
            df = st.session_state.industrial_data
            curr_rr = st.number_input("Current R", min_value=float(df['reflux_ratio'].min()), 
                                     max_value=float(df['reflux_ratio'].max()), 
                                     value=float(df['reflux_ratio'].median()), step=0.1)
            curr_fr = st.number_input("Current Feed", min_value=float(df['feed_rate'].min()),
                                     max_value=float(df['feed_rate'].max()),
                                     value=float(df['feed_rate'].median()), step=1.0)
            curr_fc = st.number_input("Current Comp.", min_value=float(df['feed_composition'].min()),
                                     max_value=float(df['feed_composition'].max()),
                                     value=float(df['feed_composition'].median()), step=0.01)

            try:
                col = get_active_column()
                curr_res = col.simulate(curr_rr, curr_fr, curr_fc)
                st.markdown(f"""
                <div class="card" style="background: #f3f4f6;">
                    <p style="margin: 0;"><strong>Current Energy:</strong> {curr_res['reboiler_duty']:.1f} kW</p>
                    <p style="margin: 0.3rem 0 0 0;"><strong>Current Purity:</strong> {curr_res['top_purity']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Current simulation error: {str(e)}")
                curr_res = None

            if st.button("🚀 Optimize Now", type="primary", use_container_width=True):
                with st.spinner("Searching plant data for optimal operation..."):
                    try:
                        df = st.session_state.industrial_data
                        # Filter for target purity
                        mask = df['top_purity'] >= target
                        if mask.sum() == 0:
                            st.warning(f"No plant data meets target purity {target}. Using best available.")
                            best_idx = df['top_purity'].idxmax()
                        else:
                            best_idx = df[mask]['reboiler_duty'].idxmin()

                        best_row = df.loc[best_idx]
                        best = {
                            'reflux_ratio': best_row['reflux_ratio'],
                            'feed_rate': best_row['feed_rate'],
                            'feed_composition': best_row['feed_composition']
                        }
                        duty = best_row['reboiler_duty']

                        opt_col = get_active_column()
                        opt_res = opt_col.simulate(best['reflux_ratio'], best['feed_rate'], best['feed_composition'])

                        st.session_state.optimization_results = {
                            'best': best, 'duty': duty,
                            'current': curr_res, 'optimal': opt_res,
                            'target': target,
                            'empirical_mode': True
                        }
                        st.success("✅ Optimization complete! (From plant data history)")
                    except Exception as e:
                        st.error(f"Optimization error: {str(e)}")

        with col_res:
            if st.session_state.optimization_results is not None:
                res = st.session_state.optimization_results
                savings = res['current']['reboiler_duty'] - res['duty']
                savings_pct = (savings / res['current']['reboiler_duty']) * 100 if res['current']['reboiler_duty'] > 0 else 0

                st.markdown("### 🎉 Optimization Results")
                st.markdown("<div class='info-pill'>Optimization from plant data history</div>", unsafe_allow_html=True)

                top3 = st.columns(3)
                with top3[0]: 
                    st.markdown(f"""
                    <div class="metric-box" style="border-color: #10b981;">
                        <h2 style="color: #10b981; margin: 0;">{savings:.1f}</h2>
                        <p style="color: #6b7280; margin: 0;">kW Saved</p>
                    </div>
                    """, unsafe_allow_html=True)
                with top3[1]: 
                    st.markdown(f"""
                    <div class="metric-box" style="border-color: #667eea;">
                        <h2 style="color: #667eea; margin: 0;">{savings_pct:.1f}%</h2>
                        <p style="color: #6b7280; margin: 0;">Energy Savings</p>
                    </div>
                    """, unsafe_allow_html=True)
                with top3[2]: 
                    st.markdown(f"""
                    <div class="metric-box" style="border-color: #f59e0b;">
                        <h2 style="color: #f59e0b; margin: 0;">{res['optimal']['top_purity']:.4f}</h2>
                        <p style="color: #6b7280; margin: 0;">Achieved Purity</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 📊 Before vs After")
                comp_df = pd.DataFrame({
                    'Parameter': ['Reflux Ratio', 'Feed Rate', 'Feed Comp.', 'Reboiler Duty (kW)', 'Top Purity'],
                    'Current': [f"{curr_rr:.2f}", f"{curr_fr:.1f}", f"{curr_fc:.2f}",
                               f"{res['current']['reboiler_duty']:.1f}", f"{res['current']['top_purity']:.4f}"],
                    'AI Optimized': [f"{res['best']['reflux_ratio']:.2f}", f"{res['best']['feed_rate']:.1f}",
                                    f"{res['best']['feed_composition']:.2f}", f"{res['duty']:.1f}",
                                    f"{res['optimal']['top_purity']:.4f}"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Current', 'AI Optimized'],
                    y=[res['current']['reboiler_duty'], res['duty']],
                    marker_color=['#ef4444', '#10b981'],
                    text=[f"{res['current']['reboiler_duty']:.1f}", f"{res['duty']:.1f}"],
                    textposition='outside'
                ))
                fig.update_layout(title="Energy Comparison", yaxis_title="kW", template='plotly_white', height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📄 Export Report")
                if st.button("📄 Generate PDF Report", key="opt_pdf", type="secondary", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        try:
                            from utils.pdf_generator import PDFReportGenerator
                            pdf_gen = PDFReportGenerator()
                            buffer, filename = pdf_gen.generate_optimization_report(
                                system, res['current'], res['optimal'], savings, savings_pct
                            )
                            pdf_download_button(buffer, filename, "Download Optimization Report")
                        except Exception as e:
                            st.error(f"PDF Error: {str(e)}")
            else:
                st.info("Click 'Optimize Now' to see results")

    else:
        # PHYSICAL MODE: Use neural network surrogate
        model_exists = os.path.exists('models/surrogate_model.pth')

        if not model_exists:
            st.markdown("""
            <div class="warning-pill" style="display: block; margin-bottom: 1rem;">
                ⚠️ No AI model found. Please train a model first using the "Industrial Data Upload" section, or use Quick Train below.
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Quick Train Model", type="primary"):
                with st.spinner("Generating data and training..."):
                    try:
                        os.makedirs('data', exist_ok=True)
                        os.makedirs('models', exist_ok=True)
                        column = DistillationColumn(system=st.session_state.current_system)
                        data = []
                        for _ in range(2000):
                            rr = np.random.uniform(1.0, 8.0)
                            fr = np.random.uniform(20, 200)
                            fc = np.random.uniform(0.1, 0.9)
                            res = column.simulate(rr, fr, fc)
                            data.append({
                                'reflux_ratio': rr, 'feed_rate': fr, 'feed_composition': fc,
                                'top_purity': res['top_purity'], 'bottom_purity': res['bottom_purity'],
                                'reboiler_duty': res['reboiler_duty'], 'condenser_duty': res['condenser_duty']
                            })
                        df = pd.DataFrame(data)
                        df.to_csv('data/quick_train.csv', index=False)
                        train_model('data/quick_train.csv', epochs=50)
                        st.session_state.trained_model = True
                        st.success("✅ Model trained! Refresh page to see optimization.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
        else:
            system = st.session_state.current_system
            col_setup, col_res = st.columns([1, 2])

            with col_setup:
                st.markdown("### 🎯 Setup")
                target = st.slider("Target Purity", 0.80, 0.99, 0.95, 0.01)
                trials = st.slider("Optimization Trials", 10, 300, 100, 10)

                st.markdown("### ⚙️ Current Operation")
                curr_rr = st.number_input("Current R", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
                curr_fr = st.number_input("Current Feed", min_value=20.0, max_value=200.0, value=100.0, step=1.0)
                curr_fc = st.number_input("Current Comp.", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

                try:
                    col = DistillationColumn(system=system)
                    curr_res = col.simulate(curr_rr, curr_fr, curr_fc)
                    st.markdown(f"""
                    <div class="card" style="background: #f3f4f6;">
                        <p style="margin: 0;"><strong>Current Energy:</strong> {curr_res['reboiler_duty']:.1f} kW</p>
                        <p style="margin: 0.3rem 0 0 0;"><strong>Current Purity:</strong> {curr_res['top_purity']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Current simulation error: {str(e)}")
                    curr_res = None

                if st.button("🚀 Optimize Now", type="primary", use_container_width=True):
                    with st.spinner("AI optimizing..."):
                        try:
                            opt = DistillationOptimizer()
                            best, duty = opt.optimize(target_purity=target, n_trials=trials)
                            opt_col = DistillationColumn(system=system)
                            opt_res = opt_col.simulate(best['reflux_ratio'], best['feed_rate'], best['feed_composition'])

                            savings = curr_res['reboiler_duty'] - duty
                            savings_pct = (savings / curr_res['reboiler_duty']) * 100 if curr_res['reboiler_duty'] > 0 else 0

                            st.session_state.optimization_results = {
                                'best': best, 'duty': duty,
                                'current': curr_res, 'optimal': opt_res,
                                'target': target,
                                'empirical_mode': False
                            }
                            st.success("✅ Optimization complete!")
                        except Exception as e:
                            st.error(f"Optimization error: {str(e)}")

            with col_res:
                if st.session_state.optimization_results is not None:
                    res = st.session_state.optimization_results
                    savings = res['current']['reboiler_duty'] - res['duty']
                    savings_pct = (savings / res['current']['reboiler_duty']) * 100 if res['current']['reboiler_duty'] > 0 else 0

                    st.markdown("### 🎉 Optimization Results")

                    top3 = st.columns(3)
                    with top3[0]: 
                        st.markdown(f"""
                        <div class="metric-box" style="border-color: #10b981;">
                            <h2 style="color: #10b981; margin: 0;">{savings:.1f}</h2>
                            <p style="color: #6b7280; margin: 0;">kW Saved</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with top3[1]: 
                        st.markdown(f"""
                        <div class="metric-box" style="border-color: #667eea;">
                            <h2 style="color: #667eea; margin: 0;">{savings_pct:.1f}%</h2>
                            <p style="color: #6b7280; margin: 0;">Energy Savings</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with top3[2]: 
                        st.markdown(f"""
                        <div class="metric-box" style="border-color: #f59e0b;">
                            <h2 style="color: #f59e0b; margin: 0;">{res['optimal']['top_purity']:.4f}</h2>
                            <p style="color: #6b7280; margin: 0;">Achieved Purity</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### 📊 Before vs After")
                    comp_df = pd.DataFrame({
                        'Parameter': ['Reflux Ratio', 'Feed Rate', 'Feed Comp.', 'Reboiler Duty (kW)', 'Top Purity'],
                        'Current': [f"{curr_rr:.2f}", f"{curr_fr:.1f}", f"{curr_fc:.2f}",
                                   f"{res['current']['reboiler_duty']:.1f}", f"{res['current']['top_purity']:.4f}"],
                        'AI Optimized': [f"{res['best']['reflux_ratio']:.2f}", f"{res['best']['feed_rate']:.1f}",
                                        f"{res['best']['feed_composition']:.2f}", f"{res['duty']:.1f}",
                                        f"{res['optimal']['top_purity']:.4f}"]
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Current', 'AI Optimized'],
                        y=[res['current']['reboiler_duty'], res['duty']],
                        marker_color=['#ef4444', '#10b981'],
                        text=[f"{res['current']['reboiler_duty']:.1f}", f"{res['duty']:.1f}"],
                        textposition='outside'
                    ))
                    fig.update_layout(title="Energy Comparison", yaxis_title="kW", template='plotly_white', height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 📄 Export Report")
                    if st.button("📄 Generate PDF Report", key="opt_pdf", type="secondary", use_container_width=True):
                        with st.spinner("Generating PDF..."):
                            try:
                                from utils.pdf_generator import PDFReportGenerator
                                pdf_gen = PDFReportGenerator()
                                buffer, filename = pdf_gen.generate_optimization_report(
                                    system, res['current'], res['optimal'], savings, savings_pct
                                )
                                pdf_download_button(buffer, filename, "Download Optimization Report")
                            except Exception as e:
                                st.error(f"PDF Error: {str(e)}")
                else:
                    st.info("Click 'Optimize Now' to see results")

# ==================== INDUSTRIAL DATA UPLOAD ====================
elif page == "📤 Industrial Data Upload":
    st.markdown('<div class="section-header">📤 Industrial Data Upload & AI Training</div>', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Plant Data")
    st.markdown("""
    <div style="border: 3px dashed #667eea; border-radius: 20px; padding: 3rem; text-align: center; background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);">
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">Drop your CSV file here</h3>
        <p style="color: #6b7280; margin: 0;">Or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=['csv'], label_visibility="collapsed")
    col_map, col_preview = st.columns([1, 2])
    with col_map:
        st.markdown("### 🗺️ Column Mapping")
        st.markdown("""
        <div style="background: #f9fafb; border-left: 4px solid #667eea; padding: 1rem; border-radius: 0 12px 12px 0; margin: 0.5rem 0;">
            <p style="margin: 0; font-size: 0.85rem;"><strong>Required columns:</strong></p>
            <code style="font-size: 0.75rem;">reflux_ratio, feed_rate, feed_composition,<br>top_purity, bottom_purity, reboiler_duty,<br>condenser_duty</code>
        </div>
        """, unsafe_allow_html=True)
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ {len(df):,} rows × {len(df.columns)} cols")
                expected = ['reflux_ratio', 'feed_rate', 'feed_composition', 'top_purity', 'bottom_purity', 'reboiler_duty', 'condenser_duty']
                mapping = {}
                for exp in expected:
                    mapping[exp] = st.selectbox(exp, df.columns, index=df.columns.tolist().index(exp) if exp in df.columns else 0, key=f"map_{exp}")
                df_mapped = df.rename(columns={v: k for k, v in mapping.items()})
                st.session_state.industrial_data = df_mapped
                if st.button("💾 Save & Use", type="primary", use_container_width=True):
                    os.makedirs('data', exist_ok=True)
                    df_mapped.to_csv('data/industrial_data.csv', index=False)
                    st.session_state.use_industrial = True
                    st.session_state.current_system = 'industrial-custom'
                    st.success("✅ Data saved! Now select '📊 Plant Data' from the sidebar dropdown.")
                    st.markdown("""
                    <div style="background: #d1fae5; color: #065f46; padding: 1rem; border-radius: 12px; margin-top: 0.5rem;">
                        <strong>✅ Ready!</strong> Your plant data is now available in the system selector.<br>
                        Go to any tab and select <strong>"📊 Plant Data"</strong> from the sidebar dropdown.
                    </div>
                    """)
                    st.balloons()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col_preview:
        if st.session_state.industrial_data is not None:
            st.markdown("### 👁️ Data Preview")
            st.dataframe(st.session_state.industrial_data.head(15), use_container_width=True, height=300)
            st.markdown("### 📊 Statistics")
            st.dataframe(st.session_state.industrial_data.describe(), use_container_width=True)
            st.markdown("### 📈 Distributions")
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Reflux Ratio', 'Feed Rate', 'Feed Composition', 'Reboiler Duty'))
            for idx, col in enumerate(['reflux_ratio', 'feed_rate', 'feed_composition', 'reboiler_duty']):
                row, col_idx = idx // 2 + 1, idx % 2 + 1
                fig.add_trace(go.Histogram(x=st.session_state.industrial_data[col], name=col, showlegend=False), row=row, col=col_idx)
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Train AI on Industrial Data</div>', unsafe_allow_html=True)
    if st.session_state.industrial_data is not None or uploaded is not None:
        train_cols = st.columns([1, 1, 1, 2])
        with train_cols[0]: epochs = st.number_input("Epochs", 10, 500, 100, 10)
        with train_cols[1]: hidden = st.selectbox("Hidden", ["[64,128]", "[128,256]", "[256,512]"], index=1)
        with train_cols[2]: lr = st.selectbox("LR", [0.001, 0.0001], index=0)
        with train_cols[3]:
            if st.button("🚀 Train AI Model", type="primary", use_container_width=True):
                with st.spinner("Training neural network..."):
                    progress = st.progress(0)
                    df_train = st.session_state.industrial_data if st.session_state.industrial_data is not None else pd.read_csv(uploaded)
                    df_train.to_csv('data/train_temp.csv', index=False)
                    for i in range(5):
                        time.sleep(0.3)
                        progress.progress((i+1)*20)
                    train_model('data/train_temp.csv', epochs=epochs)
                    st.session_state.trained_model = True
                    progress.progress(100)
                    st.success("✅ AI Model Trained Successfully!")
                    st.balloons()
                    st.markdown("<div style='background: #d1fae5; color: #065f46; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 4px;'>✅ Model saved to models/surrogate_model.pth<br>✅ Scalers saved to models/<br>✅ Ready for optimization in \"AI Optimization\" tab</div>", unsafe_allow_html=True)
    else:
        st.info("📁 Upload data first to enable training")

# ==================== LIVE MONITOR ====================
elif page == "📡 Live Monitor":
    st.markdown('<div class="section-header">📡 Real-Time Process Monitor</div>', unsafe_allow_html=True)
    show_empirical_banner()
    if 'monitor_data' not in st.session_state:
        st.session_state.monitor_data = []
        st.session_state.monitoring = False
    ctrl, disp = st.columns([1, 3])
    with ctrl:
        st.markdown("### 🎛️ Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                st.session_state.monitoring = True
        with c2:
            if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                st.session_state.monitoring = False
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.monitor_data = []
        st.markdown("---")
        st.markdown("### ⚠️ Alarm Limits")
        min_pur = st.slider("Min Purity", 0.8, 0.99, 0.90)
        max_duty = st.slider("Max Duty", 100, 500, 300)
        max_temp = st.slider("Max Temp", 50, 200, 150)
        if st.session_state.monitor_data:
            df_mon = pd.DataFrame(st.session_state.monitor_data)
            st.markdown("### 📊 Stats")
            st.metric("Points", len(df_mon))
            if len(df_mon) > 0:
                st.metric("Avg Purity", f"{df_mon['top_purity'].mean():.4f}")
                st.metric("Avg Duty", f"{df_mon['reboiler_duty'].mean():.1f} kW")
    with disp:
        if st.session_state.monitoring:
            import random
            t = len(st.session_state.monitor_data)
            base_rr = 3.0 + 0.5 * np.sin(t / 50)
            st.session_state.monitor_data.append({
                'time': t, 'reflux_ratio': base_rr + random.gauss(0, 0.1),
                'feed_rate': 100 + random.gauss(0, 5), 'top_purity': 0.95 + random.gauss(0, 0.005),
                'reboiler_duty': 150 + 10*(base_rr-3) + random.gauss(0, 3),
                'top_temp': 82 + 2*(base_rr-3) + random.gauss(0, 1)
            })
            if len(st.session_state.monitor_data) > 200:
                st.session_state.monitor_data.pop(0)
            time.sleep(1.0)
            st.rerun()
        if st.session_state.monitor_data:
            df_mon = pd.DataFrame(st.session_state.monitor_data)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Reflux Ratio', 'Top Purity', 'Reboiler Duty'), vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=df_mon['time'], y=df_mon['reflux_ratio'], mode='lines', line=dict(color='#667eea', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_mon['time'], y=df_mon['top_purity'], mode='lines', line=dict(color='#10b981', width=2)), row=2, col=1)
            fig.add_hline(y=min_pur, line_dash="dash", line_color="#ef4444", row=2, col=1)
            fig.add_trace(go.Scatter(x=df_mon['time'], y=df_mon['reboiler_duty'], mode='lines', line=dict(color='#f59e0b', width=2)), row=3, col=1)
            fig.add_hline(y=max_duty, line_dash="dash", line_color="#ef4444", row=3, col=1)
            fig.update_layout(height=700, template='plotly_white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            latest = st.session_state.monitor_data[-1]
            a1, a2, a3 = st.columns(3)
            with a1:
                if latest['top_purity'] < min_pur: st.error(f"🚨 PURITY: {latest['top_purity']:.4f}")
                else: st.success(f"✅ {latest['top_purity']:.4f}")
            with a2:
                if latest['reboiler_duty'] > max_duty: st.warning(f"⚠️ DUTY: {latest['reboiler_duty']:.1f}")
                else: st.info(f"ℹ️ {latest['reboiler_duty']:.1f} kW")
            with a3:
                if latest['top_temp'] > max_temp: st.warning(f"⚠️ TEMP: {latest['top_temp']:.1f}°C")
                else: st.info(f"ℹ️ {latest['top_temp']:.1f}°C")
        else:
            st.markdown("<div style='text-align: center; padding: 5rem; color: #9ca3af;'><h2>📡 Ready to Monitor</h2><p>Click \"Start\" to begin real-time data collection</p></div>")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<div style='text-align: center; padding: 1rem; color: #9ca3af; font-size: 0.85rem;'><p><strong>DistilAI Pro v3.0</strong> | Industrial Distillation Analysis & Optimization</p><p>Built with Streamlit, PyTorch, Plotly, Optuna | For Educational & Industrial Use</p></div>")
