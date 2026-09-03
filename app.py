import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import plotly.express as px
from xgboost import XGBClassifier
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Online Shopper Purchase Intention Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS: Galactic Dark Theme, Neon Charts & Sidebar Buttons
# ==========================================
st.markdown("""
<style>
    /* Galactic sidebar background */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 20% 10%, rgba(124, 58, 237, 0.20), transparent 34%),
            radial-gradient(circle at 80% 90%, rgba(0, 229, 255, 0.10), transparent 32%),
            #070b17;
        border-right: 1px solid rgba(129, 140, 248, 0.16);
    }

    /* Galactic main app background */
    .stApp {
        background:
            radial-gradient(circle at 82% 8%, rgba(124, 58, 237, 0.20), transparent 30%),
            radial-gradient(circle at 8% 88%, rgba(0, 229, 255, 0.12), transparent 28%),
            #0a1020;
    }

    /* Subtle star field */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.30;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.75) 0 1px, transparent 1.5px),
            radial-gradient(circle, rgba(0,229,255,0.55) 0 1px, transparent 1.5px);
        background-size: 97px 113px, 157px 181px;
        background-position: 10px 20px, 70px 90px;
        z-index: 0;
    }

    /* General Text Colors */
    h1, h2, h3, p, span {
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    h1 {
        /* Keep native emoji colors visible in page titles */
        color: #e2e8f0 !important;
        background: none !important;
        -webkit-text-fill-color: currentColor !important;
    }

    /* Bubbly Main Button (Soft Purple/Blue gradient) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #7c73e6, #5ca1e6);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 28px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        font-weight: 600;
        font-size: 16px;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(124, 115, 230, 0.4);
    }

    /* Sidebar navigation buttons: no radio controls or circles */
    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        justify-content: flex-start;
        background: transparent !important;
        color: #cbd5e1 !important;
        border: 1px solid transparent !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        padding: 12px 15px !important;
        font-size: 17px !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] div.stButton > button p {
        font-size: 17px !important;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.07) !important;
        border-color: rgba(0, 229, 255, 0.28) !important;
        color: #ffffff !important;
        transform: translateX(3px);
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.12) !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, rgba(176, 38, 255, 0.24), rgba(0, 229, 255, 0.12)) !important;
        border-left: 4px solid #b026ff !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px rgba(176, 38, 255, 0.16) !important;
    }
    
    /* Smooth, Rounded Inputs */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div, 
    div[data-baseweb="number-input"] > div {
        border-radius: 18px !important;
        border: 1px solid #2d3748 !important;
        background-color: #1a2235 !important;
        color: white !important;
    }
    
    /* Tabs inside predictor */
    button[data-baseweb="tab"] {
        border-radius: 15px 15px 0px 0px !important;
        background-color: transparent;
    }
    
    /* Info/Success/Warning bubbles */
    div.stAlert {
        border-radius: 18px !important;
        background-color: rgba(26, 34, 53, 0.78);
        border: 1px solid rgba(129, 140, 248, 0.32);
        box-shadow: 0 0 22px rgba(124, 58, 237, 0.10);
    }

    /* Glass-style metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26, 34, 53, 0.82), rgba(17, 24, 39, 0.72));
        border: 1px solid rgba(129, 140, 248, 0.24);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.10);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED DATA & MODEL LOADING
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "xgb_model.json"
    features_path = BASE_DIR / "model_features.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    model = XGBClassifier()
    model.load_model(str(model_path))
    model_features = joblib.load(features_path)

    return model, model_features

@st.cache_data
def load_data():
    data_path = BASE_DIR / "online_shoppers_intention.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

def render_hover_pie(fig):
    """Render a pie chart with a smooth, animated hover slice."""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displaylogo": False, "responsive": True}
    )
    hover_script = """
    <script>
    (() => {
        const attachHoverEffect = () => {
            const plot = document.querySelector(".plotly-graph-div");
            if (
                !plot ||
                !window.Plotly ||
                typeof plot.on !== "function" ||
                plot.dataset.smoothHoverAttached
            ) {
                return Boolean(plot && plot.dataset.smoothHoverAttached);
            }

            plot.dataset.smoothHoverAttached = "true";
            let activeSlice = null;
            let activeTraceIndex = null;
            let activePointNumber = null;
            let hoverTimer = null;
            let unhoverTimer = null;
            let pendingHover = null;

            const animationOptions = {
                mode: "immediate",
                fromcurrent: true,
                transition: {
                    duration: 600,
                    easing: "cubic-in-out"
                },
                frame: {
                    duration: 600,
                    redraw: true
                }
            };

            const animatePull = (traceIndex, pointNumber, amount) => {
                const trace = plot.data[traceIndex];
                const sliceCount = (trace.labels || trace.values || []).length;
                if (!sliceCount) return;

                const pull = Array(sliceCount).fill(0);
                if (pointNumber !== null) pull[pointNumber] = amount;

                // Plotly's animation engine makes the slice movement smooth
                // instead of redrawing it abruptly with restyle().
                Plotly.animate(
                    plot,
                    {data: [{pull: pull}]},
                    animationOptions
                );
            };

            const cancelHoverTimers = () => {
                if (hoverTimer) {
                    window.clearTimeout(hoverTimer);
                    hoverTimer = null;
                }
                if (unhoverTimer) {
                    window.clearTimeout(unhoverTimer);
                    unhoverTimer = null;
                }
                pendingHover = null;
            };

            const clearSlice = () => {
                cancelHoverTimers();
                if (activeTraceIndex !== null) {
                    animatePull(activeTraceIndex, null, 0);
                }
                if (activeSlice) activeSlice.style.filter = "";
                activeSlice = null;
                activeTraceIndex = null;
                activePointNumber = null;
            };

            plot.on("plotly_hover", (event) => {
                const point = event.points && event.points[0];
                if (!point) return;
                if (unhoverTimer) {
                    window.clearTimeout(unhoverTimer);
                    unhoverTimer = null;
                }

                const hoveredElement = event.event && event.event.target;
                const sliceFromEvent = hoveredElement && hoveredElement.closest
                    ? hoveredElement.closest(".slice")
                    : null;
                const slices = plot.querySelectorAll(".trace.pie .slice");
                const slice = sliceFromEvent || slices[point.pointNumber];
                if (!slice) return;

                if (
                    activeTraceIndex === point.curveNumber &&
                    activePointNumber === point.pointNumber
                ) {
                    return;
                }

                const hoverKey = `${point.curveNumber}:${point.pointNumber}`;
                if (pendingHover && pendingHover.key === hoverKey) return;

                // A tiny debounce prevents Plotly from rapidly switching
                // between two slices when the pointer rests exactly on
                // their shared boundary.
                pendingHover = {
                    key: hoverKey,
                    slice: slice,
                    traceIndex: point.curveNumber,
                    pointNumber: point.pointNumber
                };
                if (hoverTimer) window.clearTimeout(hoverTimer);
                hoverTimer = window.setTimeout(() => {
                    if (!pendingHover || pendingHover.key !== hoverKey) return;

                    const nextHover = pendingHover;
                    pendingHover = null;
                    hoverTimer = null;
                    if (activeTraceIndex !== null) {
                        animatePull(activeTraceIndex, null, 0);
                    }
                    if (activeSlice) activeSlice.style.filter = "";

                    activeSlice = nextHover.slice;
                    activeTraceIndex = nextHover.traceIndex;
                    activePointNumber = nextHover.pointNumber;
                    activeSlice.style.filter =
                        "drop-shadow(0 0 7px rgba(124, 115, 230, 0.70)) " +
                        "drop-shadow(0 0 14px rgba(176, 38, 255, 0.35))";
                    animatePull(activeTraceIndex, activePointNumber, 0.10);
                }, 120);
            });

            plot.on("plotly_unhover", () => {
                if (unhoverTimer) window.clearTimeout(unhoverTimer);
                unhoverTimer = window.setTimeout(clearSlice, 120);
            });
            return true;
        };

        if (!attachHoverEffect()) {
            const observer = new MutationObserver(() => {
                if (attachHoverEffect()) observer.disconnect();
            });
            observer.observe(document.body, {childList: true, subtree: true});
            window.setTimeout(attachHoverEffect, 250);
        }
    })();
    </script>
    """
    components.html(plot_html + hover_script, height=450, scrolling=False)

try:
    model, model_features = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

df = load_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

# Custom Logo Block 
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 30px; padding: 5px;">
    <div style="background: linear-gradient(135deg, #b026ff, #00f0ff); padding: 10px 14px; border-radius: 12px; margin-right: 15px; font-weight: 800; color: white; font-size: 18px; box-shadow: 0 4px 15px rgba(176, 38, 255, 0.4);">
        ML
    </div>
    <h2 style="margin: 0; padding: 0; color: #e2e8f0; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">Shopper AI</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation buttons.
# Using buttons instead of st.radio removes the radio circles completely.
if "page" not in st.session_state:
    st.session_state.page = "📖 Project Overview"

def set_page(target_page):
    st.session_state.page = target_page

st.sidebar.button(
    "📖 Project Overview",
    key="nav_overview",
    type="primary" if st.session_state.page == "📖 Project Overview" else "secondary",
    use_container_width=True,
    on_click=set_page,
    args=("📖 Project Overview",),
)

st.sidebar.button(
    "📊 Data Exploration",
    key="nav_exploration",
    type="primary" if st.session_state.page == "📊 Data Exploration" else "secondary",
    use_container_width=True,
    on_click=set_page,
    args=("📊 Data Exploration",),
)

st.sidebar.button(
    "🔮 Predictor",
    key="nav_predictor",
    type="primary" if st.session_state.page == "🔮 Predictor" else "secondary",
    use_container_width=True,
    on_click=set_page,
    args=("🔮 Predictor",),
)

page = st.session_state.page

# ==========================================
# 4. PAGE ROUTING
# ==========================================

# ----------------- PAGE 1: OVERVIEW -----------------
if page == "📖 Project Overview":
    st.title("🛍️ Online Shopper Purchase Intention")
    st.markdown("---")
    
    st.write("""
    Welcome to the **Online Shoppers Purchasing Intention** dashboard! 
    
    This end-to-end machine learning web application evaluates real-time e-commerce session metrics to predict whether a website visitor will finalize a purchase. 
    
    **Key Objectives:**
    * **Analyze Behavior:** Understand how page duration, bounce rates, and page values impact conversions.
    * **Predict Intent:** Use a tuned XGBoost model to instantly predict transaction likelihood.
    
    *This project was developed as part of the Machine Learning program at the National Telecommunication Institute (NTI).*
    """)
    
    st.subheader("How to use this app:")
    st.markdown("""
    1. **Data Exploration Tab:** Explore the underlying dataset and view distributions of key features.
    2. **Predictor Tab:** Input live metrics for a browsing session through the categorized tabs and click predict to get real-time ML inference.
    """)

# ----------------- PAGE 2: DATA EXPLORATION -----------------
elif page == "📊 Data Exploration":
    st.title("📊 Dataset Exploration")
    st.markdown("---")
    
    if df is not None:
        # Interactive filters for the exploration view
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            available_months = df["Month"].dropna().astype(str).unique().tolist()
            ordered_months = [month for month in month_order if month in available_months]
            month_options = ["All"] + ordered_months
            selected_month = st.selectbox("Filter by Month", month_options)
        with filter_col2:
            visitor_options = ["All"] + sorted(df["VisitorType"].dropna().astype(str).unique().tolist())
            selected_visitor = st.selectbox("Filter by Visitor Type", visitor_options)

        filtered_df = df.copy()
        if selected_month != "All":
            filtered_df = filtered_df[filtered_df["Month"].astype(str) == selected_month]
        if selected_visitor != "All":
            filtered_df = filtered_df[filtered_df["VisitorType"].astype(str) == selected_visitor]

        # Live summary cards update with the selected filters
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Sessions Shown", f"{len(filtered_df):,}")
        with metric_col2:
            purchase_rate = filtered_df["Revenue"].mean() * 100 if not filtered_df.empty else None
            st.metric("Purchase Rate", f"{purchase_rate:.1f}%" if purchase_rate is not None else "—")
        with metric_col3:
            returning_rate = (
                (filtered_df["VisitorType"] == "Returning_Visitor").mean() * 100
                if not filtered_df.empty else None
            )
            st.metric("Returning Visitors", f"{returning_rate:.1f}%" if returning_rate is not None else "—")
        with metric_col4:
            average_page_value = filtered_df["PageValues"].mean() if not filtered_df.empty else None
            st.metric(
                "Avg. Page Value",
                f"{average_page_value:.2f}" if average_page_value is not None else "—"
            )
        
        if filtered_df.empty:
            st.warning("No sessions match the selected filters. Try choosing 'All'.")
        else:
            st.write(
                f"Showing {len(filtered_df):,} sessions from the historical e-commerce dataset:"
            )
            st.dataframe(filtered_df.head(50), use_container_width=True)
            
            st.markdown("---")
            
            # Create columns for charts
            col1, col2 = st.columns(2)
            
            # Neon purple and blue shades for the charts
            neon_palette = ['#b026ff', '#00f0ff', '#7000ff', '#0a58ff', '#d896ff']
            
            with col1:
                st.subheader("Target Distribution")
                target_counts = filtered_df['Revenue'].value_counts().rename(
                    index={True: "Purchased", False: "No Purchase"}
                ).reset_index()
                target_counts.columns = ['Status', 'Count']
                
                # Smooth Plotly Donut Chart
                fig1 = px.pie(target_counts, names='Status', values='Count', hole=0.55, 
                              color_discrete_sequence=['#b026ff', '#00f0ff'])
                fig1.update_layout(margin=dict(t=30, b=30, l=120, r=120), 
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#e2e8f0"),
                                   hovermode="closest",
                                   hoverlabel=dict(
                                       bgcolor="#111a33",
                                       bordercolor="#7c73e6",
                                       font=dict(color="#ffffff", size=14)
                                   ))
                fig1.update_traces(
                    hovertemplate="<b>%{label}</b><br>Sessions: %{value:,}<br>Share: %{percent}<extra></extra>",
                    marker=dict(line=dict(color="#0a1020", width=2))
                )
                render_hover_pie(fig1)
                
            with col2:
                st.subheader("Visitor Type Distribution")
                visitor_counts = filtered_df['VisitorType'].value_counts().reset_index()
                visitor_counts.columns = ['Visitor Type', 'Count']
                
                # Smooth Plotly Donut Chart
                fig2 = px.pie(visitor_counts, names='Visitor Type', values='Count', hole=0.55,
                              color_discrete_sequence=neon_palette)
                fig2.update_layout(margin=dict(t=30, b=30, l=120, r=120), 
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#e2e8f0"),
                                   hovermode="closest",
                                   hoverlabel=dict(
                                       bgcolor="#111a33",
                                       bordercolor="#b026ff",
                                       font=dict(color="#ffffff", size=14)
                                   ))
                fig2.update_traces(
                    hovertemplate="<b>%{label}</b><br>Sessions: %{value:,}<br>Share: %{percent}<extra></extra>",
                    marker=dict(line=dict(color="#0a1020", width=2))
                )
                render_hover_pie(fig2)
                
            st.markdown("---")
            st.subheader("Data Summary Statistics")
            st.dataframe(filtered_df.describe(), use_container_width=True)
        
    else:
        st.warning("Dataset not found. Please ensure `online_shoppers_intention.csv` is in the same folder.")

# ----------------- PAGE 3: PREDICTOR -----------------
elif page == "🔮 Predictor":
    st.title("🔮 Purchase Predictor")
    st.markdown("---")
    st.write("Predict whether a website visitor will complete a transaction (`Revenue = True`).")

    if "predictor_reset_counter" not in st.session_state:
        st.session_state.predictor_reset_counter = 0

    def reset_predictor_inputs():
        # Changing the widget-key namespace makes Streamlit rebuild every
        # input with its default value, including the visible widget values.
        st.session_state.predictor_reset_counter += 1

    widget_key_suffix = st.session_state.predictor_reset_counter

    reset_col, _ = st.columns([1, 4])
    with reset_col:
        st.button("↺ Reset Inputs", on_click=reset_predictor_inputs, use_container_width=True)
    
    # Form layout using sub-tabs for the predictor
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 Page Activity", 
        "📈 Page Metrics", 
        "💻 System Info", 
        "📅 Session Info"
    ])

    with sub_tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            admin = st.number_input("Administrative Page Views", min_value=0, max_value=50, value=2, key=f"admin_page_views_{widget_key_suffix}")
            admin_duration = st.number_input("Administrative Duration (s)", min_value=0.0, value=80.0, key=f"admin_duration_{widget_key_suffix}")
            info = st.number_input("Informational Page Views", min_value=0, max_value=50, value=0, key=f"info_page_views_{widget_key_suffix}")
            info_duration = st.number_input("Informational Duration (s)", min_value=0.0, value=0.0, key=f"info_duration_{widget_key_suffix}")
        with col2:
            product = st.number_input("Product Related Page Views", min_value=0, max_value=1000, value=30, key=f"product_page_views_{widget_key_suffix}")
            product_duration = st.number_input("Product Related Duration (s)", min_value=0.0, value=1100.0, key=f"product_duration_{widget_key_suffix}")

    with sub_tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            bounce_rate = st.slider("Bounce Rate", min_value=0.0, max_value=0.2, value=0.01, step=0.001, key=f"bounce_rate_{widget_key_suffix}")
            exit_rate = st.slider("Exit Rate", min_value=0.0, max_value=0.2, value=0.03, step=0.001, key=f"exit_rate_{widget_key_suffix}")
        with col2:
            page_values = st.number_input("Page Value", min_value=0.0, max_value=400.0, value=6.0, key=f"page_values_{widget_key_suffix}")
            special_day = st.selectbox("Special Day Closeness", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], index=0, key=f"special_day_{widget_key_suffix}")

    with sub_tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            operating_systems = st.selectbox("Operating System ID", options=list(range(1, 9)), index=1, key=f"operating_systems_{widget_key_suffix}")
            browser = st.selectbox("Browser ID", options=list(range(1, 14)), index=1, key=f"browser_{widget_key_suffix}")
        with col2:
            region = st.selectbox("Region ID", options=list(range(1, 10)), index=2, key=f"region_{widget_key_suffix}")
            traffic_type = st.selectbox("Traffic Type ID", options=list(range(1, 21)), index=1, key=f"traffic_type_{widget_key_suffix}")

    with sub_tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            month = st.selectbox("Month", options=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=2, key=f"month_{widget_key_suffix}")
            visitor_type = st.selectbox("Visitor Type", options=['Returning_Visitor', 'New_Visitor', 'Other'], index=0, key=f"visitor_type_{widget_key_suffix}")
        with col2:
            weekend = st.selectbox("Weekend Visit", options=[False, True], index=0, key=f"weekend_{widget_key_suffix}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Build the current session dataframe on every rerun for a live prediction.
    raw_data = {
        'Administrative': [admin],
        'Administrative_Duration': [admin_duration],
        'Informational': [info],
        'Informational_Duration': [info_duration],
        'ProductRelated': [product],
        'ProductRelated_Duration': [product_duration],
        'BounceRates': [bounce_rate],
        'ExitRates': [exit_rate],
        'PageValues': [page_values],
        'SpecialDay': [special_day],
        'Month': [month],
        'OperatingSystems': [operating_systems],
        'Browser': [browser],
        'Region': [region],
        'TrafficType': [traffic_type],
        'VisitorType': [visitor_type],
        'Weekend': [weekend]
    }
    
    input_df = pd.DataFrame(raw_data)

    # Perform One-Hot Encoding to match training preprocessing
    encoded_df = pd.get_dummies(input_df, columns=['Month', 'VisitorType'], drop_first=False)
    encoded_df['Weekend'] = encoded_df['Weekend'].astype(int)

    # Reindex columns to guarantee exact feature order expected by the model
    processed_df = encoded_df.reindex(columns=model_features, fill_value=0)

    # Make a live prediction
    prediction = model.predict(processed_df)[0]
    
    # Calculate probability if model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed_df)[0][1]
    else:
        proba = None

    # Output Results
    st.markdown("---")
    st.subheader("Prediction Result")
    if prediction == 1 or prediction == True:
        st.success("✅ **Result: Customer WILL likely complete a purchase (Revenue = True)**")
    else:
        st.warning("❌ **Result: Customer WILL NOT likely complete a purchase (Revenue = False)**")

    if proba is not None:
        # Make the model confidence easier to read at a glance
        st.progress(
            float(proba),
            text=f"Purchase likelihood: {proba * 100:.1f}%"
        )

        # A concise, user-friendly reading of the submitted session signals.
        session_signals = []
        if page_values > 0:
            session_signals.append("page value is present")
        if product >= 20:
            session_signals.append("strong product-page activity")
        if bounce_rate <= 0.05:
            session_signals.append("low bounce rate")
        if exit_rate <= 0.08:
            session_signals.append("moderate exit rate")
        if visitor_type == "Returning_Visitor":
            session_signals.append("returning visitor behavior")

        if session_signals:
            st.info("🔎 **Quick read:** " + ", ".join(session_signals).capitalize() + ".")
        else:
            st.info("🔎 **Quick read:** The estimate is based on the complete session profile above.")
