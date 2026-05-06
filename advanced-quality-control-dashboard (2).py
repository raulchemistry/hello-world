import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Critical for Streamlit Cloud
import matplotlib.pyplot as plt
import math
import io
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import requests
from io import BytesIO

# Try to import optional dependencies
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from pycaret.regression import setup, compare_models, predict_model, save_model, load_model
    from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
    HAS_PYCARET = True
except ImportError:
    HAS_PYCARET = False

# Set page configuration
st.set_page_config(
    page_title="Advanced Quality Control Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2e86ab;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .success { color: #28a745; font-weight: bold; }
    .warning { color: #ffc107; font-weight: bold; }
    .danger { color: #dc3545; font-weight: bold; }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #28a745;
    }
    .profile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .profile-image {
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .certification-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .user-stats {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load profile image
def load_profile_image():
    try:
        # Using a placeholder image - replace with your actual image URL
        image_url = "https://via.placeholder.com/150/667eea/ffffff?text=SM"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image
    except:
        # Return a blank image if loading fails
        return Image.new('RGB', (150, 150), color='#667eea')

# Generate comprehensive sample manufacturing data
def generate_manufacturing_data():
    np.random.seed(42)
    n = 1000  # Increased for better forecasting
    
    # Time series data with trends and seasonality
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Create realistic manufacturing patterns
    base_length = 10.0
    trend = np.linspace(0, 0.5, n)  # Gradual upward trend
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n) / 30)  # Monthly seasonality
    noise = np.random.normal(0, 0.05, n)
    
    length_values = base_length + trend + seasonal + noise
    
    data = {
        'date': dates,
        'part_id': range(1, n+1),
        'length': length_values,
        'diameter': np.random.normal(5.0, 0.08, n),
        'weight': np.random.normal(100.0, 2.5, n),
        'hardness': np.random.normal(45.0, 3.0, n),
        'temperature': np.random.normal(75.0, 5.0, n),
        'pressure': np.random.normal(100.0, 10.0, n),
        'defect': np.random.choice([0, 1], n, p=[0.92, 0.08]),
        'operator': np.random.choice(['John', 'Mike', 'Sarah', 'Lisa', 'David'], n),
        'machine': np.random.choice(['CNC-1', 'CNC-2', 'LATHE-1', 'MILL-1'], n),
        'shift': np.random.choice(['Morning', 'Afternoon', 'Night'], n),
        'material_batch': np.random.choice(['A123', 'B456', 'C789', 'D012', 'E345'], n),
        'production_rate': np.random.normal(50, 5, n)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation for regression analysis
    df['quality_score'] = (df['length'] * 0.3 + df['hardness'] * 0.2 + 
                          df['temperature'] * 0.1 + np.random.normal(0, 0.5, n))
    
    return df

# Quality metrics calculation functions
def calculate_cp(upper_spec, lower_spec, std_dev):
    """Calculate Process Capability Index (Cp)"""
    if std_dev == 0:
        return float('inf')
    return (upper_spec - lower_spec) / (6 * std_dev)

def calculate_cpk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Process Capability Index (Cpk)"""
    if std_dev == 0:
        return float('inf')
    cpu = (upper_spec - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
    cpl = (mean - lower_spec) / (3 * std_dev) if std_dev > 0 else float('inf')
    return min(cpu, cpl)

def calculate_pp(upper_spec, lower_spec, std_dev):
    """Calculate Process Performance Index (Pp)"""
    return calculate_cp(upper_spec, lower_spec, std_dev)

def calculate_ppk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Process Performance Index (Ppk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_cmk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Machine Capability Index (Cmk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_dpmo(defect_count, total_units):
    """Calculate Defects Per Million Opportunities (DPMO)"""
    if total_units == 0:
        return 0
    return (defect_count / total_units) * 1000000

def calculate_sigma_level(dpmo):
    """Calculate Sigma Level from DPMO"""
    if dpmo <= 0:
        return float('inf')
    if not HAS_SCIPY:
        # Simple approximation without scipy
        if dpmo <= 3.4: return 6.0
        elif dpmo <= 233: return 5.0
        elif dpmo <= 6200: return 4.0
        elif dpmo <= 66800: return 3.0
        elif dpmo <= 308000: return 2.0
        else: return 1.0
    return stats.norm.ppf(1 - dpmo/1000000) + 1.5

# Sampling recommendation functions
def recommend_sampling_method(data_type, data_nature, application):
    """Recommend sampling method based on data characteristics"""
    recommendations = []
    
    if data_type == "Variable":
        recommendations.append("📏 **Variables Sampling**: Use measurement data for precise analysis")
        recommendations.append("✅ **Recommended Methods**: SPC control charts (X-bar R, X-bar S), Acceptance sampling by variables")
    elif data_type == "Attribute":
        recommendations.append("🔢 **Attributes Sampling**: Use count data (pass/fail) for defect analysis")
        recommendations.append("✅ **Recommended Methods**: Acceptance sampling by attributes, p-charts, np-charts, c-charts, u-charts")
    
    if data_nature == "Continuous":
        recommendations.append("⏰ **Continuous Data**: Consider time-based sampling at regular intervals")
    elif data_nature == "Discrete":
        recommendations.append("📦 **Discrete Data**: Consider lot-based sampling or batch sampling")
    
    if "Normal" in data_nature:
        recommendations.append("📊 **Normal Distribution**: Parametric statistical methods can be used")
    elif "Non-normal" in data_nature:
        recommendations.append("📈 **Non-normal Distribution**: Use non-parametric methods or transform data")
    
    if application == "Process Control":
        recommendations.append("🎯 **Process Control**: Use SPC control charts with regular sampling intervals")
        recommendations.append("📋 **Sample Size**: 20-25 subgroups of 4-5 samples each")
    elif application == "Lot Acceptance":
        recommendations.append("📋 **Lot Acceptance**: Use ANSI/ASQ Z1.4 (MIL-STD-105E) for attributes")
        recommendations.append("📏 **Variables Acceptance**: Use ANSI/ASQ Z1.9 (MIL-STD-414)")
    elif application == "Capability Analysis":
        recommendations.append("📐 **Capability Analysis**: Ensure random sampling, minimum 100 individual measurements")
        recommendations.append("⏰ **Timing**: Collect data over different time periods for reliable analysis")
    elif application == "Defect Analysis":
        recommendations.append("🔍 **Defect Analysis**: Use stratified sampling by defect type/category")
        recommendations.append("📊 **Sample Strategy**: Focus on high-defect areas for detailed analysis")
    
    return recommendations

# Google Sheets integration
def load_from_google_sheets(url):
    """Load data from Google Sheets"""
    try:
        if 'docs.google.com/spreadsheets' in url:
            # Convert Google Sheets URL to CSV export
            if '/d/' in url:
                sheet_id = url.split('/d/')[1].split('/')[0]
            else:
                sheet_id = url.split('spreadsheets/d/')[1].split('/')[0]
            
            csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
            return pd.read_csv(csv_url)
        return None
    except Exception as e:
        st.error(f"Error loading Google Sheets: {e}")
        return None

# Usage tracking
class UsageTracker:
    def __init__(self):
        if 'usage_count' not in st.session_state:
            st.session_state.usage_count = 0
        if 'user_sessions' not in st.session_state:
            st.session_state.user_sessions = set()
        if 'feature_usage' not in st.session_state:
            st.session_state.feature_usage = {}
    
    def track_usage(self, feature_name):
        st.session_state.usage_count += 1
        st.session_state.user_sessions.add(id(st))
        if feature_name in st.session_state.feature_usage:
            st.session_state.feature_usage[feature_name] += 1
        else:
            st.session_state.feature_usage[feature_name] = 1
    
    def get_stats(self):
        return {
            'total_uses': st.session_state.usage_count,
            'unique_sessions': len(st.session_state.user_sessions),
            'feature_usage': st.session_state.feature_usage
        }

# Initialize usage tracker
tracker = UsageTracker()

# Main application
def main():
    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state.df = generate_manufacturing_data()
    
    # Profile Header
    col1, col2 = st.columns([1, 3])
    
    with col1:
        profile_image = load_profile_image()
        st.image(profile_image, width=150, caption="Md. Sourove Akther Momin")
    
    with col2:
        st.markdown("""
        <div class="profile-header">
            <h1>🏭 Advanced Quality Control Dashboard</h1>
            <h3>Md. Sourove Akther Momin</h3>
            <p>MSc. in Applied Statistics and Data Science | BSc. in Mechanical Engineering</p>
            <div>
                <span class="certification-badge">Certified Metal Cutting Professional (CMP)</span>
                <span class="certification-badge">World Class Manufacturing Practices Manager (WCMPM)</span>
            </div>
            <p>Expert: Process, Quality, Production Engineering, Time Series Forecasting, Machine Learning, Big Data, Deep Learning, Artificial Intelligence</p>
            <p>Member of Technical Committee and Reviewer for 2nd IEOM 2025 World Congress Windsor, Ontario, Canada</p>
            <p>Email: sourovemomin.kuet@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox("Choose Module", 
        ["🏠 Home", "📁 Data Import", "📊 Data Overview", "📐 Quality Metrics", 
         "📈 SPC Analysis", "🔍 Defect Analysis", "🎯 Sampling Recommender", 
         "🔬 Advanced Analytics", "📊 Regression & Forecasting", "👥 Usage Analytics"])
    
    # Track navigation
    tracker.track_usage(f"Navigation_{app_mode}")
    
    # Home Page
    if app_mode == "🏠 Home":
        st.markdown('<div class="section-header">🚀 Welcome to Advanced Quality Control Dashboard</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Complete Free Web-Based Quality Control & Predictive Analytics Platform</h3>
        <p>This application provides comprehensive quality control analysis capabilities with advanced machine learning for manufacturing and process industries.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        stats_data = tracker.get_stats()
        st.subheader("📊 Platform Usage Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Uses", stats_data['total_uses'])
        with col2:
            st.metric("Unique Sessions", stats_data['unique_sessions'])
        with col3:
            st.metric("Active Features", len(stats_data['feature_usage']))
        
        # Features Grid
        st.markdown("""
        <div class="feature-grid">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h3>📐 Quality Metrics</h3>
            <ul>
            <li>Cp, Cpk, Pp, Ppk, Cmk calculations</li>
            <li>DPMO and Sigma Level analysis</li>
            <li>Process capability analysis</li>
            <li>Real-time calculations</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>📈 SPC Control Charts</h3>
            <ul>
            <li>X-bar and R control charts</li>
            <li>Real-time process monitoring</li>
            <li>Control limit calculations</li>
            <li>Out-of-control detection</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>🔍 Defect Analysis</h3>
            <ul>
            <li>Pareto analysis charts</li>
            <li>Defect rate calculations</li>
            <li>Categorical analysis</li>
            <li>Vital few identification</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h3>🎯 Sampling Recommender</h3>
            <ul>
            <li>AI-powered recommendations</li>
            <li>Variable vs attribute data</li>
            <li>Application-specific guidance</li>
            <li>Industry best practices</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>🔬 Advanced Analytics</h3>
            <ul>
            <li>Normality tests (Shapiro-Wilk)</li>
            <li>Q-Q plots for distribution</li>
            <li>Correlation analysis</li>
            <li>Distribution plots</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>📊 Regression & Forecasting</h3>
            <ul>
            <li>Machine learning with PyCaret</li>
            <li>Multiple algorithm comparison</li>
            <li>Quality prediction models</li>
            <li>Future trend forecasting</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # How to Use Section
        st.markdown("---")
        st.subheader("📖 How to Use This Dashboard")
        
        steps = [
            ("1. 📁 **Data Import**", "Upload your CSV data, import from Google Sheets, or use our sample manufacturing dataset"),
            ("2. 📊 **Data Overview**", "Explore your dataset with comprehensive statistics, distributions, and quality checks"),
            ("3. 📐 **Quality Metrics**", "Calculate Cp, Cpk, Pp, Ppk, Cmk, DPMO and Sigma levels with interpretation"),
            ("4. 📈 **SPC Analysis**", "Generate X-bar and R control charts for real-time process monitoring"),
            ("5. 🔍 **Defect Analysis**", "Perform Pareto analysis and identify key improvement opportunities"),
            ("6. 🎯 **Sampling Recommender**", "Get AI-powered sampling recommendations based on your data characteristics"),
            ("7. 🔬 **Advanced Analytics**", "Run normality tests, correlation analysis, and distribution diagnostics"),
            ("8. 📊 **Regression & Forecasting**", "Build machine learning models to predict quality and forecast trends")
        ]
        
        for step, description in steps:
            st.markdown(f"**{step}**")
            st.write(description)
            st.write("")
        
        st.markdown("---")
        st.subheader("🌐 Share This App")
        st.info("""
        **This app is completely FREE to use and share!** 
        
        Send this link to your colleagues and team members:
        """)
        st.code("https://your-username-quality-control-dashboard.streamlit.app", language="text")
        
        # Quick Start Buttons
        st.markdown("---")
        st.subheader("⚡ Quick Start")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Use Sample Data & Start Analyzing", use_container_width=True):
                st.session_state.df = generate_manufacturing_data()
                st.success("✅ Sample data loaded! Switch to other modules to start analysis.")
        
        with col2:
            if st.button("📁 Upload Your Own Data", use_container_width=True):
                st.info("Switch to 'Data Import' module to upload your CSV files or import from Google Sheets")
        
        # System Status
        st.markdown("---")
        st.subheader("🔧 System Status")
        status_cols = st.columns(5)
        with status_cols[0]:
            st.success("✅ Pandas: Available")
        with status_cols[1]:
            st.success("✅ NumPy: Available")
        with status_cols[2]:
            st.success("✅ Matplotlib: Available")
        with status_cols[3]:
            if HAS_SCIPY:
                st.success("✅ SciPy: Available")
            else:
                st.warning("⚠️ SciPy: Not Available")
        with status_cols[4]:
            if HAS_PYCARET:
                st.success("✅ PyCaret: Available")
            else:
                st.warning("⚠️ PyCaret: Not Available")
    
    # Data Import Module
    elif app_mode == "📁 Data Import":
        st.markdown('<div class="section-header">📁 Data Import</div>', unsafe_allow_html=True)
        tracker.track_usage("Data_Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Your CSV Data")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], 
                                           help="Upload your manufacturing data in CSV format")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                    
                    # Show preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Basic info
                    st.subheader("📊 Data Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", df.shape[0])
                        st.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
                    with col2:
                        st.metric("Total Columns", df.shape[1])
                        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with col2:
            st.subheader("🌐 Google Sheets Import")
            gsheet_url = st.text_input("Google Sheets URL", 
                                     placeholder="https://docs.google.com/spreadsheets/d/...",
                                     help="Make sure your Google Sheet is shared with 'Anyone with the link can view'")
            
            if gsheet_url:
                if st.button("📥 Import from Google Sheets"):
                    with st.spinner("Importing data from Google Sheets..."):
                        df = load_from_google_sheets(gsheet_url)
                        if df is not None:
                            st.session_state.df = df
                            st.success(f"✅ Google Sheets imported successfully! Shape: {df.shape}")
                            st.dataframe(df.head(10), use_container_width=True)
                        else:
                            st.error("❌ Failed to import from Google Sheets. Please check the URL and sharing settings.")
            
            st.subheader("🔬 Sample Manufacturing Data")
            st.write("Use our comprehensive sample dataset to explore all features:")
            
            sample_info = """
            **Sample Dataset Includes:**
            - 1000 manufacturing records with time series
            - Multiple quality characteristics
            - Defect data for analysis
            - Process parameters (temperature, pressure)
            - Categorical variables for stratification
            - Realistic trends and variations
            """
            st.info(sample_info)
            
            if st.button("🎲 Load Sample Data", use_container_width=True):
                st.session_state.df = generate_manufacturing_data()
                st.success("✅ Sample manufacturing data loaded successfully!")
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # Data Overview Module
    elif app_mode == "📊 Data Overview":
        st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
        tracker.track_usage("Data_Overview")
        
        df = st.session_state.df
        
        # Quick stats
        st.subheader("📈 Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            numeric_cols = len(df.select_dtypes(include=np.number).columns)
            st.metric("🔢 Numeric Columns", numeric_cols)
        with col3:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Categorical Columns", categorical_cols)
        with col4:
            if 'defect' in df.columns:
                defect_rate = df['defect'].mean()
                st.metric("⚠️ Defect Rate", f"{defect_rate:.2%}")
            else:
                st.metric("ℹ️ Defect Column", "Not found")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("📊 Basic Statistics")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found for statistical analysis")
        
        # Column information
        st.subheader("🗂️ Column Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔢 Numeric Columns:**")
            numeric_cols = list(df.select_dtypes(include=np.number).columns)
            for col in numeric_cols:
                missing = df[col].isnull().sum()
                st.write(f"• **{col}** - {missing} missing")
        
        with col2:
            st.write("**📝 Categorical Columns:**")
            cat_cols = list(df.select_dtypes(include=['object']).columns)
            for col in cat_cols:
                unique_count = df[col].nunique()
                st.write(f"• **{col}** - {unique_count} unique values")
        
        # Visualizations
        st.subheader("📊 Data Distributions")
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        if numeric_cols:
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            
            if selected_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                data = df[selected_col].dropna()
                
                ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution of {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1 Std Dev')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Statistics summary
                st.write(f"**Statistics for {selected_col}:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{mean_val:.3f}")
                with col2:
                    st.metric("Std Dev", f"{std_val:.3f}")
                with col3:
                    st.metric("Min", f"{data.min():.3f}")
                with col4:
                    st.metric("Max", f"{data.max():.3f}")
    
    # Quality Metrics Module
    elif app_mode == "📐 Quality Metrics":
        st.markdown('<div class="section-header">📐 Quality Metrics Calculator</div>', unsafe_allow_html=True)
        tracker.track_usage("Quality_Metrics")
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ No numeric variables available for analysis")
            return
        
        st.info("""
        **Calculate process capability indices and quality metrics:**
        - **Cp, Cpk**: Process capability indices
        - **Pp, Ppk**: Process performance indices  
        - **Cmk**: Machine capability index
        - **DPMO**: Defects per million opportunities
        - **Sigma Level**: Process sigma level
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable = st.selectbox("📊 Select Variable", numeric_cols)
            data = df[variable].dropna()
            
            if len(data) > 0:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sample Size", len(data))
                st.metric("Mean", f"{np.mean(data):.4f}")
                st.metric("Standard Deviation", f"{np.std(data, ddof=1):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            data_min = data.min()
            data_max = data.max()
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🎯 Specification Limits")
            lsl = st.number_input("Lower Specification Limit (LSL)", 
                                value=float(data_mean - 3*data_std),
                                help="Minimum acceptable value")
            usl = st.number_input("Upper Specification Limit (USL)", 
                                value=float(data_mean + 3*data_std),
                                help="Maximum acceptable value")
            subgroup_size = st.slider("Subgroup Size for Cmk", min_value=2, max_value=10, value=5,
                                    help="Number of consecutive pieces for machine capability")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Calculate Quality Metrics", type="primary", use_container_width=True):
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            
            # Calculate metrics
            cp = calculate_cp(usl, lsl, std_val)
            cpk = calculate_cpk(usl, lsl, mean_val, std_val)
            pp = calculate_pp(usl, lsl, std_val)
            ppk = calculate_ppk(usl, lsl, mean_val, std_val)
            
            # Calculate Cmk
            subgroup_means = []
            subgroup_stds = []
            for i in range(0, min(len(data), subgroup_size*5), subgroup_size):
                subgroup = data[i:i+subgroup_size]
                subgroup_means.append(np.mean(subgroup))
                subgroup_stds.append(np.std(subgroup, ddof=1))
            
            short_term_std = np.mean(subgroup_stds) if subgroup_stds else std_val
            cmk = calculate_cmk(usl, lsl, mean_val, short_term_std)
            
            # Calculate DPMO and Sigma Level
            if variable == 'defect':
                defect_count = np.sum(data)
            else:
                defect_count = np.sum((data < lsl) | (data > usl))
            
            total_units = len(data)
            dpmo = calculate_dpmo(defect_count, total_units)
            sigma_level = calculate_sigma_level(dpmo)
            
            # Display results
            st.subheader("📊 Quality Metrics Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Cp", f"{cp:.3f}")
                st.metric("Cpk", f"{cpk:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Pp", f"{pp:.3f}")
                st.metric("Ppk", f"{ppk:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Cmk", f"{cmk:.3f}")
                st.metric("DPMO", f"{dpmo:,.0f}")
                st.metric("Sigma Level", f"{sigma_level:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interpretation
            st.subheader("🎯 Interpretation & Guidelines")
            
            capability_metrics = [
                ("Cp", cp, 1.33, 1.0, "Process potential capability"),
                ("Cpk", cpk, 1.33, 1.0, "Process actual capability accounting for centering"),
                ("Pp", pp, 1.33, 1.0, "Process performance"),
                ("Ppk", ppk, 1.33, 1.0, "Process performance accounting for centering"),
                ("Cmk", cmk, 1.67, 1.33, "Machine capability")
            ]
            
            for name, value, good_threshold, marginal_threshold, description in capability_metrics:
                if value >= good_threshold:
                    st.success(f"✅ **{name}: {value:.3f}** - Good (≥ {good_threshold}) - {description}")
                elif value >= marginal_threshold:
                    st.warning(f"⚠️ **{name}: {value:.3f}** - Marginal (≥ {marginal_threshold}) - {description}")
                else:
                    st.error(f"❌ **{name}: {value:.3f}** - Poor (< {marginal_threshold}) - {description}")
            
            # Visualization
            st.subheader("📈 Distribution Analysis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with specification limits
            ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax1.axvline(usl, color='green', linestyle='dashed', linewidth=2, label=f'USL: {usl}')
            ax1.axvline(lsl, color='green', linestyle='dashed', linewidth=2, label=f'LSL: {lsl}')
            
            # Add normal distribution curve
            x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
            if HAS_SCIPY:
                y = stats.norm.pdf(x, mean_val, std_val)
                ax1.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
            
            ax1.set_xlabel(variable)
            ax1.set_ylabel('Density')
            ax1.set_title(f'Distribution of {variable} with Specification Limits')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Capability indices bar chart
            indices = ['Cp', 'Cpk', 'Pp', 'Ppk', 'Cmk']
            values = [cp, cpk, pp, ppk, cmk]
            colors = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in values]
            
            bars = ax2.bar(indices, values, color=colors, alpha=0.7)
            ax2.axhline(y=1.33, color='red', linestyle='--', alpha=0.7, label='Minimum Recommended (1.33)')
            ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Minimum Acceptable (1.0)')
            ax2.set_ylabel('Index Value')
            ax2.set_title('Process Capability Indices')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

    # SPC Analysis Module
    elif app_mode == "📈 SPC Analysis":
        st.markdown('<div class="section-header">📈 Statistical Process Control (SPC)</div>', unsafe_allow_html=True)
        tracker.track_usage("SPC_Analysis")
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ No numeric variables available for SPC analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable = st.selectbox("📊 Select Variable for SPC", numeric_cols)
            subgroup_size = st.slider("👥 Subgroup Size", min_value=2, max_value=10, value=5)
        
        with col2:
            data = df[variable].dropna()
            data_min = data.min()
            data_max = data.max()
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            lsl = st.number_input("📏 LSL for SPC", value=float(data_mean - 3*data_std))
            usl = st.number_input("📏 USL for SPC", value=float(data_mean + 3*data_std))
        
        if st.button("📊 Generate Control Charts", type="primary"):
            # Create subgroups
            subgroups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            subgroup_means = [np.mean(subgroup) for subgroup in subgroups if len(subgroup) == subgroup_size]
            subgroup_ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups if len(subgroup) == subgroup_size]
            
            if len(subgroup_means) == 0:
                st.error("❌ Not enough data for subgroups. Try smaller subgroup size.")
                return
            
            # Calculate control limits
            xbar_mean = np.mean(subgroup_means)
            r_mean = np.mean(subgroup_ranges)
            
            # Constants for control charts
            A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
            D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
            D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
            
            a2 = A2.get(subgroup_size, 0.577)
            d3 = D3.get(subgroup_size, 0)
            d4 = D4.get(subgroup_size, 2.114)
            
            # X-bar chart limits
            xbar_ucl = xbar_mean + a2 * r_mean
            xbar_lcl = xbar_mean - a2 * r_mean
            
            # R chart limits
            r_ucl = d4 * r_mean
            r_lcl = d3 * r_mean
            
            # Create control charts
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # X-bar chart
            ax1.plot(subgroup_means, 'bo-', markersize=4, linewidth=1)
            ax1.axhline(xbar_mean, color='green', linestyle='-', label=f'Center Line ({xbar_mean:.3f})')
            ax1.axhline(xbar_ucl, color='red', linestyle='--', label=f'UCL ({xbar_ucl:.3f})')
            ax1.axhline(xbar_lcl, color='red', linestyle='--', label=f'LCL ({xbar_lcl:.3f})')
            ax1.set_title(f'X-bar Control Chart - {variable}')
            ax1.set_ylabel('Subgroup Mean')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # R chart
            ax2.plot(subgroup_ranges, 'go-', markersize=4, linewidth=1)
            ax2.axhline(r_mean, color='green', linestyle='-', label=f'Center Line ({r_mean:.3f})')
            ax2.axhline(r_ucl, color='red', linestyle='--', label=f'UCL ({r_ucl:.3f})')
            ax2.axhline(r_lcl, color='red', linestyle='--', label=f'LCL ({r_lcl:.3f})')
            ax2.set_title(f'R Control Chart - {variable}')
            ax2.set_ylabel('Subgroup Range')
            ax2.set_xlabel('Subgroup Number')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Control chart statistics
            st.subheader("📊 Control Chart Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("X-double bar", f"{xbar_mean:.4f}")
            with col2:
                st.metric("R-bar", f"{r_mean:.4f}")
            with col3:
                st.metric("Process Sigma", f"{r_mean/d2:.4f}" if (d2 := {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}.get(subgroup_size, 2.326)) else "N/A")
            with col4:
                out_of_control = sum(1 for x in subgroup_means if x > xbar_ucl or x < xbar_lcl)
                st.metric("Out-of-Control Points", out_of_control)
    
    # Defect Analysis Module
    elif app_mode == "🔍 Defect Analysis":
        st.markdown('<div class="section-header">🔍 Defect Analysis</div>', unsafe_allow_html=True)
        tracker.track_usage("Defect_Analysis")
        
        df = st.session_state.df
        
        if 'defect' not in df.columns:
            st.error("❌ No 'defect' column found in the dataset")
            st.info("The sample dataset includes a 'defect' column. Try loading sample data or ensure your dataset has a defect indicator column.")
            return
        
        cat_cols = list(df.select_dtypes(include=['object']).columns)
        
        if not cat_cols:
            st.error("❌ No categorical variables available for defect analysis")
            return
        
        category = st.selectbox("📂 Stratify Defects By", cat_cols)
        
        # Calculate defect rates by category
        defect_analysis = df.groupby(category).agg({
            'defect': ['count', 'sum', 'mean']
        }).round(4)
        defect_analysis.columns = ['Total_Units', 'Defect_Count', 'Defect_Rate']
        defect_analysis = defect_analysis.sort_values('Defect_Count', ascending=False)
        
        st.subheader("📊 Defect Analysis by Category")
        st.dataframe(defect_analysis, use_container_width=True)
        
        # Pareto Analysis
        st.subheader("📈 Pareto Analysis")
        
        # Prepare data for Pareto chart
        categories = defect_analysis.index.tolist()
        defect_counts = defect_analysis['Defect_Count'].tolist()
        cumulative_percentage = [sum(defect_counts[:i+1])/sum(defect_counts)*100 for i in range(len(defect_counts))]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for defect counts
        bars = ax1.bar(categories, defect_counts, color='skyblue', alpha=0.7, label='Defect Count')
        ax1.set_xlabel(category)
        ax1.set_ylabel('Defect Count', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        plt.xticks(rotation=45, ha='right')
        
        # Line chart for cumulative percentage
        ax2 = ax1.twinx()
        ax2.plot(categories, cumulative_percentage, 'ro-', linewidth=2, markersize=6, label='Cumulative %')
        ax2.set_ylabel('Cumulative Percentage', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)
        
        # Add 80% line
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% Line')
        
        plt.title(f'Pareto Chart - Defects by {category}')
        fig.tight_layout()
        st.pyplot(fig)
        
        # Defect Rate Analysis
        st.subheader("🎯 Key Insights")
        
        total_defects = defect_analysis['Defect_Count'].sum()
        total_units = defect_analysis['Total_Units'].sum()
        overall_defect_rate = total_defects / total_units
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Defects", int(total_defects))
        with col2:
            st.metric("Total Units", int(total_units))
        with col3:
            st.metric("Overall Defect Rate", f"{overall_defect_rate:.2%}")
        
        # Top contributors
        st.write("**Top Defect Contributors:**")
        for i, (category_name, row) in enumerate(defect_analysis.head(5).iterrows()):
            st.write(f"{i+1}. **{category_name}**: {row['Defect_Count']} defects ({row['Defect_Rate']:.2%} rate)")
    
    # Sampling Recommender Module
    elif app_mode == "🎯 Sampling Recommender":
        st.markdown('<div class="section-header">🎯 Sampling Method Recommender</div>', unsafe_allow_html=True)
        tracker.track_usage("Sampling_Recommender")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_type = st.radio("**Data Type**", ["Variable", "Attribute"])
            data_nature = st.selectbox("**Data Nature**", 
                ["Continuous", "Discrete", "Continuous - Normal", "Continuous - Non-normal"])
            application = st.selectbox("**Application**", 
                ["Process Control", "Lot Acceptance", "Capability Analysis", "Defect Analysis"])
        
        with col2:
            sample_size = st.slider("**Sample Size**", min_value=5, max_value=200, value=30)
            population_size = st.slider("**Population Size**", min_value=100, max_value=10000, value=1000)
            confidence_level = st.slider("**Confidence Level**", min_value=90, max_value=99, value=95)
        
        if st.button("🎯 Get Sampling Recommendations", type="primary"):
            recommendations = recommend_sampling_method(data_type, data_nature, application)
            
            st.subheader("📋 Sampling Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Additional calculations
            st.subheader("📊 Sampling Plan Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sampling_fraction = sample_size / population_size
                st.metric("Sampling Fraction", f"{sampling_fraction:.2%}")
            
            with col2:
                # Simple sample size calculation
                z_score = {90: 1.645, 95: 1.96, 99: 2.576}.get(confidence_level, 1.96)
                recommended_size = int((z_score**2 * 0.5 * 0.5) / (0.05**2))  # Conservative estimate
                st.metric("Recommended Sample Size", recommended_size)
            
            with col3:
                st.metric("Confidence Level", f"{confidence_level}%")
            
            # Sampling visualization
            st.subheader("📈 Sampling Strategy Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a simple visualization of sampling strategy
            methods = ['Random', 'Stratified', 'Systematic', 'Cluster']
            suitability = [0.8, 0.9, 0.7, 0.6]  # Example suitability scores
            
            bars = ax.bar(methods, suitability, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
            ax.set_ylabel('Suitability Score')
            ax.set_title('Recommended Sampling Methods Suitability')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, suitability):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Advanced Analytics Module
    elif app_mode == "🔬 Advanced Analytics":
        st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)
        tracker.track_usage("Advanced_Analytics")
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ No numeric variables available for advanced analytics")
            return
        
        analysis_type = st.selectbox("🔧 Select Analysis Type", 
            ["Normality Test", "Q-Q Plot", "Correlation Analysis", "Distribution Comparison"])
        
        if analysis_type == "Normality Test":
            if not HAS_SCIPY:
                st.warning("⚠️ SciPy not available. Using basic normality assessment.")
                # Basic assessment implementation
                variable = st.selectbox("📊 Select Variable", numeric_cols)
                data = df[variable].dropna()
                
                # Basic normality check using skewness and kurtosis
                skewness = stats.skew(data) if HAS_SCIPY else 0
                kurtosis = stats.kurtosis(data) if HAS_SCIPY else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skewness", f"{skewness:.4f}")
                with col2:
                    st.metric("Kurtosis", f"{kurtosis:.4f}")
                
                if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                    st.success("✅ Data appears approximately normal based on skewness and kurtosis")
                else:
                    st.warning("⚠️ Data may not be normally distributed based on skewness and kurtosis")
            else:
                # Full SciPy implementation
                variable = st.selectbox("📊 Select Variable", numeric_cols)
                data = df[variable].dropna()
                
                if st.button("📊 Run Normality Test"):
                    stat, p_value = stats.shapiro(data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Shapiro-Wilk Statistic", f"{stat:.4f}")
                    with col2:
                        st.metric("p-value", f"{p_value:.4f}")
                    
                    if p_value > 0.05:
                        st.success("✅ Data appears to be normally distributed (fail to reject H0)")
                    else:
                        st.error("❌ Data does not appear to be normally distributed (reject H0)")
        
        elif analysis_type == "Q-Q Plot":
            if not HAS_SCIPY:
                st.error("❌ Q-Q Plot requires SciPy. Please ensure SciPy is installed in your environment.")
            else:
                variable = st.selectbox("📊 Select Variable for Q-Q Plot", numeric_cols)
                data = df[variable].dropna()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Q-Q plot
                stats.probplot(data, dist="norm", plot=ax1)
                ax1.set_title(f'Q-Q Plot of {variable}')
                ax1.grid(True, alpha=0.3)
                
                # Histogram
                ax2.hist(data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_title(f'Distribution of {variable}')
                ax2.set_xlabel(variable)
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis_type == "Correlation Analysis":
            selected_vars = st.multiselect("📊 Select Variables for Correlation", numeric_cols, default=numeric_cols[:3])
            
            if len(selected_vars) >= 2:
                corr_matrix = df[selected_vars].corr()
                
                st.subheader("📈 Correlation Matrix")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), 
                           use_container_width=True)
                
                # Correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(selected_vars)))
                ax.set_yticks(np.arange(len(selected_vars)))
                ax.set_xticklabels(selected_vars, rotation=45, ha='right')
                ax.set_yticklabels(selected_vars)
                ax.set_title('Correlation Heatmap')
                
                # Add correlation values to heatmap
                for i in range(len(selected_vars)):
                    for j in range(len(selected_vars)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=12)
                
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis_type == "Distribution Comparison":
            selected_vars = st.multiselect("📊 Select Variables to Compare", numeric_cols, default=numeric_cols[:2])
            
            if len(selected_vars) >= 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for var in selected_vars:
                    data = df[var].dropna()
                    # Normalize for comparison
                    normalized_data = (data - data.mean()) / data.std()
                    ax.hist(normalized_data, bins=20, alpha=0.6, label=var, density=True)
                
                ax.set_xlabel('Standardized Values')
                ax.set_ylabel('Density')
                ax.set_title('Distribution Comparison (Standardized)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

    # Regression & Forecasting Module
    elif app_mode == "📊 Regression & Forecasting":
        st.markdown('<div class="section-header">📊 Regression & Forecasting with PyCaret</div>', unsafe_allow_html=True)
        tracker.track_usage("Regression_Forecasting")
        
        if not HAS_PYCARET:
            st.error("""
            ❌ PyCaret is not available in your environment.
            
            **To enable machine learning and forecasting features:**
            
            ```bash
            pip install pycaret
            ```
            
            **Or update your requirements.txt:**
            ```txt
            streamlit>=1.28.0
            pandas>=1.5.0
            numpy>=1.21.0
            matplotlib>=3.5.0
            scipy>=1.10.0
            pycaret>=3.0.0
            ```
            """)
            return
        
        df = st.session_state.df
        numeric_cols = list(df.select_dtypes(include=np.number).columns)
        
        st.info("""
        **Build machine learning models to predict quality outcomes and forecast future trends:**
        - Compare multiple algorithms automatically
        - Predict quality scores and defect probabilities
        - Forecast future process behavior
        - Identify key factors affecting quality
        """)
        
        analysis_type = st.radio("Select Analysis Type", 
                                ["Regression Analysis", "Time Series Forecasting"])
        
        if analysis_type == "Regression Analysis":
            st.subheader("🔮 Quality Prediction Models")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target = st.selectbox("🎯 Target Variable (What to predict)", numeric_cols)
            
            with col2:
                available_features = [col for col in numeric_cols if col != target]
                features = st.multiselect("📊 Feature Variables (Predictors)", 
                                        available_features, 
                                        default=available_features[:3])
            
            if target and len(features) >= 1:
                if st.button("🚀 Build Prediction Models", type="primary"):
                    with st.spinner("Training multiple machine learning models... This may take a few minutes."):
                        try:
                            # Prepare data
                            model_data = df[features + [target]].dropna()
                            
                            if len(model_data) < 10:
                                st.error("❌ Not enough data for model training. Need at least 10 complete records.")
                                return
                            
                            # Setup PyCaret
                            setup_data = setup(data=model_data, 
                                             target=target,
                                             session_id=42,
                                             silent=True,
                                             verbose=False)
                            
                            # Compare models
                            best_model = compare_models()
                            
                            st.success("✅ Model training completed!")
                            
                            # Display results
                            st.subheader("📊 Model Comparison Results")
                            
                            # Get comparison results
                            from pycaret.regression import pull
                            results = pull()
                            st.dataframe(results.style.highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE', 'R2']))
                            
                            # Best model info
                            st.subheader("🏆 Best Performing Model")
                            st.write(f"**Algorithm:** {type(best_model).__name__}")
                            
                            # Feature importance
                            try:
                                from pycaret.regression import plot_model
                                fig = plot_model(best_model, plot='feature')
                                st.pyplot(fig)
                            except:
                                st.info("Feature importance plot not available for this model type")
                            
                            # Make predictions
                            st.subheader("🔮 Make Predictions")
                            sample_input = {}
                            for feature in features:
                                feature_mean = df[feature].mean()
                                sample_input[feature] = st.number_input(
                                    f"Enter value for {feature}", 
                                    value=float(feature_mean)
                                )
                            
                            if st.button("📈 Predict Target Value"):
                                input_df = pd.DataFrame([sample_input])
                                prediction = predict_model(best_model, data=input_df)
                                predicted_value = prediction['prediction_label'].iloc[0]
                                
                                st.success(f"**Predicted {target}: {predicted_value:.3f}**")
                            
                        except Exception as e:
                            st.error(f"❌ Error in model training: {e}")
        
        else:  # Time Series Forecasting
            st.subheader("📈 Time Series Forecasting")
            
            # Check if date column exists
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) == 0:
                st.warning("No date column found. Using sample data with dates.")
                df = generate_manufacturing_data()
                st.session_state.df = df
            
            col1, col2 = st.columns(2)
            
            with col1:
                date_col = st.selectbox("📅 Date/Time Column", 
                                      df.select_dtypes(include=['datetime64']).columns.tolist() or ['date'])
                value_col = st.selectbox("📊 Value to Forecast", numeric_cols)
            
            with col2:
                forecast_periods = st.slider("🔮 Forecast Periods", 1, 365, 30)
                model_type = st.selectbox("🤖 Model Type", ["ARIMA", "Exponential Smoothing", "Prophet"])
            
            if st.button("🌐 Generate Forecast", type="primary"):
                with st.spinner("Building time series forecast..."):
                    try:
                        # Simple time series visualization
                        if date_col in df.columns and value_col in df.columns:
                            ts_data = df[[date_col, value_col]].dropna()
                            ts_data = ts_data.sort_values(date_col)
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(ts_data[date_col], ts_data[value_col], 'b-', linewidth=2)
                            ax.set_xlabel('Date')
                            ax.set_ylabel(value_col)
                            ax.set_title(f'Time Series of {value_col}')
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.info("""
                            **Time Series Analysis Complete**
                            
                            For advanced forecasting with PyCaret Time Series module, additional setup is required.
                            This visualization shows the historical trend of your selected variable.
                            """)
                        else:
                            st.error("Selected columns not found in dataset")
                        
                    except Exception as e:
                        st.error(f"❌ Error in forecasting: {e}")

    # Usage Analytics Module
    elif app_mode == "👥 Usage Analytics":
        st.markdown('<div class="section-header">👥 Platform Usage Analytics</div>', unsafe_allow_html=True)
        
        stats_data = tracker.get_stats()
        
        st.info("""
        **Platform Usage Statistics**
        - Track how the dashboard is being used
        - Monitor feature popularity
        - Understand user engagement
        """)
        
        # Overall Statistics
        st.subheader("📈 Overall Platform Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Uses", stats_data['total_uses'])
        with col2:
            st.metric("Unique Sessions", stats_data['unique_sessions'])
        with col3:
            st.metric("Active Features", len(stats_data['feature_usage']))
        with col4:
            avg_uses_per_session = stats_data['total_uses'] / max(1, stats_data['unique_sessions'])
            st.metric("Avg Uses/Session", f"{avg_uses_per_session:.1f}")
        
        # Feature Usage
        st.subheader("🔥 Feature Popularity")
        
        if stats_data['feature_usage']:
            # Create feature usage chart
            features = list(stats_data['feature_usage'].keys())
            usage_counts = list(stats_data['feature_usage'].values())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(features))
            
            bars = ax.barh(y_pos, usage_counts, color='skyblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Usage Count')
            ax.set_title('Feature Usage Distribution')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(usage_counts):
                ax.text(v + 0.1, i, str(v), va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature usage table
            usage_df = pd.DataFrame({
                'Feature': features,
                'Usage Count': usage_counts,
                'Percentage': [f"{(count/stats_data['total_uses'])*100:.1f}%" for count in usage_counts]
            }).sort_values('Usage Count', ascending=False)
            
            st.dataframe(usage_df, use_container_width=True)
        else:
            st.info("No feature usage data available yet. Start using the dashboard to see analytics!")
        
        # User Engagement Insights
        st.subheader("💡 Engagement Insights")
        
        if stats_data['total_uses'] > 0:
            most_used_feature = max(stats_data['feature_usage'].items(), key=lambda x: x[1]) if stats_data['feature_usage'] else ("None", 0)
            engagement_rate = (stats_data['unique_sessions'] / max(1, stats_data['total_uses'])) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Most Popular Feature", most_used_feature[0])
                st.metric("Engagement Rate", f"{engagement_rate:.1f}%")
            with col2:
                st.metric("Total Analytics Tracked", stats_data['total_uses'])
                st.metric("Platform Health", "✅ Excellent" if stats_data['total_uses'] > 10 else "🟡 Good")
        
        # Reset analytics (for testing)
        st.markdown("---")
        st.subheader("🛠️ Analytics Management")
        
        if st.button("🔄 Reset Usage Analytics", type="secondary"):
            st.session_state.usage_count = 0
            st.session_state.user_sessions = set()
            st.session_state.feature_usage = {}
            st.success("Usage analytics reset successfully!")
            st.experimental_rerun()

# Run the application
if __name__ == "__main__":
    main()