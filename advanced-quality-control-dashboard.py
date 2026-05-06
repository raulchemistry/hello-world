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

# ===================== LANGUAGE TRANSLATIONS =====================
TRANSLATIONS = {
    "en": {
        "page_title": "Advanced Quality Control Dashboard",
        "language": "Language",
        "select_language": "Select Language",
        "welcome": "🚀 Welcome to Advanced Quality Control Dashboard",
        "complete_platform": "🎯 Complete Free Web-Based Quality Control & Predictive Analytics Platform",
        "platform_description": "This application provides comprehensive quality control analysis capabilities with advanced machine learning for manufacturing and process industries.",
        "navigation": "🔧 Navigation",
        "choose_module": "Choose Module",
        "home": "🏠 Home",
        "data_import": "📁 Data Import",
        "data_overview": "📊 Data Overview",
        "quality_metrics": "📐 Quality Metrics",
        "spc_analysis": "📈 SPC Analysis",
        "defect_analysis": "🔍 Defect Analysis",
        "sampling_recommender": "🎯 Sampling Recommender",
        "advanced_analytics": "🔬 Advanced Analytics",
        "regression_forecasting": "📊 Regression & Forecasting",
        "usage_analytics": "👥 Usage Analytics",
        "platform_usage_statistics": "📊 Platform Usage Statistics",
        "total_uses": "Total Uses",
        "unique_sessions": "Unique Sessions",
        "active_features": "Active Features",
        "quality_metrics_card": "📐 Quality Metrics",
        "cp_cpk_calculations": "Cp, Cpk, Pp, Ppk, Cmk calculations",
        "dpmo_sigma": "DPMO and Sigma Level analysis",
        "process_capability": "Process capability analysis",
        "real_time_calculations": "Real-time calculations",
        "spc_control_charts": "📈 SPC Control Charts",
        "xbar_r_charts": "X-bar and R control charts",
        "real_time_monitoring": "Real-time process monitoring",
        "control_limit_calculations": "Control limit calculations",
        "out_of_control_detection": "Out-of-control detection",
        "defect_analysis_card": "🔍 Defect Analysis",
        "pareto_charts": "Pareto analysis charts",
        "defect_rate_calculations": "Defect rate calculations",
        "categorical_analysis": "Categorical analysis",
        "vital_few_identification": "Vital few identification",
        "sampling_recommender_card": "🎯 Sampling Recommender",
        "ai_powered_recommendations": "AI-powered recommendations",
        "variable_vs_attribute": "Variable vs attribute data",
        "application_specific_guidance": "Application-specific guidance",
        "industry_best_practices": "Industry best practices",
        "advanced_analytics_card": "🔬 Advanced Analytics",
        "normality_tests": "Normality tests (Shapiro-Wilk)",
        "qq_plots": "Q-Q plots for distribution",
        "correlation_analysis": "Correlation analysis",
        "distribution_plots": "Distribution plots",
        "regression_forecasting_card": "📊 Regression & Forecasting",
        "machine_learning_pycaret": "Machine learning with PyCaret",
        "multiple_algorithm_comparison": "Multiple algorithm comparison",
        "quality_prediction_models": "Quality prediction models",
        "future_trend_forecasting": "Future trend forecasting",
        "how_to_use": "📖 How to Use This Dashboard",
        "data_import_step": "1. 📁 **Data Import**",
        "data_import_desc": "Upload your CSV data, import from Google Sheets, or use our sample manufacturing dataset",
        "data_overview_step": "2. 📊 **Data Overview**",
        "data_overview_desc": "Explore your dataset with comprehensive statistics, distributions, and quality checks",
        "quality_metrics_step": "3. 📐 **Quality Metrics**",
        "quality_metrics_desc": "Calculate Cp, Cpk, Pp, Ppk, Cmk, DPMO and Sigma levels with interpretation",
        "spc_analysis_step": "4. 📈 **SPC Analysis**",
        "spc_analysis_desc": "Generate X-bar and R control charts for real-time process monitoring",
        "defect_analysis_step": "5. 🔍 **Defect Analysis**",
        "defect_analysis_desc": "Perform Pareto analysis and identify key improvement opportunities",
        "sampling_recommender_step": "6. 🎯 **Sampling Recommender**",
        "sampling_recommender_desc": "Get AI-powered sampling recommendations based on your data characteristics",
        "advanced_analytics_step": "7. 🔬 **Advanced Analytics**",
        "advanced_analytics_desc": "Run normality tests, correlation analysis, and distribution diagnostics",
        "regression_forecasting_step": "8. 📊 **Regression & Forecasting**",
        "regression_forecasting_desc": "Build machine learning models to predict quality and forecast trends",
        "share_this_app": "🌐 Share This App",
        "app_free": "**This app is completely FREE to use and share!**",
        "send_link": "Send this link to your colleagues and team members:",
        "quick_start": "⚡ Quick Start",
        "use_sample_data": "🚀 Use Sample Data & Start Analyzing",
        "sample_data_loaded": "✅ Sample data loaded! Switch to other modules to start analysis.",
        "upload_own_data": "📁 Upload Your Own Data",
        "upload_info": "Switch to 'Data Import' module to upload your CSV files or import from Google Sheets",
        "system_status": "🔧 System Status",
        "pandas_available": "✅ Pandas: Available",
        "numpy_available": "✅ NumPy: Available",
        "matplotlib_available": "✅ Matplotlib: Available",
        "scipy_available": "✅ SciPy: Available",
        "scipy_not_available": "⚠️ SciPy: Not Available",
        "pycaret_available": "✅ PyCaret: Available",
        "pycaret_not_available": "⚠️ PyCaret: Not Available",
        "upload_csv_data": "📤 Upload Your CSV Data",
        "choose_csv_file": "Choose CSV file",
        "upload_manufacturing_data": "Upload your manufacturing data in CSV format",
        "file_uploaded_successfully": "✅ File uploaded successfully! Shape:",
        "data_preview": "📋 Data Preview",
        "data_information": "📊 Data Information",
        "total_rows": "Total Rows",
        "numeric_columns": "Numeric Columns",
        "total_columns": "Total Columns",
        "categorical_columns": "Categorical Columns",
        "error_reading_file": "❌ Error reading file:",
        "google_sheets_import": "🌐 Google Sheets Import",
        "google_sheets_url": "Google Sheets URL",
        "google_sheets_placeholder": "https://docs.google.com/spreadsheets/d/...",
        "google_sheets_help": "Make sure your Google Sheet is shared with 'Anyone with the link can view'",
        "import_google_sheets": "📥 Import from Google Sheets",
        "importing_google_sheets": "Importing data from Google Sheets...",
        "google_sheets_imported": "✅ Google Sheets imported successfully! Shape:",
        "google_sheets_failed": "❌ Failed to import from Google Sheets. Please check the URL and sharing settings.",
        "sample_manufacturing_data": "🔬 Sample Manufacturing Data",
        "use_sample_dataset": "Use our comprehensive sample dataset to explore all features:",
        "sample_dataset_includes": "**Sample Dataset Includes:**",
        "manufacturing_records": "- 1000 manufacturing records with time series",
        "quality_characteristics": "- Multiple quality characteristics",
        "defect_data": "- Defect data for analysis",
        "process_parameters": "- Process parameters (temperature, pressure)",
        "categorical_variables": "- Categorical variables for stratification",
        "realistic_trends": "- Realistic trends and variations",
        "load_sample_data": "🎲 Load Sample Data",
        "sample_data_loaded_success": "✅ Sample manufacturing data loaded successfully!",
        "dataset_summary": "📈 Dataset Summary",
        "dataset_shape": "📊 Dataset Shape",
        "numeric_columns_count": "🔢 Numeric Columns",
        "categorical_columns_count": "📝 Categorical Columns",
        "defect_rate": "⚠️ Defect Rate",
        "defect_column_not_found": "ℹ️ Defect Column",
        "not_found": "Not found",
        "basic_statistics": "📊 Basic Statistics",
        "no_numeric_columns": "No numeric columns found for statistical analysis",
        "column_information": "🗂️ Column Information",
        "numeric_columns_label": "**🔢 Numeric Columns:**",
        "missing": "missing",
        "categorical_columns_label": "**📝 Categorical Columns:**",
        "unique_values": "unique values",
        "data_distributions": "📊 Data Distributions",
        "select_column_visualize": "Select column to visualize",
        "distribution_of": "Distribution of",
        "frequency": "Frequency",
        "mean": "Mean",
        "std_dev": "Std Dev",
        "min": "Min",
        "max": "Max",
        "quality_metrics_calculator": "📐 Quality Metrics Calculator",
        "calculate_process_capability": "**Calculate process capability indices and quality metrics:**",
        "cp_cpk_indices": "- **Cp, Cpk**: Process capability indices",
        "pp_ppk_indices": "- **Pp, Ppk**: Process performance indices",
        "cmk_index": "- **Cmk**: Machine capability index",
        "dpmo_metric": "- **DPMO**: Defects per million opportunities",
        "sigma_level_metric": "- **Sigma Level**: Process sigma level",
        "select_variable": "📊 Select Variable",
        "sample_size": "Sample Size",
        "specification_limits": "🎯 Specification Limits",
        "lower_spec_limit": "Lower Specification Limit (LSL)",
        "minimum_acceptable": "Minimum acceptable value",
        "upper_spec_limit": "Upper Specification Limit (USL)",
        "maximum_acceptable": "Maximum acceptable value",
        "subgroup_size_cmk": "Subgroup Size for Cmk",
        "consecutive_pieces": "Number of consecutive pieces for machine capability",
        "calculate_metrics": "🚀 Calculate Quality Metrics",
        "quality_metrics_results": "📊 Quality Metrics Results",
        "interpretation_guidelines": "🎯 Interpretation & Guidelines",
        "good": "Good",
        "marginal": "Marginal",
        "poor": "Poor",
        "process_potential": "Process potential capability",
        "process_actual": "Process actual capability accounting for centering",
        "process_performance": "Process performance",
        "process_performance_centering": "Process performance accounting for centering",
        "machine_capability": "Machine capability",
        "distribution_analysis": "📈 Distribution Analysis",
        "spc_control": "📈 Statistical Process Control (SPC)",
        "select_variable_spc": "📊 Select Variable for SPC",
        "subgroup_size": "👥 Subgroup Size",
        "lsl_spc": "📏 LSL for SPC",
        "usl_spc": "📏 USL for SPC",
        "generate_control_charts": "📊 Generate Control Charts",
        "not_enough_data_subgroups": "❌ Not enough data for subgroups. Try smaller subgroup size.",
        "xbar_control_chart": "X-bar Control Chart",
        "r_control_chart": "R Control Chart",
        "subgroup_mean": "Subgroup Mean",
        "subgroup_range": "Subgroup Range",
        "subgroup_number": "Subgroup Number",
        "control_chart_statistics": "📊 Control Chart Statistics",
        "xdouble_bar": "X-double bar",
        "rbar": "R-bar",
        "process_sigma": "Process Sigma",
        "out_of_control_points": "Out-of-Control Points",
        "defect_column_not_found_error": "❌ No 'defect' column found in the dataset",
        "sample_dataset_has_defect": "The sample dataset includes a 'defect' column. Try loading sample data or ensure your dataset has a defect indicator column.",
        "no_categorical_variables": "❌ No categorical variables available for defect analysis",
        "stratify_defects_by": "📂 Stratify Defects By",
        "defect_analysis_category": "📊 Defect Analysis by Category",
        "pareto_analysis": "📈 Pareto Analysis",
        "defect_count": "Defect Count",
        "cumulative_percentage": "Cumulative %",
        "80_percent_line": "80% Line",
        "pareto_chart": "Pareto Chart - Defects by",
        "key_insights": "🎯 Key Insights",
        "total_defects": "Total Defects",
        "total_units": "Total Units",
        "overall_defect_rate": "Overall Defect Rate",
        "top_defect_contributors": "**Top Defect Contributors:**",
        "defects_rate": "defects",
        "rate": "rate",
        "sampling_method_recommender": "🎯 Sampling Method Recommender",
        "data_type": "**Data Type**",
        "variable": "Variable",
        "attribute": "Attribute",
        "data_nature": "**Data Nature**",
        "continuous": "Continuous",
        "discrete": "Discrete",
        "continuous_normal": "Continuous - Normal",
        "continuous_non_normal": "Continuous - Non-normal",
        "application": "**Application**",
        "process_control": "Process Control",
        "lot_acceptance": "Lot Acceptance",
        "capability_analysis": "Capability Analysis",
        "sample_size_slider": "**Sample Size**",
        "population_size": "**Population Size**",
        "confidence_level": "**Confidence Level**",
        "get_sampling_recommendations": "🎯 Get Sampling Recommendations",
        "sampling_recommendations": "📋 Sampling Recommendations",
        "sampling_plan_details": "📊 Sampling Plan Details",
        "sampling_fraction": "Sampling Fraction",
        "recommended_sample_size": "Recommended Sample Size",
        "sampling_strategy_visualization": "📈 Sampling Strategy Visualization",
        "advanced_analytics_module": "🔬 Advanced Analytics",
        "select_analysis_type": "🔧 Select Analysis Type",
        "normality_test": "Normality Test",
        "qq_plot": "Q-Q Plot",
        "correlation_analysis_option": "Correlation Analysis",
        "distribution_comparison": "Distribution Comparison",
        "scipy_not_available_normality": "⚠️ SciPy not available. Using basic normality assessment.",
        "select_variable_normality": "📊 Select Variable",
        "skewness": "Skewness",
        "kurtosis": "Kurtosis",
        "data_approximately_normal": "✅ Data appears approximately normal based on skewness and kurtosis",
        "data_not_normal": "⚠️ Data may not be normally distributed based on skewness and kurtosis",
        "run_normality_test": "📊 Run Normality Test",
        "shapiro_wilk_statistic": "Shapiro-Wilk Statistic",
        "p_value": "p-value",
        "data_normally_distributed": "✅ Data appears to be normally distributed (fail to reject H0)",
        "data_not_normally_distributed": "❌ Data does not appear to be normally distributed (reject H0)",
        "scipy_required_qq": "❌ Q-Q Plot requires SciPy. Please ensure SciPy is installed in your environment.",
        "select_variable_qq": "📊 Select Variable for Q-Q Plot",
        "qq_plot_title": "Q-Q Plot of",
        "select_variables_correlation": "📊 Select Variables for Correlation",
        "correlation_matrix": "📈 Correlation Matrix",
        "correlation_heatmap": "Correlation Heatmap",
        "select_variables_compare": "📊 Select Variables to Compare",
        "standardized_values": "Standardized Values",
        "density": "Density",
        "distribution_comparison_title": "Distribution Comparison (Standardized)",
        "regression_forecasting_module": "📊 Regression & Forecasting with PyCaret",
        "pycaret_not_available": "❌ PyCaret is not available in your environment.",
        "enable_ml_features": "**To enable machine learning and forecasting features:**",
        "build_ml_models": "**Build machine learning models to predict quality outcomes and forecast future trends:**",
        "compare_multiple_algorithms": "- Compare multiple algorithms automatically",
        "predict_quality_scores": "- Predict quality scores and defect probabilities",
        "forecast_future_behavior": "- Forecast future process behavior",
        "identify_key_factors": "- Identify key factors affecting quality",
        "select_analysis_type_regression": "Select Analysis Type",
        "regression_analysis": "Regression Analysis",
        "time_series_forecasting": "Time Series Forecasting",
        "quality_prediction_models_title": "🔮 Quality Prediction Models",
        "target_variable": "🎯 Target Variable (What to predict)",
        "feature_variables": "📊 Feature Variables (Predictors)",
        "build_prediction_models": "🚀 Build Prediction Models",
        "training_models": "Training multiple machine learning models... This may take a few minutes.",
        "model_training_completed": "✅ Model training completed!",
        "model_comparison_results": "📊 Model Comparison Results",
        "best_performing_model": "🏆 Best Performing Model",
        "algorithm": "**Algorithm:**",
        "feature_importance": "Feature Importance",
        "feature_importance_not_available": "Feature importance plot not available for this model type",
        "make_predictions": "🔮 Make Predictions",
        "enter_value_for": "Enter value for",
        "predict_target_value": "📈 Predict Target Value",
        "predicted": "**Predicted",
        "error_model_training": "❌ Error in model training:",
        "time_series_forecasting_title": "📈 Time Series Forecasting",
        "no_date_column": "No date column found. Using sample data with dates.",
        "date_time_column": "📅 Date/Time Column",
        "value_to_forecast": "📊 Value to Forecast",
        "forecast_periods": "🔮 Forecast Periods",
        "model_type": "🤖 Model Type",
        "arima": "ARIMA",
        "exponential_smoothing": "Exponential Smoothing",
        "prophet": "Prophet",
        "generate_forecast": "🌐 Generate Forecast",
        "building_forecast": "Building time series forecast...",
        "time_series_analysis_complete": "**Time Series Analysis Complete**",
        "forecast_advanced_setup": "For advanced forecasting with PyCaret Time Series module, additional setup is required.",
        "historical_trend": "This visualization shows the historical trend of your selected variable.",
        "columns_not_found": "Selected columns not found in dataset",
        "error_forecasting": "❌ Error in forecasting:",
        "usage_analytics_module": "👥 Platform Usage Analytics",
        "platform_usage_statistics_info": "**Platform Usage Statistics**",
        "track_usage": "- Track how the dashboard is being used",
        "monitor_popularity": "- Monitor feature popularity",
        "understand_engagement": "- Understand user engagement",
        "overall_platform_statistics": "📈 Overall Platform Statistics",
        "avg_uses_per_session": "Avg Uses/Session",
        "feature_popularity": "🔥 Feature Popularity",
        "no_feature_usage_data": "No feature usage data available yet. Start using the dashboard to see analytics!",
        "engagement_insights": "💡 Engagement Insights",
        "most_popular_feature": "Most Popular Feature",
        "engagement_rate": "Engagement Rate",
        "total_analytics_tracked": "Total Analytics Tracked",
        "platform_health": "Platform Health",
        "excellent": "✅ Excellent",
        "good": "🟡 Good",
        "analytics_management": "🛠️ Analytics Management",
        "reset_usage_analytics": "🔄 Reset Usage Analytics",
        "analytics_reset": "Usage analytics reset successfully!",
    },
    "pt": {
        "page_title": "Painel Avançado de Controle de Qualidade",
        "language": "Idioma",
        "select_language": "Selecionar Idioma",
        "welcome": "🚀 Bem-vindo ao Painel Avançado de Controle de Qualidade",
        "complete_platform": "🎯 Plataforma Web Completa e Gratuita de Controle de Qualidade e Análise Preditiva",
        "platform_description": "Este aplicativo fornece recursos abrangentes de análise de controle de qualidade com aprendizado de máquina avançado para indústrias de manufatura e processos.",
        "navigation": "🔧 Navegação",
        "choose_module": "Escolher Módulo",
        "home": "🏠 Início",
        "data_import": "📁 Importar Dados",
        "data_overview": "📊 Visão Geral dos Dados",
        "quality_metrics": "📐 Métricas de Qualidade",
        "spc_analysis": "📈 Análise CEP",
        "defect_analysis": "🔍 Análise de Defeitos",
        "sampling_recommender": "🎯 Recomendador de Amostragem",
        "advanced_analytics": "🔬 Análise Avançada",
        "regression_forecasting": "📊 Regressão e Previsão",
        "usage_analytics": "👥 Análise de Uso",
        "platform_usage_statistics": "📊 Estatísticas de Uso da Plataforma",
        "total_uses": "Total de Usos",
        "unique_sessions": "Sessões Únicas",
        "active_features": "Funcionalidades Ativas",
        "quality_metrics_card": "📐 Métricas de Qualidade",
        "cp_cpk_calculations": "Cálculos de Cp, Cpk, Pp, Ppk, Cmk",
        "dpmo_sigma": "Análise de DPMO e Nível Sigma",
        "process_capability": "Análise de capacidade do processo",
        "real_time_calculations": "Cálculos em tempo real",
        "spc_control_charts": "📈 Cartas de Controle CEP",
        "xbar_r_charts": "Cartas de controle X-barra e R",
        "real_time_monitoring": "Monitoramento de processo em tempo real",
        "control_limit_calculations": "Cálculos de limites de controle",
        "out_of_control_detection": "Detecção de fora de controle",
        "defect_analysis_card": "🔍 Análise de Defeitos",
        "pareto_charts": "Gráficos de Pareto",
        "defect_rate_calculations": "Cálculos de taxa de defeitos",
        "categorical_analysis": "Análise categórica",
        "vital_few_identification": "Identificação dos poucos vitais",
        "sampling_recommender_card": "🎯 Recomendador de Amostragem",
        "ai_powered_recommendations": "Recomendações baseadas em IA",
        "variable_vs_attribute": "Dados de variável vs atributo",
        "application_specific_guidance": "Orientação específica por aplicação",
        "industry_best_practices": "Melhores práticas da indústria",
        "advanced_analytics_card": "🔬 Análise Avançada",
        "normality_tests": "Testes de normalidade (Shapiro-Wilk)",
        "qq_plots": "Gráficos Q-Q para distribuição",
        "correlation_analysis": "Análise de correlação",
        "distribution_plots": "Gráficos de distribuição",
        "regression_forecasting_card": "📊 Regressão e Previsão",
        "machine_learning_pycaret": "Aprendizado de máquina com PyCaret",
        "multiple_algorithm_comparison": "Comparação de múltiplos algoritmos",
        "quality_prediction_models": "Modelos de predição de qualidade",
        "future_trend_forecasting": "Previsão de tendências futuras",
        "how_to_use": "📖 Como Usar Este Painel",
        "data_import_step": "1. 📁 **Importar Dados**",
        "data_import_desc": "Carregue seu arquivo CSV, importe do Google Sheets ou use nosso conjunto de dados de manufatura de amostra",
        "data_overview_step": "2. 📊 **Visão Geral dos Dados**",
        "data_overview_desc": "Explore seu conjunto de dados com estatísticas abrangentes, distribuições e verificações de qualidade",
        "quality_metrics_step": "3. 📐 **Métricas de Qualidade**",
        "quality_metrics_desc": "Calcule Cp, Cpk, Pp, Ppk, Cmk, DPMO e níveis Sigma com interpretação",
        "spc_analysis_step": "4. 📈 **Análise CEP**",
        "spc_analysis_desc": "Gere cartas de controle X-barra e R para monitoramento de processo em tempo real",
        "defect_analysis_step": "5. 🔍 **Análise de Defeitos**",
        "defect_analysis_desc": "Realize análise de Pareto e identifique oportunidades de melhoria",
        "sampling_recommender_step": "6. 🎯 **Recomendador de Amostragem**",
        "sampling_recommender_desc": "Obtenha recomendações de amostragem baseadas em IA com base nas características dos seus dados",
        "advanced_analytics_step": "7. 🔬 **Análise Avançada**",
        "advanced_analytics_desc": "Execute testes de normalidade, análise de correlação e diagnósticos de distribuição",
        "regression_forecasting_step": "8. 📊 **Regressão e Previsão**",
        "regression_forecasting_desc": "Construa modelos de aprendizado de máquina para prever qualidade e tendências futuras",
        "share_this_app": "🌐 Compartilhar Este Aplicativo",
        "app_free": "**Este aplicativo é COMPLETAMENTE GRATUITO para usar e compartilhar!**",
        "send_link": "Envie este link para seus colegas e membros da equipe:",
        "quick_start": "⚡ Início Rápido",
        "use_sample_data": "🚀 Usar Dados de Amostra e Começar a Analisar",
        "sample_data_loaded": "✅ Dados de amostra carregados! Mude para outros módulos para começar a análise.",
        "upload_own_data": "📁 Carregar Seus Próprios Dados",
        "upload_info": "Mude para o módulo 'Importar Dados' para carregar seus arquivos CSV ou importar do Google Sheets",
        "system_status": "🔧 Status do Sistema",
        "pandas_available": "✅ Pandas: Disponível",
        "numpy_available": "✅ NumPy: Disponível",
        "matplotlib_available": "✅ Matplotlib: Disponível",
        "scipy_available": "✅ SciPy: Disponível",
        "scipy_not_available": "⚠️ SciPy: Não Disponível",
        "pycaret_available": "✅ PyCaret: Disponível",
        "pycaret_not_available": "⚠️ PyCaret: Não Disponível",
        "upload_csv_data": "📤 Carregar Seus Dados CSV",
        "choose_csv_file": "Escolher arquivo CSV",
        "upload_manufacturing_data": "Carregue seus dados de manufatura em formato CSV",
        "file_uploaded_successfully": "✅ Arquivo carregado com sucesso! Formato:",
        "data_preview": "📋 Prévia dos Dados",
        "data_information": "📊 Informações dos Dados",
        "total_rows": "Total de Linhas",
        "numeric_columns": "Colunas Numéricas",
        "total_columns": "Total de Colunas",
        "categorical_columns": "Colunas Categóricas",
        "error_reading_file": "❌ Erro ao ler arquivo:",
        "google_sheets_import": "🌐 Importar do Google Sheets",
        "google_sheets_url": "URL do Google Sheets",
        "google_sheets_placeholder": "https://docs.google.com/spreadsheets/d/...",
        "google_sheets_help": "Certifique-se de que sua planilha Google esteja compartilhada com 'Qualquer pessoa com o link pode visualizar'",
        "import_google_sheets": "📥 Importar do Google Sheets",
        "importing_google_sheets": "Importando dados do Google Sheets...",
        "google_sheets_imported": "✅ Google Sheets importado com sucesso! Formato:",
        "google_sheets_failed": "❌ Falha ao importar do Google Sheets. Verifique a URL e as configurações de compartilhamento.",
        "sample_manufacturing_data": "🔬 Dados de Manufatura de Amostra",
        "use_sample_dataset": "Use nosso conjunto de dados de amostra abrangente para explorar todos os recursos:",
        "sample_dataset_includes": "**Conjunto de Dados de Amostra Inclui:**",
        "manufacturing_records": "- 1000 registros de manufatura com série temporal",
        "quality_characteristics": "- Múltiplas características de qualidade",
        "defect_data": "- Dados de defeitos para análise",
        "process_parameters": "- Parâmetros de processo (temperatura, pressão)",
        "categorical_variables": "- Variáveis categóricas para estratificação",
        "realistic_trends": "- Tendências e variações realistas",
        "load_sample_data": "🎲 Carregar Dados de Amostra",
        "sample_data_loaded_success": "✅ Dados de manufatura de amostra carregados com sucesso!",
        "dataset_summary": "📈 Resumo do Conjunto de Dados",
        "dataset_shape": "📊 Formato do Conjunto",
        "numeric_columns_count": "🔢 Colunas Numéricas",
        "categorical_columns_count": "📝 Colunas Categóricas",
        "defect_rate": "⚠️ Taxa de Defeitos",
        "defect_column_not_found": "ℹ️ Coluna de Defeito",
        "not_found": "Não encontrada",
        "basic_statistics": "📊 Estatísticas Básicas",
        "no_numeric_columns": "Nenhuma coluna numérica encontrada para análise estatística",
        "column_information": "🗂️ Informações das Colunas",
        "numeric_columns_label": "**🔢 Colunas Numéricas:**",
        "missing": "ausentes",
        "categorical_columns_label": "**📝 Colunas Categóricas:**",
        "unique_values": "valores únicos",
        "data_distributions": "📊 Distribuições dos Dados",
        "select_column_visualize": "Selecionar coluna para visualizar",
        "distribution_of": "Distribuição de",
        "frequency": "Frequência",
        "mean": "Média",
        "std_dev": "Desvio Padrão",
        "min": "Mínimo",
        "max": "Máximo",
        "quality_metrics_calculator": "📐 Calculadora de Métricas de Qualidade",
        "calculate_process_capability": "**Calcular índices de capacidade de processo e métricas de qualidade:**",
        "cp_cpk_indices": "- **Cp, Cpk**: Índices de capacidade do processo",
        "pp_ppk_indices": "- **Pp, Ppk**: Índices de desempenho do processo",
        "cmk_index": "- **Cmk**: Índice de capacidade da máquina",
        "dpmo_metric": "- **DPMO**: Defeitos por milhão de oportunidades",
        "sigma_level_metric": "- **Nível Sigma**: Nível sigma do processo",
        "select_variable": "📊 Selecionar Variável",
        "sample_size": "Tamanho da Amostra",
        "specification_limits": "🎯 Limites de Especificação",
        "lower_spec_limit": "Limite Inferior de Especificação (LIE)",
        "minimum_acceptable": "Valor mínimo aceitável",
        "upper_spec_limit": "Limite Superior de Especificação (LSE)",
        "maximum_acceptable": "Valor máximo aceitável",
        "subgroup_size_cmk": "Tamanho do Subgrupo para Cmk",
        "consecutive_pieces": "Número de peças consecutivas para capacidade da máquina",
        "calculate_metrics": "🚀 Calcular Métricas de Qualidade",
        "quality_metrics_results": "📊 Resultados das Métricas de Qualidade",
        "interpretation_guidelines": "🎯 Interpretação e Diretrizes",
        "good": "Bom",
        "marginal": "Marginal",
        "poor": "Ruim",
        "process_potential": "Capacidade potencial do processo",
        "process_actual": "Capacidade real do processo considerando centralização",
        "process_performance": "Desempenho do processo",
        "process_performance_centering": "Desempenho do processo considerando centralização",
        "machine_capability": "Capacidade da máquina",
        "distribution_analysis": "📈 Análise de Distribuição",
        "spc_control": "📈 Controle Estatístico de Processo (CEP)",
        "select_variable_spc": "📊 Selecionar Variável para CEP",
        "subgroup_size": "👥 Tamanho do Subgrupo",
        "lsl_spc": "📏 LIE para CEP",
        "usl_spc": "📏 LSE para CEP",
        "generate_control_charts": "📊 Gerar Cartas de Controle",
        "not_enough_data_subgroups": "❌ Dados insuficientes para subgrupos. Tente um tamanho de subgrupo menor.",
        "xbar_control_chart": "Carta de Controle X-barra",
        "r_control_chart": "Carta de Controle R",
        "subgroup_mean": "Média do Subgrupo",
        "subgroup_range": "Amplitude do Subgrupo",
        "subgroup_number": "Número do Subgrupo",
        "control_chart_statistics": "📊 Estatísticas da Carta de Controle",
        "xdouble_bar": "X-barra dupla",
        "rbar": "R-barra",
        "process_sigma": "Sigma do Processo",
        "out_of_control_points": "Pontos Fora de Controle",
        "defect_column_not_found_error": "❌ Nenhuma coluna 'defect' (defeito) encontrada no conjunto de dados",
        "sample_dataset_has_defect": "O conjunto de dados de amostra inclui uma coluna 'defect'. Tente carregar os dados de amostra ou certifique-se de que seu conjunto de dados tenha uma coluna indicadora de defeito.",
        "no_categorical_variables": "❌ Nenhuma variável categórica disponível para análise de defeitos",
        "stratify_defects_by": "📂 Estratificar Defeitos Por",
        "defect_analysis_category": "📊 Análise de Defeitos por Categoria",
        "pareto_analysis": "📈 Análise de Pareto",
        "defect_count": "Contagem de Defeitos",
        "cumulative_percentage": "% Acumulada",
        "80_percent_line": "Linha de 80%",
        "pareto_chart": "Gráfico de Pareto - Defeitos por",
        "key_insights": "🎯 Principais Insights",
        "total_defects": "Total de Defeitos",
        "total_units": "Total de Unidades",
        "overall_defect_rate": "Taxa Geral de Defeitos",
        "top_defect_contributors": "**Principais Contribuintes para Defeitos:**",
        "defects_rate": "defeitos",
        "rate": "taxa",
        "sampling_method_recommender": "🎯 Recomendador de Método de Amostragem",
        "data_type": "**Tipo de Dado**",
        "variable": "Variável",
        "attribute": "Atributo",
        "data_nature": "**Natureza do Dado**",
        "continuous": "Contínuo",
        "discrete": "Discreto",
        "continuous_normal": "Contínuo - Normal",
        "continuous_non_normal": "Contínuo - Não normal",
        "application": "**Aplicação**",
        "process_control": "Controle de Processo",
        "lot_acceptance": "Aceitação de Lote",
        "capability_analysis": "Análise de Capacidade",
        "sample_size_slider": "**Tamanho da Amostra**",
        "population_size": "**Tamanho da População**",
        "confidence_level": "**Nível de Confiança**",
        "get_sampling_recommendations": "🎯 Obter Recomendações de Amostragem",
        "sampling_recommendations": "📋 Recomendações de Amostragem",
        "sampling_plan_details": "📊 Detalhes do Plano de Amostragem",
        "sampling_fraction": "Fração de Amostragem",
        "recommended_sample_size": "Tamanho de Amostra Recomendado",
        "sampling_strategy_visualization": "📈 Visualização da Estratégia de Amostragem",
        "advanced_analytics_module": "🔬 Análise Avançada",
        "select_analysis_type": "🔧 Selecionar Tipo de Análise",
        "normality_test": "Teste de Normalidade",
        "qq_plot": "Gráfico Q-Q",
        "correlation_analysis_option": "Análise de Correlação",
        "distribution_comparison": "Comparação de Distribuições",
        "scipy_not_available_normality": "⚠️ SciPy não disponível. Usando avaliação básica de normalidade.",
        "select_variable_normality": "📊 Selecionar Variável",
        "skewness": "Assimetria",
        "kurtosis": "Curtose",
        "data_approximately_normal": "✅ Os dados parecem aproximadamente normais com base na assimetria e curtose",
        "data_not_normal": "⚠️ Os dados podem não ser normalmente distribuídos com base na assimetria e curtose",
        "run_normality_test": "📊 Executar Teste de Normalidade",
        "shapiro_wilk_statistic": "Estatística de Shapiro-Wilk",
        "p_value": "valor-p",
        "data_normally_distributed": "✅ Os dados parecem ser normalmente distribuídos (falha em rejeitar H0)",
        "data_not_normally_distributed": "❌ Os dados não parecem ser normalmente distribuídos (rejeitar H0)",
        "scipy_required_qq": "❌ O Gráfico Q-Q requer SciPy. Certifique-se de que o SciPy esteja instalado em seu ambiente.",
        "select_variable_qq": "📊 Selecionar Variável para Gráfico Q-Q",
        "qq_plot_title": "Gráfico Q-Q de",
        "select_variables_correlation": "📊 Selecionar Variáveis para Correlação",
        "correlation_matrix": "📈 Matriz de Correlação",
        "correlation_heatmap": "Mapa de Calor da Correlação",
        "select_variables_compare": "📊 Selecionar Variáveis para Comparar",
        "standardized_values": "Valores Padronizados",
        "density": "Densidade",
        "distribution_comparison_title": "Comparação de Distribuições (Padronizado)",
        "regression_forecasting_module": "📊 Regressão e Previsão com PyCaret",
        "pycaret_not_available": "❌ PyCaret não está disponível em seu ambiente.",
        "enable_ml_features": "**Para habilitar os recursos de aprendizado de máquina e previsão:**",
        "build_ml_models": "**Construa modelos de aprendizado de máquina para prever resultados de qualidade e tendências futuras:**",
        "compare_multiple_algorithms": "- Compare múltiplos algoritmos automaticamente",
        "predict_quality_scores": "- Preveja pontuações de qualidade e probabilidades de defeito",
        "forecast_future_behavior": "- Preveja o comportamento futuro do processo",
        "identify_key_factors": "- Identifique fatores-chave que afetam a qualidade",
        "select_analysis_type_regression": "Selecionar Tipo de Análise",
        "regression_analysis": "Análise de Regressão",
        "time_series_forecasting": "Previsão de Séries Temporais",
        "quality_prediction_models_title": "🔮 Modelos de Predição de Qualidade",
        "target_variable": "🎯 Variável Alvo (O que prever)",
        "feature_variables": "📊 Variáveis de Características (Preditores)",
        "build_prediction_models": "🚀 Construir Modelos de Predição",
        "training_models": "Treinando múltiplos modelos de aprendizado de máquina... Isso pode levar alguns minutos.",
        "model_training_completed": "✅ Treinamento do modelo concluído!",
        "model_comparison_results": "📊 Resultados da Comparação de Modelos",
        "best_performing_model": "🏆 Melhor Modelo",
        "algorithm": "**Algoritmo:**",
        "feature_importance": "Feature Importance",
        "feature_importance_not_available": "Gráfico de importância de características não disponível para este tipo de modelo",
        "make_predictions": "🔮 Fazer Previsões",
        "enter_value_for": "Digite o valor para",
        "predict_target_value": "📈 Prever Valor Alvo",
        "predicted": "**Previsto",
        "error_model_training": "❌ Erro no treinamento do modelo:",
        "time_series_forecasting_title": "📈 Previsão de Séries Temporais",
        "no_date_column": "Nenhuma coluna de data encontrada. Usando dados de amostra com datas.",
        "date_time_column": "📅 Coluna de Data/Hora",
        "value_to_forecast": "📊 Valor a Prever",
        "forecast_periods": "🔮 Períodos de Previsão",
        "model_type": "🤖 Tipo de Modelo",
        "arima": "ARIMA",
        "exponential_smoothing": "Suavização Exponencial",
        "prophet": "Prophet",
        "generate_forecast": "🌐 Gerar Previsão",
        "building_forecast": "Construindo previsão de série temporal...",
        "time_series_analysis_complete": "**Análise de Série Temporal Concluída**",
        "forecast_advanced_setup": "Para previsão avançada com o módulo de Séries Temporais do PyCaret, é necessária configuração adicional.",
        "historical_trend": "Esta visualização mostra a tendência histórica da variável selecionada.",
        "columns_not_found": "Colunas selecionadas não encontradas no conjunto de dados",
        "error_forecasting": "❌ Erro na previsão:",
        "usage_analytics_module": "👥 Análise de Uso da Plataforma",
        "platform_usage_statistics_info": "**Estatísticas de Uso da Plataforma**",
        "track_usage": "- Acompanhe como o painel está sendo usado",
        "monitor_popularity": "- Monitore a popularidade das funcionalidades",
        "understand_engagement": "- Entenda o engajamento do usuário",
        "overall_platform_statistics": "📈 Estatísticas Gerais da Plataforma",
        "avg_uses_per_session": "Média de Usos/Sessão",
        "feature_popularity": "🔥 Popularidade das Funcionalidades",
        "no_feature_usage_data": "Nenhum dado de uso de funcionalidade disponível ainda. Comece a usar o painel para ver as análises!",
        "engagement_insights": "💡 Insights de Engajamento",
        "most_popular_feature": "Funcionalidade Mais Popular",
        "engagement_rate": "Taxa de Engajamento",
        "total_analytics_tracked": "Total de Análises Rastreadas",
        "platform_health": "Saúde da Plataforma",
        "excellent": "✅ Excelente",
        "good": "🟡 Boa",
        "analytics_management": "🛠️ Gerenciamento de Análises",
        "reset_usage_analytics": "🔄 Redefinir Análises de Uso",
        "analytics_reset": "Análises de uso redefinidas com sucesso!",
    }
}

def get_text(key):
    """Get translated text based on current language"""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# Set page configuration
st.set_page_config(
    page_title="Advanced Quality Control Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Language selector in the top right corner
col1, col2, col3 = st.columns([3, 1, 1])
with col3:
    language_choice = st.selectbox(
        get_text("select_language"),
        options=["English", "Português"],
        index=0 if st.session_state.language == 'en' else 1,
        key="language_selector",
        label_visibility="collapsed"
    )
    if language_choice == "English":
        st.session_state.language = 'en'
    else:
        st.session_state.language = 'pt'

# Rest of the code remains the same with translations integrated
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
        image_url = "https://via.placeholder.com/150/667eea/ffffff?text=SM"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image
    except:
        return Image.new('RGB', (150, 150), color='#667eea')

# Generate comprehensive sample manufacturing data
def generate_manufacturing_data():
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    base_length = 10.0
    trend = np.linspace(0, 0.5, n)
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n) / 30)
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
    df['quality_score'] = (df['length'] * 0.3 + df['hardness'] * 0.2 + 
                          df['temperature'] * 0.1 + np.random.normal(0, 0.5, n))
    
    return df

# Quality metrics calculation functions
def calculate_cp(upper_spec, lower_spec, std_dev):
    if std_dev == 0:
        return float('inf')
    return (upper_spec - lower_spec) / (6 * std_dev)

def calculate_cpk(upper_spec, lower_spec, mean, std_dev):
    if std_dev == 0:
        return float('inf')
    cpu = (upper_spec - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
    cpl = (mean - lower_spec) / (3 * std_dev) if std_dev > 0 else float('inf')
    return min(cpu, cpl)

def calculate_pp(upper_spec, lower_spec, std_dev):
    return calculate_cp(upper_spec, lower_spec, std_dev)

def calculate_ppk(upper_spec, lower_spec, mean, std_dev):
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_cmk(upper_spec, lower_spec, mean, std_dev):
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_dpmo(defect_count, total_units):
    if total_units == 0:
        return 0
    return (defect_count / total_units) * 1000000

def calculate_sigma_level(dpmo):
    if dpmo <= 0:
        return float('inf')
    if not HAS_SCIPY:
        if dpmo <= 3.4: return 6.0
        elif dpmo <= 233: return 5.0
        elif dpmo <= 6200: return 4.0
        elif dpmo <= 66800: return 3.0
        elif dpmo <= 308000: return 2.0
        else: return 1.0
    return stats.norm.ppf(1 - dpmo/1000000) + 1.5

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

tracker = UsageTracker()

# Main application
def main():
    if 'df' not in st.session_state:
        st.session_state.df = generate_manufacturing_data()
    
    # Profile Header
    col1, col2 = st.columns([1, 3])
    
    with col1:
        profile_image = load_profile_image()
        st.image(profile_image, width=150, caption="Md. Sourove Akther Momin")
    
    with col2:
        st.markdown(f"""
        <div class="profile-header">
            <h1>🏭 {get_text("page_title")}</h1>
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
    st.sidebar.title(get_text("navigation"))
    app_mode = st.sidebar.selectbox(
        get_text("choose_module"),
        [get_text("home"), get_text("data_import"), get_text("data_overview"), get_text("quality_metrics"),
         get_text("spc_analysis"), get_text("defect_analysis"), get_text("sampling_recommender"),
         get_text("advanced_analytics"), get_text("regression_forecasting"), get_text("usage_analytics")]
    )
    
    tracker.track_usage(f"Navigation_{app_mode}")
    
    # HOME PAGE
    if app_mode == get_text("home"):
        st.markdown(f'<div class="section-header">{get_text("welcome")}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
        <h3>{get_text("complete_platform")}</h3>
        <p>{get_text("platform_description")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        stats_data = tracker.get_stats()
        st.subheader(get_text("platform_usage_statistics"))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(get_text("total_uses"), stats_data['total_uses'])
        with col2:
            st.metric(get_text("unique_sessions"), stats_data['unique_sessions'])
        with col3:
            st.metric(get_text("active_features"), len(stats_data['feature_usage']))
        
        st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="feature-card">
            <h3>{get_text("quality_metrics_card")}</h3>
            <ul>
            <li>{get_text("cp_cpk_calculations")}</li>
            <li>{get_text("dpmo_sigma")}</li>
            <li>{get_text("process_capability")}</li>
            <li>{get_text("real_time_calculations")}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="feature-card">
            <h3>{get_text("regression_forecasting_card")}</h3>
            <ul>
            <li>{get_text("machine_learning_pycaret")}</li>
            <li>{get_text("multiple_algorithm_comparison")}</li>
            <li>{get_text("quality_prediction_models")}</li>
            <li>{get_text("future_trend_forecasting")}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader(get_text("quick_start"))
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(get_text("use_sample_data"), use_container_width=True):
                st.session_state.df = generate_manufacturing_data()
                st.success(get_text("sample_data_loaded"))
        
        with col2:
            if st.button(get_text("upload_own_data"), use_container_width=True):
                st.info(get_text("upload_info"))

if __name__ == "__main__":
    main()
