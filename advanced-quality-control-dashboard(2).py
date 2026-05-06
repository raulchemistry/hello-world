import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Crítico para o Streamlit Cloud
import matplotlib.pyplot as plt
import math
import io
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import requests
from io import BytesIO

# Tentar importar dependências opcionais
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

# Configuração da página
st.set_page_config(
    page_title="Painel Avançado de Controle de Qualidade",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhor estilo
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

# Carregar imagem de perfil
def load_profile_image():
    try:
        # Usando uma imagem de placeholder - substitua pela URL da sua imagem real
        image_url = "https://via.placeholder.com/150/667eea/ffffff?text=SM"
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image
    except:
        # Retorna uma imagem em branco se o carregamento falhar
        return Image.new('RGB', (150, 150), color='#667eea')

# Gerar dados de manufatura de amostra abrangente
def generate_manufacturing_data():
    np.random.seed(42)
    n = 1000  # Aumentado para melhor previsão
    
    # Dados de série temporal com tendências e sazonalidade
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Criar padrões de manufatura realistas
    base_length = 10.0
    trend = np.linspace(0, 0.5, n)  # Tendência ascendente gradual
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n) / 30)  # Sazonalidade mensal
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
        'operator': np.random.choice(['João', 'Miguel', 'Sara', 'Lisa', 'David'], n),
        'machine': np.random.choice(['CNC-1', 'CNC-2', 'TORNO-1', 'FRESA-1'], n),
        'shift': np.random.choice(['Manhã', 'Tarde', 'Noite'], n),
        'material_batch': np.random.choice(['A123', 'B456', 'C789', 'D012', 'E345'], n),
        'production_rate': np.random.normal(50, 5, n)
    }
    
    df = pd.DataFrame(data)
    
    # Adicionar alguma correlação para análise de regressão
    df['quality_score'] = (df['length'] * 0.3 + df['hardness'] * 0.2 + 
                          df['temperature'] * 0.1 + np.random.normal(0, 0.5, n))
    
    return df

# Funções de cálculo de métricas de qualidade
def calculate_cp(upper_spec, lower_spec, std_dev):
    """Calcular Índice de Capacidade do Processo (Cp)"""
    if std_dev == 0:
        return float('inf')
    return (upper_spec - lower_spec) / (6 * std_dev)

def calculate_cpk(upper_spec, lower_spec, mean, std_dev):
    """Calcular Índice de Capacidade do Processo (Cpk)"""
    if std_dev == 0:
        return float('inf')
    cpu = (upper_spec - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
    cpl = (mean - lower_spec) / (3 * std_dev) if std_dev > 0 else float('inf')
    return min(cpu, cpl)

def calculate_pp(upper_spec, lower_spec, std_dev):
    """Calcular Índice de Desempenho do Processo (Pp)"""
    return calculate_cp(upper_spec, lower_spec, std_dev)

def calculate_ppk(upper_spec, lower_spec, mean, std_dev):
    """Calcular Índice de Desempenho do Processo (Ppk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_cmk(upper_spec, lower_spec, mean, std_dev):
    """Calcular Índice de Capacidade da Máquina (Cmk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_dpmo(defect_count, total_units):
    """Calcular Defeitos por Milhão de Oportunidades (DPMO)"""
    if total_units == 0:
        return 0
    return (defect_count / total_units) * 1000000

def calculate_sigma_level(dpmo):
    """Calcular Nível Sigma a partir do DPMO"""
    if dpmo <= 0:
        return float('inf')
    if not HAS_SCIPY:
        # Aproximação simples sem scipy
        if dpmo <= 3.4: return 6.0
        elif dpmo <= 233: return 5.0
        elif dpmo <= 6200: return 4.0
        elif dpmo <= 66800: return 3.0
        elif dpmo <= 308000: return 2.0
        else: return 1.0
    return stats.norm.ppf(1 - dpmo/1000000) + 1.5

# Funções de recomendação de amostragem
def recommend_sampling_method(data_type, data_nature, application):
    """Recomendar método de amostragem com base nas características dos dados"""
    recommendations = []
    
    if data_type == "Variável":
        recommendations.append("📏 **Amostragem por Variáveis**: Use dados de medição para análise precisa")
        recommendations.append("✅ **Métodos Recomendados**: Cartas de controle CEP (X-barra R, X-barra S), Amostragem de aceitação por variáveis")
    elif data_type == "Atributo":
        recommendations.append("🔢 **Amostragem por Atributos**: Use dados de contagem (aprovado/reprovado) para análise de defeitos")
        recommendations.append("✅ **Métodos Recomendados**: Amostragem de aceitação por atributos, cartas p, cartas np, cartas c, cartas u")
    
    if data_nature == "Contínuo":
        recommendations.append("⏰ **Dados Contínuos**: Considere amostragem baseada em tempo em intervalos regulares")
    elif data_nature == "Discreto":
        recommendations.append("📦 **Dados Discretos**: Considere amostragem por lote ou amostragem em batelada")
    
    if "Normal" in data_nature:
        recommendations.append("📊 **Distribuição Normal**: Métodos estatísticos paramétricos podem ser usados")
    elif "Não normal" in data_nature:
        recommendations.append("📈 **Distribuição Não Normal**: Use métodos não paramétricos ou transforme os dados")
    
    if application == "Controle de Processo":
        recommendations.append("🎯 **Controle de Processo**: Use cartas de controle CEP com intervalos de amostragem regulares")
        recommendations.append("📋 **Tamanho da Amostra**: 20-25 subgrupos de 4-5 amostras cada")
    elif application == "Aceitação de Lote":
        recommendations.append("📋 **Aceitação de Lote**: Use ANSI/ASQ Z1.4 (MIL-STD-105E) para atributos")
        recommendations.append("📏 **Aceitação por Variáveis**: Use ANSI/ASQ Z1.9 (MIL-STD-414)")
    elif application == "Análise de Capacidade":
        recommendations.append("📐 **Análise de Capacidade**: Garanta amostragem aleatória, mínimo de 100 medições individuais")
        recommendations.append("⏰ **Momento**: Colete dados em diferentes períodos de tempo para análise confiável")
    elif application == "Análise de Defeitos":
        recommendations.append("🔍 **Análise de Defeitos**: Use amostragem estratificada por tipo/categoria de defeito")
        recommendations.append("📊 **Estratégia de Amostragem**: Concentre-se em áreas de alto defeito para análise detalhada")
    
    return recommendations

# Integração com Google Sheets
def load_from_google_sheets(url):
    """Carregar dados do Google Sheets"""
    try:
        if 'docs.google.com/spreadsheets' in url:
            # Converter URL do Google Sheets para exportação CSV
            if '/d/' in url:
                sheet_id = url.split('/d/')[1].split('/')[0]
            else:
                sheet_id = url.split('spreadsheets/d/')[1].split('/')[0]
            
            csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
            return pd.read_csv(csv_url)
        return None
    except Exception as e:
        st.error(f"Erro ao carregar Google Sheets: {e}")
        return None

# Rastreamento de uso
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

# Inicializar rastreador de uso
tracker = UsageTracker()

# Aplicação principal
def main():
    # Inicializar estado da sessão para dados
    if 'df' not in st.session_state:
        st.session_state.df = generate_manufacturing_data()
    
    # Cabeçalho do Perfil
    col1, col2 = st.columns([1, 3])
    
    with col1:
        profile_image = load_profile_image()
        st.image(profile_image, width=150, caption="Md. Sourove Akther Momin")
    
    with col2:
        st.markdown("""
        <div class="profile-header">
            <h1>🏭 Painel Avançado de Controle de Qualidade</h1>
            <h3>Md. Sourove Akther Momin</h3>
            <p>Mestrado em Estatística Aplicada e Ciência de Dados | Bacharelado em Engenharia Mecânica</p>
            <div>
                <span class="certification-badge">Profissional Certificado em Corte de Metais (CMP)</span>
                <span class="certification-badge">Gerente de Práticas de Manufatura de Classe Mundial (WCMPM)</span>
            </div>
            <p>Especialista: Processos, Qualidade, Engenharia de Produção, Previsão de Séries Temporais, Aprendizado de Máquina, Big Data, Aprendizado Profundo, Inteligência Artificial</p>
            <p>Membro do Comitê Técnico e Revisor do 2º Congresso Mundial IEOM 2025 Windsor, Ontário, Canadá</p>
            <p>Email: sourovemomin.kuet@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Barra lateral para navegação
    st.sidebar.title("🔧 Navegação")
    app_mode = st.sidebar.selectbox("Escolher Módulo", 
        ["🏠 Início", "📁 Importar Dados", "📊 Visão Geral dos Dados", "📐 Métricas de Qualidade", 
         "📈 Análise CEP", "🔍 Análise de Defeitos", "🎯 Recomendador de Amostragem", 
         "🔬 Análise Avançada", "📊 Regressão e Previsão", "👥 Análise de Uso"])
    
    # Rastrear navegação
    tracker.track_usage(f"Navegação_{app_mode}")
    
    # Página Inicial
    if app_mode == "🏠 Início":
        st.markdown('<div class="section-header">🚀 Bem-vindo ao Painel Avançado de Controle de Qualidade</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Plataforma Web Completa e Gratuita de Controle de Qualidade e Análise Preditiva</h3>
        <p>Este aplicativo fornece recursos abrangentes de análise de controle de qualidade com aprendizado de máquina avançado para indústrias de manufatura e processos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Estatísticas Rápidas
        stats_data = tracker.get_stats()
        st.subheader("📊 Estatísticas de Uso da Plataforma")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Usos", stats_data['total_uses'])
        with col2:
            st.metric("Sessões Únicas", stats_data['unique_sessions'])
        with col3:
            st.metric("Funcionalidades Ativas", len(stats_data['feature_usage']))
        
        # Grade de Funcionalidades
        st.markdown("""
        <div class="feature-grid">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h3>📐 Métricas de Qualidade</h3>
            <ul>
            <li>Cálculos de Cp, Cpk, Pp, Ppk, Cmk</li>
            <li>Análise de DPMO e Nível Sigma</li>
            <li>Análise de capacidade do processo</li>
            <li>Cálculos em tempo real</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>📈 Cartas de Controle CEP</h3>
            <ul>
            <li>Cartas de controle X-barra e R</li>
            <li>Monitoramento de processo em tempo real</li>
            <li>Cálculos de limites de controle</li>
            <li>Detecção de fora de controle</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>🔍 Análise de Defeitos</h3>
            <ul>
            <li>Gráficos de Pareto</li>
            <li>Cálculos de taxa de defeitos</li>
            <li>Análise categórica</li>
            <li>Identificação dos poucos vitais</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h3>🎯 Recomendador de Amostragem</h3>
            <ul>
            <li>Recomendações baseadas em IA</li>
            <li>Dados de variável vs atributo</li>
            <li>Orientação específica por aplicação</li>
            <li>Melhores práticas da indústria</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>🔬 Análise Avançada</h3>
            <ul>
            <li>Testes de normalidade (Shapiro-Wilk)</li>
            <li>Gráficos Q-Q para distribuição</li>
            <li>Análise de correlação</li>
            <li>Gráficos de distribuição</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
            <h3>📊 Regressão e Previsão</h3>
            <ul>
            <li>Aprendizado de máquina com PyCaret</li>
            <li>Comparação de múltiplos algoritmos</li>
            <li>Modelos de predição de qualidade</li>
            <li>Previsão de tendências futuras</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Seção Como Usar
        st.markdown("---")
        st.subheader("📖 Como Usar Este Painel")
        
        steps = [
            ("1. 📁 **Importar Dados**", "Carregue seu arquivo CSV, importe do Google Sheets ou use nosso conjunto de dados de manufatura de amostra"),
            ("2. 📊 **Visão Geral dos Dados**", "Explore seu conjunto de dados com estatísticas abrangentes, distribuições e verificações de qualidade"),
            ("3. 📐 **Métricas de Qualidade**", "Calcule Cp, Cpk, Pp, Ppk, Cmk, DPMO e níveis Sigma com interpretação"),
            ("4. 📈 **Análise CEP**", "Gere cartas de controle X-barra e R para monitoramento de processo em tempo real"),
            ("5. 🔍 **Análise de Defeitos**", "Realize análise de Pareto e identifique oportunidades de melhoria"),
            ("6. 🎯 **Recomendador de Amostragem**", "Obtenha recomendações de amostragem baseadas em IA com base nas características dos seus dados"),
            ("7. 🔬 **Análise Avançada**", "Execute testes de normalidade, análise de correlação e diagnósticos de distribuição"),
            ("8. 📊 **Regressão e Previsão**", "Construa modelos de aprendizado de máquina para prever qualidade e tendências futuras")
        ]
        
        for step, description in steps:
            st.markdown(f"**{step}**")
            st.write(description)
            st.write("")
        
        st.markdown("---")
        st.subheader("🌐 Compartilhar Este Aplicativo")
        st.info("""
        **Este aplicativo é COMPLETAMENTE GRATUITO para usar e compartilhar!** 
        
        Envie este link para seus colegas e membros da equipe:
        """)
        st.code("https://seu-usuario-qualidade-controle-painel.streamlit.app", language="text")
        
        # Botões de Início Rápido
        st.markdown("---")
        st.subheader("⚡ Início Rápido")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Usar Dados de Amostra e Começar a Analisar", use_container_width=True):
                st.session_state.df = generate_manufacturing_data()
                st.success("✅ Dados de amostra carregados! Mude para outros módulos para começar a análise.")
        
        with col2:
            if st.button("📁 Carregar Seus Próprios Dados", use_container_width=True):
                st.info("Mude para o módulo 'Importar Dados' para carregar seus arquivos CSV ou importar do Google Sheets")
        
        # Status do Sistema
        st.markdown("---")
        st.subheader("🔧 Status do Sistema")
        status_cols = st.columns(5)
        with status_cols[0]:
            st.success("✅ Pandas: Disponível")
        with status_cols[1]:
            st.success("✅ NumPy: Disponível")
        with status_cols[2]:
            st.success("✅ Matplotlib: Disponível")
        with status_cols[3]:
            if HAS_SCIPY:
                st.success("✅ SciPy: Disponível")
            else:
                st.warning("⚠️ SciPy: Não Disponível")
        with status_cols[4]:
            if HAS_PYCARET:
                st.success("✅ PyCaret: Disponível")
            else:
                st.warning("⚠️ PyCaret: Não Disponível")
    
    # Módulo de Importação de Dados
    elif app_mode == "📁 Importar Dados":
        st.markdown('<div class="section-header">📁 Importar Dados</div>', unsafe_allow_html=True)
        tracker.track_usage("Importar_Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Carregar Seus Dados CSV")
            uploaded_file = st.file_uploader("Escolher arquivo CSV", type=['csv'], 
                                           help="Carregue seus dados de manufatura em formato CSV")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success(f"✅ Arquivo carregado com sucesso! Formato: {df.shape}")
                    
                    # Mostrar prévia
                    st.subheader("📋 Prévia dos Dados")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Informações básicas
                    st.subheader("📊 Informações dos Dados")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total de Linhas", df.shape[0])
                        st.metric("Colunas Numéricas", len(df.select_dtypes(include=np.number).columns))
                    with col2:
                        st.metric("Total de Colunas", df.shape[1])
                        st.metric("Colunas Categóricas", len(df.select_dtypes(include=['object']).columns))
                    
                except Exception as e:
                    st.error(f"❌ Erro ao ler arquivo: {e}")
        
        with col2:
            st.subheader("🌐 Importar do Google Sheets")
            gsheet_url = st.text_input("URL do Google Sheets", 
                                     placeholder="https://docs.google.com/spreadsheets/d/...",
                                     help="Certifique-se de que sua planilha Google esteja compartilhada com 'Qualquer pessoa com o link pode visualizar'")
            
            if gsheet_url:
                if st.button("📥 Importar do Google Sheets"):
                    with st.spinner("Importando dados do Google Sheets..."):
                        df = load_from_google_sheets(gsheet_url)
                        if df is not None:
                            st.session_state.df = df
                            st.success(f"✅ Google Sheets importado com sucesso! Formato: {df.shape}")
                            st.dataframe(df.head(10), use_container_width=True)
                        else:
                            st.error("❌ Falha ao importar do Google Sheets. Verifique a URL e as configurações de compartilhamento.")
            
            st.subheader("🔬 Dados de Manufatura de Amostra")
            st.write("Use nosso conjunto de dados de amostra abrangente para explorar todos os recursos:")
            
            sample_info = """
            **Conjunto de Dados de Amostra Inclui:**
            - 1000 registros de manufatura com série temporal
            - Múltiplas características de qualidade
            - Dados de defeitos para análise
            - Parâmetros de processo (temperatura, pressão)
            - Variáveis categóricas para estratificação
            - Tendências e variações realistas
            """
            st.info(sample_info)
            
            if st.button("🎲 Carregar Dados de Amostra", use_container_width=True):
                st.session_state.df = generate_manufacturing_data()
                st.success("✅ Dados de manufatura de amostra carregados com sucesso!")
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # Módulo de Visão Geral dos Dados
    elif app_mode == "📊 Visão Geral dos Dados":
        st.markdown('<div class="section-header">📊 Visão Geral dos Dados</div>', unsafe_allow_html=True)
        tracker.track_usage("Visão_Geral_Dados")
        
        df = st.session_state.df
        
        # Estatísticas rápidas
        st.subheader("📈 Resumo do Conjunto de Dados")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Formato do Conjunto", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            numeric_cols = len(df.select_dtypes(include=np.number).columns)
            st.metric("🔢 Colunas Numéricas", numeric_cols)
        with col3:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Colunas Categóricas", categorical_cols)
        with col4:
            if 'defect' in df.columns:
                defect_rate = df['defect'].mean()
                st.metric("⚠️ Taxa de Defeitos", f"{defect_rate:.2%}")
            else:
                st.metric("ℹ️ Coluna de Defeito", "Não encontrada")
        
        # Prévia dos dados
        st.subheader("📋 Prévia dos Dados")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Estatísticas básicas
        st.subheader("📊 Estatísticas Básicas")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.warning("Nenhuma coluna numérica encontrada para análise estatística")
        
        # Informações das colunas
        st.subheader("🗂️ Informações das Colunas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔢 Colunas Numéricas:**")
            numeric_cols = list(df.select_dtypes(include=np.number).columns)
            for col in numeric_cols:
                missing = df[col].isnull().sum()
                st.write(f"• **{col}** - {missing} ausentes")
        
        with col2:
            st.write("**📝 Colunas Categóricas:**")
            cat_cols = list(df.select_dtypes(include=['object']).columns)
            for col in cat_cols:
                unique_count = df[col].nunique()
                st.write(f"• **{col}** - {unique_count} valores únicos")
        
        # Visualizações
        st.subheader("📊 Distribuições dos Dados")
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        if numeric_cols:
            selected_col = st.selectbox("Selecionar coluna para visualizar", numeric_cols)
            
            if selected_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                data = df[selected_col].dropna()
                
                ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribuição de {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequência')
                
                # Adicionar estatísticas
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1 Desvio Padrão')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Resumo de estatísticas
                st.write(f"**Estatísticas para {selected_col}:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Média", f"{mean_val:.3f}")
                with col2:
                    st.metric("Desvio Padrão", f"{std_val:.3f}")
                with col3:
                    st.metric("Mínimo", f"{data.min():.3f}")
                with col4:
                    st.metric("Máximo", f"{data.max():.3f}")
    
    # Módulo de Métricas de Qualidade
    elif app_mode == "📐 Métricas de Qualidade":
        st.markdown('<div class="section-header">📐 Calculadora de Métricas de Qualidade</div>', unsafe_allow_html=True)
        tracker.track_usage("Métricas_Qualidade")
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ Nenhuma variável numérica disponível para análise")
            return
        
        st.info("""
        **Calcular índices de capacidade de processo e métricas de qualidade:**
        - **Cp, Cpk**: Índices de capacidade do processo
        - **Pp, Ppk**: Índices de desempenho do processo  
        - **Cmk**: Índice de capacidade da máquina
        - **DPMO**: Defeitos por milhão de oportunidades
        - **Nível Sigma**: Nível sigma do processo
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable = st.selectbox("📊 Selecionar Variável", numeric_cols)
            data = df[variable].dropna()
            
            if len(data) > 0:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Tamanho da Amostra", len(data))
                st.metric("Média", f"{np.mean(data):.4f}")
                st.metric("Desvio Padrão", f"{np.std(data, ddof=1):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            data_min = data.min()
            data_max = data.max()
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🎯 Limites de Especificação")
            lsl = st.number_input("Limite Inferior de Especificação (LIE)", 
                                value=float(data_mean - 3*data_std),
                                help="Valor mínimo aceitável")
            usl = st.number_input("Limite Superior de Especificação (LSE)", 
                                value=float(data_mean + 3*data_std),
                                help="Valor máximo aceitável")
            subgroup_size = st.slider("Tamanho do Subgrupo para Cmk", min_value=2, max_value=10, value=5,
                                    help="Número de peças consecutivas para capacidade da máquina")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Calcular Métricas de Qualidade", type="primary", use_container_width=True):
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            
            # Calcular métricas
            cp = calculate_cp(usl, lsl, std_val)
            cpk = calculate_cpk(usl, lsl, mean_val, std_val)
            pp = calculate_pp(usl, lsl, std_val)
            ppk = calculate_ppk(usl, lsl, mean_val, std_val)
            
            # Calcular Cmk
            subgroup_means = []
            subgroup_stds = []
            for i in range(0, min(len(data), subgroup_size*5), subgroup_size):
                subgroup = data[i:i+subgroup_size]
                subgroup_means.append(np.mean(subgroup))
                subgroup_stds.append(np.std(subgroup, ddof=1))
            
            short_term_std = np.mean(subgroup_stds) if subgroup_stds else std_val
            cmk = calculate_cmk(usl, lsl, mean_val, short_term_std)
            
            # Calcular DPMO e Nível Sigma
            if variable == 'defect':
                defect_count = np.sum(data)
            else:
                defect_count = np.sum((data < lsl) | (data > usl))
            
            total_units = len(data)
            dpmo = calculate_dpmo(defect_count, total_units)
            sigma_level = calculate_sigma_level(dpmo)
            
            # Exibir resultados
            st.subheader("📊 Resultados das Métricas de Qualidade")
            
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
                st.metric("Nível Sigma", f"{sigma_level:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interpretação
            st.subheader("🎯 Interpretação e Diretrizes")
            
            capability_metrics = [
                ("Cp", cp, 1.33, 1.0, "Capacidade potencial do processo"),
                ("Cpk", cpk, 1.33, 1.0, "Capacidade real do processo considerando centralização"),
                ("Pp", pp, 1.33, 1.0, "Desempenho do processo"),
                ("Ppk", ppk, 1.33, 1.0, "Desempenho do processo considerando centralização"),
                ("Cmk", cmk, 1.67, 1.33, "Capacidade da máquina")
            ]
            
            for name, value, good_threshold, marginal_threshold, description in capability_metrics:
                if value >= good_threshold:
                    st.success(f"✅ **{name}: {value:.3f}** - Bom (≥ {good_threshold}) - {description}")
                elif value >= marginal_threshold:
                    st.warning(f"⚠️ **{name}: {value:.3f}** - Marginal (≥ {marginal_threshold}) - {description}")
                else:
                    st.error(f"❌ **{name}: {value:.3f}** - Ruim (< {marginal_threshold}) - {description}")
            
            # Visualização
            st.subheader("📈 Análise de Distribuição")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histograma com limites de especificação
            ax1.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            ax1.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Média: {mean_val:.3f}')
            ax1.axvline(usl, color='green', linestyle='dashed', linewidth=2, label=f'LSE: {usl}')
            ax1.axvline(lsl, color='green', linestyle='dashed', linewidth=2, label=f'LIE: {lsl}')
            
            # Adicionar curva de distribuição normal
            x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
            if HAS_SCIPY:
                y = stats.norm.pdf(x, mean_val, std_val)
                ax1.plot(x, y, 'r-', linewidth=2, label='Distribuição Normal')
            
            ax1.set_xlabel(variable)
            ax1.set_ylabel('Densidade')
            ax1.set_title(f'Distribuição de {variable} com Limites de Especificação')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de barras dos índices de capacidade
            indices = ['Cp', 'Cpk', 'Pp', 'Ppk', 'Cmk']
            values = [cp, cpk, pp, ppk, cmk]
            colors = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in values]
            
            bars = ax2.bar(indices, values, color=colors, alpha=0.7)
            ax2.axhline(y=1.33, color='red', linestyle='--', alpha=0.7, label='Mínimo Recomendado (1.33)')
            ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Mínimo Aceitável (1.0)')
            ax2.set_ylabel('Valor do Índice')
            ax2.set_title('Índices de Capacidade do Processo')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adicionar rótulos de valor nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)

    # Módulo de Análise CEP
    elif app_mode == "📈 Análise CEP":
        st.markdown('<div class="section-header">📈 Controle Estatístico de Processo (CEP)</div>', unsafe_allow_html=True)
        tracker.track_usage("Análise_CEP")
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ Nenhuma variável numérica disponível para análise CEP")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable = st.selectbox("📊 Selecionar Variável para CEP", numeric_cols)
            subgroup_size = st.slider("👥 Tamanho do Subgrupo", min_value=2, max_value=10, value=5)
        
        with col2:
            data = df[variable].dropna()
            data_min = data.min()
            data_max = data.max()
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            lsl = st.number_input("📏 LIE para CEP", value=float(data_mean - 3*data_std))
            usl = st.number_input("📏 LSE para CEP", value=float(data_mean + 3*data_std))
        
        if st.button("📊 Gerar Cartas de Controle", type="primary"):
            # Criar subgrupos
            subgroups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            subgroup_means = [np.mean(subgroup) for subgroup in subgroups if len(subgroup) == subgroup_size]
            subgroup_ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups if len(subgroup) == subgroup_size]
            
            if len(subgroup_means) == 0:
                st.error("❌ Dados insuficientes para subgrupos. Tente um tamanho de subgrupo menor.")
                return
            
            # Calcular limites de controle
            xbar_mean = np.mean(subgroup_means)
            r_mean = np.mean(subgroup_ranges)
            
            # Constantes para cartas de controle
            A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
            D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
            D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
            
            a2 = A2.get(subgroup_size, 0.577)
            d3 = D3.get(subgroup_size, 0)
            d4 = D4.get(subgroup_size, 2.114)
            
            # Limites da carta X-barra
            xbar_ucl = xbar_mean + a2 * r_mean
            xbar_lcl = xbar_mean - a2 * r_mean
            
            # Limites da carta R
            r_ucl = d4 * r_mean
            r_lcl = d3 * r_mean
            
            # Criar cartas de controle
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Carta X-barra
            ax1.plot(subgroup_means, 'bo-', markersize=4, linewidth=1)
            ax1.axhline(xbar_mean, color='green', linestyle='-', label=f'Linha Central ({xbar_mean:.3f})')
            ax1.axhline(xbar_ucl, color='red', linestyle='--', label=f'LSC ({xbar_ucl:.3f})')
            ax1.axhline(xbar_lcl, color='red', linestyle='--', label=f'LIC ({xbar_lcl:.3f})')
            ax1.set_title(f'Carta de Controle X-barra - {variable}')
            ax1.set_ylabel('Média do Subgrupo')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Carta R
            ax2.plot(subgroup_ranges, 'go-', markersize=4, linewidth=1)
            ax2.axhline(r_mean, color='green', linestyle='-', label=f'Linha Central ({r_mean:.3f})')
            ax2.axhline(r_ucl, color='red', linestyle='--', label=f'LSC ({r_ucl:.3f})')
            ax2.axhline(r_lcl, color='red', linestyle='--', label=f'LIC ({r_lcl:.3f})')
            ax2.set_title(f'Carta de Controle R - {variable}')
            ax2.set_ylabel('Amplitude do Subgrupo')
            ax2.set_xlabel('Número do Subgrupo')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Estatísticas da carta de controle
            st.subheader("📊 Estatísticas da Carta de Controle")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("X-barra dupla", f"{xbar_mean:.4f}")
            with col2:
                st.metric("R-barra", f"{r_mean:.4f}")
            with col3:
                st.metric("Sigma do Processo", f"{r_mean/d2:.4f}" if (d2 := {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}.get(subgroup_size, 2.326)) else "N/A")
            with col4:
                out_of_control = sum(1 for x in subgroup_means if x > xbar_ucl or x < xbar_lcl)
                st.metric("Pontos Fora de Controle", out_of_control)
    
    # Módulo de Análise de Defeitos
    elif app_mode == "🔍 Análise de Defeitos":
        st.markdown('<div class="section-header">🔍 Análise de Defeitos</div>', unsafe_allow_html=True)
        tracker.track_usage("Análise_Defeitos")
        
        df = st.session_state.df
        
        if 'defect' not in df.columns:
            st.error("❌ Nenhuma coluna 'defect' (defeito) encontrada no conjunto de dados")
            st.info("O conjunto de dados de amostra inclui uma coluna 'defect'. Tente carregar os dados de amostra ou certifique-se de que seu conjunto de dados tenha uma coluna indicadora de defeito.")
            return
        
        cat_cols = list(df.select_dtypes(include=['object']).columns)
        
        if not cat_cols:
            st.error("❌ Nenhuma variável categórica disponível para análise de defeitos")
            return
        
        category = st.selectbox("📂 Estratificar Defeitos Por", cat_cols)
        
        # Calcular taxas de defeito por categoria
        defect_analysis = df.groupby(category).agg({
            'defect': ['count', 'sum', 'mean']
        }).round(4)
        defect_analysis.columns = ['Total_Unidades', 'Contagem_Defeitos', 'Taxa_Defeitos']
        defect_analysis = defect_analysis.sort_values('Contagem_Defeitos', ascending=False)
        
        st.subheader("📊 Análise de Defeitos por Categoria")
        st.dataframe(defect_analysis, use_container_width=True)
        
        # Análise de Pareto
        st.subheader("📈 Análise de Pareto")
        
        # Preparar dados para gráfico de Pareto
        categories = defect_analysis.index.tolist()
        defect_counts = defect_analysis['Contagem_Defeitos'].tolist()
        cumulative_percentage = [sum(defect_counts[:i+1])/sum(defect_counts)*100 for i in range(len(defect_counts))]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Gráfico de barras para contagem de defeitos
        bars = ax1.bar(categories, defect_counts, color='skyblue', alpha=0.7, label='Contagem de Defeitos')
        ax1.set_xlabel(category)
        ax1.set_ylabel('Contagem de Defeitos', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        plt.xticks(rotation=45, ha='right')
        
        # Gráfico de linha para porcentagem acumulada
        ax2 = ax1.twinx()
        ax2.plot(categories, cumulative_percentage, 'ro-', linewidth=2, markersize=6, label='% Acumulada')
        ax2.set_ylabel('Porcentagem Acumulada', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)
        
        # Adicionar linha de 80%
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Linha de 80%')
        
        plt.title(f'Gráfico de Pareto - Defeitos por {category}')
        fig.tight_layout()
        st.pyplot(fig)
        
        # Análise da Taxa de Defeitos
        st.subheader("🎯 Principais Insights")
        
        total_defects = defect_analysis['Contagem_Defeitos'].sum()
        total_units = defect_analysis['Total_Unidades'].sum()
        overall_defect_rate = total_defects / total_units
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Defeitos", int(total_defects))
        with col2:
            st.metric("Total de Unidades", int(total_units))
        with col3:
            st.metric("Taxa Geral de Defeitos", f"{overall_defect_rate:.2%}")
        
        # Principais contribuintes
        st.write("**Principais Contribuintes para Defeitos:**")
        for i, (category_name, row) in enumerate(defect_analysis.head(5).iterrows()):
            st.write(f"{i+1}. **{category_name}**: {row['Contagem_Defeitos']} defeitos ({row['Taxa_Defeitos']:.2%} taxa)")
    
    # Módulo de Recomendador de Amostragem
    elif app_mode == "🎯 Recomendador de Amostragem":
        st.markdown('<div class="section-header">🎯 Recomendador de Método de Amostragem</div>', unsafe_allow_html=True)
        tracker.track_usage("Recomendador_Amostragem")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_type = st.radio("**Tipo de Dado**", ["Variável", "Atributo"])
            data_nature = st.selectbox("**Natureza do Dado**", 
                ["Contínuo", "Discreto", "Contínuo - Normal", "Contínuo - Não normal"])
            application = st.selectbox("**Aplicação**", 
                ["Controle de Processo", "Aceitação de Lote", "Análise de Capacidade", "Análise de Defeitos"])
        
        with col2:
            sample_size = st.slider("**Tamanho da Amostra**", min_value=5, max_value=200, value=30)
            population_size = st.slider("**Tamanho da População**", min_value=100, max_value=10000, value=1000)
            confidence_level = st.slider("**Nível de Confiança**", min_value=90, max_value=99, value=95)
        
        if st.button("🎯 Obter Recomendações de Amostragem", type="primary"):
            recommendations = recommend_sampling_method(data_type, data_nature, application)
            
            st.subheader("📋 Recomendações de Amostragem")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Cálculos adicionais
            st.subheader("📊 Detalhes do Plano de Amostragem")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sampling_fraction = sample_size / population_size
                st.metric("Fração de Amostragem", f"{sampling_fraction:.2%}")
            
            with col2:
                # Cálculo simples do tamanho da amostra
                z_score = {90: 1.645, 95: 1.96, 99: 2.576}.get(confidence_level, 1.96)
                recommended_size = int((z_score**2 * 0.5 * 0.5) / (0.05**2))  # Estimativa conservadora
                st.metric("Tamanho de Amostra Recomendado", recommended_size)
            
            with col3:
                st.metric("Nível de Confiança", f"{confidence_level}%")
            
            # Visualização da amostragem
            st.subheader("📈 Visualização da Estratégia de Amostragem")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Criar uma visualização simples da estratégia de amostragem
            methods = ['Aleatória', 'Estratificada', 'Sistemática', 'Por Conglomerados']
            suitability = [0.8, 0.9, 0.7, 0.6]  # Pontuações de adequação de exemplo
            
            bars = ax.bar(methods, suitability, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
            ax.set_ylabel('Pontuação de Adequação')
            ax.set_title('Adequação dos Métodos de Amostragem Recomendados')
            ax.set_ylim(0, 1)
            
            # Adicionar rótulos de valor nas barras
            for bar, value in zip(bars, suitability):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Módulo de Análise Avançada
    elif app_mode == "🔬 Análise Avançada":
        st.markdown('<div class="section-header">🔬 Análise Avançada</div>', unsafe_allow_html=True)
        tracker.track_usage("Análise_Avançada")
        
        df = st.session_state.df
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if 'id' not in col.lower()]
        
        if not numeric_cols:
            st.error("❌ Nenhuma variável numérica disponível para análise avançada")
            return
        
        analysis_type = st.selectbox("🔧 Selecionar Tipo de Análise", 
            ["Teste de Normalidade", "Gráfico Q-Q", "Análise de Correlação", "Comparação de Distribuições"])
        
        if analysis_type == "Teste de Normalidade":
            if not HAS_SCIPY:
                st.warning("⚠️ SciPy não disponível. Usando avaliação básica de normalidade.")
                # Implementação de avaliação básica
                variable = st.selectbox("📊 Selecionar Variável", numeric_cols)
                data = df[variable].dropna()
                
                # Verificação básica de normalidade usando assimetria e curtose
                skewness = stats.skew(data) if HAS_SCIPY else 0
                kurtosis = stats.kurtosis(data) if HAS_SCIPY else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Assimetria", f"{skewness:.4f}")
                with col2:
                    st.metric("Curtose", f"{kurtosis:.4f}")
                
                if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                    st.success("✅ Os dados parecem aproximadamente normais com base na assimetria e curtose")
                else:
                    st.warning("⚠️ Os dados podem não ser normalmente distribuídos com base na assimetria e curtose")
            else:
                # Implementação completa com SciPy
                variable = st.selectbox("📊 Selecionar Variável", numeric_cols)
                data = df[variable].dropna()
                
                if st.button("📊 Executar Teste de Normalidade"):
                    stat, p_value = stats.shapiro(data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estatística de Shapiro-Wilk", f"{stat:.4f}")
                    with col2:
                        st.metric("valor-p", f"{p_value:.4f}")
                    
                    if p_value > 0.05:
                        st.success("✅ Os dados parecem ser normalmente distribuídos (falha em rejeitar H0)")
                    else:
                        st.error("❌ Os dados não parecem ser normalmente distribuídos (rejeitar H0)")
        
        elif analysis_type == "Gráfico Q-Q":
            if not HAS_SCIPY:
                st.error("❌ O Gráfico Q-Q requer SciPy. Certifique-se de que o SciPy esteja instalado em seu ambiente.")
            else:
                variable = st.selectbox("📊 Selecionar Variável para Gráfico Q-Q", numeric_cols)
                data = df[variable].dropna()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Gráfico Q-Q
                stats.probplot(data, dist="norm", plot=ax1)
                ax1.set_title(f'Gráfico Q-Q de {variable}')
                ax1.grid(True, alpha=0.3)
                
                # Histograma
                ax2.hist(data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_title(f'Distribuição de {variable}')
                ax2.set_xlabel(variable)
                ax2.set_ylabel('Frequência')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis_type == "Análise de Correlação":
            selected_vars = st.multiselect("📊 Selecionar Variáveis para Correlação", numeric_cols, default=numeric_cols[:3])
            
            if len(selected_vars) >= 2:
                corr_matrix = df[selected_vars].corr()
                
                st.subheader("📈 Matriz de Correlação")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), 
                           use_container_width=True)
                
                # Mapa de calor da correlação
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(selected_vars)))
                ax.set_yticks(np.arange(len(selected_vars)))
                ax.set_xticklabels(selected_vars, rotation=45, ha='right')
                ax.set_yticklabels(selected_vars)
                ax.set_title('Mapa de Calor da Correlação')
                
                # Adicionar valores de correlação ao mapa de calor
                for i in range(len(selected_vars)):
                    for j in range(len(selected_vars)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontsize=12)
                
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
        
        elif analysis_type == "Comparação de Distribuições":
            selected_vars = st.multiselect("📊 Selecionar Variáveis para Comparar", numeric_cols, default=numeric_cols[:2])
            
            if len(selected_vars) >= 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for var in selected_vars:
                    data = df[var].dropna()
                    # Normalizar para comparação
                    normalized_data = (data - data.mean()) / data.std()
                    ax.hist(normalized_data, bins=20, alpha=0.6, label=var, density=True)
                
                ax.set_xlabel('Valores Padronizados')
                ax.set_ylabel('Densidade')
                ax.set_title('Comparação de Distribuições (Padronizado)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

    # Módulo de Regressão e Previsão
    elif app_mode == "📊 Regressão e Previsão":
        st.markdown('<div class="section-header">📊 Regressão e Previsão com PyCaret</div>', unsafe_allow_html=True)
        tracker.track_usage("Regressão_Previsão")
        
        if not HAS_PYCARET:
            st.error("""
            ❌ PyCaret não está disponível em seu ambiente.
            
            **Para habilitar os recursos de aprendizado de máquina e previsão:**
            
            ```bash
            pip install pycaret
            '''
            
            ***Ou atualize seu requirements.txt:***
            '''txt
            streamlit>=1.28.0
            pandas>=1.5.0
            numpy>=1.21.0
            matplotlib>=3.5.0
            scipy>=1.10.0
            pycaret>=3.0.0
            '''
            """)
            return

        df = st.session_state.df
        numeric_cols = list(df.select_dtypes(include=np.number).columns)

        st.info("""
        ***Construa modelos de aprendizado de máquina para prever resultados de qualidade e tendências futuras:
        - Compare múltiplos algoritmos automaticamente
        - Preveja pontuações de qualidade e probabilidades de defeito
        - Preveja o comportamento futuro do processo
        - Identifique fatores-chave que afetam a qualidade
        """)

        analysis_type = st.radio("Selecionar Tipo de Análise",
                                 ["Análise de Regressão", "Previsão de Séries Temporais"])

        if analysis_type == "Análise de Regressão":
            st.subheader("🔮 Modelos de Predição de Qualidade")

            col1, col2 = st.columns(2)

            with col1:
                target = st.selectbox("🎯 Variável Alvo (O que prever)", numeric_cols)

            with col2:
                available_features = [col for col in numeric_cols if col != target]
                features = st.multiselect("📊 Variáveis de Características (Preditores)",
                                          available_features,
                                          default=available_features[:3])

            if target and len(features) >= 1:
                if st.button("🚀 Construir Modelos de Predição", type="primary"):
                    with st.spinner("Treinando múltiplos modelos de aprendizado de máquina... Isso pode levar alguns minutos."):
                        try:
                            #Preparar dados
                            model_data = df[features + [target]].dropna()
                            if len(model_data) < 10:
                                st.error("❌ Dados insuficientes para treinamento do modelo. Necessário pelo menos 10 registros completos.")
                                return
                            
                            #Configurar PyCaret
                            setup_data = setup(data=model_data,
                                               target=target,
                                               session_id=42,
                                               silent=True,
                                               verbose=False)
                            #Comparar modelos
                            best_model = compare_models()
                            
                            st.success("✅ Treinamento do modelo concluído!")

                            #Exibir resultados
                            st.subheader("📊 Resultados da Comparação de Modelos")

                            #Obter resultados da comparação
                            from pycaret.regression import pull
                            results = pull()
                            st.dataframe(results.style.highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE', 'R2']))

                            #Informações do melhor modelo
                            st.subheader("🏆 Melhor Modelo")
                            st.write(f"Algoritmo: {type(best_model).name}")

                            #Importância das características
                            try:
                                from pycaret.regression import plot_model
                                fig = plot_model(best_model, plot='feature')
                                st.pyplot(fig)
                            except:
                                st.info("Gráfico de importância de características não disponível para este tipo de modelo")

                            #Fazer previsões
                            st.subheader("🔮 Fazer Previsões")
                            sample_input = {}
                            for feature in features:
                                feature_mean = df[feature].mean()
                                sample_input[feature] = st.number_input(
                                    f"Digite o valor para {feature}",
                                    value=float(feature_mean)
                                )
                            
                            if st.button("📈 Prever Valor Alvo"):
                                input_df = pd.DataFrame([sample_input])
                                prediction = predict_model(best_model, data=input_df)
                                predicted_value = prediction['prediction_label'].iloc[0]
                                
                                st.success(f"{target} previsto: {predicted_value:.3f}")
                        
                        except Exception as e:
                            st.error(f"❌ Erro no treinamento do modelo: {e}")
            
            else: # Previsão de Séries Temporais
                st.subheader("📈 Previsão de Séries Temporais")
                
                #Verificar se existe coluna de data
                date_columns = df.select_dtypes(include=['datetime64']).columns
                if len(date_columns) == 0:
                    st.warning("Nenhuma coluna de data encontrada. Usando dados de amostra com datas.")
                    df = generate_manufacturing_data()
                    st.session_state.df = df
                
                col1, col2 = st.columns(2)
                
                with col1:
                    date_col = st.selectbox("📅 Coluna de Data/Hora",
                                            df.select_dtypes(include=['datetime64']).columns.tolist() or ['date'])
                    value_col = st.selectbox("📊 Valor a Prever", numeric_cols)
                
                with col2:
                    forecast_periods = st.slider("🔮 Períodos de Previsão", 1, 365, 30)
                    model_type = st.selectbox("🤖 Tipo de Modelo", ["ARIMA", "Suavização Exponencial", "Prophet"])

                if st.button("🌐 Gerar Previsão", type="primary"):
                    with st.spinner("Construindo previsão de série temporal..."):
                        try:
                            
                            #Visualização simples de série temporal
                            if date_col in df.columns and value_col in df.columns:
                                ts_data = df[[date_col, value_col]].dropna()
                                ts_data = ts_data.sort_values(date_col)
                                
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.plot(ts_data[date_col], ts_data[value_col], 'b-', linewidth=2)
                                ax.set_xlabel('Data')
                                ax.set_ylabel(value_col)
                                ax.set_title(f'Série Temporal de {value_col}')
                                ax.grid(True, alpha=0.3)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                st.info("""
                                ***Análise de Série Temporal Concluída***

                                Para previsão avançada com o módulo de Séries Temporais do PyCaret, é necessária configuração adicional.
                                Esta visualização mostra a tendência histórica da variável selecionada.
                                """)
                            else:
                                st.error("Colunas selecionadas não encontradas no conjunto de dados")
                                
                        except Exception as e:
                            st.error(f"❌ Erro na previsão: {e}")
                            
    #Módulo de Análise de Uso
    elif app_mode == "👥 Análise de Uso":
        st.markdown('<div class="section-header">👥 Análise de Uso da Plataforma</div>', unsafe_allow_html=True)
        
        stats_data = tracker.get_stats()
        
        st.info("""
        ***Estatísticas de Uso da Plataforma***
        - Acompanhe como o painel está sendo usado
        - Monitore a popularidade das funcionalidades
        - Entenda o engajamento do usuário
        """)

        #Estatísticas Gerais
        st.subheader("📈 Estatísticas Gerais da Plataforma")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Usos", stats_data['total_uses'])
        with col2:
            st.metric("Sessões Únicas", stats_data['unique_sessions'])
        with col3:
            st.metric("Funcionalidades Ativas", len(stats_data['feature_usage']))
        with col4:
            avg_uses_per_session = stats_data['total_uses'] / max(1, stats_data['unique_sessions'])
            st.metric("Média de Usos/Sessão", f"{avg_uses_per_session:.1f}")
        
        #Uso de Funcionalidades
        st.subheader("🔥 Popularidade das Funcionalidades")
        
        if stats_data['feature_usage']:
            
            #Criar gráfico de uso de funcionalidades
            features = list(stats_data['feature_usage'].keys())
            usage_counts = list(stats_data['feature_usage'].values())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(features))
            
            bars = ax.barh(y_pos, usage_counts, color='skyblue', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Contagem de Uso')
            ax.set_title('Distribuição de Uso das Funcionalidades')
            ax.grid(True, alpha=0.3, axis='x')
            
            #Adicionar rótulos de valor
            for i, v in enumerate(usage_counts):
                ax.text(v + 0.1, i, str(v), va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            #Tabela de uso de funcionalidades
            usage_df = pd.DataFrame({
                'Funcionalidade': features,
                'Contagem de Uso': usage_counts,
                'Porcentagem': [f"{(count/stats_data['total_uses'])*100:.1f}%" for count in usage_counts]
            }).sort_values('Contagem de Uso', ascending=False)
            
            st.dataframe(usage_df, use_container_width=True)
        else:
            st.info("Nenhum dado de uso de funcionalidade disponível ainda. Comece a usar o painel para ver as análises!")
        
        #Insights de Engajamento do Usuário
        st.subheader("💡 Insights de Engajamento")
        
        if stats_data['total_uses'] > 0:
            most_used_feature = max(stats_data['feature_usage'].items(), key=lambda x: x[1]) if stats_data['feature_usage'] else ("Nenhum", 0)
            engagement_rate = (stats_data['unique_sessions'] / max(1, stats_data['total_uses'])) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Funcionalidade Mais Popular", most_used_feature[0])
                st.metric("Taxa de Engajamento", f"{engagement_rate:.1f}%")
            with col2:
                st.metric("Total de Análises Rastreadas", stats_data['total_uses'])
                st.metric("Saúde da Plataforma", "✅ Excelente" if stats_data['total_uses'] > 10 else "🟡 Boa")
            
        #Redefinir análises (para teste)
        st.markdown("---")
        st.subheader("🛠️ Gerenciamento de Análises")
        
        if st.button("🔄 Redefinir Análises de Uso", type="secondary"):
            st.session_state.usage_count = 0
            st.session_state.user_sessions = set()
            st.session_state.feature_usage = {}
            st.success("Análises de uso redefinidas com sucesso!")
            st.experimental_rerun()
        
    #Executar a aplicação
    if name == "main":
        main()