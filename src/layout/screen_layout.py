import streamlit as st
import joblib
import pandas as pd
from src.utils.extract_pyfeast import extrair_features_streamlit


MODEL_PATH = 'screenshot_validation-master/models/decision_tree_preprocessing.joblib'

def predict(model_filepath : str, data_filepath : str) -> str:

    """Rotula uma imagem, a partir de seus descritores, em manipulado ou autêntico

    Parameters
    ----------
    model_filepath : str
        Caminho do arquivo do modelo
    
    data_filepath : str
        Caminho com os dados dos descritores da imagem. Deve ter extensão 'csv'.
    
    Returns
    -------
    str
        Rótulo da imagem em 'Manipulado' ou 'Autêntico'
    """

    model = joblib.load(model_filepath)
    data = pd.read_csv(data_filepath)

    # Features categóricas que o modelo foi treinado
    categoric_columns = [
        'FOS_FOS_Median',
        'FOS_FOS_Mode',
        'FOS_FOS_MinimalGrayLevel',
        'FOS_FOS_10Percentile',
        'FOS_FOS_25Percentile',
        'FOS_FOS_75Percentile',
        'FOS_FOS_90Percentile',
        'FOS_FOS_HistogramWidth',
        'cor_R_mediana',
        'cor_G_min',
        'cor_G_mediana',
        'cor_B_min',
        'cor_B_mediana',
    ]

    # Features numéricas que o modelo foi treinado
    numeric_columns = [
        'FOS_FOS_Variance',
        'GLCM_GLCM_ASM_Mean',
        'tex_gradiente_std',
        'tex_gradiente_max',
        'tex_laplacian_media',
        'tex_suavidade',
        'comp_variancia_blocos',
        'comp_media_blocos'
        ]

    # Remove colunas desnecessárias
    data = data.drop(["nome_arquivo", "rotulo"], axis=1)

    # Agrega as colunas categóricas em apenas duas categorias: zero e non_zero
    for column in categoric_columns:

        data[column] = data[column].apply(lambda x: "zero" if x == 0 else "non_zero")

    # Seleciona apenas as features que foram treinadas
    data = data.filter(numeric_columns + categoric_columns)

    return "Manipulado" if model.predict(data)[0] == 1 else "Autêntico"

# Carregamento Otimizado do Modelo
@st.cache_resource
def load_model():
    """Carrega o pipeline (pré-processamento + modelo)."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Verifique o caminho. Erro: {e}")
        st.stop()


def render_result_page():
    st.title('🛡️ Classificação de Screenshots')
    st.write('Aplicativo para predição de manipulação digital usando um modelo Decision Tree.')
    
    # Widget de upload
    arquivo = st.file_uploader("📤 Envie uma imagem (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    
    if arquivo is not None:
        # Exibe a imagem 
        st.image(arquivo, caption="Imagem enviada", width='content')
        
        # Botão para iniciar a análise
        if st.button("Executar Análise", type="primary", use_container_width=True):
            with st.spinner("Processando e extraindo features..."):
                try:
                    # 1. CARREGA O PIPELINE
                    #  pipeline = load_model()
                    
                    # 2. EXTRAI AS FEATURES DA IMAGEM CARREGADA
                    # IMPORTANTE: A função deve retornar um DataFrame (ou array) 
                    # com as features prontas para o predict.
                    extrair_features_streamlit(imagem_input=arquivo) 
                    
                    features_df = pd.read_csv('screenshot_validation-master/data/features_imagem.csv')

                    print(features_df.isnull().any(axis=1))

                    # 3. FAZ A PREVISÃO (Corrigido o input)
                    # previsao = pipeline.predict(features_df)[0]
                    previsao = predict(MODEL_PATH, 'screenshot_validation-master/data/features_imagem.csv')
                    print("--------")
                    # Opcional: Obtém probabilidades para melhor diagnóstico
                    # probabilidades = pipeline.predict_proba(features_df)[0]
                    
                    # 4. EXIBE RESULTADOS
                    st.success('Resultado da Classificação')
                    
                    if previsao == "Manipulada": # Assumindo 1 = Manipulada
                        st.error(f'⚠️ **A Screenshot é Provavelmente: {previsao}**')
                    else: # Assumindo 0 = Real
                        st.balloons()
                        st.success(f'✅ **A Screenshot é Provavelmente: {previsao}**')
                        
                    #st.write(f'Probabilidade (Real): **{probabilidades[0]*100:.2f}%**')
                    #st.write(f'Probabilidade (Manipulada): **{probabilidades[1]*100:.2f}%**')
                    
                    st.subheader('Features Enviadas ao Modelo')
                    st.dataframe(features_df)

                except Exception as e:
                    # Melhor feedback em caso de falha
                    st.error("❌ Ocorreu um erro durante a análise das features ou previsão.")
                    st.code(f"Detalhes do Erro: {e}")

