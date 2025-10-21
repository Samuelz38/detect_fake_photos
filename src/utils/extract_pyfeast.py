import os
import cv2
import pandas as pd
import numpy as np
from pyfeats import fos, glcm_features
from PIL import Image
from datetime import datetime
from .util import preprocessar_screeshot
from scipy.stats import entropy
import tempfile

def extrair_todas_features(diretorio_base='dataset', csv_saida='features_completas.csv'):
    """
    Extrai TODAS as features (PyFeats + geométricas + cor + textura + metadados + compressão)
    e salva em um único CSV.
    """
    pastas_e_rotulos = {
        os.path.join(diretorio_base, 'manipulados'): 'manipulado',
        os.path.join(diretorio_base, 'autenticos'): 'autentico'
    }

    lista_de_caracteristicas = []
    imagens_processadas_sucesso = 0

    print(f"\nIniciando o processamento...")

    # =================================================================
    # Percorrer as pastas e processar as imagens
    # =================================================================
    for caminho_pasta, rotulo in pastas_e_rotulos.items():
        if not os.path.exists(caminho_pasta):
            print(f"Aviso: Diretório {caminho_pasta} não encontrado. Pulando.")
            continue

        for nome_arquivo in os.listdir(caminho_pasta):
            if not nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            caminho_completo = os.path.join(caminho_pasta, nome_arquivo)

            try:
                imagem = cv2.imread(caminho_completo)
                if imagem is None:
                    raise FileNotFoundError("Imagem não foi lida corretamente.")

                # ======================
                # PRÉ-PROCESSAMENTO
                # ======================
            
                imagem = preprocessar_screeshot(imagem)

                # ======================
                # PYFEATS 
                # ======================
                mascara = np.ones(imagem.shape, dtype=np.uint8)
                fos_features, fos_labels = fos(imagem, mascara)
                glcm_mean, _, glcm_labels_mean, _ = glcm_features(f=imagem, ignore_zeros=True)

                resultados_img = {
                    'nome_arquivo': nome_arquivo,
                    'rotulo': rotulo
                }

                for label, feature in zip(fos_labels, fos_features):
                    resultados_img[f'FOS_{label}'] = feature

                for label, feature in zip(glcm_labels_mean, glcm_mean):
                    resultados_img[f'GLCM_{label}'] = feature

                # =================================================================
                # OUTRAS FEATURES (GEOMETRIA, COR, TEXTURA, METADADOS, COMPRESSÃO)
                # =================================================================
                img_color = cv2.imread(caminho_completo)
                if img_color is not None:
                    altura, largura, canais = img_color.shape

                    # Geometria
                    resultados_img.update({
                        'geo_largura': largura,
                        'geo_altura': altura,
                        'geo_area_pixels': largura * altura,
                        'geo_aspect_ratio': largura / altura if altura > 0 else 0,
                        'geo_diagonal': np.sqrt(largura**2 + altura**2),
                        'geo_num_canais': canais,
                        'geo_perimetro': 2 * (largura + altura),
                        'geo_compacidade': (4 * np.pi * (largura * altura)) / ((2 * (largura + altura))**2),
                    })

                    # Cor
                    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                    for i, cor in enumerate(['R', 'G', 'B']):
                        canal = img_rgb[:, :, i]
                        resultados_img[f'cor_{cor}_media'] = np.mean(canal)
                        resultados_img[f'cor_{cor}_std'] = np.std(canal)
                        resultados_img[f'cor_{cor}_min'] = np.min(canal)
                        resultados_img[f'cor_{cor}_max'] = np.max(canal)
                        resultados_img[f'cor_{cor}_mediana'] = np.median(canal)

                    for i, comp in enumerate(['H', 'S', 'V']):
                        canal = img_hsv[:, :, i]
                        resultados_img[f'cor_{comp}_media'] = np.mean(canal)
                        resultados_img[f'cor_{comp}_std'] = np.std(canal)

                    resultados_img['cor_brilho_geral'] = np.mean(img_gray)
                    resultados_img['cor_contraste_geral'] = np.std(img_gray)

                    # Textura simples
                    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
                    gradiente = np.sqrt(sobelx**2 + sobely**2)
                    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

                    hist, _ = np.histogram(img_gray.flatten(), bins=256, range=[0, 256])
                    hist = hist / hist.sum()

                    resultados_img.update({
                        'tex_gradiente_medio': np.mean(gradiente),
                        'tex_gradiente_std': np.std(gradiente),
                        'tex_gradiente_max': np.max(gradiente),
                        'tex_laplacian_media': np.mean(np.abs(laplacian)),
                        'tex_laplacian_std': np.std(laplacian),
                        'tex_entropia': entropy(hist),
                        'tex_suavidade': 1 - (1 / (1 + np.var(img_gray))),
                        'tex_uniformidade': np.sum(hist**2)
                    })

                # Metadados do arquivo
                try:
                    stat_info = os.stat(caminho_completo)
                    data_mod = datetime.fromtimestamp(stat_info.st_mtime)
                    resultados_img.update({
                        'meta_tamanho_bytes': stat_info.st_size,
                        'meta_tamanho_kb': stat_info.st_size / 1024,
                        'meta_ano_modificacao': data_mod.year,
                        'meta_mes_modificacao': data_mod.month,
                        'meta_dia_semana': data_mod.weekday()
                    })
                    try:
                        img_pil = Image.open(caminho_completo)
                        exif_data = img_pil._getexif()
                        resultados_img['meta_tem_exif'] = 1 if exif_data else 0
                    except:
                        resultados_img['meta_tem_exif'] = 0
                except Exception as e:
                    print(f"Erro metadados: {e}")

                # Compressão / artefatos
                blocos_8x8 = []
                h, w = imagem.shape
                for i in range(0, h - 8, 8):
                    for j in range(0, w - 8, 8):
                        bloco = imagem[i:i + 8, j:j + 8]
                        blocos_8x8.append(np.std(bloco))

                if blocos_8x8:
                    resultados_img['comp_variancia_blocos'] = np.std(blocos_8x8)
                    resultados_img['comp_media_blocos'] = np.mean(blocos_8x8)

                dct = cv2.dct(np.float32(imagem))
                resultados_img['comp_energia_dct'] = np.sum(dct ** 2)
                resultados_img['comp_coef_dc'] = dct[0, 0]

                lista_de_caracteristicas.append(resultados_img)
                imagens_processadas_sucesso += 1

            except Exception as e:
                print(f"Erro ao processar {nome_arquivo}: {e}")

    # =================================================================
    # Salvar CSV
    # =================================================================
    if lista_de_caracteristicas:
        df = pd.DataFrame(lista_de_caracteristicas)
        df.to_csv(csv_saida, index=False)
        print(f"\n✅ {imagens_processadas_sucesso} imagens processadas com sucesso.")
        print(f"CSV salvo em: {csv_saida}")
        return df
    else:
        print("\n❌ Nenhuma imagem processada.")
        return None


def extrair_features_streamlit(imagem_input, rotulo='desconhecido', csv_saida='screenshot_validation-master/data/features_imagem.csv'):
    """
    Extrai TODAS as features (PyFeats + geométricas + cor + textura + metadados + compressão)
    de UMA imagem (vinda de st.file_uploader ou caminho local) e salva em um CSV.

    Parâmetros:
        imagem_input (str | UploadedFile): Caminho da imagem ou arquivo carregado via Streamlit.
        rotulo (str): Classe associada (ex: 'autentico', 'manipulado').
        csv_saida (str): Nome do arquivo CSV de saída.

    Retorna:
        pd.DataFrame com todas as features extraídas.
    """

    # ==========================================================
    # TRATAR ENTRADA (streamlit ou caminho local)
    # ==========================================================
    if hasattr(imagem_input, 'read'):  # caso seja um UploadedFile do Streamlit
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp.write(imagem_input.read())
        temp_path = temp.name
        temp.close()
        caminho_imagem = temp_path
        nome_arquivo = imagem_input.name
    else:
        caminho_imagem = imagem_input
        nome_arquivo = os.path.basename(caminho_imagem)

    try:
        # ==========================================================
        # LEITURA E PRÉ-PROCESSAMENTO
        # ==========================================================
        print("---------")
        imagem = cv2.imread(temp_path)
        if imagem is None:
            raise FileNotFoundError("Imagem não foi lida corretamente (None).")
        print("--------")
    
        imagem, mascara = preprocessar_screeshot(imagem)

        resultados_img = {
            'nome_arquivo': nome_arquivo,
            'rotulo': rotulo
        }

        # ==========================================================
        # PYFEATS (FOS + GLCM)
        # ==========================================================
        mascara2 = np.ones(imagem.shape, dtype=np.uint8)

        fos_features, fos_labels = fos(imagem, mascara)
        glcm_mean, _, glcm_labels_mean, _ = glcm_features(f=imagem, ignore_zeros=True)

        for label, feature in zip(fos_labels, fos_features):
            resultados_img[f'FOS_{label}'] = feature

        for label, feature in zip(glcm_labels_mean, glcm_mean):
            resultados_img[f'GLCM_{label}'] = feature

        # ==========================================================
        # OUTRAS FEATURES (GEOMETRIA, COR, TEXTURA, METADADOS, COMPRESSÃO)
        # ==========================================================
        img_color = cv2.imread(caminho_imagem)
        if img_color is not None:
            altura, largura, canais = img_color.shape

            # Geometria
            resultados_img.update({
                'geo_largura': largura,
                'geo_altura': altura,
                'geo_area_pixels': largura * altura,
                'geo_aspect_ratio': largura / altura if altura > 0 else 0,
                'geo_diagonal': np.sqrt(largura**2 + altura**2),
                'geo_num_canais': canais,
                'geo_perimetro': 2 * (largura + altura),
                'geo_compacidade': (4 * np.pi * (largura * altura)) / ((2 * (largura + altura))**2),
            })

            # Cor
            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            for i, cor in enumerate(['R', 'G', 'B']):
                canal = img_rgb[:, :, i]
                resultados_img[f'cor_{cor}_media'] = np.mean(canal)
                resultados_img[f'cor_{cor}_std'] = np.std(canal)
                resultados_img[f'cor_{cor}_min'] = np.min(canal)
                resultados_img[f'cor_{cor}_max'] = np.max(canal)
                resultados_img[f'cor_{cor}_mediana'] = np.median(canal)

            for i, comp in enumerate(['H', 'S', 'V']):
                canal = img_hsv[:, :, i]
                resultados_img[f'cor_{comp}_media'] = np.mean(canal)
                resultados_img[f'cor_{comp}_std'] = np.std(canal)

            resultados_img['cor_brilho_geral'] = np.mean(img_gray)
            resultados_img['cor_contraste_geral'] = np.std(img_gray)

            # Textura simples
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradiente = np.sqrt(sobelx**2 + sobely**2)
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

            hist, _ = np.histogram(img_gray.flatten(), bins=256, range=[0, 256])
            hist = hist / hist.sum()

            resultados_img.update({
                'tex_gradiente_medio': np.mean(gradiente),
                'tex_gradiente_std': np.std(gradiente),
                'tex_gradiente_max': np.max(gradiente),
                'tex_laplacian_media': np.mean(np.abs(laplacian)),
                'tex_laplacian_std': np.std(laplacian),
                'tex_entropia': entropy(hist),
                'tex_suavidade': 1 - (1 / (1 + np.var(img_gray))),
                'tex_uniformidade': np.sum(hist**2)
            })

        # Metadados
        try:
            stat_info = os.stat(caminho_imagem)
            data_mod = datetime.fromtimestamp(stat_info.st_mtime)
            resultados_img.update({
                'meta_tamanho_bytes': stat_info.st_size,
                'meta_tamanho_kb': stat_info.st_size / 1024,
                'meta_ano_modificacao': data_mod.year,
                'meta_mes_modificacao': data_mod.month,
                'meta_dia_semana': data_mod.weekday()
            })
            try:
                img_pil = Image.open(caminho_imagem)
                exif_data = img_pil._getexif()
                resultados_img['meta_tem_exif'] = 1 if exif_data else 0
            except:
                resultados_img['meta_tem_exif'] = 0
        except Exception as e:
            print(f"Erro metadados: {e}")

        # Compressão / artefatos
        blocos_8x8 = []
        h, w = imagem.shape
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                bloco = imagem[i:i + 8, j:j + 8]
                blocos_8x8.append(np.std(bloco))

        if blocos_8x8:
            resultados_img['comp_variancia_blocos'] = np.std(blocos_8x8)
            resultados_img['comp_media_blocos'] = np.mean(blocos_8x8)

        dct = cv2.dct(np.float32(imagem))
        resultados_img['comp_energia_dct'] = np.sum(dct ** 2)
        resultados_img['comp_coef_dc'] = dct[0, 0]

        # ==========================================================
        # SALVAR CSV
        # ==========================================================
        df = pd.DataFrame([resultados_img])
        df.to_csv(csv_saida, index=False)

        print(f"\n✅ Features extraídas e salvas em '{csv_saida}'")
        return df

    except Exception as e:
        print(f"❌ Erro ao processar imagem: {e}")
        return None

    finally:
        # Se for arquivo temporário, remover
        if hasattr(imagem_input, 'read'):
            os.remove(temp_path)



