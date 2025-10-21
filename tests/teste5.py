import os
import cv2
import pandas as pd
import numpy as np
from pyfeats import fos, glcm_features, glds_features, fps
from PIL import Image
from datetime import datetime
from scipy.stats import entropy

def preprocessar_screeshot(imagem):
    """
    Pré-processamento leve para capturas de tela (screenshots):
    - Redimensiona para padronizar entradas
    - Suaviza artefatos de compressão com filtro bilateral
    - Normaliza intensidades (0–255)
    - Equaliza histograma de forma global (para contraste geral)
    """
    
    # Converte para gray_scale
    gray_scale = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)    

    # Redimensiona (mantendo formato quadrado)
    imagem = cv2.resize(gray_scale, (256, 256), interpolation=cv2.INTER_AREA)

    # Filtro bilateral preserva bordas e suaviza compressão
    imagem = cv2.bilateralFilter(imagem, d=5, sigmaColor=50, sigmaSpace=50)

    # Normalização linear (ajuste de brilho/contraste geral)
    imagem = cv2.normalize(imagem, None, 0, 255, cv2.NORM_MINMAX)

    # Equalização de histograma (somente se contraste for baixo)
    if np.std(imagem) < 40:  # Evita supercontraste em imagens já fortes
        imagem = cv2.equalizeHist(imagem)



    _, mascara = cv2.threshold(imagem, 150, 255, cv2.THRESH_OTSU)

    return imagem, mascara 



def extrair_todas_features(diretorio_base='capturas-de-tela-main/capturas-de-tela-main/dataset', csv_saida='features_completas.csv'):
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
                
                imagem,mascara = preprocessar_screeshot(imagem)

                # ======================
                # PYFEATS 
                # ======================
                
                
                fos_features, fos_labels = fos(imagem, mascara)
                glcm_mean, _, glcm_labels_mean, _ = glcm_features(f=imagem, ignore_zeros=True)
                gls_features, gls_labels = glds_features(imagem, mascara, Dx=[0,1,1,1], Dy=[1,1,0,-1])
                fps_features, fps_labels = fps(imagem, mascara)

                resultados_img = {
                    'nome_arquivo': nome_arquivo,
                    'rotulo': rotulo
                }

                for label, feature in zip(fos_labels, fos_features):
                    resultados_img[f'FOS_{label}'] = feature

                for label, feature in zip(glcm_labels_mean, glcm_mean):
                    resultados_img[f'GLCM_{label}'] = feature

                for label, feature in zip(gls_labels, gls_features):
                    resultados_img[f'GLS_{label}'] = feature
                
                for label, feature in zip(fps_labels,fps_features):
                    resultados_img[f'FPS_{label}'] = feature
                
                # =================================================================
                # OUTRAS FEATURES (GEOMETRIA, METADADOS)
                # =================================================================
                
                # Cor
                    img_color = cv2.imread(caminho_completo)
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

extrair_todas_features()