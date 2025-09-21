import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import time


def draw_flow(img,flow,step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x,y,x-fx,y-fy]).T.reshape(-1,2,2)
    lines = np.int32(lines + 0.5)

    img_bar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bar,lines,0,(0,255,0))

    for (x1,y1), (_x2, _y2) in lines:
        cv2.circle(img_bar, (x1,y1), 1, (0,255,0), -1)
    
    return img_bar

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]

    # Ângulo em radianos -> [0, 2π]
    ang = np.arctan2(fy, fx) + np.pi

    # Magnitude
    v = np.sqrt(fx * fx + fy * fy)

    # Criar imagem HSV
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2   # Hue (direção)
    hsv[..., 1] = 255                     # Saturação
    hsv[..., 2] = np.minimum(v * 4, 255)    # Magnitude -> Value

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def covert_to_rgb(img):
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_image(img,option, K=10, limite_desvio=2.0):
    match option:
        case 'k-means':
            img = cv2.resize(img, (800, 600))
            Z = img.reshape((-1, 3))
            Z = np.float32(Z)
            
            # Critérios de parada
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # K-Means
            ret_kmeans, label, center = cv2.kmeans(
                Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            center = np.uint8(center)
            res = center[label.flatten()]
            img_kmeans = res.reshape((img.shape))

            return img_kmeans
    
        case 'analise ruido':
        
                 # Converter para escala de cinza
            cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calcular a média e desvio padrão local usando um kernel
            kernel_size = 15  # Tamanho da vizinhança para análise
            media_local = cv2.blur(cinza, (kernel_size, kernel_size))
            
            # Calcular o quadrado das diferenças para a variância
            diferenca_quad = (cinza.astype(np.float32) - media_local.astype(np.float32)) ** 2
            variancia_local = cv2.blur(diferenca_quad, (kernel_size, kernel_size))
            desvio_local = np.sqrt(variancia_local)
            
            # Identificar pixels anômalos (aqueles que estão muito longe da média local)
            limiar_anomalo = limite_desvio * np.mean(desvio_local)
            mascara_anomalos = (desvio_local > limiar_anomalo).astype(np.uint8) * 255
            
            # Calcular percentual de pixels anômalos
            total_pixels = cinza.size
            pixels_anomalos = np.sum(mascara_anomalos > 0)
            percentual_anomalos = (pixels_anomalos / total_pixels) * 100
            
            # Destacar pixels anômalos na imagem original (em vermelho)
            img_com_marcacoes = img.copy()
            img_com_marcacoes[mascara_anomalos > 0] = [0, 0, 255]  # Vermelho BGR
            
            return img_com_marcacoes, mascara_anomalos, percentual_anomalos



def process_video(data,option,codec='mp4v'):
    
    tfile = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
    tfile.write(data.read())
    tpm_path = tfile.name
    tfile.close()
    
    video = cv2.VideoCapture(tfile.name)
    ret, frame = video.read()

    match option:
        case 'Análise de Metadados':
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = int(video.get(cv2.CAP_PROP_FOURCC))
            
            return frame,fps,frame_count,width,height,codec
        
        case 'Bordas (Canny)':
                edge_inconsistencies = []
                for i in range(0, 100, 10):
                    video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = video.read()
                    
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        edges = cv2.Canny(gray, 100, 200)
                        
                        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
                        edge_inconsistencies.append(edge_density)
                
                video.release()
                
                inconsistency = np.var(edge_inconsistencies)
                return inconsistency

        case 'Análise de Inconsistências de Compressão':
            compression_artifacts = []
            for i in range(0, 100, 10):
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    artifact_score = np.var(laplacian)
                
                    compression_artifacts.append(artifact_score)
        
            video.release()
            inconsistency = np.var(compression_artifacts)
            return inconsistency

        case 'Análise de Padrões de Ruído':
                noise_patterns = []
    
                for i in range(0, 100, 10):  
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        noise = gray - blurred
                        
                
                        noise_variance = np.var(noise)
                        noise_patterns.append(noise_variance)
                
                video.release()
                
            
                inconsistency = np.var(noise_patterns)
                print(f"Inconsistência de ruído: {inconsistency:.6f}")
                
                return inconsistency
    

        case 'Optical Analise':
            fps = video.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(data, fourcc, fps, (w, h))

            
            if not suc:
                video.release()
                out.release()
                raise RuntimeError("Vídeo vazio")

            prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            while True:
                suc, img = video.read()
                if not suc:
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # calcula fluxo Farneback (ajuste parâmetros se precisar)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                prevgray = gray

                # desenha resultado (por exemplo: draw_flow ou draw_hsv)
                # aqui suponho que draw_flow/ draw_hsv retornam BGR com mesmo tamanho (w,h)
                vis = draw_flow(gray, flow)          # ou draw_hsv(flow)

                # garante tipo e tamanho corretos
                if vis.shape[1] != w or vis.shape[0] != h:
                    vis = cv2.resize(vis, (w, h))

                out.write(vis)

            video.release()
            out.release()
            return tpm_path

