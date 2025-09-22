import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

# ---- CONFIGURAÇÃO ----
VIDEO1_PATH = "video.mp4"   # coloque o caminho do primeiro vídeo aqui
VIDEO2_PATH = "videofake.mp4"   # coloque o caminho do segundo vídeo aqui
TARGET_WIDTH = 640   # largura para redimensionar (ou None para usar original)
OUT_DIR = "data"

FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# ----------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def gray_frame_from_capture(cap, target_width=None):
    ret, frame = cap.read()
    if not ret:
        return None
    if target_width is not None:
        h, w = frame.shape[:2]
        if w != target_width:
            new_h = int(h * (target_width / w))
            frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def compute_flow_sequence(video_path, target_width=None, fb_params=FB_PARAMS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erro ao abrir vídeo: {video_path}")

    flows = []
    prev = gray_frame_from_capture(cap, target_width)
    if prev is None:
        cap.release()
        return flows, 0

    frame_count = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    pbar = tqdm(total=total_frames, desc=f"Processando {os.path.basename(video_path)}")
    while True:
        curr = gray_frame_from_capture(cap, target_width)
        if curr is None:
            break
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            fb_params['pyr_scale'],
                                            fb_params['levels'],
                                            fb_params['winsize'],
                                            fb_params['iterations'],
                                            fb_params['poly_n'],
                                            fb_params['poly_sigma'],
                                            fb_params['flags'])
        flows.append(flow)
        prev = curr
        frame_count += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return flows, frame_count

def framewise_inconsistency(flows1, flows2):
    n = min(len(flows1), len(flows2))
    metrics = []
    heatmaps = []
    for i in range(n):
        diff = flows1[i] - flows2[i]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        metrics.append(np.mean(dist))
        heatmaps.append(dist)
    return np.array(metrics), heatmaps

def save_metric_plot(metrics, out_path, fps=None):
    plt.figure(figsize=(10,4))
    x = np.arange(len(metrics)) / fps if fps is not None else np.arange(len(metrics))
    plt.plot(x, metrics, linewidth=1.2)
    plt.xlabel("Tempo (s)" if fps else "Quadro")
    plt.ylabel("Inconsistência média (distância euclidiana)")
    plt.title("Inconsistência quadro a quadro")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_heatmap(heatmap, out_path, cmap='hot', vmin=None, vmax=None):
    plt.figure(figsize=(6,4))
    plt.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="distância")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    ensure_dir(OUT_DIR)
    
    flows1, _ = compute_flow_sequence(VIDEO1_PATH, TARGET_WIDTH, FB_PARAMS)
    flows2, _ = compute_flow_sequence(VIDEO2_PATH, TARGET_WIDTH, FB_PARAMS)

    metrics, heatmaps = framewise_inconsistency(flows1, flows2)

    # salvar CSV
    df = pd.DataFrame({"frame": np.arange(len(metrics)), "inconsistencia": metrics})
    df.to_csv(os.path.join(OUT_DIR, "metricas.csv"), index=False)

    # salvar gráfico
    cap = cv2.VideoCapture(VIDEO1_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    save_metric_plot(metrics, os.path.join(OUT_DIR, "grafico_inconsistencia.png"), fps if fps > 0 else None)

    # salvar alguns heatmaps
    heatmap_dir = os.path.join(OUT_DIR, "heatmaps")
    ensure_dir(heatmap_dir)
    vmin, vmax = np.percentile(metrics, 5), np.percentile(metrics, 95)
    for i in range(0, len(heatmaps), max(1, len(heatmaps)//10)):
        save_heatmap(heatmaps[i], os.path.join(heatmap_dir, f"heatmap_{i:04d}.png"))

    print(f"✅ Finalizado! Resultados em: {OUT_DIR}")

if __name__ == "__main__":
    main()
