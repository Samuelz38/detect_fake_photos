# -*- coding: utf-8 -*-
"""
Visualização ao vivo do Fluxo Óptico (Farnebäck) para vídeo com REAL (esq.) e DEEPFAKE (dir.) no mesmo frame.
- Calcula fluxo separadamente em cada metade para evitar "vazamento" no corte central.
- Mostra modos: original, overlay colorido do fluxo, vetores (quiver) e heatmap de magnitude.

Teclas:
  m: alterna modo de visualização
  espaço: pausa/continua
  s: salva snapshot
  q: sai
"""

import cv2
import numpy as np
import os, time

# ========== CONFIG ==========
VIDEO_PATH = r"./Anderson Cooper [1].mp4"  # <-- ajuste aqui
DISPLAY_MAX_WIDTH = 1280                  # redimensiona p/ caber na tela (0 = não redimensiona)
FLOW_DOWNSCALE = 0.5                      # calcula fluxo numa escala menor p/ ganho de desempenho
ALPHA_OVERLAY = 0.55                      # transparência do overlay colorido
VECTOR_STEP = 16                          # espaçamento entre vetores (px) no modo "Vetores"
VECTOR_SCALE = 2.0                        # escala do tamanho dos vetores desenhados

# ========== MODOS ==========
MODES = ["Original", "Overlay", "Vetores", "Heatmap"]
mode_idx = 1  # começa em "Overlay"

def compute_flow(prev_bgr, curr_bgr):
    """Calcula fluxo Farnebäck + artefatos de visualização (mapa HSV e magnitude)."""
    # opcionalmente reduz para acelerar
    if FLOW_DOWNSCALE != 1.0:
        prev_bgr_small = cv2.resize(prev_bgr, None, fx=FLOW_DOWNSCALE, fy=FLOW_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
        curr_bgr_small = cv2.resize(curr_bgr, None, fx=FLOW_DOWNSCALE, fy=FLOW_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
    else:
        prev_bgr_small = prev_bgr
        curr_bgr_small = curr_bgr

    prev_g = cv2.cvtColor(prev_bgr_small, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr_small, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3,
        winsize=25, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )
    fx, fy = flow[...,0], flow[...,1]
    mag, ang = cv2.cartToPolar(fx, fy)

    # HSV: H=direção, S=255, V=mag normalizada (0..255)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv = np.zeros((*ang.shape, 3), dtype=np.uint8)
    hsv[...,0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # 0..2π -> 0..180
    hsv[...,1] = 255
    hsv[...,2] = mag_u8
    flow_bgr_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # reescala para o tamanho original do half-frame
    flow_bgr = cv2.resize(flow_bgr_small, (prev_bgr.shape[1], prev_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    mag_big  = cv2.resize(mag,            (prev_bgr.shape[1], prev_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    return flow_bgr, mag_big, fx, fy, flow_bgr_small, mag  # retorna também versões small pra vetores

def draw_vectors(img_bgr, fx_small, fy_small, color=(255,255,255)):
    """Desenha um campo de vetores (setinhas) amostrando um grid."""
    h_s, w_s = fx_small.shape
    step = max(8, int(VECTOR_STEP * FLOW_DOWNSCALE))  # mantém densidade parecida após downscale
    for y in range(step//2, h_s, step):
        for x in range(step//2, w_s, step):
            dx = fx_small[y, x]
            dy = fy_small[y, x]
            # ponto origem na imagem grande
            X = int(x / FLOW_DOWNSCALE)
            Y = int(y / FLOW_DOWNSCALE)
            # destino (escala do vetor)
            X2 = int(X + VECTOR_SCALE * dx / FLOW_DOWNSCALE)
            Y2 = int(Y + VECTOR_SCALE * dy / FLOW_DOWNSCALE)
            cv2.arrowedLine(img_bgr, (X, Y), (X2, Y2), color, 1, tipLength=0.3)

def put_label(img, text, org=(10, 28), color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def main():
    global mode_idx

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERRO] Arquivo não encontrado: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERRO] Não consegui abrir o vídeo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = 1

    ok, prev = cap.read()
    if not ok:
        print("[ERRO] Vídeo vazio.")
        return

    # redimensiona para tela se necessário
    if DISPLAY_MAX_WIDTH and prev.shape[1] > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / prev.shape[1]
        prev = cv2.resize(prev, (DISPLAY_MAX_WIDTH, int(prev.shape[0]*scale)))

    win_name = "Fluxo Óptico — Esquerda: REAL | Direita: DEEPFAKE"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    paused = False
    frame_count = 0
    snap_idx = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if DISPLAY_MAX_WIDTH and frame.shape[1] > DISPLAY_MAX_WIDTH:
                scale = DISPLAY_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(frame.shape[0]*scale)))

            H, W = frame.shape[:2]
            mid = W // 2

            # halves
            prevL, prevR = prev[:, :mid], prev[:, mid:]
            currL, currR = frame[:, :mid], frame[:, mid:]

            # calcula fluxo por metade
            flowL_bgr, magL, fxL_big, fyL_big, flowL_small, magL_small = compute_flow(prevL, currL)
            flowR_bgr, magR, fxR_big, fyR_big, flowR_small, magR_small = compute_flow(prevR, currR)

            # métricas simples por lado (usando float64 p/ evitar erros de tipo)
            mL = float(np.asarray(magL, dtype=np.float64).mean())
            sL = float(np.asarray(magL, dtype=np.float64).std())
            mR = float(np.asarray(magR, dtype=np.float64).mean())
            sR = float(np.asarray(magR, dtype=np.float64).std())

            # prepara canvas de exibição conforme modo
            mode = MODES[mode_idx]
            if mode == "Original":
                left_vis  = currL.copy()
                right_vis = currR.copy()

            elif mode == "Overlay":
                left_vis  = cv2.addWeighted(currL, 1-ALPHA_OVERLAY, flowL_bgr, ALPHA_OVERLAY, 0)
                right_vis = cv2.addWeighted(currR, 1-ALPHA_OVERLAY, flowR_bgr, ALPHA_OVERLAY, 0)

            elif mode == "Vetores":
                left_vis  = currL.copy()
                right_vis = currR.copy()
                draw_vectors(left_vis,  flowL_small[...,0], flowL_small[...,1], color=(255,255,255))
                draw_vectors(right_vis, flowR_small[...,0], flowR_small[...,1], color=(255,255,255))

            elif mode == "Heatmap":
                # heatmap da magnitude: escala para 0..255 e aplica colormap
                magL_u8 = cv2.normalize(magL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                magR_u8 = cv2.normalize(magR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                left_vis  = cv2.applyColorMap(magL_u8, cv2.COLORMAP_TURBO)
                right_vis = cv2.applyColorMap(magR_u8, cv2.COLORMAP_TURBO)

            # concatena e rotula
            vis = np.hstack([left_vis, right_vis])
            put_label(vis, f"REAL  | mean={mL:.3f}  std={sL:.3f}", (10, 28), (80, 240, 80))
            put_label(vis, f"{mode}", (W//2 - 70, 28), (255, 255, 0))
            put_label(vis, f"DEEPFAKE | mean={mR:.3f}  std={sR:.3f}", (W//2 + 10, 28), (60, 140, 255))

            cv2.imshow(win_name, vis)
            prev = frame.copy()
            frame_count += 1

        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(MODES)
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            fname = f"snapshot_flow_{mode_idx}_{snap_idx:03d}.png"
            cv2.imwrite(fname, vis)
            print(f"[snap] salvo: {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()