"""
Prueba de concepto (PoC) - Caso 10, Grupo 10.
Supuesto técnico de mayor riesgo: "una CNN puede detectar defectos en imágenes de PCB
con exhaustividad alta, dentro del presupuesto de tiempo del takt (6 s/tarjeta), y el
umbral puede fijarse por costo de error".
Dataset público proxy: DeepPCB (Tang et al., 2019) - defectos de cobre en PCB desnuda,
NO soldaduras. Valida la cadena técnica y el método, no la precisión final en soldaduras.
"""
import os, sys, json, time, random
import numpy as np
from PIL import Image
import tensorflow as tf

ROOT = sys.argv[1]            # ruta a DeepPCB/PCBData
OUT = sys.argv[2]             # carpeta de resultados
P = 64                        # tamaño de parche (px)
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.makedirs(OUT, exist_ok=True)

def read_list(name):
    pairs = []
    for line in open(os.path.join(ROOT, name)):
        img, ann = line.split()
        stem = img.replace('.jpg', '')
        pairs.append((os.path.join(ROOT, stem + '_test.jpg'),
                      os.path.join(ROOT, stem + '_temp.jpg'),
                      os.path.join(ROOT, ann)))
    return pairs

def boxes(path):
    b = []
    for l in open(path):
        v = l.replace(',', ' ').split()
        if len(v) >= 5:
            b.append(tuple(int(x) for x in v[:5]))
    return b

def crop(img, cx, cy):
    h, w = img.shape
    x0 = int(np.clip(cx - P // 2, 0, w - P)); y0 = int(np.clip(cy - P // 2, 0, h - P))
    return img[y0:y0 + P, x0:x0 + P]

def overlaps(x0, y0, bs):
    for (a, b, c, d, _) in bs:
        if x0 < c and x0 + P > a and y0 < d and y0 + P > b:
            return True
    return False

def build(pairs, neg_per_img, jitter=True):
    X, y = [], []
    for test, temp, ann in pairs:
        t = np.asarray(Image.open(test).convert('L'), dtype=np.uint8)
        tp = np.asarray(Image.open(temp).convert('L'), dtype=np.uint8)
        bs = boxes(ann)
        for (a, b, c, d, k) in bs:                       # positivos: parche centrado en el defecto
            cx, cy = (a + c) // 2, (b + d) // 2
            if jitter:
                cx += random.randint(-8, 8); cy += random.randint(-8, 8)
            X.append(crop(t, cx, cy)); y.append(1)
        h, w = t.shape; n = 0; tries = 0
        while n < neg_per_img and tries < 200:           # negativos: zonas sin defecto
            tries += 1
            x0, y0 = random.randint(0, w - P), random.randint(0, h - P)
            if overlaps(x0, y0, bs):
                continue
            src = t if random.random() < 0.5 else tp
            X.append(src[y0:y0 + P, x0:x0 + P]); y.append(0); n += 1
        # negativos "difíciles": misma posición del defecto pero en la plantilla sin defecto
        for (a, b, c, d, k) in bs[:2]:
            X.append(crop(tp, (a + c) // 2, (b + d) // 2)); y.append(0)
    X = np.stack(X)[..., None].astype('float32') / 255.0
    return X, np.array(y, dtype='int32')

train_pairs = read_list('trainval.txt'); test_pairs = read_list('test.txt')
t0 = time.time()
Xtr, ytr = build(train_pairs, neg_per_img=6)
# prueba con desbalance realista: muchos más parches buenos que defectuosos
Xte, yte = build(test_pairs, neg_per_img=40, jitter=False)
prep_s = time.time() - t0
print('train', Xtr.shape, ytr.mean(), 'test', Xte.shape, yte.mean())

aug = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal_and_vertical'),
                           tf.keras.layers.RandomRotation(0.25, fill_mode='reflect'),
                           tf.keras.layers.RandomContrast(0.2)])
def model():
    i = tf.keras.Input((P, P, 1)); x = aug(i)
    for f in (16, 32, 64):
        x = tf.keras.layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    o = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(i, o)

m = model()
m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
          metrics=[tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])
pos = ytr.mean(); cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}       # ponderación por desbalance
idx = np.random.permutation(len(ytr)); nv = len(ytr) // 10
t0 = time.time()
h = m.fit(Xtr[idx[nv:]], ytr[idx[nv:]], validation_data=(Xtr[idx[:nv]], ytr[idx[:nv]]),
          epochs=12, batch_size=128, class_weight=cw, verbose=2)
train_s = time.time() - t0
s = m.predict(Xte, batch_size=512, verbose=0).ravel()

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, confusion_matrix, precision_recall_curve)
def at(th):
    p = (s >= th).astype(int); tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    return dict(th=round(float(th), 4), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                recall=round(tp / (tp + fn), 4), precision=round(tp / max(tp + fp, 1), 4),
                f1=round(2 * tp / (2 * tp + fp + fn), 4), fpr=round(fp / (fp + tn), 4))

# Umbral por costo: C_FN = escape al cliente; C_FP = re-inspección/rechazo de pieza buena
C_FN, C_FP = 400000.0, 4000.0      # COP, supuestos del modelo financiero (ver Anexo)
grid = np.linspace(0.01, 0.99, 99)
costs = []
for th in grid:
    r = at(th); costs.append(r['fn'] * C_FN + r['fp'] * C_FP)
th_cost = float(grid[int(np.argmin(costs))])
prec, rec, thr = precision_recall_curve(yte, s)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9); th_f1 = float(thr[int(np.argmax(f1s[:-1]))])

# Latencia: CPU del entorno (2 núcleos), lote de ROIs por tarjeta
lat = {}
for n_roi in (1, 100, 300):
    xb = Xte[:n_roi] if n_roi <= len(Xte) else np.repeat(Xte[:1], n_roi, 0)
    m.predict(xb, batch_size=n_roi, verbose=0)
    ts = []
    for _ in range(15):
        t1 = time.perf_counter(); m(xb, training=False); ts.append(time.perf_counter() - t1)
    lat[n_roi] = dict(p50_ms=round(1000 * float(np.median(ts)), 2), p95_ms=round(1000 * float(np.percentile(ts, 95)), 2))

res = dict(dataset='DeepPCB (proxy, PCB desnuda)', patch=P, n_train=int(len(ytr)), pos_rate_train=round(float(pos), 4),
           n_test=int(len(yte)), pos_rate_test=round(float(yte.mean()), 4), params=int(m.count_params()),
           auc=round(float(roc_auc_score(yte, s)), 4), ap=round(float(average_precision_score(yte, s)), 4),
           at_0_5=at(0.5), at_best_f1=at(th_f1), at_min_cost=at(th_cost),
           cost_ratio_FN_FP=C_FN / C_FP, latency_cpu_2cores=lat, train_seconds=round(train_s, 1),
           prep_seconds=round(prep_s, 1), tf=tf.__version__,
           history={k: [round(float(v), 4) for v in vals] for k, vals in h.history.items()})
json.dump(res, open(os.path.join(OUT, 'poc_results.json'), 'w'), indent=2, ensure_ascii=False)
np.save(os.path.join(OUT, 'scores.npy'), s); np.save(os.path.join(OUT, 'labels.npy'), yte)
np.save(os.path.join(OUT, 'cost_curve.npy'), np.array([grid, costs]))
print(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1, ensure_ascii=False))
