# InspecSol 4.0: prueba de concepto de visión artificial

Prueba de concepto del proyecto **InspecSol 4.0** (Caso 10, «Industria 4.0: visión artificial para control de calidad»), trabajo final de Gestión de Proyectos, Universidad El Bosque, periodo 2026-2. Grupo 10: Sergio Hernández, Andrés Amariles y Samuel Novoa.

El código, el método, los resultados y sus limitaciones se describen en el **Anexo M** del documento de anexos. Este repositorio contiene el mismo script que aparece impreso en ese anexo.

## Qué valida

El supuesto técnico de mayor riesgo del proyecto: *una red neuronal convolucional puede detectar defectos en imágenes de circuito impreso con exhaustividad alta, dentro del takt de 6 s por tarjeta, y el umbral de decisión puede fijarse por el costo del error.*

Como la planta del caso es ficticia, se usa como sustituto el dataset público **DeepPCB** (Tang et al., 2019). Ese dataset tiene defectos de cobre en tarjeta desnuda, no soldaduras: la prueba valida la cadena técnica y el método de umbral por costo, no la precisión final en soldaduras.

## Cómo ejecutarlo

1. Descargue el dataset: <https://github.com/tangsanli5201/DeepPCB> (carpeta `PCBData`).
2. Instale las dependencias (Python 3.10 o superior):

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecute:

   ```bash
   python3 poc_vision.py <ruta a DeepPCB/PCBData> <carpeta de resultados>
   ```

El script genera en la carpeta de resultados `poc_results.json` (métricas, latencias e historial de entrenamiento), `scores.npy`, `labels.npy` y `cost_curve.npy`. La semilla está fija (42); en otro equipo los números pueden variar levemente.

## Resultados reportados (Anexo M)

| Métrica | Umbral 0,50 | Umbral de máx. F1 (0,987) | Umbral de mín. costo (0,41) |
|---|---|---|---|
| Exhaustividad | 99,36 % | 94,08 % | 99,52 % |
| Precisión | 65,08 % | 92,34 % | 60,98 % |
| F1 | 0,786 | 0,932 | 0,756 |
| Falsas alarmas por parche | 7,97 % | 1,17 % | 9,52 % |

AUC = 0,9958. 14.869 parches de entrenamiento y 24.140 de prueba (13,0 % defectuosos).

Latencia de inferencia en CPU de 2 núcleos (TensorFlow 2.21.0): 100 regiones por lote, p95 = 71,5 ms; 300 regiones, p95 = 201,8 ms.

**Lectura:** la latencia cabe con holgura en el presupuesto de 500 ms y el umbral por costo cambia la operación de forma marcada. La tasa de falsas alarmas por junta todavía no cumple la meta del proyecto; por eso se mantiene como condición de la puerta H3 (modo sombra).

## Limitaciones

1. DeepPCB contiene defectos de cobre en tarjeta desnuda y defectos artificiales, no soldaduras.
2. No se probó el aprendizaje por transferencia (sin pesos preentrenados en el entorno de ejecución).
3. Los parches negativos son más fáciles que una tarjeta real con componentes.

## Referencia del dataset

Tang, S., He, F., Huang, X. y Yang, J. (2019). *Online PCB defect detector on a new PCB defect dataset* [Preprint]. arXiv. https://arxiv.org/abs/1902.06197
