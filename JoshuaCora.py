import numpy as np
import random
import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import chi2

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

print(f"ALGORITMOS GENÉTICOS Y EVOLUCIÓN DE LA POBLACIÓN")
print(f"LABORATORIO 11")
print(f"Integrantes")
print(f"- Cora Salazar, Joshua Jian\n- Olortegui Perez, Kevin Estiben\n")

print(f"\n{'='*50}\n{'='*50}\n")
print(f"a. Realice el preprocesamiento de la información que incluya el análisis de datos faltantes,tratamiento de outliers a nivel univariado y multivariado y escalamiento de datos para las variables numéricas.\n")

# Carga del archivo 'Advertising'
url = "https://drive.google.com/file/d/1Gn2_7CGAX0D0M3Ka3JKSjlEdrWbKCxH0/view?usp=sharing"
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

df = pl.read_csv(url)
print("Dataset cargado:")
print(df.head())

print(f"\n{'='*50}\n")
print("Análisis de datos faltantes:")
nulos = df.null_count()
print(nulos)
print("- No existe ningún dato faltante en ninguna variable\n")

print(f"\n{'='*50}\n")
print("Tratamiento de outliers univariados:\n")

# Factor IQR
factor_iqr = 1.5
mascara = pl.lit(True)

# Recorrer cada columna numérica
for col in df.select(pl.col(pl.NUMERIC_DTYPES)).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    limite_inf = q1 - factor_iqr * iqr
    limite_sup = q3 + factor_iqr * iqr
    
    print(f"\nColumna: {col}")
    print(f"• Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"• Límites: [{limite_inf:.2f}, {limite_sup:.2f}]")
    
    mascara_col = (pl.col(col) >= limite_inf) & (pl.col(col) <= limite_sup)
    mascara = mascara & mascara_col
    
    outliers = df.filter(~mascara_col)
    print(f"• Outliers detectados: {outliers.height} fila(s)")

df1 = df.filter(mascara)
print(f"\nFilas eliminadas: {df.height - df1.height}\n")

print("Outliers multivariados:\n")

# Seleccionar columnas numéricas
n_col = df1.select(pl.col(pl.NUMERIC_DTYPES)).columns
data = df1.select(pl.col(n_col)).to_numpy()

# Calcular matriz de covarianza y distancias de Mahalanobis
cov_matrix = np.cov(data, rowvar=False)
try:
    inv_cov_matrix = np.linalg.inv(cov_matrix)
except np.linalg.LinAlgError:
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

mean_vec = np.mean(data, axis=0)
diff = data - mean_vec
mahalanobis_dist = np.array([np.sqrt(d.dot(inv_cov_matrix).dot(d)) for d in diff])

threshold = np.sqrt(chi2.ppf(0.95, df=len(n_col)))
print(f"Umbral de Mahalanobis (95% confianza): {threshold:.4f}")

es_outlier = mahalanobis_dist > threshold
outliers_count = np.sum(es_outlier)
print(f"Outliers detectados: {outliers_count}")

df1 = df1.with_columns(
    pl.Series("mahalanobis", mahalanobis_dist),
    pl.Series("es_outlier", es_outlier)
)

df_clean = df1.filter(~pl.col("es_outlier"))
print(f"DataFrame limpio: {df_clean.height} filas (eliminados {outliers_count} outliers)\n")

print(f"\n{'='*50}\n{'='*50}\n")
print("b. Cree un algoritmo genético para seleccionar las mejores variables para un modelo de regresión lineal incluyendo diversos valores para los parámetros: tamaño de la población, probabilidad de cruce, probabilidad de mutación y número de generaciones.\n")

# Preparar datos para el algoritmo genético
X = df_clean.drop(["Sales", "mahalanobis", "es_outlier"]).to_numpy()
y = df_clean["Sales"].to_numpy()

# Escalamiento de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo de regresión lineal
modelo = linear_model.LinearRegression()

n_features = X_scaled.shape[1]
verbose = 0

# Función de fitness usando R2 con validación cruzada
def calculate_fitness(modelo, x, y):
    if x.shape[1] == 0:  # Si no hay características seleccionadas
        return 0.0
    try:
        scores = cross_val_score(modelo, x, y, cv=5, scoring='r2')
        return np.mean(scores)
    except:
        return 0.0

# Función de evaluación
def evaluate(individual):
    np_ind = np.asarray(individual)
    if np.sum(np_ind) == 0:
        fitness = 0.0
    else:
        feature_idx = np.where(np_ind == 1)[0]
        fitness = calculate_fitness(modelo, X_scaled[:, feature_idx], y)
        if verbose:
            print("Individuo: {} Fitness Score: {:.4f}".format(individual, fitness))
    return (fitness,)

# CORRECCIÓN PRINCIPAL: weights en lugar de weigths
creator.create("FeatureSelect", base.Fitness, weights=(1.0,))  # CORREGIDO: weights, no weigths
creator.create("Individual", list, fitness=creator.FeatureSelect)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)  # CORREGIDO: tournsize, no tournzise
toolbox.register("evaluate", evaluate)

# Parámetros del algoritmo genético
N_POP = 50
CX_PB = 0.5
MUT_PB = 0.2
N_GEN = 20

# Función para estadísticas
def build_stats(gen, pop, fits):
    record = {}
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    record['gen'] = gen + 1
    record['min'] = min(fits)
    record['max'] = max(fits)
    record['avg'] = mean
    record['std'] = std
    print("Gen: {} | Min: {:.4f} | Max: {:.4f} | Avg: {:.4f} | Std: {:.4f}"
          .format(record['gen'], min(fits), max(fits), mean, std))
    return record

# Inicializar población
pop = toolbox.population(N_POP)

print("Evaluando individuos iniciales...")
fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

stats_records = []
print("-- GENERACIÓN 0 --")
fits = [ind.fitness.values[0] for ind in pop]
stats_records.append(build_stats(0, pop, fits))

# Evolución
for g in range(N_GEN):
    print("-- GENERACIÓN {} --".format(g + 1))
    
    # Selección
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    
    # Cruce
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CX_PB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # Mutación
    for mutant in offspring:
        if random.random() < MUT_PB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluación de individuos modificados
    weak_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, weak_ind))
    for ind, fit in zip(weak_ind, fitnesses):
        ind.fitness.values = fit
    
    # Reemplazar población
    pop[:] = offspring
    
    # Estadísticas
    fits = [ind.fitness.values[0] for ind in pop]
    stats_records.append(build_stats(g + 1, pop, fits))

print(f"\n{'='*50}\n{'='*50}\n")
print("c. Realice un gráfico de la evolución del coeficiente de determinación (R2), seleccione las mejores variables, cree un modelo de regresión lineal con base a estas tomando un 80% de datos para el entrenamiento y mida su desempeño con el R2.\n")

# Encontrar el mejor individuo
best_ind = tools.selBest(pop, 1)[0]
print("Mejor individuo:", best_ind)
print("Fitness del mejor individuo:", best_ind.fitness.values[0])

# Seleccionar mejores variables
feature_names = df_clean.drop(["Sales", "mahalanobis", "es_outlier"]).columns
selected_features = np.where(np.array(best_ind) == 1)[0]
selected_feature_names = [feature_names[i] for i in selected_features]

print("Variables seleccionadas:", selected_feature_names)

# Crear modelo final con las mejores variables
X_selected = X_scaled[:, selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

modelo_final = linear_model.LinearRegression()
modelo_final.fit(X_train, y_train)

# Predicciones y evaluación
y_pred_train = modelo_final.predict(X_train)
y_pred_test = modelo_final.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nR² en entrenamiento: {r2_train:.4f}")
print(f"R² en prueba: {r2_test:.4f}")

# Gráfico de evolución del R²
plt.figure(figsize=(12, 8))
generations = [record['gen'] for record in stats_records]
avg_fitness = [record['avg'] for record in stats_records]
max_fitness = [record['max'] for record in stats_records]

plt.subplot(2, 1, 1)
plt.plot(generations, avg_fitness, 'b-o', label='Promedio', markersize=4)
plt.plot(generations, max_fitness, 'r-o', label='Máximo', markersize=4)
plt.xlabel('Generación')
plt.ylabel('R² Score')
plt.title('Evolución del Algoritmo Genético')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico de comparación real vs predicho
plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title(f'Predicciones vs Valores Reales (R² = {r2_test:.4f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nAnálisis completado exitosamente!")