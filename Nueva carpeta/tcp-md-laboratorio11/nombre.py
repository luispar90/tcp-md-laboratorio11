
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from deap import base, creator, tools, algorithms
import random
import requests
from io import BytesIO



# cargando los datos desde Google Drive
URL = "https://drive.google.com/file/d/1Gn2_7CGAX0D0M3Ka3JKSjlEdrWbKCxH0/view?usp=sharing"
URL = 'https://drive.google.com/uc?id=' + URL.split('/')[-2]

df = pl.read_csv(URL)
print(df.head())



# === 1. PREPROCESAMIENTO DE DATOS ===
print("\nValores nulos:\n", df.null_count())

# Análisis de outliers univariado
for col in df.columns[:-1]:
    sns.boxplot(y=df[col].to_pandas())
    plt.title(f'Boxplot - {col}')
    plt.show()

# Escalamiento
X = df.drop("Sales")
y = df["Sales"].to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.to_numpy())

# === 2. ALGORITMO GENÉTICO PARA SELECCIÓN DE VARIABLES ===

# Función de aptitud
def eval_individual(individual):
    if sum(individual) == 0:
        return 0.,
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    X_sel = X_scaled[:, selected_features]
    model = LinearRegression()
    scores = cross_val_score(model, X_sel, y, cv=5, scoring='r2')
    return scores.mean(),

# Configurar DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
n_features = X.shape[1]

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parámetros del algoritmo genético
N_POP = 30
N_GEN = 20
CX_PB = 0.5
MUT_PB = 0.2

pop = toolbox.population(n=N_POP)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

# Evolución
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=N_GEN,
                                   stats=stats, halloffame=hof, verbose=True)

# === 3. GRAFICAR EVOLUCIÓN DE R2 ===
gen = logbook.select("gen")
maxs = logbook.select("max")

plt.plot(gen, maxs, label="Máx R2")
plt.xlabel("Generación")
plt.ylabel("R²")
plt.title("Evolución del R² en generaciones")
plt.legend()
plt.grid()
plt.show()

# === 4. MODELO FINAL CON VARIABLES SELECCIONADAS ===
best_individual = hof[0]
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
col_names = X.columns
print("Variables seleccionadas:", [col_names[i] for i in selected_features])

# Entrenando el  modelo final
X_final = X_scaled[:, selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R² en test:", r2_score(y_test, y_pred))
















