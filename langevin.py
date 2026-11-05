import numpy as np
import matplotlib.pyplot as plt
import time

#Parametros computacionales
N_simulaciones = 100


#Parametros y constantes fisicas
k = 1.380649e-23
Temp = 20. + 273.15
eta = 1.0016e-3  # Pa s, viscosidad aproximada a 20 °C
NA = 6.02214076e23
m = 18.01528e-3 / NA  # kg, masa de una molécula de H2O
radio = 0.14e-9  # m, radio efectivo supuesto
diam = 2. * radio
gamma = 6. * np.pi * eta * radio #Usando aproximacion de Stokes
alpha = gamma / m
tauB = 1. / alpha
Deltat = tauB / 10.

a = 1. - alpha * Deltat #Esto es para no hacer la misma multiplicacion una y otra vez
b = np.sqrt(
    (k * Temp / m) * alpha
    * (2. - alpha * Deltat) * Deltat
)


# OBJETIVO
MSD_objetivo = 100. * diam**2


# CONDICIONES INICIALES
r = np.zeros((N_simulaciones, 3)) #Hacemos las simulaciones a la vez. r sera una matriz Nsimulacionesx3

v = np.random.normal(
    0, np.sqrt(k * Temp / m),
    size=(N_simulaciones, 3)
) #Empiezan con una velocidad dada por la distribución de Boltzmann a partir del teorema de equiparticion. v sera una matriz Nsimulacionesx3

MSD = [0.] #Una lista cuyo primer elemento es 0.0
N_pasos = 0


# SIMULACIÓN
inicio = time.process_time() #Activamos el crono

while MSD[-1] < MSD_objetivo: #Eso es el ultimo elemento de la lista

    Z = np.random.normal(
        0, 1, size=(N_simulaciones, 3)
    )

    r = r + v * Deltat #Langevin discret
    v = a * v + b * Z

    desp_cuad_med = np.mean(np.sum(r**2, axis=1))

    MSD.append(desp_cuad_med) #Añado el MSD de este paso a la lista
    N_pasos += 1

fin = time.process_time() #Desactivamos el crono


# TIEMPOS DE COMPUTACIÓN
tiempo_cpu = fin - inicio
tiempo_medio = tiempo_cpu / (N_simulaciones * N_pasos) #Tiempo para cada pasito

print(f"Pasos realizados: {N_pasos}") 
print(f"Tiempo total de CPU: {tiempo_cpu:.3f} s")
print(f"Tiempo CPU medio por paso: {tiempo_medio:.3e} s") #Esto es lo que pide iii)


#PREPARO GRAFICO MSD
pasos = np.arange(1, N_pasos + 1)
MSD_array = np.array(MSD[1:]) / diam**2


#CALCULO DE PENDIENTES
# Régimen balístico (pasos 1 a 10)
seleccion_b = (pasos >= 1) & (pasos <= 10)

pendiente_b = np.polyfit(
    np.log10(pasos[seleccion_b]),
    np.log10(MSD_array[seleccion_b]),
    1
)[0]

# Régimen difusivo (pasos 1000 a 100000)
seleccion_d = (pasos >= 10000) & (pasos <= 100000)

pendiente_d = np.polyfit(
    np.log10(pasos[seleccion_d]),
    np.log10(MSD_array[seleccion_d]),
    1
)[0]

print(f"Pendiente balística: {pendiente_b:.3f}")
print(f"Pendiente difusiva: {pendiente_d:.3f}")


#COMIENZO GRAFICA
fig, ax = plt.subplots(figsize=(10, 6.5))

#Solución teórica de Langevin
t = pasos * Deltat
D = k * Temp / (m * alpha)

MSD_teorico = 6. * D * (
    t - tauB * (1. - np.exp(-t / tauB))
)

MSD_teorico = MSD_teorico / diam**2

ax.loglog(
    pasos, MSD_teorico,
    color="#E69F00",
    linewidth=2.2,
    label="Langevin teórico"
)

#Simulación numérica
ax.loglog(
    pasos, MSD_array,
    color="blue",
    linewidth=2,
    label="Simulación"
)

#Referencia balística (pendiente 2)
n_b = np.logspace(0, 1, 100)
y_b = MSD_array[0] * (n_b / pasos[0])**2

ax.loglog(
    n_b, y_b,
    color="red",
    linestyle="--",
    linewidth=2.3,
    label=r"Régimen balístico $\left(\mathrm{MSD}\propto t^2\right)$"
)

#Referencia difusiva (pendiente 1)
n_d = np.logspace(5, np.log10(N_pasos), 100)
y_d = MSD_array[-1] * (n_d / N_pasos)

ax.loglog(
    n_d, y_d,
    color="limegreen",
    linestyle="--",
    linewidth=2.3,
    label=r"Régimen difusivo $\left(\mathrm{MSD}\propto t\right)$"
)


#Etiquetas de los ejes
ax.set_xlabel(
    "Número de pasos de integración",
    fontsize=16,
    labelpad=12
)

ax.set_ylabel(
    r"$\mathrm{MSD}/\sigma^2$",
    fontsize=16,
    labelpad=12
)

#Tamaño de números y marcas
ax.tick_params(
    axis="both",
    which="major",
    labelsize=13,
    length=6,
    width=1.1
)

ax.tick_params(
    axis="both",
    which="minor",
    length=3,
    width=0.6
)

#Cuadrícula
ax.grid(
    True,
    which="major",
    color="#B0B0B0",
    linestyle="-",
    linewidth=0.7,
    alpha=0.65
)

ax.grid(
    True,
    which="minor",
    color="#D0D0D0",
    linestyle=":",
    linewidth=0.45,
    alpha=0.45
)

#Leyenda
ax.legend(
    fontsize=12,
    loc="upper left",
    frameon=True,
    framealpha=0.95,
    edgecolor="#CCCCCC"
)

#Aspecto general
for spine in ax.spines.values():
    spine.set_linewidth(1.1)

plt.tight_layout()

#Guardar figura
plt.savefig(
    "MSD_Langevin.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "MSD_Langevin.pdf",
    bbox_inches="tight"
)

plt.show()