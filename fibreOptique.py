import numpy as np
import matplotlib.pyplot as plt

# Caractéristiques du laser
W_0_laser = (0.63 * 10**-3)/2
divergence_laser = (1.3 * 10**-3)/2
longueur_onde_laser = 632.8*10**-9

# Caractéristiques de la lentille
longeur_focale_lentille = 4.5*10**-3

# Caractéristiques de la fibre optique
longueur_onde_critique = 620*10**-9
V_c = 2.405
NA = 0.12
a = (V_c*longueur_onde_critique)/(2*np.pi*NA)
V = (2*np.pi*a/longueur_onde_laser)*NA

# Génération des valeurs de la distance Z
Z = np.linspace(0, 2, 2000)

# Calcul des valeurs de l'efficacité du couplage T
W_objectif = W_0_laser*(1 + ((Z*divergence_laser)/W_0_laser)**2)**(1/2)
W_0_image = (longueur_onde_laser*longeur_focale_lentille)/(np.pi*W_objectif)
W_1 = W_0_image
W_2 = a*(0.65 + 1.619/V**(3/2) + 2.879/V**6)
T = (((2*W_1*W_2)/(W_1**2+W_2**2))**2)*100
        
# Création du graphique
plt.figure(figsize=(8, 5))
plt.plot(Z, T, color="blue")
plt.axhline(0, color='black', linewidth=0.8)  # Axe horizontal
plt.axvline(0, color='black', linewidth=0.8)  # Axe vertical
plt.grid(True, linestyle="--", alpha=0.6)
plt.title(f"Éfficacité du couplage T en fonction de la distance Z")
plt.xlabel("Z (m)")
plt.ylabel("T (%)")
plt.show()

# Affichage des résultats
T_max = np.max(T)
print(f"L'efficacité maximale du couplage T est de {T_max:.2f} %")
Z_optimal = Z[np.argmax(T)]
print(f"La distance Z optimale pour un couplage maximal est de {Z_optimal:.4f} m")