import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Importation des données
# =========================

data = np.loadtxt("bruit01.csv", delimiter=",", skiprows=1)

time = data[:, 0]
signal = data[:, 1]
sync = data[:, 2]

# =========================
# 2. Séparation fond / signal avec numpy.where()
# =========================

indices_fond = np.where(sync < 0.5) # sélectionne quand sync est bas
indices_signal = np.where(sync > 0.5)

fond = signal[indices_fond]
signal_haut = signal[indices_signal]


# =========================
# 3. Question 3 — calcule du SNR
# =========================

# analyse statistique du fond
mu_fond = np.mean(fond)
sigma_fond = np.std(fond)

print("Fond :")
print("Moyenne =", mu_fond)
print("Écart-type =", sigma_fond)

# analyse statistique du signal haut
mu_haut = np.mean(signal_haut)  
sigma_haut = np.std(signal_haut)
print("\nSignal haut :")
print("Moyenne =", mu_haut)
print("Écart-type =", sigma_haut)

# calcul du rapport signal sur bruit (SNR)
snr = (mu_haut - mu_fond) / sigma_fond
print("\nRapport signal sur bruit (SNR) =", snr)


# =========================
# 4. Question 4a — caractérisation du signal haut
# =========================

mu_signal = np.mean(signal_haut)
sigma_signal = np.std(signal_haut)

print("Signal haut :")
print("Moyenne =", mu_signal)
print("Écart-type =", sigma_signal)

# histogramme de la distribution
plt.figure()
plt.hist(signal_haut, bins=50)
plt.xlabel("Tension (V)")
plt.ylabel("Nombre de points")
plt.title("Distribution du signal en mode haut")
plt.show()

# =========================
# 5. Question 4b — impact du nombre de points N
# =========================

# valeurs de N testées
N_values = np.array([2, 5, 10, 20, 50, 100, 200, 500])

# nombre de répétitions statistiques
n_trials = 1000

errors = []

for N in N_values:
    means = []
    
    for _ in range(n_trials):
        sample = np.random.choice(signal_haut, size=N, replace=False)
        means.append(np.mean(sample))
    
    # erreur sur la moyenne = écart-type des moyennes
    errors.append(np.std(means))

errors = np.array(errors)

# =========================
# 6. Comparaison avec la théorie (SEM = sigma / sqrt(N))
# =========================

theory = sigma_signal / np.sqrt(N_values)

plt.figure()
plt.plot(N_values, errors, 'o', label="Erreur expérimentale")
plt.plot(N_values, theory, '--', label=r"$\sigma/\sqrt{N}$")
plt.xlabel("Nombre de points N")
plt.ylabel("Erreur sur la moyenne")
plt.legend()
plt.show()

# =========================
# 7. Graphe log-log 
# =========================

plt.figure()
plt.loglog(N_values, errors, 'o', label="Expérimental")
plt.loglog(N_values, theory, '--', label=r"$\sigma/\sqrt{N}$")
plt.xlabel("N")
plt.ylabel("Erreur")
plt.legend()
plt.show()

# =========================
# 7. Question 5 - amélioration du SNR par moyennage
# =========================

mu_fond = np.mean(fond)
sigma_fond = np.std(fond)

mu_signal = np.mean(signal_haut)

SNR_initial = (mu_signal - mu_fond) / sigma_fond
print("SNR initial =", SNR_initial)

N_avg = [1, 2, 5, 10, 20, 50, 100]

SNR_values = []

for N in N_avg:
    means = []
    
    for _ in range(1000):
        sample = np.random.choice(signal_haut, size=N, replace=False)
        means.append(np.mean(sample))
    
    means = np.array(means)
    
    mu_avg = np.mean(means)
    sigma_avg = np.std(means)
    
    SNR_values.append((mu_avg - mu_fond) / sigma_avg)


SNR_theory = SNR_initial * np.sqrt(N_avg)

plt.figure()
plt.plot(N_avg, SNR_values, 'o', label="SNR mesuré")
plt.plot(N_avg, SNR_theory, '--', label=r"$\mathrm{SNR}_0 \sqrt{N}$")
plt.xlabel("Nombre de points moyennés N")
plt.ylabel("SNR")
plt.legend()
plt.show()


gain = SNR_values[-1] / SNR_initial
print("Gain de SNR =", gain)

sigma_new = sigma_fond / np.sqrt(100)
print("Nouvelle variation minimale détectable =", sigma_new)
