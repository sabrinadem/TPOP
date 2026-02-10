import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Chargement ---
df = pd.read_csv(r"C:\Users\sabri\OneDrive\Desktop\uni\vscode\tpop\TPOP\fibre\Values.csv")

# --- Sécurité: conversion numérique ---
for col in ["X", "data", "fit"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["X", "data", "fit"])

# --- Normalisation 0–100 avec min -> 0 ---
def normalize_0_100(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.nan, index=series.index)
    if s_max == s_min:
        # Série constante -> tout à 0 (on pourrait choisir 100 selon le contexte)
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min) * 100

df["data_norm"] = normalize_0_100(df["data"])
df["fit_norm"]  = normalize_0_100(df["fit"])

# --- Centrage de l'axe X : centre de la gaussienne = 0 ---
# On utilise le maximum de la courbe "fit" (ajustement) comme centre
# (Si tu préfères centrer sur les données: remplace "fit_norm" par "data_norm")
idx_center = df["fit_norm"].idxmax()
x_center = df.loc[idx_center, "X"]

df["X_centered"] = df["X"] - x_center

# --- Figure ---
fig, ax = plt.subplots(figsize=(7, 4))

# Données expérimentales : points noirs
ax.plot(df["X_centered"], df["data_norm"],
        linestyle="None", marker="o", markersize=4,
        color="black", label="Données expérimentales")

# Ajustement : triangles gris
ax.plot(df["X_centered"], df["fit_norm"],
        linestyle="None", marker="^", markersize=4,
        color="grey", label="Ajustement de courbe")

# --- Axes & légendes ---
ax.set_xlabel("Position le long du profil (pixels)", fontsize=13)
ax.set_ylabel("Intensité moyennée (%)", fontsize=13)

ax.grid(True, alpha=0.3)
ax.legend()

# Y : on veut voir un peu plus haut que 100
# Option simple : plafond fixe (ex. 110)
ax.set_ylim(0, 110)

# (Option auto : +5% au-dessus du max réellement présent)
# ymax = np.nanmax([df["data_norm"].max(), df["fit_norm"].max()])
# ax.set_ylim(0, ymax * 1.05)

# X : si tu veux imposer un rayon r (symétrique -r à +r), décommente :
# r = 50  # en pixels (à adapter)
# ax.set_xlim(-r, r)

fig.tight_layout()
fig.canvas.draw()  # Forcer le rendu avant sauvegarde

# --- Sortie robuste ---
try:
    base_path = Path(__file__).parent
except NameError:
    base_path = Path.cwd()

out_path = base_path / "fibre_analyseimage.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print("✅ Image sauvegardée:", out_path.resolve())
print(f"ℹ️ Centre gaussienne (X original) = {x_center}")
