"""
Funciones de riesgo crediticio usadas en el notebook del modelo PD thin-file.

- Discriminación: AUC, Gini, KS, intervalos de confianza por bootstrap, tabla de deciles.
- Estabilidad: PSI (score) y CSI (variables).
- Binning supervisado con WoE e Information Value (IV).

Convención: WoE = ln(%malos / %buenos). WoE alto => más riesgo, por lo que los
coeficientes de una regresión logística sobre WoE deben ser POSITIVOS.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

EPS = 0.5          # suavizado para bins con 0 malos o 0 buenos
MISSING = "Missing"


# ---------------------------------------------------------------- Discriminación
def auc(y, score):
    return float(roc_auc_score(y, score))


def gini(y, score):
    return 2 * auc(y, score) - 1


def ks(y, score):
    """Máxima distancia entre las distribuciones acumuladas de buenos y malos."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(np.abs(tpr - fpr)))


def auc_univariada(y, x):
    """AUC orientada (>= 0,5) y dirección del riesgo (+ : a mayor valor, más riesgo)."""
    m = pd.notna(x)
    a = roc_auc_score(np.asarray(y)[m], np.asarray(x)[m])
    return (a, "+") if a >= 0.5 else (1 - a, "-")


def bootstrap_ci(y, score, metrica=auc, n_boot=1000, seed=42, alpha=0.05):
    """Intervalo de confianza por bootstrap (percentiles) para una métrica."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(metrica(y[idx], score[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def tabla_deciles(y, score, n_bins=10):
    """Tabla de ganancia por deciles de score (decil 1 = más riesgoso)."""
    df = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    df["decil"] = pd.qcut(df["score"].rank(method="first", ascending=False),
                          n_bins, labels=range(1, n_bins + 1))
    t = df.groupby("decil", observed=True).agg(n=("y", "size"), malos=("y", "sum"))
    t["buenos"] = t["n"] - t["malos"]
    t["tasa_malos"] = t["malos"] / t["n"]
    t["pct_malos_acum"] = t["malos"].cumsum() / t["malos"].sum()
    t["pct_buenos_acum"] = t["buenos"].cumsum() / t["buenos"].sum()
    t["ks"] = (t["pct_malos_acum"] - t["pct_buenos_acum"]).abs()
    t["lift"] = t["tasa_malos"] / df["y"].mean()
    return t


# ---------------------------------------------------------------- Estabilidad
def psi(esperado, actual, n_bins=10, return_table=False):
    """PSI con cortes por deciles de la población de referencia. Nulos = bin propio."""
    esperado = pd.Series(esperado).astype(float)
    actual = pd.Series(actual).astype(float)
    cortes = np.unique(np.nanquantile(esperado.dropna(), np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = pd.cut(esperado, cortes, include_lowest=True).astype(str).where(esperado.notna(), MISSING)
    a = pd.cut(actual, cortes, include_lowest=True).astype(str).where(actual.notna(), MISSING)
    tab = pd.concat([e.value_counts(normalize=True).rename("pct_ref"),
                     a.value_counts(normalize=True).rename("pct_comp")], axis=1).fillna(0).clip(lower=1e-4)
    tab["psi"] = (tab["pct_comp"] - tab["pct_ref"]) * np.log(tab["pct_comp"] / tab["pct_ref"])
    valor = float(tab["psi"].sum())
    return (valor, tab) if return_table else valor


def psi_categorica(esperado, actual):
    pe = pd.Series(esperado).astype(str).value_counts(normalize=True)
    pa = pd.Series(actual).astype(str).value_counts(normalize=True)
    tab = pd.concat([pe, pa], axis=1, keys=["ref", "comp"]).fillna(0).clip(lower=1e-4)
    return float(((tab.comp - tab.ref) * np.log(tab.comp / tab.ref)).sum())


def lectura_psi(v):
    if v < 0.10:
        return "Estable"
    if v < 0.25:
        return "Cambio moderado"
    return "Cambio significativo"


# ---------------------------------------------------------------- Binning WoE / IV
def lectura_iv(iv):
    if iv < 0.02:
        return "No predictiva"
    if iv < 0.10:
        return "Débil"
    if iv < 0.30:
        return "Media"
    if iv < 0.50:
        return "Fuerte"
    return "Sospechosa (revisar leakage)"


class WoEBinner:
    """Binning supervisado de una variable.

    - Numéricas: cortes de un árbol de decisión (máx. `max_bins` hojas, cada una con al
      menos `min_bin_pct` de la población). Opcionalmente fuerza tendencia monótona
      fusionando bins adyacentes que la violen.
    - Categóricas: cada categoría es un bin; las de menos de `min_bin_pct` van a "Otros".
    - Nulos: bin propio "Missing". Si es muy pequeño (< min_bin_pct), su WoE no es
      confiable y se le asigna el del bin más riesgoso (política conservadora: la falta
      de información nunca debe premiar al solicitante).
    """

    def __init__(self, max_bins=6, min_bin_pct=0.05, monotonic="auto", categorical=False, random_state=42):
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.monotonic = monotonic        # "auto", "asc", "desc" o None
        self.categorical = categorical
        self.random_state = random_state

    # ---- utilidades internas
    @staticmethod
    def _tabla_woe(bins, y):
        df = pd.DataFrame({"bin": bins.values, "y": y.values})
        t = df.groupby("bin", observed=True, sort=False)["y"].agg(n="size", malos="sum")
        t["buenos"] = t["n"] - t["malos"]
        tot_m, tot_b = t["malos"].sum(), t["buenos"].sum()
        t["pct_n"] = t["n"] / t["n"].sum()
        t["tasa_malos"] = t["malos"] / t["n"]
        t["dist_malos"] = (t["malos"] + EPS) / (tot_m + EPS * len(t))
        t["dist_buenos"] = (t["buenos"] + EPS) / (tot_b + EPS * len(t))
        t["woe"] = np.log(t["dist_malos"] / t["dist_buenos"])
        t["iv"] = (t["dist_malos"] - t["dist_buenos"]) * t["woe"]
        return t

    def _cortes_arbol(self, x, y):
        m = x.notna()
        arbol = DecisionTreeClassifier(max_leaf_nodes=self.max_bins,
                                       min_samples_leaf=max(int(self.min_bin_pct * len(x)), 1),
                                       random_state=self.random_state)
        arbol.fit(x[m].to_frame(), y[m])
        umbrales = sorted(t for t in arbol.tree_.threshold if t != -2)
        cortes = [-np.inf] + umbrales + [np.inf]
        if self.monotonic:
            cortes = self._forzar_monotonia(x[m], y[m], cortes)
        return cortes

    def _forzar_monotonia(self, x, y, cortes):
        direccion = self.monotonic
        if direccion == "auto":
            direccion = "asc" if pd.Series(x).corr(pd.Series(y), method="spearman") >= 0 else "desc"
        while len(cortes) > 2:
            tasas = pd.Series(y.values).groupby(
                pd.cut(x.values, cortes, include_lowest=True), observed=False).mean().values
            dif = np.diff(tasas)
            malos = np.where(dif < 0)[0] if direccion == "asc" else np.where(dif > 0)[0]
            if len(malos) == 0:
                break
            cortes = cortes[:malos[0] + 1] + cortes[malos[0] + 2:]   # fusiona el par que viola
        return cortes

    def _etiquetas(self):
        f = lambda v: f"{v:.4g}" if np.isfinite(v) else ("-inf" if v < 0 else "inf")
        return [f"({f(self.cortes_[i])}, {f(self.cortes_[i + 1])}]" for i in range(len(self.cortes_) - 1)]

    def bins(self, x):
        if self.categorical:
            x = pd.Series(x).astype(str)
            return x.where(x.isin(self.categorias_), "Otros")
        x = pd.Series(x).astype(float)
        b = pd.cut(x, self.cortes_, labels=self._etiquetas(), include_lowest=True).astype(str)
        return b.where(x.notna(), MISSING)

    # ---- API
    def fit(self, x, y):
        x = pd.Series(x).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True).astype(int)
        if self.categorical:
            freq = x.astype(str).value_counts(normalize=True)
            self.categorias_ = set(freq[freq >= self.min_bin_pct].index)
        else:
            self.cortes_ = self._cortes_arbol(x.astype(float), y)
        self.table_ = self._tabla_woe(self.bins(x), y)
        self.iv_ = float(self.table_["iv"].sum())
        if MISSING in self.table_.index and self.table_.loc[MISSING, "pct_n"] < self.min_bin_pct:
            self.table_.loc[MISSING, "woe"] = self.table_.drop(index=MISSING)["woe"].max()
        if not self.categorical:
            orden = [b for b in self._etiquetas() if b in self.table_.index]
            if MISSING in self.table_.index:
                orden.append(MISSING)
            self.table_ = self.table_.loc[orden]
        self.woe_map_ = self.table_["woe"].to_dict()
        return self

    def transform(self, x):
        """Reemplaza cada valor por el WoE de su bin (bin no visto => 0, neutral)."""
        return self.bins(pd.Series(x).reset_index(drop=True)).map(self.woe_map_).fillna(0.0).values


def ajustar_binners(df, y, num_cols, cat_cols=(), monotonic=None, **kw):
    monotonic = monotonic or {}
    binners = {c: WoEBinner(monotonic=monotonic.get(c, "auto"), **kw).fit(df[c], y) for c in num_cols}
    binners.update({c: WoEBinner(categorical=True, monotonic=None, **kw).fit(df[c], y) for c in cat_cols})
    return binners


def woe_transform(df, binners):
    return pd.DataFrame({c: b.transform(df[c]) for c, b in binners.items()}, index=df.index)


def resumen_iv(binners):
    s = pd.DataFrame({"iv": {c: b.iv_ for c, b in binners.items()},
                      "n_bins": {c: len(b.table_) for c, b in binners.items()}})
    s["poder_predictivo"] = s["iv"].map(lectura_iv)
    return s.sort_values("iv", ascending=False)