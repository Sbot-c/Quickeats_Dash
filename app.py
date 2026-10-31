"""
QuickBite · Diner Intelligence
------------------------------
Run:  streamlit run app.py
Needs diners.csv and orders.csv next to this file (or upload them in the sidebar).

Models (K-Means segments, Apriori rules, reorder classifier) are trained on the fly and cached.

Cola balancing
  The raw order log is skewed: Budget students place most of the baskets and almost all of them hold cola, so cola
  looks like a menu-wide favourite. The sidebar switch "Balance the order log" fixes that in three steps:
    1. drink-only baskets are dropped (not a meal occasion),
    2. every diner is given the same number of baskets (heavy diners are down-sampled, light ones re-sampled),
    3. an item found in >50% of the baskets (cola) is treated as an add-on and kept out of dish rankings,
       basket rules and recommendations; it is still reported as an attach rate.
  Switch it off to see the numbers exactly as logged.

The Diner predictor uses food_models.joblib (same folder) when present and valid, otherwise the live models.
"""
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="QuickBite · Diner Intelligence", page_icon="🍔",
                   layout="wide", initial_sidebar_state="expanded")

# ───────────────────────────── design tokens ─────────────────────────────
INK, MUTED, LINE, PAPER, SURFACE = "#12202B", "#5B6770", "#D9DFD8", "#F2F4F1", "#FFFFFF"
SEG_ORDER = ["Budget students", "Families", "Premium foodies"]
SEG_COLORS = {"Budget students": "#2F6FDE", "Families": "#D98A00", "Premium foodies": "#8B3FA8"}
SEG_TEXT = {"Budget students": "#FFFFFF", "Families": INK, "Premium foodies": "#FFFFFF"}
C_OK, C_BAD, C_NEUTRAL = "#1B998B", "#E4572E", "#3D5A6C"
STATUS_COLORS = {"Reordered": C_OK, "Did not reorder": C_BAD}
DISPLAY = "Bricolage Grotesque"
BODY = "Hanken Grotesk"

FEATURES = ["orders_per_month", "avg_order_value", "avg_items_per_order", "days_since_last_order", "discount_rate"]
CLF_FEATURES = FEATURES + ["age", "total_spend"]
NICE = {"age": "Age (years)", "orders_per_month": "Orders per month", "avg_order_value": "Avg order value (AED)",
        "avg_items_per_order": "Avg items per order", "days_since_last_order": "Days since last order",
        "discount_rate": "Discount rate", "total_spend": "Total spend (AED)"}
RS = 42
BEVERAGES = {"cola", "coffee", "green_tea", "wine"}
ADDON_THRESHOLD = 0.50

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: '{BODY}', system-ui, sans-serif; color: {INK}; }}
.stApp {{ background: {PAPER}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ font-family: '{DISPLAY}', 'Hanken Grotesk', system-ui, sans-serif !important; color: {INK} !important; letter-spacing: -.02em; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
.brand {{ font-family: '{DISPLAY}', 'Hanken Grotesk', system-ui, sans-serif; font-weight: 800; font-size: 1.55rem; letter-spacing: -.03em; line-height: 1; }}
.brand i {{ font-style: normal; color: {C_BAD}; }}
.side-h {{ font-family: '{DISPLAY}', 'Hanken Grotesk', system-ui, sans-serif; font-weight: 700; font-size: .95rem; margin: 18px 0 2px; }}

/* hero */
.hero h1 {{ color: {INK} !important; font-size: clamp(2rem, 3.6vw, 3.1rem); line-height: 1.04; font-weight: 800; margin: 0 0 .5rem; max-width: 22ch; }}
.hero h1 .hl {{ color: {C_BAD} !important; }}
.hero p {{ color: {MUTED}; font-size: 1.05rem; max-width: 64ch; margin: 0 0 1.4rem; line-height: 1.5; }}
.strip {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 6px; padding: 18px 22px 14px; margin-bottom: 1.4rem; }}
.strip-legend {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 12px; font-size: .86rem; font-weight: 600; }}
.strip-legend b {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }}
.strip-row {{ display: grid; grid-template-columns: 170px 1fr; gap: 14px; align-items: center; margin: 7px 0; }}
.strip-label {{ font-size: .86rem; color: {MUTED}; }}
.strip-bar {{ display: flex; gap: 2px; height: 34px; }}
.strip-bar div {{ display: flex; align-items: center; padding-left: 10px; font-weight: 700; font-size: .85rem; overflow: hidden; white-space: nowrap; }}
.strip-bar div:first-child {{ border-radius: 4px 0 0 4px; }} .strip-bar div:last-child {{ border-radius: 0 4px 4px 0; }}
@media (max-width: 760px) {{ .strip-row {{ grid-template-columns: 1fr; gap: 4px; }} }}

/* stats + notes */
.stat {{ border-top: 2px solid {INK}; padding: 10px 4px 6px 0; }}
.stat-v {{ font-family: '{DISPLAY}', 'Hanken Grotesk', system-ui, sans-serif; font-weight: 800; font-size: 2.05rem; line-height: 1.1; letter-spacing: -.03em; }}
.stat-l {{ font-weight: 600; font-size: .9rem; margin-top: 2px; }}
.stat-s {{ color: {MUTED}; font-size: .82rem; margin-top: 1px; }}
.note {{ border-left: 3px solid {INK}; padding: 4px 0 4px 14px; margin: 12px 0; font-size: .96rem; line-height: 1.5; max-width: 95ch; }}
.note.bad {{ border-left-color: {C_BAD}; }} .note.ok {{ border-left-color: {C_OK}; }}
.sec {{ border-top: 1px solid {LINE}; margin: 2.2rem 0 .6rem; padding-top: 1rem; }}
.sec h3 {{ font-size: 1.35rem; font-weight: 700; margin: 0 0 .15rem; }}
.sec p {{ color: {MUTED}; margin: 0; font-size: .95rem; max-width: 80ch; line-height: 1.45; }}
.pill {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin: 2px 5px 2px 0; }}
.plan {{ border-top: 4px solid {INK}; background: {SURFACE}; padding: 16px 18px 14px; height: 100%; border-radius: 0 0 6px 6px; border-left: 1px solid {LINE}; border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE}; }}
.plan h4 {{ margin: 0 0 2px; font-size: 1.15rem; }} .plan .why {{ color: {MUTED}; font-size: .85rem; }}
.meter {{ position: relative; height: 16px; background: {LINE}; border-radius: 8px; margin: 10px 0 4px; }}
.meter-fill {{ height: 100%; border-radius: 8px; }}
.meter-tick {{ position: absolute; left: 50%; top: -5px; bottom: -5px; width: 2px; background: {INK}; }}

/* tabs */
div[data-baseweb="tab-list"] {{ gap: 28px; }}
button[data-baseweb="tab"] {{ padding: 10px 0; background: transparent; }}
button[data-baseweb="tab"] p {{ font-family: '{DISPLAY}', 'Hanken Grotesk', system-ui, sans-serif; font-weight: 700; font-size: 1.05rem; color: {MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: {INK}; }}
div[data-baseweb="tab-highlight"] {{ background: {INK} !important; height: 3px !important; }}
div[data-baseweb="tab-border"] {{ background: {LINE} !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 6px; overflow: hidden; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 6px; }}
:focus-visible {{ outline: 2px solid {C_NEUTRAL}; outline-offset: 2px; }}
</style>
""", unsafe_allow_html=True)


def style(fig, height=360, legend=False, margin=None):
    """One light chart look: thin grid, no chart-junk, labels in ink."""
    fig.update_layout(
        template="simple_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=f"{BODY}, system-ui, sans-serif", color=INK, size=13), showlegend=legend,
        margin=margin or dict(l=6, r=18, t=46 if legend else 14, b=8),
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", title_text="", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=BODY, bgcolor=INK, font_color="#fff", bordercolor=INK))
    fig.update_xaxes(automargin=True, gridcolor="#E6EAE5", linecolor=LINE, zeroline=False, ticks="", title_standoff=10)
    fig.update_yaxes(automargin=True, gridcolor="#E6EAE5", linecolor=LINE, zeroline=False, ticks="", title_standoff=10)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_coloraxes(colorbar=dict(thickness=10, len=.8, outlinewidth=0))
    return fig


def plot(fig, **kw):
    st.plotly_chart(style(fig, **kw), width="stretch", config={"displayModeBar": False})


def stat(col, label, value, sub=""):
    col.markdown(f'<div class="stat"><div class="stat-v">{value}</div><div class="stat-l">{label}</div>'
                 f'<div class="stat-s">{sub}</div></div>', unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def head(title, sub=""):
    st.markdown(f'<div class="sec"><h3>{title}</h3>' + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def pct(x, d=0):
    return f"{x:.{d}%}"


def pills(items, color, text=INK):
    return "".join(f'<span class="pill" style="background:{color};color:{text}">{i.replace("_", " ")}</span>' for i in items)


def seg_filter(key):
    sel = st.pills("Segments", SEG_ORDER, selection_mode="multi", default=SEG_ORDER, key=key)
    return list(sel) or SEG_ORDER


# ───────────────────────────── data ─────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_bytes(b):
    return pd.read_csv(io.BytesIO(b))


def load_data():
    here = Path(__file__).parent
    d_path, o_path = here / "diners.csv", here / "orders.csv"
    if d_path.exists() and o_path.exists():
        return pd.read_csv(d_path), pd.read_csv(o_path)
    st.sidebar.warning("diners.csv / orders.csv not found next to app.py. Upload them:")
    d_up = st.sidebar.file_uploader("diners.csv", type="csv")
    o_up = st.sidebar.file_uploader("orders.csv", type="csv")
    if d_up and o_up:
        return read_csv_bytes(d_up.getvalue()), read_csv_bytes(o_up.getvalue())
    st.info("Add **diners.csv** and **orders.csv** to continue.")
    st.stop()


# ───────────────────────────── models (cached) ─────────────────────────────
@st.cache_resource(show_spinner="Training segmentation and churn models…")
def build_models(diners: pd.DataFrame):
    d = diners.copy()
    scaler = StandardScaler().fit(d[FEATURES])
    Xs = scaler.transform(d[FEATURES])
    ks = list(range(2, 9))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RS).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    kmeans = KMeans(3, n_init=10, random_state=RS).fit(Xs)
    d["cluster"] = kmeans.labels_

    means = d.groupby("cluster")[FEATURES].mean()            # rule-based naming: stays right if cluster numbers change
    budget = means["avg_order_value"].idxmin()
    families = means["avg_items_per_order"].idxmax()
    premium = ({0, 1, 2} - {budget, families}).pop()
    names = {int(budget): "Budget students", int(families): "Families", int(premium): "Premium foodies"}
    d["segment"] = d.cluster.map(names)

    pca = PCA(2, random_state=RS).fit(Xs)
    pts = pca.transform(Xs)
    d["pc1"], d["pc2"] = pts[:, 0], pts[:, 1]

    X, y = d[CLF_FEATURES], d["reordered"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)

    def make_clf():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight={0: 2, 1: 1}))

    ev = make_clf().fit(X_tr, y_tr)
    pred, proba = ev.predict(X_te), ev.predict_proba(X_te)[:, 1]
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
    fpr, tpr, _ = roc_curve(y_te, proba)
    metrics = dict(acc=accuracy_score(y_te, pred), base_acc=accuracy_score(y_te, base),
                   churn_recall=recall_score(y_te, pred, pos_label=0), auc=roc_auc_score(y_te, proba),
                   cm=confusion_matrix(y_te, pred, labels=[0, 1]), fpr=fpr, tpr=tpr, n_test=len(y_te))

    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    plain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    d["p_churn"] = cross_val_predict(plain, X, y, cv=cv, method="predict_proba")[:, 0]

    clf_final = make_clf().fit(X, y)
    coefs = pd.Series(clf_final[-1].coef_[0], index=CLF_FEATURES)
    return dict(df=d, scaler=scaler, kmeans=kmeans, names=names, ks=ks, inertia=inertia, sil=sil, pca=pca,
                clf=clf_final, metrics=metrics, coefs=coefs)


# ───────────────────────────── order log: cleaning + balancing ─────────────────────────────
def balance_log(log: pd.DataFrame, k: int, seed: int = RS) -> pd.DataFrame:
    """Every diner contributes exactly k baskets. Diners with more are down-sampled; diners with fewer keep all
    theirs and the rest are drawn with replacement (copies get a ~n suffix so order ids stay unique)."""
    rng = np.random.default_rng(seed)
    picks = []
    for did, oids in log.groupby("diner_id").order_id.unique().items():
        if len(oids) >= k:
            chosen = rng.choice(oids, k, replace=False)
        else:
            chosen = np.concatenate([oids, rng.choice(oids, k - len(oids), replace=True)])
        picks += [(did, o) for o in chosen]
    sel = pd.DataFrame(picks, columns=["diner_id", "order_id"])
    sel["rep"] = sel.groupby("order_id").cumcount()
    out = sel.merge(log, on=["diner_id", "order_id"])
    out["order_id"] = np.where(out.rep > 0, out.order_id + "~" + out.rep.astype(str), out.order_id)
    return out[["order_id", "diner_id", "item"]]


@st.cache_data(show_spinner=False)
def prepare_orders(raw: pd.DataFrame, k: int):
    """Returns dict: raw (as logged), clean (drink-only dropped), balanced, addons, n_drink_only."""
    o = raw.copy()
    o["item"] = o["item"].astype(str).str.strip().str.lower()
    o = o.dropna(subset=["order_id", "diner_id", "item"]).drop_duplicates(["order_id", "item"])
    items = o.groupby("order_id").item.agg(set)
    drink_only = items.index[items.apply(lambda s: s <= BEVERAGES)]
    clean = o[~o.order_id.isin(drink_only)].copy()
    share = clean.groupby("item").order_id.nunique() / clean.order_id.nunique()
    addons = sorted(share[share > ADDON_THRESHOLD].index)
    return dict(raw=o, clean=clean, balanced=balance_log(clean, k), addons=addons, n_drink_only=int(len(drink_only)))


@st.cache_data(show_spinner=False)
def to_basket(orders_df):
    return (orders_df.assign(v=True).pivot_table(index="order_id", columns="item", values="v",
                                                aggfunc="max", fill_value=False).astype(bool))


@st.cache_data(show_spinner=False)
def mine_rules(orders_df, min_support, min_lift=1.0):
    basket = to_basket(orders_df)
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return basket, pd.DataFrame()
    try:
        rules = association_rules(freq, num_itemsets=len(basket), metric="lift", min_threshold=min_lift)
    except TypeError:
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    if rules.empty:
        return basket, rules
    nice = lambda s: " + ".join(sorted(s)).replace("_", " ")
    rules["rule"] = rules.antecedents.apply(nice) + "  →  " + rules.consequents.apply(nice)
    rules["itemset"] = rules.apply(lambda r: frozenset(r.antecedents | r.consequents), axis=1)
    rules["bundle"] = rules.itemset.apply(nice)
    return basket, rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def pair_stats(orders_df):
    basket = to_basket(orders_df)
    B = basket.values.astype(float)
    p = B.mean(0)
    co = B.T @ B / len(B)
    return (pd.DataFrame(co / np.outer(p, p), index=basket.columns, columns=basket.columns),
            pd.DataFrame(co / p[:, None], index=basket.columns, columns=basket.columns))


def basket_share(lines, ref):
    """Share of baskets that contain each item."""
    return lines.drop_duplicates(["order_id", "item"]).groupby("item").order_id.nunique() / ref.order_id.nunique()


def attach_audit(log, seg_map, item):
    """Segment mix of the baskets and how often `item` is attached, for one version of the log."""
    m = log.merge(seg_map, on="diner_id").drop_duplicates(["order_id", "item"])
    nb = m.groupby("segment").order_id.nunique().reindex(SEG_ORDER)
    with_item = m[m.item == item].groupby("segment").order_id.nunique().reindex(SEG_ORDER, fill_value=0)
    overall = m[m.item == item].order_id.nunique() / m.order_id.nunique()
    return nb / nb.sum(), with_item / nb, overall


# ───────────────────────────── load + sidebar (part 1) ─────────────────────────────
diners_raw, orders_raw = load_data()

with st.sidebar:
    st.markdown('<div class="brand">QuickBite<i>.</i></div>', unsafe_allow_html=True)
    st.caption("Diner intelligence: segments, bundles, churn")
    st.markdown('<div class="side-h">Cola balance</div>', unsafe_allow_html=True)
    balance_on = st.toggle("Balance the order log", value=True,
                           help="Drops drink-only baskets, gives every diner the same number of baskets and sets cola "
                                "aside as an add-on, so heavy cola-buying students stop dominating the menu, the "
                                "rules and the recommendations. Off = numbers exactly as logged.")
    k_per_diner = st.slider("Baskets counted per diner", 2, 6, 4, disabled=not balance_on,
                            help="4 is the median diner. Diners with more baskets are down-sampled; diners with fewer are re-sampled.")

PO = prepare_orders(orders_raw, k_per_diner)
ADDONS = PO["addons"] if balance_on else []
ADDON_TXT = ", ".join(a.replace("_", " ") for a in ADDONS)
orders = PO["balanced"] if balance_on else PO["raw"]
orders_meal = orders[~orders.item.isin(ADDONS)].copy()
n_dropped = PO["n_drink_only"] if balance_on else 0

M = build_models(diners_raw)
D = M["df"].copy()
D["status"] = D.reordered.map({1: "Reordered", 0: "Did not reorder"})
SEG_MAP = D[["diner_id", "segment"]]
orders_seg = orders.merge(SEG_MAP, on="diner_id")
orders_seg_meal = orders_meal.merge(SEG_MAP, on="diner_id")
n_baskets = orders.order_id.nunique()
reorder_rate = D.reordered.mean()

seg_tbl = (D.groupby("segment").agg(diners=("diner_id", "size"), spend=("total_spend", "sum"), reorder=("reordered", "mean"),
                                    age=("age", "mean"), opm=("orders_per_month", "mean"), aov=("avg_order_value", "mean"),
                                    nitems=("avg_items_per_order", "mean"), days=("days_since_last_order", "mean"),
                                    disc=("discount_rate", "mean"), avg_spend=("total_spend", "mean")).loc[SEG_ORDER])
seg_tbl["diner_share"] = seg_tbl.diners / seg_tbl.diners.sum()
seg_tbl["spend_share"] = seg_tbl.spend / seg_tbl.spend.sum()
D["spend_churned"] = np.where(D.reordered == 0, D.total_spend, 0.0)
seg_tbl["spend_churned"] = D.groupby("segment").spend_churned.sum().loc[SEG_ORDER]
seg_tbl["churn_share"] = seg_tbl.spend_churned / seg_tbl.spend_churned.sum()

reco = (orders_seg_meal.groupby("segment").item.value_counts().groupby(level=0).head(4).reset_index(name="n")
        .groupby("segment").item.apply(list).to_dict())

# ───────────────────────────── food_models.joblib ─────────────────────────────
MODEL_FILE = Path(__file__).parent / "food_models.joblib"
REQUIRED_KEYS = {"scaler", "kmeans", "segment_names", "clf", "features", "reco"}


def models_bundle():
    return {"scaler": M["scaler"], "kmeans": M["kmeans"], "segment_names": {int(k): v for k, v in M["names"].items()},
            "clf": M["clf"], "features": list(FEATURES), "reco": reco}


def bundle_bytes():
    buf = io.BytesIO()
    joblib.dump(models_bundle(), buf)
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def load_saved_models(path_str, mtime):
    return joblib.load(path_str)


# ───────────────────────────── sidebar (part 2) ─────────────────────────────
with st.sidebar:
    st.caption(f"{len(D):,} diners · {len(orders):,} order lines · {n_baskets:,} baskets")
    if balance_on:
        st.caption(f"{n_dropped} drink-only baskets dropped"
                   + (f" · {ADDON_TXT} set aside as add-on" if ADDONS else ""))
    else:
        st.caption("Raw order log, balancing is off")
    st.markdown('<div class="side-h">Saved models</div>', unsafe_allow_html=True)
    use_saved = st.checkbox("Predictor uses food_models.joblib", value=True, disabled=not MODEL_FILE.exists(),
                            help="Loads the file next to app.py (6 required keys) instead of the live models.")
    if st.button("Save food_models.joblib", width="stretch"):
        try:
            joblib.dump(models_bundle(), MODEL_FILE)
            st.success("Saved next to app.py.")
        except Exception as e:
            st.error(f"Could not write the file here ({e}). Use the download button instead.")
    st.download_button("Download food_models.joblib", bundle_bytes(), "food_models.joblib",
                       "application/octet-stream", width="stretch")

PM = dict(scaler=M["scaler"], kmeans=M["kmeans"], names=M["names"], clf=M["clf"], reco=reco, source="live models trained in this app")
if MODEL_FILE.exists() and use_saved:
    try:
        sv = load_saved_models(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
        if not isinstance(sv, dict) or REQUIRED_KEYS - set(sv):
            raise ValueError(f"missing keys: {sorted(REQUIRED_KEYS - set(sv)) if isinstance(sv, dict) else 'not a dict'}")
        if list(sv["features"]) != FEATURES:
            raise ValueError("'features' differs from the 5 required clustering features")
        PM = dict(scaler=sv["scaler"], kmeans=sv["kmeans"], names={int(k): v for k, v in sv["segment_names"].items()},
                  clf=sv["clf"], reco=sv["reco"], source="food_models.joblib")
    except Exception as e:
        st.sidebar.warning(f"food_models.joblib ignored: {e}. Using live models.")


# ───────────────────────────── hero ─────────────────────────────
lost = D.loc[D.reordered == 0, "total_spend"].sum()
st.markdown(f"""<div class="hero"><h1>{1 - reorder_rate:.0%} of our diners did not reorder. They hold <span class="hl">AED {lost / 1000:,.0f}k</span> of lifetime spend.</h1>
<p>QuickBite has {len(D):,} diners in three behavioural segments. Each segment orders from a different menu, and the ones with the most money are the ones leaving.</p></div>""",
            unsafe_allow_html=True)


def strip_row(label, shares):
    cells = "".join(f'<div style="width:{shares[s] * 100:.2f}%;background:{SEG_COLORS[s]};color:{SEG_TEXT[s]}" '
                    f'title="{s}: {shares[s]:.0%}">{pct(shares[s]) if shares[s] > .06 else ""}</div>' for s in SEG_ORDER)
    return f'<div class="strip-row"><div class="strip-label">{label}</div><div class="strip-bar">{cells}</div></div>'


legend = "".join(f'<span><b style="background:{SEG_COLORS[s]}"></b>{s}</span>' for s in SEG_ORDER)
st.markdown(f'<div class="strip"><div class="strip-legend">{legend}</div>'
            + strip_row("Share of diners", seg_tbl.diner_share)
            + strip_row("Share of lifetime spend", seg_tbl.spend_share)
            + strip_row("Share of spend that left", seg_tbl.churn_share) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Segments", "Baskets", "Churn", "Predictor", "Actions"])

# ═════════════════════════ OVERVIEW ═════════════════════════
with tabs[0]:
    sel = seg_filter("flt_overview")
    DF = D[D.segment.isin(sel)]
    c = st.columns(5)
    stat(c[0], "Diners", f"{len(DF):,}", f"of {len(D):,}")
    stat(c[1], "Reorder rate", pct(DF.reordered.mean()), "ordered again")
    stat(c[2], "Avg order value", f"AED {DF.avg_order_value.mean():,.0f}", "per order")
    stat(c[3], "Lifetime spend", f"AED {DF.total_spend.sum() / 1000:,.0f}k", f"AED {DF.total_spend.mean():,.0f} per diner")
    stat(c[4], "Baskets", f"{n_baskets:,}", f"{len(orders) / n_baskets:.1f} items per basket")

    a, b = st.columns(2, gap="large")
    with a:
        head("Who comes back", "Reorder rate inside each segment.")
        rr = DF.groupby("segment").reordered.mean().reindex([s for s in SEG_ORDER if s in sel])
        fig = go.Figure()
        fig.add_bar(y=rr.index, x=rr.values, orientation="h", name="Reordered", marker_color=C_OK,
                    text=[pct(v) for v in rr.values], textposition="inside", insidetextanchor="start", textfont=dict(color="#fff", size=14))
        fig.add_bar(y=rr.index, x=1 - rr.values, orientation="h", name="Did not reorder", marker_color=C_BAD,
                    text=[pct(1 - v) for v in rr.values], textposition="inside", insidetextanchor="start", textfont=dict(color="#fff", size=14))
        fig.update_layout(barmode="stack", bargap=.35)
        fig.update_xaxes(tickformat=".0%", showgrid=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        plot(fig, height=250, legend=True)
    with b:
        head("Silence predicts churn", "Reorder rate by days since the last order. Green is above the overall rate, red below.")
        bands = pd.cut(DF.days_since_last_order, [-1, 3, 7, 14, 21, 60], labels=["0–3", "4–7", "8–14", "15–21", "22+"])
        rb = DF.groupby(bands, observed=True).reordered.agg(rate="mean", n="size").reset_index()
        rb.columns = ["band", "rate", "n"]
        fig = go.Figure(go.Bar(x=rb.band, y=rb.rate, marker_color=[C_OK if r >= DF.reordered.mean() else C_BAD for r in rb.rate],
                               text=[f"{r:.0%}<br>n={n}" for r, n in zip(rb.rate, rb.n)], textposition="outside"))
        fig.add_hline(y=DF.reordered.mean(), line_dash="dot", line_color=INK,
                      annotation_text=f"overall {DF.reordered.mean():.0%}", annotation_position="top right")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.25], title="Reorder rate")
        fig.update_xaxes(title="Days since last order", showgrid=False)
        plot(fig, height=250)

    # ---- the cola block ----
    addon_name = (PO["addons"][0] if PO["addons"] else "cola")
    nm = addon_name.replace("_", " ")
    seg_raw, rate_raw, all_raw = attach_audit(PO["raw"], SEG_MAP, addon_name)
    seg_bal, rate_bal, all_bal = attach_audit(PO["balanced"], SEG_MAP, addon_name)
    head(f"The {nm} skew, rebalanced",
         f"Left: who writes the baskets. Right: how often {nm} is in them. Balancing gives every diner the same weight, so the overall {nm} rate "
         "reflects the diner mix instead of the heaviest orderers.")
    a, b = st.columns(2, gap="large")
    with a:
        fig = go.Figure()
        fig.add_bar(name="Share of diners", x=SEG_ORDER, y=seg_tbl.diner_share.values, marker_color=INK,
                    text=[pct(v) for v in seg_tbl.diner_share], textposition="outside")
        fig.add_bar(name="Share of baskets, as logged", x=SEG_ORDER, y=seg_raw.values, marker_color=C_BAD,
                    text=[pct(v) for v in seg_raw], textposition="outside")
        fig.add_bar(name="Share of baskets, balanced", x=SEG_ORDER, y=seg_bal.values, marker_color=C_OK,
                    text=[pct(v) for v in seg_bal], textposition="outside")
        fig.update_layout(barmode="group", bargap=.28)
        fig.update_yaxes(tickformat=".0%", range=[0, max(seg_raw.max(), seg_bal.max(), seg_tbl.diner_share.max()) * 1.2], title="Share of total")
        fig.update_xaxes(showgrid=False)
        plot(fig, height=360, legend=True)
    with b:
        cats = ["All diners"] + SEG_ORDER
        fig = go.Figure()
        fig.add_bar(name="As logged", x=cats, y=[all_raw] + list(rate_raw.values), marker_color=C_BAD,
                    text=[pct(v) for v in [all_raw] + list(rate_raw.values)], textposition="outside")
        fig.add_bar(name="Balanced", x=cats, y=[all_bal] + list(rate_bal.values), marker_color=C_OK,
                    text=[pct(v) for v in [all_bal] + list(rate_bal.values)], textposition="outside")
        fig.update_layout(barmode="group", bargap=.3)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.15], title=f"Baskets containing {nm}")
        fig.update_xaxes(showgrid=False)
        plot(fig, height=360, legend=True)
    top_seg = seg_raw.idxmax()
    note(f"<b>The imbalance:</b> {top_seg} are {seg_tbl.diner_share[top_seg]:.0%} of diners but write {seg_raw[top_seg]:.0%} of the logged baskets, "
         f"and {rate_raw[top_seg]:.0%} of those hold {nm}. That is why the raw log says {nm} is in <b>{all_raw:.0%}</b> of all baskets.", "bad")
    note(f"<b>After balancing:</b> basket shares match diner shares and the overall {nm} rate is <b>{all_bal:.0%}</b>. "
         f"What remains is real behaviour: {nm} is a {top_seg.lower()} habit ({rate_bal[top_seg]:.0%} of their baskets, "
         f"{rate_bal['Premium foodies']:.0%} for Premium foodies). It is kept out of the dish ranking, rules and recommendations, and reported here as an attach rate."
         + ("" if balance_on else " <i>Balancing is currently switched off in the sidebar, so the rest of the dashboard still uses the raw log.</i>"), "ok")

    a, b = st.columns(2, gap="large")
    with a:
        head("Most ordered dishes" if ADDONS else "Most ordered items",
             f"Share of baskets containing the item{f' ({ADDON_TXT} excluded)' if ADDONS else ''}.")
        sel_log = orders_seg[orders_seg.segment.isin(sel)]
        sel_meal = orders_seg_meal[orders_seg_meal.segment.isin(sel)]
        pop = basket_share(sel_meal, sel_log).sort_values().rename_axis("item").reset_index(name="share")
        pop["item"] = pop["item"].str.replace("_", " ")
        fig = go.Figure(go.Bar(x=pop.share, y=pop.item, orientation="h", marker_color=C_NEUTRAL,
                               text=pop.share.map(pct), textposition="outside"))
        fig.update_xaxes(tickformat=".0%", range=[0, pop.share.max() * 1.18], title="Share of baskets")
        fig.update_yaxes(dtick=1, showgrid=False)
        plot(fig, height=420)
    with b:
        head("Retained vs churned", "Compare any behaviour. Distributions are normalised so group sizes do not matter.")
        feat = st.selectbox("Behaviour", list(NICE), index=4, format_func=lambda k: NICE[k], label_visibility="collapsed")
        fig = px.histogram(DF, x=feat, color="status", barmode="overlay", opacity=.7, nbins=30, histnorm="probability density",
                           color_discrete_map=STATUS_COLORS, labels={feat: NICE[feat], "status": ""})
        fig.update_yaxes(title="Density", showgrid=True)
        fig.update_xaxes(showgrid=False)
        plot(fig, height=370, legend=True)

    r_hi = DF[DF.days_since_last_order > 14].reordered.mean() if (DF.days_since_last_order > 14).any() else np.nan
    r_lo = DF[DF.days_since_last_order <= 7].reordered.mean()
    note(f"<b>Class balance:</b> {reorder_rate:.0%} of diners reorder, so always guessing “will reorder” is already {reorder_rate:.0%} accurate "
         f"and catches <b>0%</b> of churners. Accuracy alone is misleading.")
    note(f"<b>Recency is the early warning:</b> diners idle for more than 14 days reorder <b>{r_hi:.0%}</b> of the time, against {r_lo:.0%} for those active in the last week.", "bad")
    with st.expander("What the balancing did, step by step"):
        st.markdown(f"""
- **Drink-only baskets:** {PO['n_drink_only']} of {orders_raw.order_id.nunique():,} baskets held nothing but drinks. They are dropped, because they are not a meal occasion.
- **Equal weight per diner:** each diner contributes {k_per_diner} baskets. Diners with more are randomly down-sampled (fixed seed); diners with fewer keep all theirs and the rest are re-drawn with replacement. This removes the volume skew without inventing new items.
- **Add-on set aside:** {', '.join(PO['addons']) or 'nothing'} appears in over {ADDON_THRESHOLD:.0%} of the cleaned baskets, so it co-occurs with everything and creates trivial rules. It stays in the order history; bundles and recommendations are built on dishes.
- **Known limit:** the order log never holds more than {int(orders_raw.groupby('order_id').size().max())} items per basket although many Families report 6+, so large baskets look cut off.
""")

# ═════════════════════════ SEGMENTS ═════════════════════════
with tabs[1]:
    head("Who are our diners?", "K-Means on five behavioural features, standardised. Age and total spend are held out to validate the segments.")
    a, b = st.columns(2, gap="large")
    best_k = M["ks"][int(np.argmax(M["sil"]))]
    with a:
        fig = go.Figure(go.Scatter(x=M["ks"], y=M["inertia"], mode="lines+markers", line=dict(color=C_NEUTRAL, width=2.5), marker=dict(size=9)))
        fig.add_vline(x=3, line_dash="dot", line_color=INK, annotation_text="chosen k = 3", annotation_position="top right")
        fig.update_xaxes(title="Number of clusters (k)", dtick=1)
        fig.update_yaxes(title="Inertia (elbow method)")
        plot(fig, height=290)
    with b:
        fig = go.Figure(go.Scatter(x=M["ks"], y=M["sil"], mode="lines+markers", line=dict(color=C_NEUTRAL, width=2.5), marker=dict(size=9)))
        fig.add_vline(x=3, line_dash="dot", line_color=INK, annotation_text="chosen k = 3", annotation_position="top right")
        fig.update_xaxes(title="Number of clusters (k)", dtick=1)
        fig.update_yaxes(title="Mean silhouette score")
        plot(fig, height=290)
    note(f"<b>Why 3 segments?</b> Silhouette at k=3 is <b>{M['sil'][1]:.2f}</b> (best k = {best_k}) and the inertia gain beyond k=3 is small. "
         "Three groups are separable and easy to act on.")

    seg_sc = seg_filter("flt_segments")
    DFs = D[D.segment.isin(seg_sc)]
    a, b = st.columns(2, gap="large")
    with a:
        head("Segments in two dimensions", "PCA of the five scaled features.")
        fig = px.scatter(DFs, x="pc1", y="pc2", color="segment", color_discrete_map=SEG_COLORS, opacity=.75,
                         category_orders={"segment": SEG_ORDER},
                         hover_data={"diner_id": True, "orders_per_month": True, "avg_order_value": ":.0f", "pc1": False, "pc2": False},
                         labels={"pc1": f"Component 1 ({M['pca'].explained_variance_ratio_[0]:.0%} of variance)",
                                 "pc2": f"Component 2 ({M['pca'].explained_variance_ratio_[1]:.0%})", "segment": ""})
        fig.update_traces(marker=dict(size=8, line=dict(color="#fff", width=1)))
        plot(fig, height=400, legend=True)
    with b:
        head("Frequency against ticket size", "Bubble size is items per order.")
        fig = px.scatter(DFs, x="orders_per_month", y="avg_order_value", color="segment", color_discrete_map=SEG_COLORS,
                         size="avg_items_per_order", size_max=14, opacity=.7, category_orders={"segment": SEG_ORDER},
                         labels={"orders_per_month": "Orders per month", "avg_order_value": "Average order value (AED)", "segment": "",
                                 "avg_items_per_order": "Items / order"})
        plot(fig, height=400, legend=True)

    head("What makes each segment different", "Each cell is how many standard deviations the segment sits from the average diner.")
    z_cols = FEATURES + ["age", "total_spend"]
    z = D.groupby("segment")[z_cols].mean().loc[SEG_ORDER]
    z = (z - D[z_cols].mean()) / D[z_cols].std()
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        zr = z.rename(columns={"orders_per_month": "Orders/month", "avg_order_value": "Order value", "avg_items_per_order": "Items/order",
                               "days_since_last_order": "Days idle", "discount_rate": "Discount rate", "age": "Age", "total_spend": "Total spend"})
        fig = px.imshow(zr, text_auto=".1f", aspect="auto", zmin=-2, zmax=2,
                        color_continuous_scale=[[0, "#2F6FDE"], [.5, "#F2F4F1"], [1, "#E4572E"]], labels=dict(color="Std. dev."))
        fig.update_traces(xgap=3, ygap=3)
        fig.update_xaxes(tickangle=-40, side="bottom")
        plot(fig, height=300)
    with b:
        fig = go.Figure()
        fig.add_bar(name="Share of diners", x=SEG_ORDER, y=seg_tbl.diner_share, marker_color=INK, text=seg_tbl.diner_share.map(pct), textposition="outside")
        fig.add_bar(name="Share of lifetime spend", x=SEG_ORDER, y=seg_tbl.spend_share, marker_color=C_NEUTRAL, text=seg_tbl.spend_share.map(pct), textposition="outside")
        fig.update_layout(barmode="group", bargap=.3)
        fig.update_yaxes(tickformat=".0%", range=[0, seg_tbl[["diner_share", "spend_share"]].values.max() * 1.25], title="Share of total")
        fig.update_xaxes(showgrid=False)
        plot(fig, height=300, legend=True)

    prof = seg_tbl.rename(columns={"diners": "Diners", "age": "Age", "opm": "Orders / month", "aov": "Avg order (AED)", "nitems": "Items / order",
                                   "days": "Days since last order", "disc": "Discount rate", "avg_spend": "Avg lifetime spend (AED)",
                                   "reorder": "Reorder rate"})[
        ["Diners", "Age", "Orders / month", "Avg order (AED)", "Items / order", "Days since last order", "Discount rate",
         "Avg lifetime spend (AED)", "Reorder rate"]]
    st.dataframe(prof, width="stretch", column_config={
        "Age": st.column_config.NumberColumn(format="%.0f"), "Orders / month": st.column_config.NumberColumn(format="%.1f"),
        "Avg order (AED)": st.column_config.NumberColumn(format="%.0f"), "Items / order": st.column_config.NumberColumn(format="%.1f"),
        "Days since last order": st.column_config.NumberColumn(format="%.0f"),
        "Discount rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Avg lifetime spend (AED)": st.column_config.NumberColumn(format="%.0f"),
        "Reorder rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

    head("Where the money sits", "Lifetime spend of diners who reordered against those who did not, in AED thousands.")
    fig = go.Figure()
    fig.add_bar(name="Reordered", y=SEG_ORDER, x=(seg_tbl.spend - seg_tbl.spend_churned) / 1000, orientation="h", marker_color=C_OK,
                text=[f"{v / 1000:.0f}k" for v in seg_tbl.spend - seg_tbl.spend_churned], textposition="inside", textfont=dict(color="#fff"))
    fig.add_bar(name="Did not reorder", y=SEG_ORDER, x=seg_tbl.spend_churned / 1000, orientation="h", marker_color=C_BAD,
                text=[f"{v / 1000:.0f}k" for v in seg_tbl.spend_churned], textposition="inside", textfont=dict(color="#fff"))
    fig.update_layout(barmode="stack", bargap=.35)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Lifetime spend (AED thousands)")
    plot(fig, height=260, legend=True)

    pf, fam, bud = seg_tbl.loc["Premium foodies"], seg_tbl.loc["Families"], seg_tbl.loc["Budget students"]
    note(f"<b>Premium foodies</b> are {pf.diner_share:.0%} of diners but {pf.spend_share:.0%} of lifetime spend, rarely use discounts ({pf.disc:.0%}), "
         f"and only <b>{pf.reorder:.0%}</b> reorder: the most valuable group with the weakest hold.", "bad")
    note(f"<b>Families</b> order rarely ({fam.opm:.0f} a month) in big baskets ({fam.nitems:.1f} items, AED {fam.aov:.0f}); only <b>{fam.reorder:.0%}</b> come back. A re-engagement problem.")
    note(f"<b>Budget students</b> order {bud.opm:.0f} times a month with small, discount-driven tickets (AED {bud.aov:.0f}, {bud.disc:.0%} discounted) and "
         f"<b>{bud.reorder:.0%}</b> reorder. Already loyal; blanket discounts mostly give away margin.", "ok")

# ═════════════════════════ BASKETS ═════════════════════════
with tabs[2]:
    head("What do diners order together?", "Apriori association rules, one basket per order. Lift above 1 means the items are ordered together more than by chance."
         + (f" Mined on dishes only: {ADDON_TXT} is an add-on that pairs with everything." if ADDONS else ""))
    c1, c2, c3 = st.columns([1.2, 1, 1], gap="large")
    scope = c1.selectbox("Analyse baskets of", ["All diners"] + SEG_ORDER)
    min_sup = c2.slider("Minimum support", 0.01, 0.15, 0.03 if scope == "All diners" else 0.05, 0.005, format="%.3f",
                        help="Share of baskets an item combination must appear in. Segments have fewer baskets, so use a smaller value.")
    min_lift = c3.slider("Minimum lift", 1.0, 5.0, 1.2, 0.1)

    sub = orders_meal if scope == "All diners" else orders_seg_meal[orders_seg_meal.segment == scope][["order_id", "diner_id", "item"]]
    basket, rules = mine_rules(sub, min_sup, 1.0)
    if rules.empty:
        st.warning("No rules at this support. Lower the minimum support.")
    else:
        rules_f = rules[rules.lift >= min_lift]
        uniq = rules_f.drop_duplicates("itemset")
        k = st.columns(4)
        stat(k[0], "Baskets analysed", f"{len(basket):,}", scope)
        stat(k[1], "Rules found", f"{len(rules_f):,}", f"lift ≥ {min_lift:.1f}")
        stat(k[2], "Unique combinations", f"{len(uniq):,}", "A→B and B→A merged")
        stat(k[3], "Strongest lift", f"{rules_f.lift.max():.1f}" if len(rules_f) else "–", "times more than chance")
        if rules_f.empty:
            st.warning("No rules above this lift. Lower the minimum lift.")
        else:
            a, b = st.columns([1.1, 1], gap="large")
            with a:
                head("Strongest bundles", "Top 10 by lift. Bar label: lift · confidence.")
                t = uniq.head(10).iloc[::-1]
                fig = go.Figure(go.Bar(x=t.lift, y=t.rule, orientation="h", marker_color=SEG_COLORS.get(scope, C_NEUTRAL),
                                       text=[f"{l:.1f} · {c:.0%}" for l, c in zip(t.lift, t.confidence)], textposition="outside"))
                fig.update_xaxes(title="Lift", range=[0, t.lift.max() * 1.3])
                fig.update_yaxes(showgrid=False)
                plot(fig, height=440)
            with b:
                head("Common against reliable", "Each dot is a rule; size and colour show lift.")
                fig = px.scatter(rules_f, x="support", y="confidence", color="lift", size="lift", size_max=20,
                                 color_continuous_scale=["#CFE0F7", "#2F6FDE", "#12202B"], hover_name="rule",
                                 labels={"support": "Support (share of baskets)", "confidence": "Confidence", "lift": "Lift"})
                fig.update_xaxes(tickformat=".0%")
                fig.update_yaxes(tickformat=".0%")
                plot(fig, height=440)
            tbl = uniq[["rule", "support", "confidence", "lift"]].head(15).rename(
                columns={"rule": "Rule", "support": "Support", "confidence": "Confidence", "lift": "Lift"})
            st.dataframe(tbl, hide_index=True, width="stretch", column_config={
                "Support": st.column_config.NumberColumn(format="percent"),
                "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Lift": st.column_config.NumberColumn(format="%.1f")})
            best = uniq.iloc[0]
            note(f"<b>Top bundle for {scope.lower() if scope != 'All diners' else 'all diners'}:</b> {best.bundle}, lift <b>{best.lift:.1f}</b>. "
                 f"When “{' + '.join(sorted(best.antecedents)).replace('_', ' ')}” is in the basket, “{' + '.join(sorted(best.consequents)).replace('_', ' ')}” "
                 f"follows <b>{best.confidence:.0%}</b> of the time. Promote it as one combo.")
            if scope in SEG_ORDER and uniq.lift.max() < 1.6:
                note(f"<b>Weak cross-sell signal:</b> inside {scope} nearly everyone orders the same few items, so lifts stay low. The lever is a fixed-price meal deal, not recommendations.", "bad")

        head("Item-to-item lift", "Which items are bought together (1 = independent). Use the finder to see the best add-ons for any cart item.")
        B = basket.values.astype(float)
        p = B.mean(0)
        pl = (B.T @ B / len(B)) / np.outer(p, p)
        np.fill_diagonal(pl, np.nan)
        lab = [c.replace("_", " ") for c in basket.columns]
        order = np.argsort(-np.nan_to_num(pl).sum(0))
        pl_df = pd.DataFrame(pl, index=lab, columns=lab).iloc[order, order]
        a, b = st.columns([1.3, 1], gap="large")
        with a:
            fig = px.imshow(pl_df, text_auto=".1f", aspect="auto", color_continuous_scale=["#F2F4F1", "#9DBCEF", "#12202B"], labels=dict(color="Lift"))
            fig.update_traces(xgap=2, ygap=2)
            fig.update_xaxes(tickangle=-40)
            plot(fig, height=500)
        with b:
            cart = st.selectbox("If the cart contains…", sorted(lab))
            row = pl_df.loc[cart].drop(cart, errors="ignore").sort_values(ascending=False).head(5)
            cm_ = B[:, basket.columns.get_loc(cart.replace(" ", "_"))] > 0
            out = pd.DataFrame({"Suggest": row.index, "Lift": row.values,
                                "P": [B[cm_, basket.columns.get_loc(i.replace(" ", "_"))].mean() for i in row.index]})
            fig = go.Figure(go.Bar(x=out.P.iloc[::-1], y=out.Suggest.iloc[::-1], orientation="h", marker_color=C_NEUTRAL,
                                   text=[f"{v:.0%}" for v in out.P.iloc[::-1]], textposition="outside"))
            fig.update_xaxes(tickformat=".0%", range=[0, 1.12], title=f"Baskets with {cart} that also hold this")
            fig.update_yaxes(showgrid=False)
            plot(fig, height=320)

    head("Menu fingerprint of each segment", "Share of each segment's baskets that contain the item"
         + (f", {ADDON_TXT} included so you can see who actually orders it." if ADDONS else "."))
    seg_baskets = orders_seg.drop_duplicates("order_id").segment.value_counts()
    share = (orders_seg.drop_duplicates(["order_id", "item"]).groupby(["segment", "item"]).size().unstack(fill_value=0)
             .div(seg_baskets, axis=0).loc[SEG_ORDER])
    share = share[share.max().sort_values(ascending=False).index]
    fig = px.imshow(share.rename(columns=lambda c: c.replace("_", " ")), text_auto=".0%", aspect="auto",
                    color_continuous_scale=["#F2F4F1", "#9DBCEF", "#12202B"], labels=dict(color="Share"))
    fig.update_traces(xgap=2, ygap=2)
    plot(fig, height=260)
    own = (share > 0.05).sum()
    note(f"<b>Segments barely overlap:</b> {int((own == 1).sum())} of {len(own)} menu items are ordered (more than 5% of baskets) by only one segment, "
         "so bundles should be targeted per segment.")

# ═════════════════════════ CHURN ═════════════════════════
with tabs[3]:
    head("Who will reorder?", None)
    mt, cm = M["metrics"], M["metrics"]["cm"]
    st.caption(f"Logistic regression with churn weighted ×2, seven features, stratified 75/25 split, {mt['n_test']} unseen test diners.")
    c = st.columns(4)
    stat(c[0], "Model accuracy", pct(mt["acc"]), f"baseline {pct(mt['base_acc'])} (always “reorder”)")
    stat(c[1], "Churners caught", pct(mt["churn_recall"]), "baseline catches 0%")
    stat(c[2], "ROC AUC", f"{mt['auc']:.2f}", "1.0 is a perfect ranking")
    stat(c[3], "Missed churners", f"{cm[0, 1]} of {cm[0].sum()}", f"{cm[1, 0]} false alarms among {cm[1].sum()} loyal diners")

    a, b = st.columns(2, gap="large")
    with a:
        head("Confusion matrix", "Test set. Each row adds to 100%.")
        cmn = cm / cm.sum(axis=1, keepdims=True)
        txt = [[f"<b>{cm[i, j]}</b><br>{cmn[i, j]:.0%} of actual" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=cmn, x=["Predicted: churn", "Predicted: reorder"], y=["Actual: churned", "Actual: reordered"],
                                   text=txt, texttemplate="%{text}", colorscale=[[0, "#E7EBE6"], [1, "#3D5A6C"]], showscale=False, xgap=3, ygap=3))
        fig.update_yaxes(autorange="reversed")
        plot(fig, height=330)
    with b:
        head("ROC curve", "How well the model ranks churners above loyal diners.")
        fig = go.Figure()
        fig.add_scatter(x=mt["fpr"], y=mt["tpr"], mode="lines", name=f"Model (AUC {mt['auc']:.2f})", line=dict(color=C_NEUTRAL, width=3),
                        fill="tozeroy", fillcolor="rgba(61,90,108,.12)")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(color=MUTED, dash="dot"))
        fig.update_xaxes(title="False-positive rate (loyal diners wrongly flagged)")
        fig.update_yaxes(title="True-positive rate")
        plot(fig, height=330, legend=True)
    note(f"<b>Beats the baseline:</b> {mt['acc']:.0%} accuracy against {mt['base_acc']:.0%}, and it catches <b>{mt['churn_recall']:.0%}</b> of churners where the baseline catches none.", "ok")
    note(f"<b>Which mistake hurts more?</b> Missing a churner loses a customer; a false alarm costs a small voucher. The model misses {cm[0, 1]} churners "
         f"and raises {cm[1, 0]} false alarms, the cheaper error, by design.", "bad")

    a, b = st.columns(2, gap="large")
    with a:
        head("What drives reordering", "Green raises the chance of reordering, red lowers it.")
        co = M["coefs"].rename(index=NICE).sort_values()
        fig = go.Figure(go.Bar(x=co.values, y=co.index, orientation="h", marker_color=[C_BAD if v < 0 else C_OK for v in co.values],
                               text=[f"{v:+.2f}" for v in co.values], textposition="outside"))
        fig.update_xaxes(title="Standardised coefficient (log-odds per +1 std. dev.)", range=[co.min() - .5, co.max() + .5])
        fig.update_yaxes(showgrid=False)
        plot(fig, height=380)
    with b:
        head("Choose the alert threshold by cost", "Out-of-fold predictions, so the numbers are honest.")
        s1, s2 = st.columns(2)
        cfn = s1.slider("Cost of a missed churner", 1, 10, 3)
        cfp = s2.slider("Cost of a false alarm", 1, 10, 1)
        isc = (D.reordered == 0).values
        rows = []
        for t in np.linspace(0.05, 0.95, 91):
            flag = D.p_churn.values >= t
            fn, fp = int((isc & ~flag).sum()), int((~isc & flag).sum())
            rows.append((t, fn, fp, cfn * fn + cfp * fp))
        th = pd.DataFrame(rows, columns=["t", "fn", "fp", "cost"])
        best_t = th.loc[th.cost.idxmin()]
        fig = go.Figure(go.Scatter(x=th.t, y=th.cost, mode="lines", line=dict(color=C_BAD, width=3)))
        fig.add_vline(x=best_t.t, line_dash="dot", line_color=INK, annotation_text=f"best ≈ {best_t.t:.2f}", annotation_position="top right")
        fig.update_xaxes(title="Flag a diner as at risk when P(churn) ≥")
        fig.update_yaxes(title="Total expected cost")
        plot(fig, height=250)
        note(f"A missed churner costs <b>{cfn / cfp:.1f}×</b> a false alarm, so flag diners from <b>P(churn) ≥ {best_t.t:.2f}</b>: {int(best_t.fn)} missed, {int(best_t.fp)} false alarms.")

    head("At-risk win-back list", "Highest-spending diners first.")
    thr = st.slider("Flag diners with P(churn) of at least", 0.05, 0.95, float(round(best_t.t, 2)), 0.01)
    seg_risk = seg_filter("flt_risk")
    risk = D[(D.p_churn >= thr) & D.segment.isin(seg_risk)].sort_values("total_spend", ascending=False)
    risk_out = risk[["diner_id", "segment", "p_churn", "total_spend", "days_since_last_order", "orders_per_month", "status"]].rename(
        columns={"p_churn": "P(churn)", "total_spend": "Total spend (AED)", "days_since_last_order": "Days since last order",
                 "orders_per_month": "Orders / month", "status": "Actual outcome", "diner_id": "Diner", "segment": "Segment"})
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        st.dataframe(risk_out.head(50), hide_index=True, width="stretch", height=360, column_config={
            "P(churn)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Total spend (AED)": st.column_config.NumberColumn(format="%,.0f")})
        st.download_button("Download full at-risk list (CSV)", risk_out.to_csv(index=False).encode(), "at_risk_diners.csv", "text/csv")
    with b:
        fl = risk.groupby("segment").agg(n=("diner_id", "size"), spend=("total_spend", "sum")).reindex(SEG_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=fl.index, y=fl.n, marker_color=[SEG_COLORS[s] for s in fl.index],
                               text=[f"{int(n)} diners<br>AED {s / 1000:.0f}k" for n, s in zip(fl.n, fl.spend)], textposition="outside"))
        fig.update_yaxes(range=[0, max(fl.n.max() * 1.3, 1)], title="Diners flagged")
        fig.update_xaxes(showgrid=False)
        st.markdown(f"**{len(risk)} diners would get a win-back offer**")
        plot(fig, height=330)

# ═════════════════════════ PREDICTOR ═════════════════════════
def predictor_tab():
    head("Diner predictor", "Look up a diner or enter a profile to get a segment, a reorder likelihood and recommended items.")
    st.caption(f"Models in use: {PM['source']}")
    med, lo, hi = D[CLF_FEATURES].median(), D[CLF_FEATURES].min(), D[CLF_FEATURES].max()
    mode = st.radio("Input", ["Look up a diner ID", "Enter a profile"], horizontal=True, label_visibility="collapsed")
    lookup = mode.startswith("Look")
    diner_id, diner_row = None, None

    if lookup:
        diner_id = st.selectbox("Diner ID", sorted(D.diner_id), index=None, placeholder="Type to search, e.g. D1034")
        if diner_id is None:
            st.info("Pick a diner ID to see their segment, reorder likelihood and recommendations.")
            return
        diner_row = D[D.diner_id == diner_id].iloc[0]
        base = {k: float(diner_row[k]) for k in CLF_FEATURES}
        ns = diner_id
        fmt = {"age": "{:.0f}", "orders_per_month": "{:.0f}", "avg_order_value": "{:,.2f}", "avg_items_per_order": "{:.0f}",
               "days_since_last_order": "{:.0f}", "discount_rate": "{:.0%}", "total_spend": "{:,.0f}"}
        m = st.columns(7)
        for col, k in zip(m, CLF_FEATURES):
            col.metric(NICE[k], fmt[k].format(base[k]))
        what_if = st.toggle("Adjust values (what-if)", value=False, key=f"wi_{diner_id}")
    else:
        presets = {
            "Custom (use controls)": None,
            "Student: frequent, cheap, recent": dict(age=21, orders_per_month=14, avg_order_value=40, avg_items_per_order=2, days_since_last_order=3, discount_rate=0.7, total_spend=550),
            "Family: big rare basket, long gap": dict(age=41, orders_per_month=3, avg_order_value=175, avg_items_per_order=6, days_since_last_order=20, discount_rate=0.3, total_spend=620),
            "Premium foodie: ordered 6 days ago": dict(age=33, orders_per_month=9, avg_order_value=100, avg_items_per_order=3, days_since_last_order=6, discount_rate=0.1, total_spend=900),
            "Premium foodie: silent for 22 days": dict(age=33, orders_per_month=9, avg_order_value=100, avg_items_per_order=3, days_since_last_order=22, discount_rate=0.1, total_spend=900),
        }
        choice = st.selectbox("Start from an example", list(presets))
        pre = presets[choice] or {k: float(med[k]) for k in CLF_FEATURES}
        base = {k: float(pre[k]) for k in CLF_FEATURES}
        ns, what_if = choice, True

    if what_if:
        s1, s2, s3, s4 = st.columns(4)
        age = s1.slider("Age", 5, 80, int(round(base["age"])), key=f"a_{ns}")
        opm = s2.slider("Orders per month", 1, 20, int(round(base["orders_per_month"])), key=f"o_{ns}")
        aov = s3.number_input("Avg order value (AED)", 10.0, 500.0, float(base["avg_order_value"]), 1.0, key=f"v_{ns}")
        items = s4.slider("Avg items per order", 1, 10, int(round(base["avg_items_per_order"])), key=f"i_{ns}")
        s5, s6, s7, _ = st.columns(4)
        days = s5.slider("Days since last order", 0, 60, int(round(base["days_since_last_order"])), key=f"d_{ns}")
        disc = s6.slider("Discount rate", 0.0, 1.0, float(base["discount_rate"]), 0.01, key=f"r_{ns}")
        spend = s7.number_input("Total spend (AED)", 0.0, 10000.0, float(base["total_spend"]), 10.0, key=f"s_{ns}")
        prof = dict(age=age, orders_per_month=opm, avg_order_value=aov, avg_items_per_order=items,
                    days_since_last_order=days, discount_rate=disc, total_spend=spend)
    else:
        prof = dict(base)

    def analyse(p):
        row = pd.DataFrame([p])
        seg_id = int(PM["kmeans"].predict(PM["scaler"].transform(row[FEATURES]))[0])
        return PM["names"].get(seg_id, f"Segment {seg_id}"), float(PM["clf"].predict_proba(row[CLF_FEATURES])[0][1])

    seg_name, p_re = analyse(prof)
    color = SEG_COLORS.get(seg_name, C_NEUTRAL)
    tcolor = SEG_TEXT.get(seg_name, "#fff")
    seg_items = list(PM["reco"].get(seg_name, []))

    fav, cross = [], []
    if lookup:
        mine = orders[orders.diner_id == diner_id]
        n_baskets_d = mine.order_id.nunique()
        cnt = orders_meal[orders_meal.diner_id == diner_id].item.value_counts()
        fav = list(cnt.index)
        lift_df, conf_df = pair_stats(orders_meal)
        owned = set(fav)
        for c_ in lift_df.columns:
            if c_ in owned:
                continue
            best = None
            for i in owned:
                l, cf = float(lift_df.loc[i, c_]), float(conf_df.loc[i, c_])
                if cf >= 0.15 and l >= 1.5 and (best is None or l > best[1]):
                    best = (i, l, cf)
            if best:
                cross.append((c_,) + best)
        cross.sort(key=lambda x: -x[2])

    st.write("")
    verdict = "Likely to reorder" if p_re >= 0.5 else "At risk of churning"
    shown = (fav[:3] + [i for i in seg_items if i not in fav[:3]])[:4] if lookup else seg_items
    k = st.columns([1, 1.3, 1.2], gap="large")
    stat(k[0], "Segment", f'<span class="pill" style="background:{color};color:{tcolor};font-size:1.1rem;padding:5px 14px">{seg_name}</span>', "nearest K-Means centre")
    mcol = C_OK if p_re >= 0.5 else C_BAD
    k[1].markdown(f'<div class="stat"><div class="stat-v" style="color:{mcol}">{p_re:.0%}</div><div class="stat-l">{verdict}</div>'
                  f'<div class="meter"><div class="meter-fill" style="width:{p_re * 100:.0f}%;background:{mcol}"></div><div class="meter-tick"></div></div>'
                  f'<div class="stat-s">Probability of reordering. The tick marks 50%.</div></div>', unsafe_allow_html=True)
    k[2].markdown(f'<div class="stat"><div class="stat-l" style="margin-bottom:6px">Recommended items</div>{pills(shown, color, tcolor)}</div>', unsafe_allow_html=True)

    outside = [NICE[c_] for c_ in CLF_FEATURES if not (lo[c_] <= prof[c_] <= hi[c_])]
    if outside:
        note("<b>Heads-up:</b> outside the range the model saw in training for " + ", ".join(outside) + ". Treat the prediction with caution.", "bad")

    if lookup:
        seg_s, p_s = analyse(base)
        actual = int(diner_row.reordered)
        oof = 1 - float(diner_row.p_churn)
        match = (p_s >= 0.5) == (actual == 1)
        note(f"<b>{diner_id}</b> actually <b>{'reordered' if actual else 'did not reorder'}</b>. On the stored profile the model says P(reorder) = <b>{p_s:.0%}</b>, "
             f"{'a correct call' if match else 'a miss'}. A model that never saw this diner (5-fold out-of-fold) says <b>{oof:.0%}</b>; "
             "the live model is trained on every diner, so its own number is optimistic.", "ok" if match else "bad")
        head(f"Recommendations for {diner_id}")
        c1, c2, c3 = st.columns(3, gap="large")
        c1.markdown(f'<div class="plan" style="border-top-color:{color}"><h4>Their favourites</h4><div class="why">{n_baskets_d} baskets on record'
                    + (", dishes only" if ADDONS else "") + '</div><p style="margin-top:10px">'
                    + "".join(f'<span class="pill" style="background:{color};color:{tcolor}">{i.replace("_", " ")} ×{n}</span>' for i, n in cnt.head(5).items())
                    + '</p><div class="why">Keep these front and centre in reminders.</div></div>', unsafe_allow_html=True)
        cs_html = "".join(f'<div style="margin:5px 0"><span class="pill" style="background:{C_NEUTRAL};color:#fff">{c_.replace("_", " ")}</span>'
                          f'<span class="why">with {i.replace("_", " ")}, lift {l:.1f}</span></div>' for c_, i, l, cf in cross[:3]) \
            or '<span class="why">No strong add-on found (lift ≥ 1.5).</span>'
        c2.markdown(f'<div class="plan" style="border-top-color:{C_NEUTRAL}"><h4>Cross-sell at checkout</h4><div class="why">Not yet ordered, from basket rules</div>'
                    f'<div style="margin-top:10px">{cs_html}</div></div>', unsafe_allow_html=True)
        new_seg = [i for i in seg_items if i not in fav]
        c3.markdown(f'<div class="plan" style="border-top-color:{INK}"><h4>Popular in {seg_name}</h4><div class="why">Best-sellers they have not tried</div>'
                    f'<p style="margin-top:10px">' + (pills(new_seg, color, tcolor) if new_seg else '<span class="why">Already ordered all of them.</span>') + "</p></div>",
                    unsafe_allow_html=True)
        with st.expander(f"Order history · {n_baskets_d} baskets"):
            hist = mine.groupby("order_id").item.apply(lambda x: ", ".join(i.replace("_", " ") for i in x)).reset_index()
            hist.columns = ["Order", "Items"]
            st.dataframe(hist, hide_index=True, width="stretch")

    a, b = st.columns([1, 1.4], gap="large")
    with a:
        head("Next step")
        if p_re < 0.5:
            offer = fav[:2] if lookup and fav else seg_items[:2]
            note(f"<b>Send a win-back offer</b> featuring {', '.join(i.replace('_', ' ') for i in offer)}" + (" (their own favourites)." if lookup and fav else "."), "bad")
        else:
            note("<b>No incentive needed now.</b> Keep recommending their favourites.", "ok")
    with b:
        head("What-if: recency against frequency", "P(reorder) if this diner waited longer, or ordered more often.")
        days_grid, opm_grid = np.arange(0, 31, 3), np.arange(2, 19, 2)
        grid = pd.DataFrame([[analyse({**prof, "days_since_last_order": int(d_), "orders_per_month": int(o_)})[1] for d_ in days_grid]
                             for o_ in opm_grid], index=opm_grid, columns=days_grid)
        fig = px.imshow(grid, text_auto=".0%", aspect="auto", zmin=0, zmax=1, origin="lower",
                        color_continuous_scale=[[0, C_BAD], [.5, "#F2F4F1"], [1, C_OK]],
                        labels=dict(x="Days since last order", y="Orders per month", color="P(reorder)"))
        fig.update_traces(xgap=2, ygap=2)
        plot(fig, height=380)


with tabs[4]:
    predictor_tab()

# ═════════════════════════ ACTIONS ═════════════════════════
with tabs[5]:
    head("What to do next", None)
    _, rules_all = mine_rules(orders_meal, 0.03, 1.0)
    top3 = rules_all.drop_duplicates("itemset").head(3) if not rules_all.empty else pd.DataFrame()
    pf, fam, bud = seg_tbl.loc["Premium foodies"], seg_tbl.loc["Families"], seg_tbl.loc["Budget students"]
    mt = M["metrics"]
    c = st.columns(3, gap="large")
    for col, seg, why, play in [
        (c[0], "Budget students", f"{bud.diner_share:.0%} of diners · {bud.spend_share:.0%} of spend · {bud.reorder:.0%} reorder",
         "Protect margin: a fixed-price student meal (burger or wrap, fries, cola) and no more blanket discounts."),
        (c[1], "Families", f"{fam.diner_share:.0%} of diners · {fam.spend_share:.0%} of spend · {fam.reorder:.0%} reorder",
         "Re-engage: family bundles and a reminder once the gap passes about two weeks."),
        (c[2], "Premium foodies", f"{pf.diner_share:.0%} of diners · {pf.spend_share:.0%} of spend · {pf.reorder:.0%} reorder",
         "Priority segment: loyalty perks, curated premium bundles and early win-back tests."),
    ]:
        col.markdown(f'<div class="plan" style="border-top-color:{SEG_COLORS[seg]}"><h4>{seg}</h4><div class="why">{why}</div>'
                     f'<p style="margin:10px 0">{play}</p><div>{pills(reco.get(seg, []), SEG_COLORS[seg], SEG_TEXT[seg])}</div></div>', unsafe_allow_html=True)
    st.write("")
    note(f"<b>1. Focus on Premium foodies.</b> {pf.diner_share:.0%} of diners but {pf.spend_share:.0%} of lifetime spend (AED {pf.avg_spend:,.0f} each), almost no discount use, "
         f"yet {1 - pf.reorder:.0%} do not reorder. AED {pf.spend_churned / 1000:,.0f}k sits with churned diners here. Stayers and leavers look alike inside the segment, "
         "so run small retention tests rather than one blanket rule.", "bad")
    if len(top3):
        bl = "; ".join(f"<b>{r.bundle}</b> (lift {r.lift:.1f}, confidence {r.confidence:.0%})" for r in top3.itertuples())
        note(f"<b>2. Promote these bundles:</b> {bl}. Target each at the segment that buys it (see Baskets, menu fingerprint) and suggest the partner item at checkout.")
    note(f"<b>3. Use the reorder model as an early-warning alert.</b> Diners idle for more than 14 days reorder only {D[D.days_since_last_order > 14].reordered.mean():.0%} of the time. "
         f"The model reaches {mt['acc']:.0%} accuracy (baseline {mt['base_acc']:.0%}) and catches {mt['churn_recall']:.0%} of churners; its errors are mostly false alarms, the cheaper mistake. "
         "Send a cheap voucher on the segment's favourite items and tune the threshold once the real cost ratio is known.")
    note(f"<b>4. Budget students need no incentive</b> ({bud.reorder:.0%} retained); discounts there mostly give away margin. "
         f"<b>Families</b> ({1 - fam.reorder:.0%} churn) need their own re-engagement play.", "ok")
    note(f"<b>5. Treat {ADDON_TXT or 'cola'} as an add-on, not a headline product.</b> It is a Budget-student habit, so offer it inside the student meal deal "
         "instead of recommending it to everyone.")

    head("Caveats")
    under18 = int((D.age < 18).sum())
    st.markdown(f"""
- Small data: **{len(D)} diners** and **{n_baskets:,} baskets**. Validate on a later period before rolling out.
- Balancing re-weights the order log; it cannot create information the log never held. {'The balanced log re-draws baskets for diners with fewer than ' + str(k_per_diner) + ', so treat rules from small segments as directional.' if balance_on else 'Balancing is off, so item popularity here is skewed toward frequent, low-ticket diners.'}
- The order log caps baskets at 4 items, so item popularity is approximate.
- Recency may partly reflect churn (a diner who already left looks idle), so the model's strength can be flattered.
- **{under18} diners are under 18**. Check the data and any marketing rules for minors.
- Segment names are rule-based labels for behaviour clusters, not verified demographics.
- A/B-test bundles and win-back offers before scaling.
""")
