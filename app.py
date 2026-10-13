"""
QuickBite · Diner Intelligence Dashboard
----------------------------------------
Run:  streamlit run app.py
Needs diners.csv and orders.csv in the same folder (or upload them in the sidebar).
Models (K-Means segments, Apriori rules, reorder classifier) are trained on the fly and cached,
so no food_models.joblib is required. If one is present it is ignored.
"""
from pathlib import Path

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

# ───────────────────────────── page + theme ─────────────────────────────
st.set_page_config(page_title="QuickBite · Diner Intelligence", page_icon="🍔",
                   layout="wide", initial_sidebar_state="expanded")

SEG_ORDER = ["Budget students", "Families", "Premium foodies"]
SEG_COLORS = {"Budget students": "#38bdf8", "Families": "#fb923c", "Premium foodies": "#c084fc"}
STATUS_COLORS = {"Reordered (retained)": "#34d399", "Did not reorder (churned)": "#f87171"}
FEATURES = ["orders_per_month", "avg_order_value", "avg_items_per_order",
            "days_since_last_order", "discount_rate"]
CLF_FEATURES = FEATURES + ["age", "total_spend"]
NICE = {"age": "Age (years)", "orders_per_month": "Orders per month",
        "avg_order_value": "Avg order value (AED)", "avg_items_per_order": "Avg items per order",
        "days_since_last_order": "Days since last order", "discount_rate": "Discount rate",
        "total_spend": "Total spend (AED)"}
RS = 42

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background: radial-gradient(1200px 600px at 8% -10%, rgba(251,146,60,.18), transparent 60%),
              radial-gradient(1000px 600px at 100% 0%, rgba(192,132,252,.16), transparent 55%),
              linear-gradient(180deg, #0b0f1a 0%, #0f1424 100%);
  color: #e5e7eb;
}
.block-container { padding-top: 1.4rem; max-width: 1400px; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; letter-spacing: -.01em; }
section[data-testid="stSidebar"] { background: rgba(15,20,36,.92); border-right: 1px solid rgba(255,255,255,.07); }
.hero { padding: 26px 30px; border-radius: 22px; margin-bottom: 14px;
  background: linear-gradient(120deg, rgba(251,146,60,.95), rgba(244,63,94,.92) 55%, rgba(192,132,252,.95));
  box-shadow: 0 18px 50px rgba(244,63,94,.25); }
.hero h1 { margin: 0; color: #fff; font-size: 2.1rem; font-weight: 800; }
.hero p { margin: 6px 0 0; color: rgba(255,255,255,.92); font-size: 1.02rem; }
.kpi { background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09); border-radius: 18px;
  padding: 16px 18px; backdrop-filter: blur(6px); height: 100%; }
.kpi .lbl { font-size: .74rem; text-transform: uppercase; letter-spacing: .09em; color: #9ca3af; }
.kpi .val { font-family: 'Poppins'; font-size: 1.85rem; font-weight: 700; color: #fff; line-height: 1.25; }
.kpi .sub { font-size: .8rem; color: #9ca3af; }
.insight { border-left: 4px solid #fb923c; background: rgba(251,146,60,.09); border-radius: 0 14px 14px 0;
  padding: 12px 16px; margin: 8px 0; font-size: .95rem; color: #f3f4f6; }
.insight b { color: #fdba74; }
.warn { border-left-color: #f87171; background: rgba(248,113,113,.09); }
.warn b { color: #fca5a5; }
.good { border-left-color: #34d399; background: rgba(52,211,153,.09); }
.good b { color: #6ee7b7; }
.card { background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09); border-radius: 18px; padding: 18px 20px; height: 100%; }
.card h4 { margin: 0 0 6px; }
.pill { display:inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; color:#0b0f1a; margin: 2px 4px 2px 0; }
button[data-baseweb="tab"] { font-family: 'Poppins'; font-weight: 600; font-size: .98rem; }
div[data-baseweb="tab-list"] { gap: 6px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

AXIS = dict(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.15)", linecolor="rgba(255,255,255,.15)")


def style(fig, height=380, title=None, legend=True):
    """Apply the shared dark look to every Plotly figure."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e5e7eb", size=13), margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        title=dict(text=title, font=dict(family="Poppins", size=16), x=0.0) if title else None,
        showlegend=legend, legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text, kind=""):
    st.markdown(f'<div class="insight {kind}">{text}</div>', unsafe_allow_html=True)


# ───────────────────────────── data ─────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_bytes(b):
    import io
    return pd.read_csv(io.BytesIO(b))


def load_data():
    here = Path(__file__).parent
    d_path, o_path = here / "diners.csv", here / "orders.csv"
    if d_path.exists() and o_path.exists():
        return pd.read_csv(d_path), pd.read_csv(o_path)
    st.sidebar.warning("diners.csv / orders.csv not found next to app.py – upload them:")
    d_up = st.sidebar.file_uploader("diners.csv", type="csv")
    o_up = st.sidebar.file_uploader("orders.csv", type="csv")
    if d_up and o_up:
        return read_csv_bytes(d_up.getvalue()), read_csv_bytes(o_up.getvalue())
    st.info("Add **diners.csv** and **orders.csv** to continue.")
    st.stop()


# ───────────────────────────── modelling (cached) ─────────────────────────────
@st.cache_resource(show_spinner="Training segmentation & churn models…")
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

    # Rule-based naming so labels stay right even if cluster numbers change
    means = d.groupby("cluster")[FEATURES].mean()
    budget = means["avg_order_value"].idxmin()
    families = means["avg_items_per_order"].idxmax()
    premium = ({0, 1, 2} - {budget, families}).pop()
    names = {int(budget): "Budget students", int(families): "Families", int(premium): "Premium foodies"}
    d["segment"] = d.cluster.map(names)

    pca = PCA(2, random_state=RS).fit(Xs)
    pts = pca.transform(Xs)
    d["pc1"], d["pc2"] = pts[:, 0], pts[:, 1]

    # Classifier: scaler + logistic regression, churn weighted x2 (as in the notebook)
    X, y = d[CLF_FEATURES], d["reordered"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)

    def make_clf():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight={0: 2, 1: 1}))

    clf_eval = make_clf().fit(X_tr, y_tr)
    pred = clf_eval.predict(X_te)
    proba = clf_eval.predict_proba(X_te)[:, 1]
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
    fpr, tpr, _ = roc_curve(y_te, proba)
    metrics = dict(acc=accuracy_score(y_te, pred), base_acc=accuracy_score(y_te, base),
                   churn_recall=recall_score(y_te, pred, pos_label=0), auc=roc_auc_score(y_te, proba),
                   cm=confusion_matrix(y_te, pred, labels=[0, 1]), fpr=fpr, tpr=tpr, n_test=len(y_te))

    # Out-of-fold churn probability for every diner (unbiased risk list + threshold study)
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    plain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    d["p_churn"] = cross_val_predict(plain, X, y, cv=cv, method="predict_proba")[:, 0]

    clf_final = make_clf().fit(X, y)
    coefs = pd.Series(clf_final[-1].coef_[0], index=CLF_FEATURES)
    return dict(df=d, scaler=scaler, kmeans=kmeans, names=names, ks=ks, inertia=inertia, sil=sil, pca=pca,
                clf=clf_final, metrics=metrics, coefs=coefs)


@st.cache_data(show_spinner=False)
def to_basket(orders_df: pd.DataFrame):
    return (orders_df.assign(v=True).pivot_table(index="order_id", columns="item", values="v",
                                                aggfunc="max", fill_value=False).astype(bool))


@st.cache_data(show_spinner=False)
def mine_rules(orders_df: pd.DataFrame, min_support: float, min_lift: float = 1.0):
    basket = to_basket(orders_df)
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return basket, pd.DataFrame()
    try:
        rules = association_rules(freq, num_itemsets=len(basket), metric="lift", min_threshold=min_lift)
    except TypeError:                                    # older mlxtend
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    if rules.empty:
        return basket, rules
    nice = lambda s: " + ".join(sorted(s)).replace("_", " ")
    rules["rule"] = rules.antecedents.apply(nice) + "  →  " + rules.consequents.apply(nice)
    rules["itemset"] = rules.apply(lambda r: frozenset(r.antecedents | r.consequents), axis=1)
    rules["bundle"] = rules.itemset.apply(nice)
    return basket, rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def pct(x, d=0):
    return f"{x:.{d}%}"


# ───────────────────────────── load + prepare ─────────────────────────────
diners_raw, orders = load_data()
M = build_models(diners_raw)
D = M["df"]
D["status"] = D.reordered.map({1: "Reordered (retained)", 0: "Did not reorder (churned)"})
orders_seg = orders.merge(D[["diner_id", "segment"]], on="diner_id")
n_baskets = orders.order_id.nunique()
reorder_rate = D.reordered.mean()

# Segment business table
seg_tbl = (D.groupby("segment").agg(diners=("diner_id", "size"), spend=("total_spend", "sum"),
                                    reorder=("reordered", "mean"), age=("age", "mean"),
                                    opm=("orders_per_month", "mean"), aov=("avg_order_value", "mean"),
                                    nitems=("avg_items_per_order", "mean"), days=("days_since_last_order", "mean"),
                                    disc=("discount_rate", "mean"), avg_spend=("total_spend", "mean"))
           .loc[SEG_ORDER])
seg_tbl["diner_share"] = seg_tbl.diners / seg_tbl.diners.sum()
seg_tbl["spend_share"] = seg_tbl.spend / seg_tbl.spend.sum()
seg_tbl["churned"] = seg_tbl.diners * (1 - seg_tbl.reorder)
D["spend_churned"] = np.where(D.reordered == 0, D.total_spend, 0.0)
seg_tbl["spend_churned"] = D.groupby("segment").spend_churned.sum().loc[SEG_ORDER]

# Recommended items (most ordered per segment)
reco = (orders_seg.groupby("segment").item.value_counts().groupby(level=0).head(4).reset_index(name="n")
        .groupby("segment").item.apply(list).to_dict())

# ───────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.markdown("## 🍔 QuickBite")
    st.caption("Diner intelligence · segments · bundles · churn")
    st.markdown("---")
    st.markdown("**Global filter**")
    seg_filter = st.multiselect("Segments shown in Overview & charts", SEG_ORDER, default=SEG_ORDER)
    if not seg_filter:
        seg_filter = SEG_ORDER
    st.markdown("---")
    st.caption(f"📁 {len(D):,} diners · {len(orders):,} order lines · {n_baskets:,} baskets")
    st.caption("Models retrain automatically if the CSVs change.")

DF = D[D.segment.isin(seg_filter)]

# ───────────────────────────── hero ─────────────────────────────
st.markdown("""
<div class="hero"><h1>🍔 QuickBite · Know Your Diners</h1>
<p>Who are our diners? What do they order together? Who is about to stop ordering?</p></div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "🧩 Segments", "🛒 Basket analysis", "⚠️ Churn model",
                "🎛️ Diner predictor", "💡 Insights & actions"])

# ═════════════════════════ TAB 1 · OVERVIEW ═════════════════════════
with tabs[0]:
    c = st.columns(5)
    kpi(c[0], "Diners", f"{len(DF):,}", f"of {len(D):,} total")
    kpi(c[1], "Reorder rate", pct(DF.reordered.mean()), "share who ordered again")
    kpi(c[2], "Avg order value", f"AED {DF.avg_order_value.mean():,.0f}", "per order")
    kpi(c[3], "Lifetime spend", f"AED {DF.total_spend.sum()/1000:,.0f}k", f"avg AED {DF.total_spend.mean():,.0f} / diner")
    kpi(c[4], "Baskets", f"{n_baskets:,}", f"{len(orders)/n_baskets:.1f} items per basket")
    st.write("")

    a, b = st.columns([1, 1.25])
    with a:
        cnt = DF.status.value_counts().reset_index()
        cnt.columns = ["Outcome", "Diners"]
        fig = px.pie(cnt, names="Outcome", values="Diners", hole=.6, color="Outcome", color_discrete_map=STATUS_COLORS)
        fig.update_traces(textinfo="percent+value", textfont_size=14)
        fig.add_annotation(text=f"<b>{DF.reordered.mean():.0%}</b><br>reorder", showarrow=False, font=dict(size=20))
        st.plotly_chart(style(fig, 360, "Did the diner reorder? (share of diners)"), width="stretch")
    with b:
        pop = (orders_seg[orders_seg.segment.isin(seg_filter)].drop_duplicates(["order_id", "item"])
               .groupby("item").size() / orders_seg[orders_seg.segment.isin(seg_filter)].order_id.nunique())
        pop = pop.sort_values().rename_axis("Menu item").reset_index(name="Share of baskets")
        pop["Menu item"] = pop["Menu item"].str.replace("_", " ")
        fig = px.bar(pop, x="Share of baskets", y="Menu item", orientation="h", text=pop["Share of baskets"].map(pct),
                     color="Share of baskets", color_continuous_scale=["#38bdf8", "#c084fc", "#fb923c"])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%", title="Share of baskets containing the item")
        st.plotly_chart(style(fig, 360, "Menu popularity", legend=False), width="stretch")

    a, b = st.columns(2)
    with a:
        bands = pd.cut(DF.days_since_last_order, [-1, 3, 7, 14, 21, 60], labels=["0–3", "4–7", "8–14", "15–21", "22+"])
        rb = DF.groupby(bands, observed=True).reordered.agg(rate="mean", n="size").reset_index()
        rb.columns = ["Days since last order", "Reorder rate", "Diners"]
        fig = px.bar(rb, x="Days since last order", y="Reorder rate", text=[f"{r:.0%}<br>n={n}" for r, n in zip(rb["Reorder rate"], rb["Diners"])],
                     color="Reorder rate", color_continuous_scale=["#f87171", "#fbbf24", "#34d399"], range_color=[0, 1])
        fig.add_hline(y=DF.reordered.mean(), line_dash="dash", line_color="#9ca3af",
                      annotation_text=f"overall {DF.reordered.mean():.0%}", annotation_position="top right")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.2], title="Share of diners who reordered")
        fig.update_xaxes(title="Days since last order (band)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 360, "Reorder rate falls as diners go quiet", legend=False), width="stretch")
    with b:
        feat = st.selectbox("Compare a behaviour for retained vs churned diners", list(NICE), index=4,
                            format_func=lambda k: NICE[k])
        fig = px.histogram(DF, x=feat, color="status", barmode="overlay", opacity=.65, nbins=30,
                           histnorm="probability density", color_discrete_map=STATUS_COLORS,
                           labels={feat: NICE[feat], "status": "Outcome"})
        fig.update_yaxes(title="Density")
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(style(fig, 360, f"{NICE[feat]} by outcome"), width="stretch")

    grp = DF.groupby("status")[feat].median()
    r_hi = DF[DF.days_since_last_order > 14].reordered.mean() if (DF.days_since_last_order > 14).any() else np.nan
    insight(f"<b>Class balance:</b> {reorder_rate:.0%} of all diners reorder, so a model that always says “will reorder” is already "
            f"{reorder_rate:.0%} accurate – but it would catch <b>0%</b> of churners. Accuracy alone is misleading here.")
    insight(f"<b>Recency is the early-warning signal:</b> diners idle for more than 14 days reorder only "
            f"<b>{r_hi:.0%}</b> of the time vs {DF[DF.days_since_last_order <= 7].reordered.mean():.0%} for those active in the last week.", "warn")
    top_item = pop.iloc[-1]
    insight(f"<b>Menu:</b> “{top_item['Menu item']}” is in {top_item['Share of baskets']:.0%} of baskets – but the long tail only "
            f"appears when you look inside a segment (see Basket analysis).")

# ═════════════════════════ TAB 2 · SEGMENTS ═════════════════════════
with tabs[1]:
    st.subheader("Who are our diners?")
    st.caption("K-Means on the 5 behavioural features (standardised). Age and total spend are held out to validate the segments.")

    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=M["ks"], y=M["inertia"], mode="lines+markers", line=dict(color="#fb923c", width=3),
                                   marker=dict(size=9)))
        fig.add_vline(x=3, line_dash="dash", line_color="#9ca3af", annotation_text="chosen k = 3")
        fig.update_xaxes(title="Number of clusters (k)", dtick=1)
        fig.update_yaxes(title="Inertia (within-cluster sum of squares)")
        st.plotly_chart(style(fig, 330, "Elbow method – the curve flattens after k = 3", legend=False), width="stretch")
    with b:
        best_k = M["ks"][int(np.argmax(M["sil"]))]
        fig = go.Figure(go.Scatter(x=M["ks"], y=M["sil"], mode="lines+markers", line=dict(color="#38bdf8", width=3),
                                   marker=dict(size=9)))
        fig.add_vline(x=3, line_dash="dash", line_color="#9ca3af", annotation_text="chosen k = 3")
        fig.update_xaxes(title="Number of clusters (k)", dtick=1)
        fig.update_yaxes(title="Mean silhouette score (higher = cleaner separation)")
        st.plotly_chart(style(fig, 330, f"Silhouette score – best at k = {best_k}", legend=False), width="stretch")
    sil3 = M["sil"][1]
    insight(f"<b>Why 3 segments?</b> Elbow and silhouette agree: silhouette at k=3 is <b>{sil3:.2f}</b> "
            f"(best k = {best_k}), and the inertia gain beyond k=3 is small. Three groups are separable <i>and</i> easy to act on.")

    a, b = st.columns(2)
    with a:
        fig = px.scatter(DF, x="pc1", y="pc2", color="segment", color_discrete_map=SEG_COLORS, opacity=.75,
                         category_orders={"segment": SEG_ORDER},
                         hover_data={"diner_id": True, "orders_per_month": True, "avg_order_value": ":.0f", "pc1": False, "pc2": False},
                         labels={"pc1": f"Principal component 1 ({M['pca'].explained_variance_ratio_[0]:.0%} of variance)",
                                 "pc2": f"Principal component 2 ({M['pca'].explained_variance_ratio_[1]:.0%} of variance)",
                                 "segment": "Segment"})
        st.plotly_chart(style(fig, 400, "Segments in 2-D (PCA of the 5 scaled features)"), width="stretch")
    with b:
        fig = px.scatter(DF, x="orders_per_month", y="avg_order_value", color="segment", color_discrete_map=SEG_COLORS,
                         size="avg_items_per_order", size_max=14, opacity=.7, category_orders={"segment": SEG_ORDER},
                         labels={"orders_per_month": "Orders per month", "avg_order_value": "Average order value (AED)",
                                 "segment": "Segment", "avg_items_per_order": "Items / order"})
        st.plotly_chart(style(fig, 400, "Frequency vs ticket size (bubble = basket size)"), width="stretch")

    st.markdown("#### Segment profiles")
    z_cols = FEATURES + ["age", "total_spend"]
    z = D.groupby("segment")[z_cols].mean().loc[SEG_ORDER]
    z = (z - D[z_cols].mean()) / D[z_cols].std()
    a, b = st.columns([1.15, 1])
    with a:
        fig = px.imshow(z.rename(columns=NICE), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-2, zmax=2, labels=dict(color="Std. dev. vs average", x="Behaviour", y="Segment"))
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(style(fig, 330, "What makes each segment different? (z-score vs all diners)"), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_bar(name="Share of diners", x=SEG_ORDER, y=seg_tbl.diner_share, marker_color="#94a3b8",
                    text=seg_tbl.diner_share.map(pct), textposition="outside")
        fig.add_bar(name="Share of lifetime spend", x=SEG_ORDER, y=seg_tbl.spend_share, marker_color="#fb923c",
                    text=seg_tbl.spend_share.map(pct), textposition="outside")
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", title="Share of total", range=[0, seg_tbl[["diner_share", "spend_share"]].values.max() * 1.25])
        fig.update_xaxes(title="Segment")
        st.plotly_chart(style(fig, 330, "Segment size vs money"), width="stretch"
                        )

    prof = seg_tbl.rename(columns={"diners": "Diners", "age": "Age", "opm": "Orders / month", "aov": "Avg order (AED)",
                                   "nitems": "Items / order", "days": "Days since last order", "disc": "Discount rate",
                                   "avg_spend": "Avg lifetime spend (AED)", "reorder": "Reorder rate"})
    prof = prof[["Diners", "Age", "Orders / month", "Avg order (AED)", "Items / order", "Days since last order",
                 "Discount rate", "Avg lifetime spend (AED)", "Reorder rate"]]
    st.dataframe(prof.style.format({"Age": "{:.0f}", "Orders / month": "{:.1f}", "Avg order (AED)": "{:.0f}", "Items / order": "{:.1f}",
                                    "Days since last order": "{:.0f}", "Discount rate": "{:.0%}",
                                    "Avg lifetime spend (AED)": "{:,.0f}", "Reorder rate": "{:.0%}"})
                 .background_gradient(cmap="magma", subset=["Reorder rate"], vmin=0, vmax=1.4), width="stretch")

    fig = go.Figure()
    fig.add_bar(name="Spend of diners who reordered", x=SEG_ORDER, y=(seg_tbl.spend - seg_tbl.spend_churned) / 1000, marker_color="#34d399")
    fig.add_bar(name="Spend of diners who churned", x=SEG_ORDER, y=seg_tbl.spend_churned / 1000, marker_color="#f87171",
                text=[f"{v/1000:.0f}k" for v in seg_tbl.spend_churned], textposition="inside")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Lifetime spend (AED thousands)")
    fig.update_xaxes(title="Segment")
    st.plotly_chart(style(fig, 340, "Where does lifetime spend sit – with retained or churned diners?"), width="stretch")

    pf = seg_tbl.loc["Premium foodies"]
    fam = seg_tbl.loc["Families"]
    bud = seg_tbl.loc["Budget students"]
    insight(f"<b>Premium foodies</b> are {pf.diner_share:.0%} of diners but {pf.spend_share:.0%} of lifetime spend, rarely use discounts "
            f"({pf.disc:.0%}) and only <b>{pf.reorder:.0%}</b> reorder – the most valuable group with a coin-flip retention.", "warn")
    insight(f"<b>Families</b> order rarely ({fam.opm:.0f}/month) but in big baskets ({fam.nitems:.1f} items, AED {fam.aov:.0f}); only "
            f"<b>{fam.reorder:.0%}</b> come back → a re-engagement problem.")
    insight(f"<b>Budget students</b> order {bud.opm:.0f}×/month with tiny, discount-driven tickets (AED {bud.aov:.0f}, {bud.disc:.0%} discounted) "
            f"and <b>{bud.reorder:.0%}</b> reorder → loyal already; blanket discounts mostly give away margin.", "good")

# ═════════════════════════ TAB 3 · BASKETS ═════════════════════════
with tabs[2]:
    st.subheader("What do diners order together?")
    st.caption("Apriori association rules. Each order_id is one basket. Lift > 1 means the items are ordered together more than by chance.")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    scope = c1.selectbox("Analyse baskets of", ["All diners"] + SEG_ORDER)
    min_sup = c2.slider("Minimum support (share of baskets)", 0.01, 0.15, 0.03 if scope == "All diners" else 0.05, 0.005,
                        format="%.3f", help="Lower = more (rarer) rules; each segment has fewer baskets so use a smaller value.")
    min_lift = c3.slider("Minimum lift", 1.0, 5.0, 1.2, 0.1)

    sub = orders if scope == "All diners" else orders_seg[orders_seg.segment == scope][["order_id", "diner_id", "item"]]
    basket, rules = mine_rules(sub, min_sup, 1.0)
    if rules.empty:
        st.warning("No rules at this support threshold – lower the minimum support.")
    else:
        rules_f = rules[rules.lift >= min_lift]
        uniq = rules_f.drop_duplicates("itemset")
        k = st.columns(4)
        kpi(k[0], "Baskets analysed", f"{len(basket):,}", scope)
        kpi(k[1], "Rules found", f"{len(rules_f):,}", f"lift ≥ {min_lift:.1f}")
        kpi(k[2], "Unique item combos", f"{len(uniq):,}", "A→B and B→A merged")
        kpi(k[3], "Strongest lift", f"{rules_f.lift.max():.1f}" if len(rules_f) else "–", "× more than chance")
        st.write("")
        if rules_f.empty:
            st.warning("No rules above this lift – lower the minimum lift.")
        else:
            a, b = st.columns([1.1, 1])
            with a:
                t = uniq.head(10).iloc[::-1]
                fig = go.Figure(go.Bar(x=t.lift, y=t.rule, orientation="h", marker_color=SEG_COLORS.get(scope, "#c084fc"),
                                       text=[f"lift {l:.1f} · conf {c:.0%}" for l, c in zip(t.lift, t.confidence)],
                                       textposition="outside", cliponaxis=False))
                fig.add_vline(x=1, line_dash="dash", line_color="#f87171", annotation_text="lift = 1 (no link)")
                fig.update_xaxes(title="Lift", range=[0, t.lift.max() * 1.45])
                fig.update_yaxes(title="Rule  (basket has … → also has …)")
                st.plotly_chart(style(fig, 460, "Top 10 rules by lift", legend=False), width="stretch")
            with b:
                fig = px.scatter(rules_f, x="support", y="confidence", color="lift", size="lift", size_max=22,
                                 color_continuous_scale="Turbo", hover_name="rule",
                                 labels={"support": "Support (share of baskets)", "confidence": "Confidence",
                                         "lift": "Lift"})
                fig.update_xaxes(tickformat=".0%")
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(style(fig, 460, "How common vs how reliable"), width="stretch")

            tbl = uniq[["rule", "support", "confidence", "lift"]].head(15).rename(
                columns={"rule": "Rule", "support": "Support", "confidence": "Confidence", "lift": "Lift"})
            st.dataframe(tbl.style.format({"Support": "{:.1%}", "Confidence": "{:.0%}", "Lift": "{:.1f}"})
                         .background_gradient(cmap="viridis", subset=["Lift"]), hide_index=True, width="stretch")

            best = uniq.iloc[0]
            insight(f"<b>Top bundle for {scope.lower() if scope != 'All diners' else 'all diners'}:</b> {best.bundle} – "
                    f"lift <b>{best.lift:.1f}</b>, i.e. {best.lift:.0f}× more likely together than by chance; "
                    f"when “{' + '.join(sorted(best.antecedents)).replace('_', ' ')}” is in the basket, "
                    f"“{' + '.join(sorted(best.consequents)).replace('_', ' ')}” follows <b>{best.confidence:.0%}</b> of the time → promote as one combo.")
            if scope in SEG_ORDER and uniq.lift.max() < 1.6:
                insight(f"<b>Weak cross-sell signal:</b> inside {scope} nearly everyone orders the same few items, so lifts stay low. "
                        "The lever is a fixed-price meal deal, not recommendations.", "warn")

        # Pairwise lift heatmap
        st.markdown("#### Item-to-item lift map")
        B = basket.values.astype(float)
        p = B.mean(0)
        pl = (B.T @ B / len(B)) / np.outer(p, p)
        np.fill_diagonal(pl, np.nan)
        lab = [c.replace("_", " ") for c in basket.columns]
        order = np.argsort(-np.nan_to_num(pl).sum(0))
        pl_df = pd.DataFrame(pl, index=lab, columns=lab).iloc[order, order]
        a, b = st.columns([1.3, 1])
        with a:
            fig = px.imshow(pl_df, text_auto=".1f", aspect="auto", color_continuous_scale="Magma",
                            labels=dict(x="Menu item", y="Menu item", color="Pairwise lift"))
            fig.update_xaxes(tickangle=-40)
            st.plotly_chart(style(fig, 520, "Which items are bought together? (pairwise lift, 1 = independent)"), width="stretch")
        with b:
            st.markdown("##### 🛎️ Checkout cross-sell finder")
            cart = st.selectbox("If the cart contains…", sorted(lab))
            row = pl_df.loc[cart].drop(cart, errors="ignore").sort_values(ascending=False).head(5)
            cm_ = B[:, basket.columns.get_loc(cart.replace(" ", "_"))] > 0
            out = pd.DataFrame({"Suggest": row.index, "Lift": row.values,
                                "P(also in basket)": [B[cm_, basket.columns.get_loc(i.replace(" ", "_"))].mean() for i in row.index]})
            fig = px.bar(out.iloc[::-1], x="P(also in basket)", y="Suggest", orientation="h", color="Lift",
                         color_continuous_scale="Sunsetdark", text=out.iloc[::-1]["P(also in basket)"].map(pct))
            fig.update_xaxes(tickformat=".0%", title="Share of baskets with the cart item that also contain this", range=[0, 1.1])
            fig.update_yaxes(title="Suggested item")
            st.plotly_chart(style(fig, 330, f"Best add-ons for “{cart}”", legend=False), width="stretch")

    # Menu fingerprint
    st.markdown("#### Menu fingerprint of each segment")
    seg_baskets = orders_seg.drop_duplicates("order_id").segment.value_counts()
    share = (orders_seg.drop_duplicates(["order_id", "item"]).groupby(["segment", "item"]).size().unstack(fill_value=0)
             .div(seg_baskets, axis=0).loc[SEG_ORDER])
    share = share[share.max().sort_values(ascending=False).index]
    fig = px.imshow(share.rename(columns=lambda c: c.replace("_", " ")), text_auto=".0%", aspect="auto",
                    color_continuous_scale="YlOrRd", labels=dict(x="Menu item", y="Segment", color="Share of segment's baskets"))
    st.plotly_chart(style(fig, 280, "Do segments order from different menus? Yes – almost completely"), width="stretch")
    own = (share > 0.05).sum()
    insight(f"<b>Segments barely overlap:</b> {int((own == 1).sum())} of {len(own)} menu items are ordered (>5% of baskets) by only one segment. "
            "That is why bundles should be targeted per segment instead of one-size-fits-all.")

# ═════════════════════════ TAB 4 · CHURN MODEL ═════════════════════════
with tabs[3]:
    st.subheader("Who will reorder?")
    mt = M["metrics"]
    cm = mt["cm"]
    st.caption(f"Logistic regression (churn weighted ×2) on 7 features · stratified 75/25 split · {mt['n_test']} unseen test diners.")
    c = st.columns(4)
    kpi(c[0], "Model accuracy", pct(mt["acc"]), f"baseline {pct(mt['base_acc'])} (always 'reorder')")
    kpi(c[1], "Churners caught (recall)", pct(mt["churn_recall"]), "baseline catches 0%")
    kpi(c[2], "ROC AUC", f"{mt['auc']:.2f}", "1.0 = perfect ranking")
    kpi(c[3], "Missed churners", f"{cm[0,1]} of {cm[0].sum()}", f"{cm[1,0]} false alarms of {cm[1].sum()} loyal")
    st.write("")

    a, b = st.columns(2)
    with a:
        cmn = cm / cm.sum(axis=1, keepdims=True)
        txt = [[f"<b>{cm[i,j]}</b><br>{cmn[i,j]:.0%} of actual" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=cmn, x=["Predicted: churn", "Predicted: reorder"], y=["Actual: churned", "Actual: reordered"],
                                   text=txt, texttemplate="%{text}", colorscale=[[0, "#1e293b"], [1, "#fb923c"]], showscale=False))
        fig.update_xaxes(title="Model prediction")
        fig.update_yaxes(title="What really happened", autorange="reversed")
        st.plotly_chart(style(fig, 360, "Confusion matrix (test set)", legend=False), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_scatter(x=mt["fpr"], y=mt["tpr"], mode="lines", name=f"Model (AUC {mt['auc']:.2f})", line=dict(color="#c084fc", width=3),
                        fill="tozeroy", fillcolor="rgba(192,132,252,.15)")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(color="#9ca3af", dash="dash"))
        fig.update_xaxes(title="False-positive rate (loyal diners wrongly flagged)")
        fig.update_yaxes(title="True-positive rate")
        st.plotly_chart(style(fig, 360, "ROC curve"), width="stretch")

    insight(f"<b>Beats the baseline:</b> {mt['acc']:.0%} accuracy vs {mt['base_acc']:.0%}, and it catches <b>{mt['churn_recall']:.0%}</b> of churners "
            f"where the baseline catches none.", "good")
    insight(f"<b>Which mistake hurts more?</b> Missing a churner (lost customer) costs far more than a false alarm (a small voucher). "
            f"The model misses {cm[0,1]} churners but raises {cm[1,0]} false alarms – the cheaper error, by design.", "warn")

    a, b = st.columns(2)
    with a:
        co = M["coefs"].rename(index=NICE).sort_values()
        fig = go.Figure(go.Bar(x=co.values, y=co.index, orientation="h",
                               marker_color=["#f87171" if v < 0 else "#34d399" for v in co.values],
                               text=[f"{v:+.2f}" for v in co.values], textposition="outside"))
        fig.update_xaxes(title="Standardised coefficient (log-odds of reordering per +1 std. dev.)")
        fig.update_yaxes(title="Feature")
        st.plotly_chart(style(fig, 400, "What drives reordering? (green = more likely, red = less)", legend=False), width="stretch")
    with b:
        st.markdown("##### 💰 Choose the alert threshold by cost")
        cfn = st.slider("Cost of a missed churner (relative units)", 1, 10, 3)
        cfp = st.slider("Cost of a false alarm / voucher", 1, 10, 1)
        ths = np.linspace(0.05, 0.95, 91)
        isc = (D.reordered == 0).values
        rows = []
        for t in ths:
            flag = D.p_churn.values >= t
            fn = int((isc & ~flag).sum())
            fp = int((~isc & flag).sum())
            rows.append((t, fn, fp, cfn * fn + cfp * fp))
        th = pd.DataFrame(rows, columns=["t", "fn", "fp", "cost"])
        best_t = th.loc[th.cost.idxmin()]
        fig = go.Figure(go.Scatter(x=th.t, y=th.cost, mode="lines", line=dict(color="#f87171", width=3)))
        fig.add_vline(x=best_t.t, line_dash="dash", line_color="#34d399", annotation_text=f"best ≈ {best_t.t:.2f}")
        fig.update_xaxes(title="Flag diner as 'at risk' when P(churn) ≥ threshold")
        fig.update_yaxes(title="Total expected cost")
        st.plotly_chart(style(fig, 300, "Total cost vs threshold (out-of-fold predictions)", legend=False), width="stretch")
        insight(f"With a missed churner costing <b>{cfn/cfp:.1f}×</b> a false alarm, flag diners from <b>P(churn) ≥ {best_t.t:.2f}</b>: "
                f"{int(best_t.fn)} missed, {int(best_t.fp)} false alarms.")

    st.markdown("#### 📋 At-risk win-back list")
    thr = st.slider("Risk threshold – flag diners with P(churn) at least", 0.05, 0.95, float(round(best_t.t, 2)), 0.01)
    risk = D[D.p_churn >= thr].sort_values("total_spend", ascending=False)
    risk_out = risk[["diner_id", "segment", "p_churn", "total_spend", "days_since_last_order", "orders_per_month", "status"]].rename(
        columns={"p_churn": "P(churn)", "total_spend": "Total spend (AED)", "days_since_last_order": "Days since last order",
                 "orders_per_month": "Orders / month", "status": "Actual outcome", "diner_id": "Diner", "segment": "Segment"})
    a, b = st.columns([1.6, 1])
    with a:
        st.dataframe(risk_out.head(50).style.format({"P(churn)": "{:.0%}", "Total spend (AED)": "{:,.0f}"})
                     .background_gradient(cmap="Reds", subset=["P(churn)"]), hide_index=True, width="stretch", height=360)
        st.download_button("⬇️ Download full at-risk list (CSV)", risk_out.to_csv(index=False).encode(), "at_risk_diners.csv", "text/csv")
    with b:
        fl = risk.groupby("segment").agg(n=("diner_id", "size"), spend=("total_spend", "sum")).reindex(SEG_ORDER).fillna(0)
        fig = px.bar(fl.reset_index(), x="segment", y="n", color="segment", color_discrete_map=SEG_COLORS,
                     text=[f"{int(n)} diners<br>AED {s/1000:.0f}k" for n, s in zip(fl.n, fl.spend)],
                     labels={"segment": "Segment", "n": "Diners flagged"})
        fig.update_yaxes(range=[0, max(fl.n.max() * 1.3, 1)])
        st.plotly_chart(style(fig, 360, f"{len(risk)} diners would get a win-back offer", legend=False), width="stretch")

# ═════════════════════════ TAB 5 · PREDICTOR ═════════════════════════
with tabs[4]:
    st.subheader("Diner predictor")
    st.caption("Enter a diner profile to see their segment, reorder likelihood and recommended items. Sliders default to the dataset medians.")

    med = D[CLF_FEATURES].median()
    presets = {
        "Custom (use sliders)": None,
        "Student: frequent, cheap, recent": dict(age=21, orders_per_month=14, avg_order_value=40, avg_items_per_order=2, days_since_last_order=3, discount_rate=0.7, total_spend=550),
        "Family: big rare basket, long gap": dict(age=41, orders_per_month=3, avg_order_value=175, avg_items_per_order=6, days_since_last_order=20, discount_rate=0.3, total_spend=620),
        "Premium foodie: ordered 6 days ago": dict(age=33, orders_per_month=9, avg_order_value=100, avg_items_per_order=3, days_since_last_order=6, discount_rate=0.1, total_spend=900),
        "Premium foodie: silent for 22 days": dict(age=33, orders_per_month=9, avg_order_value=100, avg_items_per_order=3, days_since_last_order=22, discount_rate=0.1, total_spend=900),
    }
    choice = st.selectbox("Quick start with an example profile", list(presets))
    pre = presets[choice] or {k: float(med[k]) for k in CLF_FEATURES}

    lo, hi = D[CLF_FEATURES].min(), D[CLF_FEATURES].max()
    s1, s2, s3, s4 = st.columns(4)
    age = s1.slider("Age", 10, 70, int(pre["age"]), key=f"a{choice}")
    opm = s2.slider("Orders per month", 1, 20, int(pre["orders_per_month"]), key=f"o{choice}")
    aov = s3.slider("Avg order value (AED)", 15, 250, int(pre["avg_order_value"]), key=f"v{choice}")
    items = s4.slider("Avg items per order", 1, 10, int(pre["avg_items_per_order"]), key=f"i{choice}")
    s5, s6, s7, _ = st.columns(4)
    days = s5.slider("Days since last order", 0, 60, int(pre["days_since_last_order"]), key=f"d{choice}")
    disc = s6.slider("Discount rate", 0.0, 1.0, float(pre["discount_rate"]), 0.01, key=f"r{choice}")
    spend = s7.slider("Total spend (AED)", 50, 5000, int(min(pre["total_spend"], 5000)), 10, key=f"s{choice}")

    prof = dict(age=age, orders_per_month=opm, avg_order_value=aov, avg_items_per_order=items,
                days_since_last_order=days, discount_rate=disc, total_spend=spend)

    def analyse(p):
        row = pd.DataFrame([p])
        seg_id = int(M["kmeans"].predict(M["scaler"].transform(row[FEATURES]))[0])
        name = M["names"].get(seg_id, f"Segment {seg_id}")
        pr = float(M["clf"].predict_proba(row[CLF_FEATURES])[0][1])
        return name, pr

    seg_name, p_re = analyse(prof)
    st.write("")
    k = st.columns(3)
    kpi(k[0], "Segment", f'<span style="color:{SEG_COLORS[seg_name]}">{seg_name}</span>', "nearest K-Means centre")
    verdict = "Likely to reorder ✅" if p_re >= 0.5 else "At risk of churning ⚠️"
    kpi(k[1], "Reorder prediction", verdict, f"P(reorder) = {p_re:.0%}")
    kpi(k[2], "Recommended items", "", "")
    k[2].markdown("".join(f'<span class="pill" style="background:{SEG_COLORS[seg_name]}">{i.replace("_", " ")}</span>'
                          for i in reco.get(seg_name, [])), unsafe_allow_html=True)

    a, b = st.columns([1, 1.4])
    with a:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p_re * 100, number=dict(suffix="%"),
                                     title=dict(text="Probability the diner reorders"),
                                     gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#fb923c"),
                                                steps=[dict(range=[0, 33], color="rgba(248,113,113,.45)"),
                                                       dict(range=[33, 66], color="rgba(251,191,36,.35)"),
                                                       dict(range=[66, 100], color="rgba(52,211,153,.4)")])))
        st.plotly_chart(style(fig, 320, legend=False), width="stretch")
        outside = [NICE[c] for c in CLF_FEATURES if not (lo[c] <= prof[c] <= hi[c])]
        if outside:
            insight("<b>Heads-up:</b> outside the range the model saw in training for " + ", ".join(outside) +
                    " – treat the prediction with caution.", "warn")
        if p_re < 0.5:
            insight(f"<b>Action:</b> send a win-back offer featuring {', '.join(i.replace('_', ' ') for i in reco.get(seg_name, [])[:2])}.", "warn")
        else:
            insight("<b>Action:</b> no incentive needed now – keep recommending their favourites.", "good")
    with b:
        days_grid = np.arange(0, 31, 3)
        opm_grid = np.arange(2, 19, 2)
        grid = pd.DataFrame([[analyse({**prof, "days_since_last_order": int(d_), "orders_per_month": int(o_)})[1] for d_ in days_grid]
                             for o_ in opm_grid], index=opm_grid, columns=days_grid)
        fig = px.imshow(grid, text_auto=".0%", aspect="auto", color_continuous_scale="RdYlGn", zmin=0, zmax=1, origin="lower",
                        labels=dict(x="Days since last order", y="Orders per month", color="P(reorder)"))
        st.plotly_chart(style(fig, 400, "What-if: how recency and frequency move reorder probability (other inputs as set)"), width="stretch")

# ═════════════════════════ TAB 6 · INSIGHTS ═════════════════════════
with tabs[5]:
    st.subheader("Business takeaways")
    # computed content
    _, rules_all = mine_rules(orders, 0.03, 1.0)
    top3 = rules_all.drop_duplicates("itemset").head(3) if not rules_all.empty else pd.DataFrame()
    pf = seg_tbl.loc["Premium foodies"]
    fam = seg_tbl.loc["Families"]
    bud = seg_tbl.loc["Budget students"]
    mt = M["metrics"]

    c = st.columns(3)
    for col, seg, why, play in [
        (c[0], "Budget students", f"{bud.diner_share:.0%} of diners · {bud.spend_share:.0%} of spend · {bud.reorder:.0%} reorder",
         "Protect margin: fixed-price student meal (burger/wrap + fries + cola); stop blanket discounts."),
        (c[1], "Families", f"{fam.diner_share:.0%} of diners · {fam.spend_share:.0%} of spend · {fam.reorder:.0%} reorder",
         "Re-engage: family bundles and a reminder once the gap passes ~2 weeks."),
        (c[2], "Premium foodies", f"{pf.diner_share:.0%} of diners · {pf.spend_share:.0%} of spend · {pf.reorder:.0%} reorder",
         "Priority segment: loyalty perks, curated premium bundles, early win-back tests."),
    ]:
        col.markdown(f'<div class="card" style="border-top:4px solid {SEG_COLORS[seg]}"><h4>{seg}</h4>'
                     f'<div style="color:#9ca3af;font-size:.85rem">{why}</div><p style="margin-top:10px">{play}</p>'
                     f'<div>{"".join(f"<span class=pill style=background:{SEG_COLORS[seg]}>{i.replace(chr(95), chr(32))}</span>" for i in reco[seg])}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    insight(f"<b>1 · Focus on Premium foodies.</b> {pf.diner_share:.0%} of diners but {pf.spend_share:.0%} of lifetime spend (AED {pf.avg_spend:,.0f} each), "
            f"almost no discount use, yet {1-pf.reorder:.0%} do not reorder. AED {pf.spend_churned/1000:,.0f}k sits with churned diners here. "
            "Within the segment stayers and leavers look alike, so run small retention tests rather than one blanket rule.", "warn")
    if len(top3):
        bl = "; ".join(f"<b>{r.bundle}</b> (lift {r.lift:.1f}, confidence {r.confidence:.0%})" for r in top3.itertuples())
        insight(f"<b>2 · Promote these bundles:</b> {bl}. Target each at the segment that actually buys it (see Basket analysis → menu fingerprint) "
                "and suggest the partner item at checkout (e.g. miso soup → green tea, steak → wine, biryani → raita).")
    insight(f"<b>3 · Use the reorder model as an early-warning alert.</b> Diners idle more than 14 days reorder only "
            f"{D[D.days_since_last_order > 14].reordered.mean():.0%} of the time. The model hits {mt['acc']:.0%} accuracy "
            f"(baseline {mt['base_acc']:.0%}) and catches {mt['churn_recall']:.0%} of churners; its errors are mostly false alarms – the cheaper mistake. "
            "Send a cheap win-back voucher on the segment’s favourite items and tune the threshold once the real cost ratio is known.")
    insight(f"<b>4 · Budget students need no incentive</b> ({bud.reorder:.0%} retained) – discounts there mostly give away margin; "
            f"<b>Families</b> ({1-fam.reorder:.0%} churn) need a separate re-engagement play.", "good")

    st.markdown("#### ⚠️ Caveats")
    under18 = int((D.age < 18).sum())
    st.markdown(f"""
- Small data: **{len(D)} diners** and **{n_baskets:,} baskets** – validate on a later period before rolling out.
- Recency may partly *reflect* churn (a diner who already left looks “idle”), so the model’s strength can be flattered.
- **{under18} diners are under 18** – check the data and any marketing rules for minors.
- Segment names are rule-based labels for behaviour clusters, not verified demographics.
- A/B-test bundles and win-back offers before scaling.
""")
