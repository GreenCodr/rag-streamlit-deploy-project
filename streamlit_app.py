# Self-contained Streamlit app (improved retrieval)
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="RAG-DB (Self-contained)", layout="wide")
st.title("RAG-DB — Self-contained Instacart Analytics")
st.write("This app uses CSVs stored inside the repo — no FastAPI backend needed.")

DATA_DIR = os.path.join("data", "data", "instacart")

def load_csv_safe(name, nrows=None):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, nrows=nrows)

st.sidebar.header("Options")
sample_nrows_prior = st.sidebar.number_input(
    "Number of rows to load from order_products__prior.csv (0 = full)", 
    min_value=0, 
    value=50000, 
    step=10000
)

mode_override = st.sidebar.selectbox("Force mode", ["auto","retrieval","aggregation"])

q = st.text_area("Ask a question:", 
                 value="Which products appear most frequently in prior orders?",
                 height=140)
run = st.button("Run Query")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior=None):
    products = load_csv_safe("products")
    aisles = load_csv_safe("aisles")
    departments = load_csv_safe("departments")
    orders = load_csv_safe("orders")
    prior = load_csv_safe("order_products__prior", nrows=nrows_prior)
    return {
        "products": products,
        "aisles": aisles,
        "departments": departments,
        "orders": orders,
        "prior": prior
    }

# simple normalizer/tokenizer
_non_alnum = re.compile(r"[^0-9a-z]+")
def normalize(text):
    if text is None:
        return ""
    t = str(text).lower()
    t = _non_alnum.sub(" ", t)
    tokens = [tok for tok in t.split() if len(tok)>1]
    return " ".join(tokens), tokens

def score_tokens(query_tokens, target_tokens):
    if not query_tokens or not target_tokens:
        return 0.0
    set_q = set(query_tokens)
    set_t = set(target_tokens)
    inter = set_q.intersection(set_t)
    # score = fraction of query tokens found in target, penalize by length
    return len(inter) / max(len(set_q), 1)

if run:
    st.info("Loading CSVs from the repo...")
    nrows = None if sample_nrows_prior == 0 else int(sample_nrows_prior)
    tables = load_tables(nrows_prior=nrows)

    products = tables["products"]
    aisles = tables["aisles"]
    prior = tables["prior"]
    orders = tables["orders"]

    if products is None or prior is None:
        st.error("Missing CSVs! Upload products.csv and order_products__prior.csv to data/data/instacart/")
        st.stop()

    ql_raw = q or ""
    ql = ql_raw.lower().strip()

    # --- Basic intent detection ---
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean","how many"]
    retrieval_keywords = ["show","list","example","what","which","find","give","where","example"]
    intent = ("aggregation" if any(k in ql for k in agg_keywords) else "retrieval")
    if mode_override != "auto":
        intent = mode_override

    st.markdown(f"### Intent detected: `{intent}`")

    # --------------- AGGREGATION ----------------
    if intent == "aggregation":
        if "most" in ql and ("product" in ql or "order" in ql or "frequent" in ql or "top" in ql):
            st.subheader("Top Products in Prior Orders")

            counts = prior["product_id"].value_counts().head(10)
            df = counts.rename_axis("product_id").reset_index(name="count")

            # join product names
            prod_idx = products.set_index("product_id")
            df["product_name"] = df["product_id"].apply(
                lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else "Unknown"
            )

            df = df[["product_name", "count"]].rename(columns={"product_name":"x","count":"y"})
            st.dataframe(df)

            fig, ax = plt.subplots(figsize=(10,4))
            ax.bar(df["x"], df["y"])
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        else:
            st.info("Aggregation detected, but I don't have a handler for this query.")

    # --------------- RETRIEVAL -----------------
    else:
        st.subheader("Smart Retrieval Results")

        # normalize query tokens
        q_norm, q_tokens = normalize(ql_raw)

        results = []

        # 1) Exact substring search in product_name (fast)
        mask_prod = products["product_name"].str.lower().str.contains(ql, na=False)
        prod_hits = products[mask_prod]
        for _, r in prod_hits.head(50).iterrows():
            results.append({"type":"product", "score":1.0, "text": f"{r['product_id']} | {r['product_name']}"})

        # 2) If none, use token-overlap scoring on product names
        if not results:
            # precompute normalized tokens for product names (cache in-memory)
            prod_candidates = []
            for _, r in products.iterrows():
                name = r.get("product_name", "")
                nname, ntoks = normalize(name)
                s = score_tokens(q_tokens, ntoks)
                if s>0:
                    prod_candidates.append((s, r["product_id"], name))
            prod_candidates.sort(key=lambda x: (-x[0], -x[1]))
            for s, pid, name in prod_candidates[:30]:
                results.append({"type":"product", "score":s, "text": f"{pid} | {name}"})

        # 3) Also search aisles (name match) and include as fallback/context
        aisle_results = []
        if aisles is not None:
            mask_aisle = aisles["aisle"].str.lower().str.contains(ql, na=False)
            for _, r in aisles[mask_aisle].iterrows():
                aisle_results.append(f"AISLE: {r['aisle_id']} | {r['aisle']}")
            if not aisle_results:
                # token overlap for aisles
                for _, r in aisles.iterrows():
                    n, toks = normalize(r["aisle"])
                    s = score_tokens(q_tokens, toks)
                    if s>0:
                        aisle_results.append(f"AISLE: {r['aisle_id']} | {r['aisle']}")

        # 4) If nothing found, do a loose partial match: any product_name token contains any query token
        if not results and products is not None:
            loose = []
            qtset = set(q_tokens)
            for _, r in products.iterrows():
                name = r.get("product_name","").lower()
                name_tokens = [t for t in re.split(r'[^0-9a-z]+', name) if t]
                if any(any(qt in nt for nt in name_tokens) for qt in qtset):
                    loose.append((r["product_id"], r["product_name"]))
            for pid, nm in loose[:30]:
                results.append({"type":"product", "score":0.3, "text": f"{pid} | {nm}"})

        # 5) Format output to user
        if results:
            # dedupe by text and sort by score
            seen = set()
            ordered = []
            for it in sorted(results, key=lambda x: -x["score"]):
                if it["text"] in seen:
                    continue
                seen.add(it["text"])
                ordered.append(it)
            st.write(f"Found {len(ordered)} matches (showing top results).")
            for it in ordered[:30]:
                st.write(it["text"])
            # also show aisle suggestions if any
            if aisle_results:
                st.markdown("**Related aisles (context):**")
                for a in aisle_results[:10]:
                    st.write(a)
        else:
            st.warning("No matches found. Try broader keywords like 'snacks', 'frozen', 'milk', or try fewer words.")

