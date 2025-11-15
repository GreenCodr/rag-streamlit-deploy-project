# RAG-DB Streamlit app — improved aggregation handlers + retrieval
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="RAG-DB (Self-contained)", layout="wide")
st.title("RAG-DB — Instacart demo (improved)")
st.write("Self-contained: CSVs in repo. Aggregation handlers for common queries + smart retrieval.")

DATA_DIR = os.path.join("data", "data", "instacart")

def load_csv_safe(name, nrows=None):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, nrows=nrows)

# --- text helpers ---
_non_alnum = re.compile(r"[^0-9a-z]+")
def normalize(text):
    if text is None:
        return "", []
    t = str(text).lower()
    t = _non_alnum.sub(" ", t)
    tokens = [tok for tok in t.split() if len(tok)>1]
    return " ".join(tokens), tokens

def score_tokens(query_tokens, target_tokens):
    if not query_tokens or not target_tokens:
        return 0.0
    set_q = set(query_tokens)
    set_t = set(target_tokens)
    return len(set_q & set_t) / max(len(set_q), 1)

# --- UI controls ---
st.sidebar.header("Options")
sample_nrows_prior = st.sidebar.number_input(
    "Rows to load from order_products__prior (0 = full)",
    min_value=0, value=50000, step=10000
)
mode_override = st.sidebar.selectbox("Force mode", ["auto","retrieval","aggregation"])
q = st.text_area("Ask a question:", value="Which products appear most frequently in prior orders?", height=140)
run = st.button("Run Query")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior=None):
    products = load_csv_safe("products")
    aisles = load_csv_safe("aisles")
    departments = load_csv_safe("departments")
    orders = load_csv_safe("orders")
    prior = load_csv_safe("order_products__prior", nrows=nrows_prior)
    return {"products":products,"aisles":aisles,"departments":departments,"orders":orders,"prior":prior}

# --- aggregation handlers ---
def agg_top_products(prior, products, top_k=10):
    counts = prior["product_id"].value_counts().head(top_k)
    df = counts.rename_axis("product_id").reset_index(name="count")
    if products is not None:
        prod_idx = products.set_index("product_id")
        df["product_name"] = df["product_id"].apply(lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else "Unknown")
    else:
        df["product_name"] = df["product_id"].astype(str)
    return df[["product_name","count"]].rename(columns={"product_name":"x","count":"y"})

def agg_total_orders(prior):
    return len(prior)

def agg_average_orders_per_product(prior):
    total_rows = len(prior)
    unique_products = prior["product_id"].nunique()
    avg = total_rows / unique_products if unique_products>0 else 0
    return {"total_prior_rows": total_rows, "unique_products": unique_products, "avg_orders_per_product": round(avg,4)}

def agg_top_by_aisle(prior, products, aisles, top_k=10):
    # join prior -> products -> aisle
    if products is None or aisles is None:
        return None
    merged = prior.merge(products[["product_id","aisle_id"]], on="product_id", how="left")
    counts = merged["aisle_id"].value_counts().head(top_k)
    df = counts.rename_axis("aisle_id").reset_index(name="count")
    aisle_idx = aisles.set_index("aisle_id")
    df["aisle_name"] = df["aisle_id"].apply(lambda x: aisle_idx.loc[x]["aisle"] if x in aisle_idx.index else "Unknown")
    return df[["aisle_name","count"]].rename(columns={"aisle_name":"x","count":"y"})

def agg_top_by_department(prior, products, departments, top_k=10):
    if products is None or departments is None:
        return None
    merged = prior.merge(products[["product_id","department_id"]], on="product_id", how="left")
    counts = merged["department_id"].value_counts().head(top_k)
    df = counts.rename_axis("department_id").reset_index(name="count")
    dep_idx = departments.set_index("department_id")
    df["department_name"] = df["department_id"].apply(lambda x: dep_idx.loc[x]["department"] if x in dep_idx.index else "Unknown")
    return df[["department_name","count"]].rename(columns={"department_name":"x","count":"y"})

def agg_least_ordered(prior, products, top_k=10):
    counts = prior["product_id"].value_counts()
    tail = counts.tail(top_k)
    df = tail.rename_axis("product_id").reset_index(name="count")
    if products is not None:
        prod_idx = products.set_index("product_id")
        df["product_name"] = df["product_id"].apply(lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else "Unknown")
    else:
        df["product_name"] = df["product_id"].astype(str)
    return df[["product_name","count"]].rename(columns={"product_name":"x","count":"y"})

def agg_orders_by_day(orders):
    if orders is None or "order_dow" not in orders.columns:
        return None
    counts = orders["order_dow"].value_counts().sort_index()
    df = counts.rename_axis("order_dow").reset_index(name="count")
    return df.rename(columns={"order_dow":"x","count":"y"})

def agg_orders_by_hour(orders):
    if orders is None or "order_hour_of_day" not in orders.columns:
        return None
    counts = orders["order_hour_of_day"].value_counts().sort_index()
    df = counts.rename_axis("hour").reset_index(name="count")
    return df.rename(columns={"hour":"x","count":"y"})

# --- main run ---
if run:
    st.info("Loading CSVs from the repo...")
    nrows = None if sample_nrows_prior==0 else int(sample_nrows_prior)
    tables = load_tables(nrows_prior=nrows)
    products = tables["products"]
    aisles = tables["aisles"]
    departments = tables["departments"]
    orders = tables["orders"]
    prior = tables["prior"]

    if products is None or prior is None:
        st.error("Missing CSVs! Upload products.csv and order_products__prior.csv to data/data/instacart/")
        st.stop()

    ql_raw = q or ""
    ql = ql_raw.lower().strip()

    # simple intent detection
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean","how many","least"]
    retr_keywords = ["show","list","example","what","which","find","give","where","example"]
    intent = ("aggregation" if any(k in ql for k in agg_keywords) else "retrieval")
    if mode_override != "auto":
        intent = mode_override

    st.markdown(f"### Intent: `{intent}`")

    if intent == "aggregation":
        # many specific handlers using keyword heuristics
        # 1) average orders per product
        if "average" in ql and "product" in ql and ("order" in ql or "orders" in ql):
            res = agg_average_orders_per_product(prior)
            st.write(f"Total prior rows: {res['total_prior_rows']:,}")
            st.write(f"Unique products: {res['unique_products']:,}")
            st.success(f"Average orders per product (sampled): {res['avg_orders_per_product']}")
        # 2) total orders
        elif "total" in ql and ("order" in ql or "orders" in ql) and "product" not in ql:
            total = agg_total_orders(prior)
            st.success(f"Total prior order-product rows (sampled): {total:,}")
        # 3) top products
        elif ("most" in ql or "top" in ql or "frequent" in ql or "highest" in ql) and ("product" in ql or "items" in ql or "ordered" in ql):
            topk = 10
            df = agg_top_products(prior, products, top_k=topk)
            st.subheader("Top products (sampled)")
            st.dataframe(df)
            fig,ax = plt.subplots(figsize=(10,4))
            ax.bar(df["x"], df["y"])
            plt.xticks(rotation=45,ha="right")
            st.pyplot(fig)
        # 4) least ordered
        elif "least" in ql or "least ordered" in ql:
            df = agg_least_ordered(prior, products, top_k=10)
            st.subheader("Least-ordered products (sampled)")
            st.dataframe(df)
        # 5) top by aisle
        elif "aisle" in ql and ("top" in ql or "most" in ql or "highest" in ql):
            df = agg_top_by_aisle(prior, products, aisles, top_k=10)
            if df is None:
                st.error("Need aisles.csv and products.csv for aisle-based aggregation.")
            else:
                st.subheader("Top aisles by prior orders (sampled)")
                st.dataframe(df)
                fig,ax = plt.subplots(figsize=(10,4))
                ax.bar(df["x"], df["y"])
                plt.xticks(rotation=45,ha="right")
                st.pyplot(fig)
        # 6) top by department
        elif "department" in ql and ("top" in ql or "most" in ql or "highest" in ql):
            df = agg_top_by_department(prior, products, departments, top_k=10)
            if df is None:
                st.error("Need departments.csv and products.csv for department-based aggregation.")
            else:
                st.subheader("Top departments by prior orders (sampled)")
                st.dataframe(df)
                fig,ax = plt.subplots(figsize=(10,4))
                ax.bar(df["x"], df["y"])
                plt.xticks(rotation=45,ha="right")
                st.pyplot(fig)
        # 7) orders by day/hour
        elif "day" in ql or "weekday" in ql or "order_dow" in ql:
            df = agg_orders_by_day(orders)
            if df is None:
                st.error("orders.csv missing or doesn't have 'order_dow' column.")
            else:
                st.subheader("Orders by day of week")
                st.dataframe(df)
                fig,ax = plt.subplots(figsize=(8,4))
                ax.bar(df["x"].astype(str), df["y"])
                st.pyplot(fig)
        elif "hour" in ql or "order_hour" in ql or "order_hour_of_day" in ql:
            df = agg_orders_by_hour(orders)
            if df is None:
                st.error("orders.csv missing or doesn't have 'order_hour_of_day' column.")
            else:
                st.subheader("Orders by hour")
                st.dataframe(df)
                fig,ax = plt.subplots(figsize=(10,4))
                ax.bar(df["x"].astype(str), df["y"])
                st.pyplot(fig)
        else:
            st.info("Aggregation detected, but I don't have a precise handler for this exact question. Try rephrasing or ask one of the example aggregation queries.")
    else:
        # --- retrieval (same improved logic as before) ---
        st.subheader("Smart Retrieval Results")
        q_norm, q_tokens = normalize(ql_raw)

        results = []
        # exact substring in products
        mask_prod = products["product_name"].str.lower().str.contains(ql, na=False)
        prod_hits = products[mask_prod]
        for _, r in prod_hits.head(50).iterrows():
            results.append({"type":"product","score":1.0,"text":f'{r["product_id"]} | {r["product_name"]}'})
        # token overlap
        if not results:
            prod_candidates=[]
            for _, r in products.iterrows():
                name = r.get("product_name","")
                nname, ntoks = normalize(name)
                s = score_tokens(q_tokens, ntoks)
                if s>0:
                    prod_candidates.append((s,r["product_id"],name))
            prod_candidates.sort(key=lambda x:(-x[0],x[1]))
            for s,pid,name in prod_candidates[:30]:
                results.append({"type":"product","score":s,"text":f"{pid} | {name}"})
        # loose partial token contains
        if not results and products is not None:
            loose=[]
            qtset=set(q_tokens)
            for _,r in products.iterrows():
                name=r.get("product_name","").lower()
                name_tokens=[t for t in re.split(r'[^0-9a-z]+', name) if t]
                if any(any(qt in nt for nt in name_tokens) for qt in qtset):
                    loose.append((r["product_id"], r["product_name"]))
            for pid,nm in loose[:30]:
                results.append({"type":"product","score":0.3,"text":f"{pid} | {nm}"})
        # aisles fallback/context
        aisle_results=[]
        if aisles is not None:
            mask_aisle = aisles["aisle"].str.lower().str.contains(ql, na=False)
            for _,r in aisles[mask_aisle].iterrows():
                aisle_results.append(f"AISLE: {r['aisle_id']} | {r['aisle']}")
            if not aisle_results:
                for _,r in aisles.iterrows():
                    n, toks = normalize(r["aisle"])
                    if score_tokens(q_tokens, toks)>0:
                        aisle_results.append(f"AISLE: {r['aisle_id']} | {r['aisle']}")
        if results:
            st.write(f"Found {len(results)} matches (top shown):")
            seen=set()
            for it in sorted(results, key=lambda x:-x["score"])[:30]:
                if it["text"] in seen: continue
                seen.add(it["text"])
                st.write(it["text"])
            if aisle_results:
                st.markdown("**Related aisles:**")
                for a in aisle_results[:8]:
                    st.write(a)
        else:
            st.warning("No matches found. Try broader keywords like 'frozen', 'snacks', 'milk'.")
