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

# ------------------- LLM + Rich-Visualization helpers (Hugging Face) -------------------
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

st.sidebar.markdown("----")
st.sidebar.subheader("LLM options (optional)")
HF_API_KEY = st.sidebar.text_input("Hugging Face API key (optional)", type="password", help="If provided, app will ask HF LLM to summarise results and suggest chart types.")
HF_MODEL = st.sidebar.selectbox("HF model", ["google/flan-t5-large","bigscience/bloomz-560m","tiiuae/falcon-7b-instruct"], index=0)

def call_hf_llm(prompt, model, token, max_tokens=512, timeout=30):
    """
    Simple HF Inference call using the Inference API endpoint.
    Returns the generated text (string) or raises.
    """
    if not token:
        return None
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.0}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # The Inference API sometimes returns a list of dicts with "generated_text"
    if isinstance(data, list):
        txt = data[0].get("generated_text") if isinstance(data[0], dict) else str(data[0])
    elif isinstance(data, dict) and "generated_text" in data:
        txt = data["generated_text"]
    else:
        # fallback: convert response to text
        txt = json.dumps(data)
    return txt


def make_llm_prompt_for_agg(question, short_context):
    """
    Produce a short instruction for the LLM: we already computed exact numbers locally,
    so LLM should explain the results and produce a small JSON with keys:
      - answer_text
      - chart_type (bar|line|pie|sunburst|none)
      - chart_spec: an object with fields: table, x, y, agg, top_k
      - followups: list of suggested follow-ups
    """
    prompt = f"""
You are a helpful assistant helping interpret analytical results.
Use ONLY the context below — do not invent values outside the context.

Context:
{json.dumps(short_context, indent=2)}

The user asked: "{question}"

Return ONLY valid JSON with the following keys:
- "answer_text": one short paragraph summarizing the insight.
- "chart_type": one of ["bar","line","pie","sunburst","none"].
- "chart_spec": {{ "table": "order_products__prior", "x": "product_id", "y": "count", "agg": "count", "top_k": 10 }}
    (Note: this is an example. Use quoted column names; do NOT use angle brackets.)
- "followups": list of 0..3 brief suggested follow-up questions.

Important: return ONLY the JSON object (no extra commentary).
"""
    return prompt
def compute_chart_df_from_spec(spec, dfs):
    """
    Given chart_spec and our loaded dfs dict, compute an exact DataFrame to plot.
    Returns a pandas DataFrame with columns [x,y] or None on error.
    """
    try:
        table = spec.get("table")
        x = spec.get("x")
        y = spec.get("y")
        agg = spec.get("agg")
        top_k = int(spec.get("top_k", 10))
        # only support a small set of safe ops
        if table == "order_products__prior" and agg == "count" and y in ("product_id","*"):
            # count product_id occurrences and optionally join to product_name
            prior = dfs.get("prior")
            products = dfs.get("products")
            counts = prior[x if x in prior.columns else "product_id"].value_counts().head(top_k)
            df = counts.rename_axis("val").reset_index(name="count")
            # if x is product_id, attempt to join product_name
            if x == "product_id" and products is not None:
                prod_idx = products.set_index("product_id")
                df["name"] = df["val"].apply(lambda pid: prod_idx.loc[pid]["product_name"] if pid in prod_idx.index else str(pid))
                df = df.rename(columns={"name":"x","count":"y"})[["x","y"]]
            elif x in prior.columns:
                df = df.rename(columns={"val":"x","count":"y"})[["x","y"]]
            else:
                # fallback: return value as x
                df = df.rename(columns={"val":"x","count":"y"})[["x","y"]]
            return df
        # support orders by day/hour
        if table == "orders" and x in ("order_dow","order_hour_of_day") and agg == "count":
            orders = dfs.get("orders")
            counts = orders[x].value_counts().sort_index()
            df = counts.rename_axis(x).reset_index(name="count").rename(columns={x:"x","count":"y"})
            return df.head(top_k)
    except Exception as e:
        print("compute_chart_df_from_spec error:", e)
    return None

def render_chart_from_df(df, chart_type, title="Chart"):
    """
    Render various chart types using Plotly and return the component (st.plotly_chart).
    """
    if df is None or df.empty:
        st.info("No data available for chart.")
        return
    if chart_type == "bar":
        fig = px.bar(df, x="x", y="y", title=title)
    elif chart_type == "line":
        fig = px.line(df, x="x", y="y", title=title)
    elif chart_type == "pie":
        if "y" not in df.columns:
            st.info("Pie requires a 'y' column.")
            return
        fig = px.pie(df, names="x", values="y", title=title)
    elif chart_type == "sunburst":
        # requires hierarchical columns; attempt simple product->aisle if present
        if "x" in df.columns and "y" in df.columns:
            # If df has 'x' with ' / ' splitable categories, use that
            # fallback: simple pie
            fig = px.sunburst(df, names="x", values="y", title=title)
        else:
            fig = px.pie(df, names="x", values="y", title=title)
    else:
        st.info("Chart type not supported; showing table instead.")
        st.dataframe(df)
        return
    st.plotly_chart(fig, use_container_width=True)

def show_retrieval_table(results_list):
    """
    Accept list of dicts {text, metadata, score} and render as a nice table + download.
    """
    if not results_list:
        st.info("No retrieval results.")
        return
    # build DataFrame
    rows = []
    for r in results_list:
        rows.append({"text": r.get("text"), "table": r.get("metadata",{}).get("table"), "row_id": r.get("metadata",{}).get("row_id"), "score": r.get("score", None)})
    df = pd.DataFrame(rows)
    st.dataframe(df.head(200))
    # download csv
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="retrieval.csv">Download retrieval results (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

# ------------------ End of LLM + Visualization helpers -------------------

# integrate LLM step AFTER aggregation or retrieval handling
# place a small UI: "Ask LLM to summarize & suggest visual"
if run and HF_API_KEY:
    try:
        # build short context summary for the LLM
        # For aggregation: prefer to show a short top-k or stats; for retrieval: show top few retrieval results
        short_context = ""
        if intent == "aggregation":
            # attempt to compute a local short context summary from prior tables
            try:
                # compute a small top-6 sample if top-products handler ran
                top_df = None
                if "product" in ql or "most" in ql or "top" in ql:
                    top_df = agg_top_products(prior, products, top_k=6)
                elif "least" in ql:
                    top_df = agg_least_ordered(prior, products, top_k=6)
                if top_df is not None:
                    # convert to simple json table
                    short_context = top_df.to_dict(orient="records")
                else:
                    short_context = {"note":"no top sample available"}
            except Exception as e:
                short_context = {"error":"could not compute short context", "e": str(e)}
        else:
            # retrieval: show first 6 retrieval result texts if available
            try:
                # reuse earlier 'results' if present in memory; else do quick product name search
                quick = []
                if 'results' in locals() and isinstance(results, list) and results:
                    quick = [r.get("text") for r in results[:8]]
                else:
                    # quick local product search
                    mask = products["product_name"].str.lower().str.contains(ql, na=False)
                    quick = products[mask].head(6)["product_name"].tolist()
                short_context = {"retrieved_examples": quick}
            except Exception as e:
                short_context = {"error":"retrieval short context failed","e":str(e)}

        prompt = make_llm_prompt_for_agg(ql_raw, json.dumps(short_context, ensure_ascii=False, indent=0))
        hf_out = call_hf_llm(prompt, HF_MODEL, HF_API_KEY)
        if hf_out:
            # attempt to extract JSON block
            m = None
            try:
                m = json.loads(hf_out)
            except Exception:
                # try to find first JSON-like substring
                import re as _re
                mm = _re.search(r"\{.*\}", hf_out, flags=_re.DOTALL)
                if mm:
                    try:
                        m = json.loads(mm.group(0))
                    except Exception:
                        m = None
            if m:
                st.subheader("LLM summary & visualization suggestion")
                st.write(m.get("answer_text","(no answer_text)"))
                chart_type = m.get("chart_type","none")
                chart_spec = m.get("chart_spec",{})
                # compute chart df (locally exact) and render
                chart_df = compute_chart_df_from_spec(chart_spec, {"prior":prior,"products":products,"orders":orders})
                render_chart_from_df(chart_df, chart_type, title="LLM suggested chart")
                # followups
                if "followups" in m:
                    st.markdown("**Follow-up suggestions:**")
                    for f in m["followups"][:3]:
                        st.write("- "+f)
            else:
                st.warning("LLM returned text but JSON parse failed — showing raw output:")
                st.code(hf_out[:2000])
    except requests.HTTPError as he:
        st.error(f"Hugging Face API error: {he} — check model name and HF token.")
    except Exception as e:
        st.error(f"LLM step failed: {e}")

