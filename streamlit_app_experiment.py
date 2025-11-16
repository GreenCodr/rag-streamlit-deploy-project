# streamlit_app_experiment.py
"""
Self-contained Streamlit app (NO FastAPI) for RAG-DB (Instacart).
- Put CSVs in data/data/instacart/
- Optional: provide HF_API_KEY in .env or paste in the sidebar for LLM summaries.
"""
import os, json, re, time
from pathlib import Path
from typing import Optional, Any, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv

# Optional HF client
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

load_dotenv()

# --- Config ---
st.set_page_config(page_title="RAG-DB — Instacart (experiment)", layout="wide")
BASE = Path(".").resolve()
DATA_DIR = BASE / "data" / "data" / "instacart"

# ---------- Helpers ----------
def load_csv_if_exists(name: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, nrows=nrows)

def detect_intent(q: str) -> str:
    ql = q.lower()
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean","least","how many","per product","reorder","ratio","percentage","orders by","which days","day of week"]
    retrieval_keywords = ["show","list","example","what are","what is","which aisles","give me","find","show me","list items","list products","contain","contains"]
    if any(k in ql for k in agg_keywords):
        return "aggregation"
    if any(k in ql for k in retrieval_keywords):
        return "retrieval"
    return "retrieval"

def tokenize(s: str):
    return [t for t in re.split(r"[^0-9a-z]+", str(s).lower()) if t]

def fuzzy_search_products(products: pd.DataFrame, q: str, top_k: int = 50):
    if products is None or q.strip()=="":
        return []
    qtokens = tokenize(q)
    out=[]
    for _,r in products.iterrows():
        name = str(r.get("product_name",""))
        ntoks = tokenize(name)
        if not ntoks: continue
        overlap = sum(1 for qt in qtokens if any(qt in nt for nt in ntoks))
        if overlap>0:
            score = overlap/len(ntoks)
            out.append({"product_id": r["product_id"], "product_name": name, "score": score})
    out = sorted(out, key=lambda x:-x["score"])
    seen=set(); res=[]
    for it in out:
        if it["product_id"] in seen: continue
        seen.add(it["product_id"]); res.append(it)
        if len(res)>=top_k: break
    return res

# --- Aggregation functions (exact local) ---
def top_products_prior(prior: pd.DataFrame, products: pd.DataFrame, top_k: int = 10):
    counts = prior["product_id"].value_counts().head(top_k)
    df = counts.rename_axis("product_id").reset_index(name="count")
    if products is not None:
        prod_idx = products.set_index("product_id")
        df["product_name"] = df["product_id"].apply(lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else str(x))
    else:
        df["product_name"] = df["product_id"].astype(str)
    df = df[["product_name","count"]].rename(columns={"product_name":"x","count":"y"})
    return df

def least_products_prior(prior: pd.DataFrame, products: pd.DataFrame, top_k: int = 10):
    counts = prior["product_id"].value_counts()
    tail = counts[counts>0].tail(top_k)
    df = tail.rename_axis("product_id").reset_index(name="count")
    if products is not None:
        prod_idx = products.set_index("product_id")
        df["product_name"] = df["product_id"].apply(lambda x: prod_idx.loc[x]["product_name"] if x in prod_idx.index else str(x))
    else:
        df["product_name"] = df["product_id"].astype(str)
    df = df[["product_name","count"]].rename(columns={"product_name":"x","count":"y"})
    return df

def avg_orders_per_product(prior: pd.DataFrame):
    counts = prior["product_id"].value_counts()
    return float(counts.mean())

def avg_reorder_ratio(prior: pd.DataFrame):
    if "reordered" in prior.columns:
        return float(prior.groupby("product_id")["reordered"].mean().mean())
    return None

def orders_by_day_of_week(orders: pd.DataFrame):
    if orders is None or "order_dow" not in orders.columns:
        return None
    counts = orders["order_dow"].value_counts().sort_index()
    df = counts.rename_axis("day").reset_index(name="count")
    df["x"] = df["day"].astype("str"); df["y"] = df["count"]
    return df[["x","y"]]

def compute_chart_from_spec(chart_spec: Dict[str,Any], tables: Dict[str,pd.DataFrame]):
    try:
        table_name = chart_spec.get("table")
        if table_name not in tables or tables[table_name] is None:
            return None
        df = tables[table_name].copy()
        js = chart_spec.get("join")
        if js:
            right_table = js.get("table")
            left_on = js.get("left_on")
            right_on = js.get("right_on")
            right_label = js.get("right_label")
            if right_table in tables and tables[right_table] is not None:
                right_df = tables[right_table][[right_on,right_label]].drop_duplicates()
                df = df.merge(right_df, left_on=left_on, right_on=right_on, how="left")
        xcol = chart_spec.get("x")
        ycol = chart_spec.get("y")
        agg = chart_spec.get("agg","count")
        top_k = int(chart_spec.get("top_k", 10))
        if agg == "count":
            grouped = df.groupby(xcol).size().rename("y").reset_index().sort_values("y",ascending=False).head(top_k)
        elif agg == "sum":
            grouped = df.groupby(xcol)[ycol].sum().rename("y").reset_index().sort_values("y",ascending=False).head(top_k)
        elif agg == "avg":
            grouped = df.groupby(xcol)[ycol].mean().rename("y").reset_index().sort_values("y",ascending=False).head(top_k)
        else:
            return None
        grouped = grouped.rename(columns={xcol:"x"})[["x","y"]]
        return grouped
    except Exception as e:
        st.exception(e)
        return None

def render_chart(df: pd.DataFrame, chart_type: str="bar", title: str=""):
    if df is None or df.empty:
        st.info("No chart data to render.")
        return
    if chart_type=="bar":
        fig = px.bar(df, x="x", y="y", title=title)
    elif chart_type=="line":
        fig = px.line(df, x="x", y="y", title=title)
    elif chart_type=="pie":
        fig = px.pie(df, names="x", values="y", title=title)
    elif chart_type=="treemap":
        fig = px.treemap(df, path=["x"], values="y", title=title)
    else:
        fig = px.bar(df, x="x", y="y", title=title)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ LLM wrapper (conservative, JSON) ------------------
def _extract_json_from_text(text: str):
    if not isinstance(text, str):
        return None
    # try direct json
    try:
        return json.loads(text)
    except Exception:
        pass
    # find braces blocks
    matches = re.findall(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if matches:
        matches = sorted(matches, key=lambda s:-len(s))
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue
    try:
        alt = text.strip().replace("'", "\"")
        return json.loads(alt)
    except Exception:
        return None

def hf_chat_wrapper(hf_token: str, model_id: str, user_question: str, short_context: Any=None, max_tokens:int=400):
    SAFE = {"answer_text":"(LLM unavailable)","chart_type":"none","chart_spec":{},"followups":[],"confidence":0.0}
    if not hf_token or not HF_AVAILABLE:
        return {**SAFE, "answer_text": "LLM disabled (HF token missing or huggingface_hub not available)."}
    # Build careful prompt: ask for JSON only, be conservative, include retrieved examples
    ctxtxt = ""
    if short_context:
        try:
            ctxtxt = json.dumps(short_context, ensure_ascii=False, indent=2)
        except Exception:
            ctxtxt = str(short_context)
    system = (
        "You are a cautious data assistant. USE ONLY the numeric and textual values provided in the short context "
        "and the list of retrieved examples. Do NOT make up numbers. If unsure, say 'I may be mistaken' and keep confidence low. "
        "Return ONLY a JSON object (no explanation) with keys: answer_text (string, 1-3 sentences), confidence (0.0-1.0), chart_type (bar|line|pie|treemap|none), chart_spec (object with table,x,y,agg,optional join,optional top_k), followups (list)."
    )
    user = f"Short context:\n{ctxtxt}\n\nUser question:\n{user_question}\n\nReturn JSON only. If describing a chart_spec, use table names: prior, products, orders, aisles, departments. Be conservative."
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    try:
        client = InferenceClient(model=model_id, token=hf_token)
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.0)
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = str(resp)
        parsed = _extract_json_from_text(content)
        if not parsed:
            # fallback: return raw text as answer_text but low confidence
            return {"answer_text": content.strip()[:1000], "chart_type":"none","chart_spec":{},"followups":[],"confidence":0.3}
        # normalize
        ans = parsed.get("answer_text") or parsed.get("answer") or parsed.get("text") or ""
        chart_type = parsed.get("chart_type","none")
        chart_spec = parsed.get("chart_spec", {}) or {}
        followups = parsed.get("followups", []) or []
        confidence = float(parsed.get("confidence", 0.9 if ans else 0.5))
        return {"answer_text": str(ans).strip(), "chart_type": chart_type if chart_type in {"bar","line","pie","treemap","none"} else "none", "chart_spec": chart_spec if isinstance(chart_spec, dict) else {}, "followups": followups if isinstance(followups,list) else [], "confidence": max(0.0, min(1.0, float(confidence)))}
    except Exception as e:
        return {"answer_text": f"LLM call failed: {repr(e)}", "chart_type":"none","chart_spec":{},"followups":[],"confidence":0.0}

# ---------------- Streamlit UI ----------------
st.title("RAG-DB — Instacart (experiment, no backend)")
st.sidebar.header("Options / Settings")
sample_nrows_prior = st.sidebar.number_input("Rows to load from order_products__prior (0 = full)", min_value=0, value=50000, step=10000)
mode = st.sidebar.selectbox("Mode", ["local-only","with-llm"])
hf_token_input = st.sidebar.text_input("Hugging Face API key (optional)", value=os.getenv("HF_API_KEY",""), type="password")
hf_model_input = st.sidebar.text_input("HF model id (chat-capable)", value="meta-llama/Meta-Llama-3-8B-Instruct")
st.sidebar.markdown("---")
st.sidebar.write("Notes:")
st.sidebar.write("- Local-only = deterministic exact answers + charts.")
st.sidebar.write("- with-llm = ask HF to produce a short friendly JSON summary and (optional) chart_spec; app computes chart exactly.")
st.sidebar.write("- If HF fails the app shows deterministic fallback.")

q = st.text_area("Ask a question", value="Which products appear most frequently in prior orders?", height=140)
run = st.button("Run")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior: Optional[int]=None):
    products = load_csv_if_exists("products")
    aisles = load_csv_if_exists("aisles")
    departments = load_csv_if_exists("departments")
    orders = load_csv_if_exists("orders")
    prior = load_csv_if_exists("order_products__prior", nrows=nrows_prior)
    return {"products":products,"aisles":aisles,"departments":departments,"orders":orders,"prior":prior}

if run:
    st.info("Loading CSVs...")
    nrows = None if sample_nrows_prior==0 else int(sample_nrows_prior)
    tables = load_tables(nrows_prior=nrows)
    products = tables["products"]; aisles = tables["aisles"]; departments = tables["departments"]; orders = tables["orders"]; prior = tables["prior"]
    if prior is None or products is None:
        st.error("Missing CSVs. Ensure CSVs are in data/data/instacart/")
        st.stop()

    ql_raw = str(q).strip()
    st.markdown("### Intent detection & execution")
    intent = detect_intent(ql_raw)
    st.write(f"Detected intent: **{intent}**")

    # short context for LLM
    short_context = {}
    try:
        short_context["rows_in_prior"] = int(len(prior))
        short_context["top5"] = top_products_prior(prior, products, top_k=5).to_dict(orient="records")
    except Exception as e:
        short_context["error"] = str(e)

    use_llm = (mode=="with-llm") and hf_token_input.strip()!="" and HF_AVAILABLE
    tables_map = {"prior":prior,"products":products,"orders":orders,"aisles":aisles,"departments":departments}

    # ---------------- AGGREGATION ----------------
    if intent=="aggregation":
        ql = ql_raw.lower()
        if any(x in ql for x in ["most frequently","most frequent","top products","most ordered","appear most frequently"]):
            df = top_products_prior(prior, products, top_k=15)
            st.subheader("Top products (exact counts)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}).head(15))
            render_chart(df, chart_type="bar", title="Top products (count)")
            with st.expander("More visualizations"):
                render_chart(df, chart_type="treemap", title="Top products (treemap)")
                render_chart(df, chart_type="pie", title="Top products (pie)")
        elif any(x in ql for x in ["least frequently","least ordered"]):
            df = least_products_prior(prior, products, top_k=15)
            st.subheader("Least frequently ordered (sample)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}))
            render_chart(df, chart_type="bar", title="Least frequently ordered")
        elif "average number of orders per product" in ql or "average orders per product" in ql or "avg orders per product" in ql:
            avg = avg_orders_per_product(prior)
            st.write(f"**Average occurrences per product**: {avg:.2f}")
        elif "reorder ratio" in ql or "average reorder" in ql:
            r = avg_reorder_ratio(prior)
            if r is not None:
                st.write(f"Average reorder ratio across products: **{r:.3f}**")
            else:
                st.info("No 'reordered' column present.")
        elif "day" in ql or "day of week" in ql or "which days" in ql:
            df = orders_by_day_of_week(orders)
            if df is not None:
                st.subheader("Orders by day of week (0=Sunday..6=Saturday)")
                st.dataframe(df.rename(columns={"x":"day","y":"count"}))
                render_chart(df, chart_type="bar", title="Orders by day")
        else:
            # fallback: compute top products locally + optionally ask LLM for polished summary
            local_df = top_products_prior(prior, products, top_k=10)
            local_answer = f"Top product: {local_df.iloc[0]['x']} with {int(local_df.iloc[0]['y'])} occurrences."
            if use_llm:
                with st.spinner("Asking LLM for a helpful summary and chart suggestion..."):
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, {**short_context, "retrieved_examples": [r["product_name"] for r in local_df.head(10).to_dict(orient='records')]})
                    st.subheader("LLM suggested (parsed)")
                    st.json(jobj)
                    llm_text = jobj.get("answer_text","")
                    # Defensive override: if LLM says "no" but we have results -> correct it
                    if isinstance(llm_text, str) and any(p in llm_text.lower() for p in ["no items","no matches","none found"]) and len(local_df)>0:
                        st.warning("LLM claimed no results but deterministic computation found matches — showing accurate results below.")
                        st.write(local_answer)
                        render_chart(local_df, chart_type="bar", title="Local top products (exact)")
                        st.dataframe(local_df.rename(columns={"x":"product_name","y":"count"}))
                    else:
                        st.write("LLM answer:")
                        st.write(llm_text)
                        # compute LLM chart if present
                        cs = jobj.get("chart_spec", {})
                        ct = jobj.get("chart_type", "bar")
                        if cs:
                            chart_df = compute_chart_from_spec(cs, tables_map)
                            if chart_df is not None:
                                render_chart(chart_df, chart_type=ct, title="LLM suggested chart (computed exactly)")
                            else:
                                st.warning("LLM suggested chart_spec couldn't be computed locally.")
                        # fallback show local
                        if not cs:
                            st.write(local_answer)
                            render_chart(local_df, chart_type="bar", title="Local top products (exact)")
            else:
                st.info("LLM disabled. Showing deterministic fallback.")
                st.write(local_answer)
                render_chart(local_df, chart_type="bar", title="Local top products (exact)")

    # ---------------- RETRIEVAL ----------------
    else:
        st.subheader("Retrieval / keyword search")
        proc = ql_raw
        prod_hits = fuzzy_search_products(products, proc, top_k=50)
        if prod_hits:
            st.write(f"Products matching query (top {len(prod_hits)}):")
            df = pd.DataFrame(prod_hits)[["product_id","product_name","score"]]
            st.dataframe(df)
            # LLM summary of retrieved list (optional)
            if use_llm:
                # include top retrieved example names in LLM context
                retrieved_names = [r["product_name"] for r in prod_hits[:20]]
                sc_for_llm = {**short_context, "retrieved_examples": retrieved_names}
                jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, sc_for_llm)
                st.subheader("LLM summary (optional)")
                llm_text = jobj.get("answer_text","")
                # defensive: if LLM says "no results" but prod_hits exists -> override
                if isinstance(llm_text, str) and any(p in llm_text.lower() for p in ["no items","no matches","none found"]) and len(prod_hits)>0:
                    st.warning("LLM summary contradicted retrieved items — showing deterministic summary:")
                    top_names = retrieved_names[:8]
                    st.write(f"Found {len(prod_hits)} matching product(s). Top examples: {', '.join(top_names)}")
                    st.dataframe(pd.DataFrame([{"product_name":n} for n in top_names]))
                else:
                    st.write(llm_text)
        else:
            st.info("No product fuzzy match found. Trying aisles substring match.")
            if aisles is not None:
                mask = aisles["aisle"].str.lower().str.contains(proc.lower(), na=False)
                if mask.any():
                    st.write("Matching aisles (examples):")
                    st.dataframe(aisles[mask].head(30))
                else:
                    st.warning("No matches found. Try simpler keywords like 'frozen', 'snacks', 'produce'.")
            else:
                st.warning("Aisles data not available.")

    st.success("Done.")