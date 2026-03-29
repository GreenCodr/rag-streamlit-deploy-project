# streamlit_app_experiment.py
"""
Experimental Streamlit app for RAG-DB (Instacart) — experiment branch.
- Local exact aggregation + retrieval
- Optional robust Hugging Face LLM for friendly summary & chart_spec suggestion
- Defensive: local facts always win over LLM contradictions
"""
import os, json, re, time
from pathlib import Path
from typing import Optional, Any, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

# --- HF client availability ---
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

# load .env
load_dotenv()

# Config
st.set_page_config(page_title="RAG-DB — Instacart (experiment)", layout="wide")
BASE = Path(".").resolve()
DATA_DIR = BASE / "data" / "data" / "instacart"

# Backend check (optional, keep for your app)
BACKEND_BASE = "http://127.0.0.1:8000"
def backend_is_healthy(timeout: float = 0.7) -> bool:
    try:
        resp = requests.get(f"{BACKEND_BASE}/ping", timeout=timeout)
        return resp.status_code == 200
    except RequestException:
        return False

# ---------- Helpers ----------
def load_csv_if_exists(name: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, nrows=nrows)

def detect_intent(q: str) -> str:
    ql = (q or "").lower()
    agg_keywords = ["count","top","most","frequent","total","sum","avg","average","mean","least","how many","per product","reorder","ratio","percentage","orders by","which days","day of week"]
    retrieval_keywords = ["show","list","example","what are","what is","which aisles","give me","find","show me","list items","list products","contain"]
    if any(k in ql for k in agg_keywords):
        return "aggregation"
    if any(k in ql for k in retrieval_keywords):
        return "retrieval"
    return "retrieval"

def tokenize(s: str):
    return [t for t in re.split(r"[^0-9a-z]+", (s or "").lower()) if t]

def fuzzy_search_products(products: pd.DataFrame, q: str, top_k: int = 30):
    if products is None: return []
    qtokens = tokenize(q)
    out=[]
    for _,r in products.iterrows():
        name = str(r.get("product_name",""))
        ntoks = tokenize(name)
        if not ntoks: continue
        overlap = sum(1 for qt in qtokens if any(qt in nt for nt in ntoks))
        if overlap>0:
            score = overlap/len(ntoks)
            out.append({"product_id": int(r["product_id"]), "product_name": name, "score": score})
    out = sorted(out, key=lambda x:-x["score"])
    seen=set(); res=[]
    for it in out:
        if it["product_id"] in seen: continue
        seen.add(it["product_id"]); res.append(it)
        if len(res)>=top_k: break
    return res

# --- Aggregation functions ---
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

# compute generic chart from a LLM-suggested "chart_spec"
_ALLOWED_TABLES = {"prior","products","orders","aisles","departments"}

def _validate_chart_spec(raw):
    if not isinstance(raw, dict):
        return {}
    table = raw.get("table")
    x = raw.get("x")
    y = raw.get("y")
    agg = raw.get("agg","count")
    top_k = int(raw.get("top_k",10)) if raw.get("top_k") is not None else 10
    join = raw.get("join")
    if table not in _ALLOWED_TABLES:
        return {}
    if not x:
        if table == "prior":
            x = "product_id"
        elif table == "orders":
            x = "order_dow"
        else:
            return {}
    if agg not in {"count","sum","avg"}:
        agg = "count"
    cleaned = {"table":table,"x":x,"y":y,"agg":agg,"top_k":top_k}
    if join and isinstance(join, dict):
        jt = join.get("table")
        if jt in _ALLOWED_TABLES:
            cleaned["join"] = {"table": jt, "left_on": join.get("left_on"), "right_on": join.get("right_on"), "right_label": join.get("right_label")}
    return cleaned

def compute_chart_from_spec(chart_spec: dict, tables: dict):
    try:
        cs = _validate_chart_spec(chart_spec)
        if not cs:
            return None
        table_name = cs["table"]
        if table_name not in tables or tables[table_name] is None:
            return None
        df = tables[table_name].copy()
        if "join" in cs:
            js = cs["join"]
            if js.get("table") in tables and tables[js.get("table")] is not None:
                right_df = tables[js["table"]][[js["right_on"], js["right_label"]]].drop_duplicates()
                df = df.merge(right_df, left_on=js["left_on"], right_on=js["right_on"], how="left")
        xcol = cs["x"]; ycol = cs.get("y"); agg = cs["agg"]; top_k = cs["top_k"]
        if agg == "count":
            grouped = df.groupby(xcol).size().rename("y").reset_index().sort_values("y", ascending=False).head(top_k)
        elif agg == "sum" and ycol:
            grouped = df.groupby(xcol)[ycol].sum().rename("y").reset_index().sort_values("y", ascending=False).head(top_k)
        elif agg == "avg" and ycol:
            grouped = df.groupby(xcol)[ycol].mean().rename("y").reset_index().sort_values("y", ascending=False).head(top_k)
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
    if chart_type == "bar":
        fig = px.bar(df, x="x", y="y", title=title, color="y",
                     color_continuous_scale="Viridis", text="y")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                          marker_line_color="rgb(8,48,107)", marker_line_width=1.5)
        fig.update_layout(title_font_size=18, title_x=0.5, xaxis_tickangle=-45,
                          showlegend=False, height=500, template="plotly_white",
                          margin=dict(t=80, b=120))
    elif chart_type == "line":
        fig = px.line(df, x="x", y="y", title=title, markers=True)
        fig.update_traces(line=dict(width=3, color="#1f77b4"), marker=dict(size=10, color="#ff7f0e"))
        fig.update_layout(title_font_size=18, title_x=0.5, height=450, template="plotly_white")
    elif chart_type == "pie":
        fig = px.pie(df, names="x", values="y", title=title, hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.03]*len(df))
        fig.update_layout(title_font_size=18, title_x=0.5, height=500,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    elif chart_type == "treemap":
        fig = px.treemap(df, path=["x"], values="y", title=title, color="y",
                         color_continuous_scale="RdYlGn")
        fig.update_layout(title_font_size=18, title_x=0.5, height=600)
    else:
        fig = px.bar(df, x="x", y="y", title=title)
    st.plotly_chart(fig, use_container_width=True)


def generate_fallback_summary(query: str, df: pd.DataFrame, metadata: dict = None) -> dict:
    """Generate a summary from local results when LLM is disabled or fails."""
    metadata = metadata or {}
    query_type = metadata.get("query_type", "general")

    if df is None or df.empty:
        return {
            "summary": f"Analysis completed for: {query}. No data found matching the criteria.",
            "key_insights": ["No matching data found"],
            "data_highlights": {},
            "chart_type": "bar",
            "chart_spec": {},
            "follow_up_questions": ["Try a different query?", "Would you like to see all products?"],
            "confidence_score": 0.5
        }

    top_item = str(df.iloc[0].get("x", df.columns[0] and df.iloc[0, 0]))
    top_value = int(df.iloc[0].get("y", df.iloc[0, 1])) if len(df.columns) > 1 else "N/A"
    total_items = len(df)

    if query_type == "top_products":
        summary = (
            f"Analysis of prior orders reveals {total_items} top products. "
            f"'{top_item}' leads with {top_value:,} order occurrences. "
            f"The top 3 products account for a significant share of all orders in the dataset. "
            f"This pattern suggests strong customer preference concentration around a few popular items."
        )
        insights = [
            f"Top product: '{top_item}' with {top_value:,} occurrences",
            f"Total of {total_items} distinct products in the top results",
            f"2nd place: '{df.iloc[1].get('x', 'N/A')}' with {int(df.iloc[1].get('y', 0)):,} occurrences" if len(df) > 1 else "Only one product found"
        ]
        trend = "Stable"
    elif query_type == "bottom_products":
        summary = (
            f"The least-ordered products analysis shows {total_items} items with very low order frequency. "
            f"'{top_item}' is among the least popular with only {top_value:,} occurrences. "
            f"These products may benefit from promotional efforts or reconsideration of inventory levels."
        )
        insights = [
            f"Least ordered item found: '{top_item}' with {top_value:,} occurrences",
            f"{total_items} products identified with minimal order frequency",
            "Low-frequency items may indicate niche products or poor discoverability"
        ]
        trend = "Decreasing"
    elif query_type == "day_analysis":
        summary = (
            f"Order distribution analysis across days of the week shows {total_items} data points. "
            f"Day {top_item} has the highest order volume with {top_value:,} orders. "
            f"This temporal pattern can help optimize staffing and inventory management."
        )
        insights = [
            f"Peak order day: Day {top_item} with {top_value:,} orders",
            f"Weekly order data analyzed across {total_items} days",
            "Order volume varies significantly across days of the week"
        ]
        trend = "Stable"
    else:
        summary = (
            f"Based on your query '{query}', {total_items} results were found. "
            f"The top result is '{top_item}' with a value of {top_value:,}. "
            f"The data shows clear patterns in the distribution of the analyzed metric."
        )
        insights = [
            f"Top result: '{top_item}' with value {top_value:,}",
            f"Total of {total_items} items in the result set",
            "Local analysis completed with exact data counts"
        ]
        trend = "Stable"

    return {
        "summary": summary,
        "key_insights": insights,
        "data_highlights": {
            "top_item": top_item[:30],
            "top_value": str(top_value),
            "trend": trend,
            "notable_pattern": f"{total_items} items analyzed"
        },
        "chart_type": "bar",
        "chart_spec": {},
        "follow_up_questions": [
            "Would you like to see more details about these results?",
            "Should I analyze a different aspect of the data?"
        ],
        "confidence_score": 0.85
    }


def display_enhanced_summary(summary_data: dict, query: str):
    """Display a rich, formatted summary in Streamlit."""
    st.markdown("---")
    st.markdown("## Analysis Summary")

    summary_text = summary_data.get("summary", "No summary available.")
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 20px;
    ">
        <strong>Summary:</strong><br>{summary_text}
    </div>
    """, unsafe_allow_html=True)

    insights = summary_data.get("key_insights", [])
    if insights:
        st.markdown("### Key Insights")
        for i, insight in enumerate(insights, 1):
            st.markdown(f"""
            <div style="
                background: #f0f2f6;
                padding: 12px 15px;
                border-left: 4px solid #667eea;
                margin: 8px 0;
                border-radius: 0 8px 8px 0;
                color: #333;
            ">
                <strong>{i}.</strong> {insight}
            </div>
            """, unsafe_allow_html=True)

    highlights = summary_data.get("data_highlights", {})
    if highlights:
        st.markdown("### Data Highlights")
        col1, col2, col3 = st.columns(3)
        with col1:
            top_item = str(highlights.get("top_item", "N/A"))
            st.metric(label="Top Item", value=top_item[:20] if len(top_item) > 20 else top_item)
        with col2:
            st.metric(label="Top Value", value=highlights.get("top_value", "N/A"), delta=highlights.get("trend", ""))
        with col3:
            confidence = summary_data.get("confidence_score", 0)
            st.metric(label="Confidence", value=f"{confidence * 100:.0f}%")

    followups = summary_data.get("follow_up_questions", [])
    if followups:
        st.markdown("### Suggested Follow-up Questions")
        for q_item in followups:
            st.markdown(f"- {q_item}")

    confidence = summary_data.get("confidence_score", 0.5)
    st.progress(float(confidence), text=f"Analysis Confidence: {confidence * 100:.0f}%")

def _extract_json_from_text(text: str):
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the largest {...} block
    m = re.search(r'\{[\s\S]*\}', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    try:
        alt = text.strip().replace("'", '"')
        return json.loads(alt)
    except Exception:
        return None


# ---------- Minimal robust HF chat wrapper (use the working pattern) ----------
def hf_chat_wrapper(hf_token: str, model_id: str, user_question: str, short_context: Any = None, max_tokens: int = 400):
    SAFE = {"summary":"(LLM unavailable)","key_insights":[],"data_highlights":{},"chart_type":"none","chart_spec":{},"follow_up_questions":[],"confidence_score":0.0}
    if not hf_token:
        return {**SAFE, "summary":"HF token missing — LLM disabled."}
    try:
        ctx = ""
        if short_context:
            try:
                ctx = "\n\nShort context:\n" + json.dumps(short_context, ensure_ascii=False, indent=2)
            except Exception:
                ctx = "\n\nShort context: (unserializable)"

        system = (
            "You are an expert data analyst assistant. Use ONLY the numeric and textual values provided in the short context. "
            "Do NOT make up numbers. Return ONLY a JSON object (no explanation) with these exact keys: "
            "summary (string, 3-5 sentences with specific numbers), "
            "key_insights (list of 3 strings, each with a specific data point), "
            "data_highlights (object with top_item, top_value, trend, notable_pattern), "
            "chart_type (bar|line|pie|treemap|none), "
            "chart_spec (object with table,x,y,agg,optional join,optional top_k), "
            "follow_up_questions (list of 2 strings), "
            "confidence_score (0.0-1.0)."
        )

        messages = [
            {"role":"system", "content": system},
            {"role":"user", "content": f"{user_question}{ctx}\n\nReturn JSON only."}
        ]

        client = InferenceClient(model=model_id, token=hf_token)
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.0)

        try:
            model_text = resp.choices[0].message.content
        except Exception:
            model_text = str(resp)

        # try parse JSON
        parsed = _extract_json_from_text(model_text)

        if not parsed:
            # second chance: ask model to return strict JSON (assistant continuation)
            followup_req = "Return ONLY a JSON object now with keys summary, key_insights, data_highlights, chart_type, chart_spec, follow_up_questions, confidence_score."
            messages2 = messages + [{"role":"assistant","content": model_text}, {"role":"user","content": followup_req}]
            try:
                resp2 = client.chat_completion(messages=messages2, max_tokens=250, temperature=0.0)
                try:
                    model_text2 = resp2.choices[0].message.content
                except Exception:
                    model_text2 = str(resp2)
                parsed = _extract_json_from_text(model_text2)
                if parsed:
                    model_text = model_text2
            except Exception:
                parsed = None

        content = model_text if isinstance(model_text, str) else str(model_text)
        if not parsed:
            return {"summary": content.strip()[:1000], "key_insights":[], "data_highlights":{}, "chart_type":"none","chart_spec":{},"follow_up_questions":[],"confidence_score":0.3}

        ans = parsed.get("summary") or parsed.get("answer_text") or parsed.get("answer") or parsed.get("text") or ""
        chart_type = parsed.get("chart_type","none")
        chart_spec = parsed.get("chart_spec", {}) or {}
        follow_up_questions = parsed.get("follow_up_questions") or parsed.get("followups", []) or []
        key_insights = parsed.get("key_insights", []) or []
        data_highlights = parsed.get("data_highlights", {}) or {}
        confidence_score = float(parsed.get("confidence_score") or parsed.get("confidence", 0.9 if ans else 0.5))
        return {
            "summary": str(ans).strip(),
            "key_insights": key_insights if isinstance(key_insights, list) else [],
            "data_highlights": data_highlights if isinstance(data_highlights, dict) else {},
            "chart_type": chart_type if chart_type in {"bar","line","pie","treemap","none"} else "none",
            "chart_spec": chart_spec if isinstance(chart_spec, dict) else {},
            "follow_up_questions": follow_up_questions if isinstance(follow_up_questions, list) else [],
            "confidence_score": max(0.0, min(1.0, float(confidence_score)))
        }

    except Exception as e:
        return {"summary": f"LLM call failed: {repr(e)}", "key_insights":[], "data_highlights":{}, "chart_type":"none","chart_spec":{},"follow_up_questions":[],"confidence_score":0.0}

# ---------------- Streamlit UI ----------------
st.title("RAG-DB — Instacart experiment (local + optional LLM)")

st.sidebar.header("Options / Settings")
sample_nrows_prior = st.sidebar.number_input("Rows to load from order_products__prior (0 = full)", min_value=0, value=50000, step=10000)
mode = st.sidebar.selectbox("Mode", ["local-only","with-llm"])
hf_token_input = st.sidebar.text_input("Hugging Face API key (optional)", value=os.getenv("HF_API_KEY",""), type="password")
hf_model_input = st.sidebar.text_input("HF model id (chat-capable)", value="meta-llama/Meta-Llama-3-8B-Instruct")
st.sidebar.markdown("---")
st.sidebar.write("Notes: local-only runs exact queries. with-llm asks HF to suggest friendly text & chart_spec; chart computed locally.")

q = st.text_area("Ask a question", value="Which products appear most frequently in prior orders?", height=150)
run = st.button("Run")

@st.cache_data(ttl=3600)
def load_tables(nrows_prior: Optional[int] = None):
    products = load_csv_if_exists("products")
    aisles = load_csv_if_exists("aisles")
    departments = load_csv_if_exists("departments")
    orders = load_csv_if_exists("orders")
    prior = load_csv_if_exists("order_products__prior", nrows=nrows_prior)
    return {"products":products,"aisles":aisles,"departments":departments,"orders":orders,"prior":prior}

if run:
    st.info("Loading CSVs...")
    nrows = None if sample_nrows_prior == 0 else int(sample_nrows_prior)
    tables = load_tables(nrows_prior=nrows)
    products = tables["products"]; aisles = tables["aisles"]; departments = tables["departments"]; orders = tables["orders"]; prior = tables["prior"]
    if prior is None or products is None:
        st.error("Missing CSVs. Put products.csv and order_products__prior.csv in data/data/instacart/")
        st.stop()

    ql_raw = str(q).strip()
    intent = detect_intent(ql_raw)
    st.markdown(f"### Detected intent: **{intent}**")

    # small short context
    short_context = {}
    try:
        short_context["rows_in_prior"] = int(len(prior))
        sc_top5 = top_products_prior(prior, products, top_k=5)
        short_context["top5"] = sc_top5.to_dict(orient="records")
    except Exception as e:
        short_context["error"] = str(e)

    use_llm = (mode == "with-llm") and hf_token_input.strip() != "" and HF_HUB_AVAILABLE
    tables_map = {"prior":prior,"products":products,"orders":orders,"aisles":aisles,"departments":departments}

    # -------- Aggregation --------
    if intent == "aggregation":
        ql = ql_raw.lower()
        if any(x in ql for x in ["most frequently","most frequent","top products","most ordered","appear most frequently"]):
            df = top_products_prior(prior, products, top_k=15)
            st.subheader("Top products (exact counts)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}).head(15))
            render_chart(df, chart_type="bar", title="Top products (count)")
            with st.expander("More visualizations"):
                render_chart(df, chart_type="treemap", title="Top products (treemap)")
                render_chart(df, chart_type="pie", title="Top products (pie)")
            # Generate summary
            if use_llm:
                with st.spinner("Generating AI summary..."):
                    summary_ctx = {**short_context, "query_type": "top_products", "top_results": df.head(10).to_dict(orient="records")}
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, summary_ctx)
                    if jobj.get("summary") and jobj["summary"] not in ["(LLM unavailable)"]:
                        display_enhanced_summary(jobj, ql_raw)
                    else:
                        display_enhanced_summary(generate_fallback_summary(ql_raw, df, {"query_type":"top_products"}), ql_raw)
            else:
                display_enhanced_summary(generate_fallback_summary(ql_raw, df, {"query_type":"top_products"}), ql_raw)
        elif any(x in ql for x in ["least frequently","least ordered"]):
            df = least_products_prior(prior, products, top_k=15)
            st.subheader("Least frequently ordered (sample)")
            st.dataframe(df.rename(columns={"x":"product_name","y":"count"}))
            render_chart(df, chart_type="bar", title="Least frequently ordered")
            # Generate summary
            if use_llm:
                with st.spinner("Generating AI summary..."):
                    summary_ctx = {**short_context, "query_type": "bottom_products", "results": df.head(10).to_dict(orient="records")}
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, summary_ctx)
                    if jobj.get("summary") and jobj["summary"] not in ["(LLM unavailable)"]:
                        display_enhanced_summary(jobj, ql_raw)
                    else:
                        display_enhanced_summary(generate_fallback_summary(ql_raw, df, {"query_type":"bottom_products"}), ql_raw)
            else:
                display_enhanced_summary(generate_fallback_summary(ql_raw, df, {"query_type":"bottom_products"}), ql_raw)
        elif "average number of orders per product" in ql or "avg orders per product" in ql:
            avg = avg_orders_per_product(prior)
            st.write(f"**Average occurrences per product:** {avg:.2f}")
        elif "reorder ratio" in ql or "reorder rate" in ql:
            rr = avg_reorder_ratio(prior)
            if rr is not None:
                st.write(f"Average reorder ratio across products: **{rr:.3f}**")
                reorder_df = pd.DataFrame([{"x": "Reorder Ratio", "y": round(rr, 3)}])
                display_enhanced_summary(generate_fallback_summary(ql_raw, reorder_df, {"query_type":"general"}), ql_raw)
            else:
                st.info("No 'reordered' column present in this dataset sample.")
        elif "day" in ql or "day of week" in ql:
            df = orders_by_day_of_week(orders)
            if df is not None:
                st.subheader("Orders by day of week")
                st.dataframe(df.rename(columns={"x":"day","y":"count"}))
                render_chart(df, chart_type="bar", title="Orders by day")
                if use_llm:
                    with st.spinner("Generating AI summary..."):
                        summary_ctx = {**short_context, "query_type": "day_analysis", "day_data": df.to_dict(orient="records")}
                        jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, summary_ctx)
                        if jobj.get("summary") and jobj["summary"] not in ["(LLM unavailable)"]:
                            display_enhanced_summary(jobj, ql_raw)
                        else:
                            display_enhanced_summary(generate_fallback_summary(ql_raw, df, {"query_type":"day_analysis"}), ql_raw)
                else:
                    display_enhanced_summary(generate_fallback_summary(ql_raw, df, {"query_type":"day_analysis"}), ql_raw)
            else:
                st.info("orders.csv missing or 'order_dow' not present.")
        else:
            # fallback: local summary + optional LLM suggestion
            local_df = top_products_prior(prior, products, top_k=10)
            if use_llm:
                with st.spinner("Asking LLM for a friendly summary and suggested chart..."):
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), ql_raw, short_context)
                    cs = jobj.get("chart_spec", {})
                    ct = jobj.get("chart_type", "bar")
                    if cs:
                        chart_df = compute_chart_from_spec(cs, tables_map)
                        if chart_df is not None:
                            render_chart(chart_df, chart_type=ct, title="LLM suggested chart (computed exactly)")
                        else:
                            st.warning("LLM suggested chart_spec cannot be computed locally; preview:")
                            st.json(cs)
                    display_enhanced_summary(jobj, ql_raw)
            else:
                st.info("LLM disabled — showing local fallback.")
                render_chart(local_df, chart_type="bar", title="Local fallback: top products")
                display_enhanced_summary(generate_fallback_summary(ql_raw, local_df, {"query_type":"top_products"}), ql_raw)

    # -------- Retrieval --------
    else:
        st.subheader("Retrieval / Keyword search")
        proc = ql_raw
        prod_hits = fuzzy_search_products(products, proc, top_k=50)
        if prod_hits:
            st.write(f"Products matching query (top {len(prod_hits)}):")
            df = pd.DataFrame(prod_hits)[["product_id","product_name","score"]]
            st.dataframe(df)
            # Build short_context with retrieved examples to pass to LLM
            retrieved_names = [r["product_name"] for r in prod_hits[:20]]
            sc_for_llm = dict(short_context) if isinstance(short_context, dict) else {"rows_in_prior": short_context}
            sc_for_llm["retrieved_examples"] = retrieved_names[:10]

            # LLM summary (defensive)
            if use_llm:
                with st.spinner("Asking LLM to summarize retrieved results..."):
                    jobj = hf_chat_wrapper(hf_token_input.strip(), hf_model_input.strip(), f"Summarize these retrieved examples for the user: {proc}", sc_for_llm, max_tokens=220)

                    llm_text = (jobj.get("summary") or "").strip()
                    # defensive override: if LLM says "no results" but we found items, prefer local
                    lower = llm_text.lower()
                    negative_phrases = ["no items", "no results", "nothing found", "no matches", "none found"]
                    contradicted = any(phrase in lower for phrase in negative_phrases) and len(prod_hits) > 0

                    if contradicted:
                        st.warning("LLM summary contradicted local retrieval — preferring local facts.")
                        top_names = retrieved_names[:8]
                        fallback_summary = {
                            "summary": f"Found {len(prod_hits)} matching product(s). Top examples: {', '.join(top_names)}.",
                            "key_insights": [f"Total matches: {len(prod_hits)}", f"Top match: {top_names[0] if top_names else 'N/A'}"],
                            "data_highlights": {"top_item": top_names[0] if top_names else "N/A", "top_value": str(len(prod_hits)), "trend": "", "notable_pattern": "Local retrieval used"},
                            "chart_type": "none", "chart_spec": {},
                            "follow_up_questions": ["Would you like to filter by aisle?", "Should I show product details?"],
                            "confidence_score": 0.95
                        }
                        display_enhanced_summary(fallback_summary, ql_raw)
                        st.dataframe(pd.DataFrame([{"product_name":n} for n in top_names]))
                    else:
                        display_enhanced_summary(jobj, ql_raw)
        else:
            st.info("No product fuzzy match found. Trying aisles substring...")
            if aisles is not None:
                mask = aisles["aisle"].str.lower().str.contains(proc.lower(), na=False)
                if mask.any():
                    st.write("Matching aisles (examples):")
                    st.dataframe(aisles[mask].head(30))
                else:
                    st.warning("No matches. Try simpler keywords like 'frozen', 'snacks', 'produce'.")
            else:
                st.warning("Aisles data not available.")

    st.success("Done.")