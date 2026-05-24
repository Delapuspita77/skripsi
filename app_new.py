import streamlit as st
import re

st.set_page_config(layout="wide", page_title="Pencarian Skripsi", initial_sidebar_state="collapsed")

# ===============================
# LOAD RETRIEVAL
# ===============================
@st.cache_resource
def load_retrieval():
    from retrieval_new import (
        docs, expand_query_hybrid,
        search_bm25_qe,
        search_base, search_base_qe,
        search_ft, search,
    )
    return docs, expand_query_hybrid, search_bm25_qe, search_base, search_base_qe, search_ft, search

(docs, expand_query_hybrid,
 search_bm25_qe,
 search_base, search_base_qe,
 search_ft, search_final) = load_retrieval()

# ===============================
# PALETTE
# ===============================
C_DARK  = "#2C4A1E"
C_MID   = "#4A7C2F"
C_SAND  = "#D6C9A8"
C_CREAM = "#F7F3EB"
C_WHITE = "#FFFFFF"
C_QY    = "#FFE566"
C_QE    = "#B8F0A0"

PER_PAGE = 10

TAB_META = [
    {"key": "bm25_qe",  "label": "BM25 + QE",        "qe": True,  "color": "#8B6914"},
    {"key": "base",     "label": "Baseline",           "qe": False, "color": "#2B5BA8"},
    {"key": "base_qe",  "label": "Baseline + QE",      "qe": True,  "color": "#5B3DA8"},
    {"key": "ft",       "label": "Finetuned",          "qe": False, "color": "#A83D2B"},
    {"key": "ft_qe",    "label": "★ Finetuned + QE",   "qe": True,  "color": C_DARK},
]

# ===============================
# HIGHLIGHT
# ===============================
def highlight_text(text, q_tokens, e_tokens):
    if not text:
        return text
    hmap = {}
    for w in e_tokens:
        hmap[w.lower()] = "e"
    for w in q_tokens:
        hmap[w.lower()] = "q"
    if not hmap:
        return text
    pattern = r'\b(' + '|'.join(re.escape(w) for w in sorted(hmap, key=len, reverse=True)) + r')\b'
    def rep(m):
        wrd   = m.group(0)
        wtype = hmap.get(wrd.lower())
        bg    = C_QY if wtype == "q" else C_QE
        fw    = "600" if wtype == "q" else "400"
        return (f"<mark style='background:{bg};color:#222;border-radius:2px;"
                f"padding:0 2px;font-weight:{fw}'>{wrd}</mark>")
    return re.sub(pattern, rep, text, flags=re.IGNORECASE)

# ===============================
# CSS
# ===============================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
    background: {C_CREAM} !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: #1a1a1a;
}}
[data-testid="stHeader"]    {{ background: transparent !important; height: auto !important; }}
[data-testid="stToolbar"]   {{ display: none !important; }}
[data-testid="stDecoration"]{{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none; }}

.block-container {{
    padding: 1rem 1.25rem 1.5rem !important;
    max-width: 100% !important;
}}

/* ---- MASTHEAD ---- */
.masthead {{
    border-bottom: 3px solid {C_DARK};
    padding: 8px 0 7px;
    margin-bottom: 12px;
    display: flex;
    align-items: baseline;
    gap: 12px;
}}
.masthead-title {{
    font-family: 'Source Serif 4', serif;
    font-size: 21px;
    font-weight: 600;
    color: {C_DARK};
    letter-spacing: -0.3px;
    line-height: 1;
}}
.masthead-sub {{
    font-size: 11px;
    color: #888;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}}

/* ---- ROW ALIGNMENT ---- */
[data-testid="stHorizontalBlock"] {{
    align-items: flex-end !important;
    gap: 8px !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-end !important;
}}

/* ---- TEXT INPUT ---- */
div[data-testid="stTextInput"] label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #666 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 3px !important;
    height: 17px !important;
    display: block !important;
}}
div[data-testid="stTextInput"] > div > div > input {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px !important;
    border: 1.5px solid {C_SAND} !important;
    border-radius: 3px !important;
    background: {C_WHITE} !important;
    padding: 0 10px !important;
    height: 36px !important;
    color: #111 !important;
    box-shadow: none !important;
}}
div[data-testid="stTextInput"] > div > div > input:focus {{
    border-color: {C_MID} !important;
    box-shadow: 0 0 0 2px {C_MID}28 !important;
}}

/* ---- SELECTBOX ---- */
div[data-testid="stSelectbox"] label {{
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #666 !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 3px !important;
    height: 17px !important;
    display: block !important;
}}
div[data-testid="stSelectbox"] > div > div {{
    font-size: 12px !important;
    border: 1.5px solid {C_SAND} !important;
    border-radius: 3px !important;
    background: {C_WHITE} !important;
    height: 36px !important;
    padding: 0 8px !important;
    display: flex !important;
    align-items: center !important;
}}

/* ---- BUTTON ---- */
div[data-testid="stButton"]::before {{
    content: '';
    display: block;
    height: 20px;
}}
div[data-testid="stButton"] > button {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    background: {C_DARK} !important;
    color: {C_WHITE} !important;
    border: none !important;
    border-radius: 3px !important;
    height: 36px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background .15s !important;
}}
div[data-testid="stButton"] > button:hover {{
    background: {C_MID} !important;
}}

/* ---- QE STRIP ---- */
.qe-strip {{
    background: {C_WHITE};
    border: 1px solid {C_SAND};
    border-left: 3px solid {C_MID};
    border-radius: 3px;
    padding: 5px 10px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    font-size: 12px;
}}
.qe-label {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    color: {C_MID};
    letter-spacing: 0.5px;
    margin-right: 6px;
    white-space: nowrap;
}}
.qe-token {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    background: {C_QY};
    color: #333;
    border-radius: 2px;
    padding: 1px 6px;
}}
.qe-exp {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    background: {C_QE};
    color: #333;
    border-radius: 2px;
    padding: 1px 6px;
}}
.qe-sep {{
    color: #ccc;
    font-size: 11px;
    margin: 0 2px;
}}

/* ---- TABS ---- */
div[data-testid="stTabs"] > div:first-child {{
    border-bottom: 2px solid {C_SAND} !important;
    gap: 0 !important;
    margin-bottom: 0 !important;
}}
button[data-baseweb="tab"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    border-radius: 0 !important;
    color: #666 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    background: transparent !important;
}}
button[data-baseweb="tab"]:hover {{
    color: {C_DARK} !important;
    background: #f0ebe1 !important;
}}
button[aria-selected="true"][data-baseweb="tab"] {{
    color: {C_DARK} !important;
    font-weight: 700 !important;
    border-bottom: 2px solid {C_DARK} !important;
    background: transparent !important;
}}

/* ---- RESULT HEADER ---- */
.result-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0 5px;
    border-bottom: 1px solid {C_SAND};
    margin-bottom: 2px;
}}
.result-count {{
    font-size: 12px;
    color: #555;
}}
.result-count b {{ color: #111; }}
.model-pill {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-radius: 2px;
    padding: 2px 8px;
    color: white;
}}

/* ---- RESULT ITEMS ---- */
.result-item {{
    border-bottom: 1px solid #EAE4D8;
    padding: 6px 0 5px;
    display: flex;
    gap: 6px;
}}
.result-item:last-child {{ border-bottom: none; }}

.result-rank {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #bbb;
    min-width: 20px;
    padding-top: 3px;
    text-align: right;
    flex-shrink: 0;
}}
.result-body {{ flex: 1; min-width: 0; }}

.result-title {{
    font-family: 'Source Serif 4', serif;
    font-size: 14px;
    font-weight: 600;
    color: {C_DARK};
    line-height: 1.35;
    text-decoration: none;
    display: block;
}}
.result-title:hover {{ text-decoration: underline; color: {C_MID}; }}

.result-meta {{
    font-size: 11px;
    color: #999;
    margin: 2px 0 3px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}}
.result-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    background: {C_DARK};
    color: white;
    border-radius: 2px;
    padding: 1px 5px;
    flex-shrink: 0;
}}
.result-abstract {{
    font-size: 12px;
    color: #555;
    line-height: 1.5;
}}

/* ---- PAGINATION buttons ---- */
div[data-testid="stButton"].pg-wrap > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    background: {C_WHITE} !important;
    color: {C_DARK} !important;
    border: 1px solid {C_SAND} !important;
    border-radius: 2px !important;
    height: 26px !important;
    padding: 0 8px !important;
    width: 100% !important;
}}
div[data-testid="stButton"].pg-wrap > button:hover {{
    background: {C_SAND} !important;
}}

/* ---- CHECKBOX ---- */
div[data-testid="stCheckbox"] {{
    margin: 0 !important;
    padding: 0 !important;
}}
div[data-testid="stCheckbox"] label {{
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    gap: 4px !important;
    color: #333 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    height: auto !important;
    visibility: visible !important;
    font-weight: 400 !important;
}}
div[data-testid="stCheckbox"] label p {{
    font-size: 12px !important;
    margin: 0 !important;
}}

/* ---- MISC ---- */
div[data-testid="stSpinner"] p {{ font-size: 12px !important; color: {C_DARK} !important; }}
div[data-testid="stAlert"]   {{ font-size: 12px !important; padding: 5px 10px !important; border-radius: 3px !important; }}
hr {{ border-color: {C_SAND}; margin: 5px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ===============================
# MASTHEAD
# ===============================
st.markdown("""
<div class="masthead">
    <span class="masthead-title">Repositori Skripsi</span>
    <span class="masthead-sub">Sistem Pencarian Semantik — Perbandingan Model</span>
</div>
""", unsafe_allow_html=True)

# ===============================
# FILTER MAPS
# ===============================
FAKULTAS_MAP = {
    "Semua Fakultas": None,
    "Teknik": "Faculty of Engineering",
    "Ekonomi & Bisnis": "Faculty of Economic and Business",
    "Ilmu Komputer": "Faculty of Computer Science",
    "Hukum": "Faculty of Law",
    "Pertanian": "Faculty of Agriculture",
    "Ilmu Sosial & Politik": "Faculty of Social and Political Sciences",
    "Arsitektur & Desain": "Faculty of Architecture and Design",
}
PRODI_MAP = {
    "Semua Prodi": None,
    "Teknik Industri": "Departement of Industrial Engineering",
    "Teknik Sipil": "Departement of Civil Engineering",
    "Teknik Kimia": "Departement of Chemical Engineering",
    "Teknik Pangan": "Departement of Food Engineering",
    "Teknik Mesin": "Departement of Mechanical Engineering",
    "Teknik Lingkungan": "Departement of Environmental Engineering",
    "Akuntansi": "Departement of Accounting",
    "Ekonomi": "Departement of Economics",
    "Manajemen": "Departement of Management",
    "Administrasi Bisnis": "Departement of Business Administration",
    "Informatika": "Departemen of Informatics",
    "Sistem Informasi": "Departemen of Information Systems",
    "Data Science": "Departemen of Data Science",
    "Hukum": "Departement of Law",
    "Agribisnis": "Departement of Agribusiness",
    "Agroteknologi": "Departement of Agritechnology",
    "Administrasi Publik": "Departement of Public Administration",
    "Hubungan Internasional": "Departement of International Relations",
    "Ilmu Komunikasi": "Departement of Communication",
    "Pariwisata": "Department of Tourism",
    "DKV": "Departement of Visual Communication Design",
    "Arsitektur": "Departement of Architecture",
}

# ===============================
# SESSION STATE
# ===============================
_page_defaults = {m["key"]: 1 for m in TAB_META}
for k, v in {
    "triggered":  False,
    "last_query": "",
    "sel_exp":    [],
    "last_exp":   [],
    "cached":     {},
    "pages":      _page_defaults.copy(),
    "last_sel_exp_tuple": (),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# SEARCH BAR
# ===============================
c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 0.7])
with c1:
    query = st.text_input("Kata Kunci", placeholder="Contoh: sistem rekomendasi berbasis deep learning", key="q_input")
with c2:
    fak_label = st.selectbox("Fakultas", list(FAKULTAS_MAP.keys()), key="fak")
with c3:
    prodi_label = st.selectbox("Program Studi", list(PRODI_MAP.keys()), key="prodi")
with c4:
    search_clicked = st.button("Cari", use_container_width=True)

faculty    = FAKULTAS_MAP[fak_label]
department = PRODI_MAP[prodi_label]

# ===============================
# TRIGGER & RESET LOGIC
# ===============================
if search_clicked and query.strip():
    st.session_state.triggered = True
    st.session_state.pages     = {m["key"]: 1 for m in TAB_META}

# Reset semua jika query berubah
if query != st.session_state.last_query:
    st.session_state.sel_exp             = []
    st.session_state.last_exp            = []
    st.session_state.last_query          = query
    st.session_state.cached              = {}
    st.session_state.pages               = {m["key"]: 1 for m in TAB_META}
    st.session_state.last_sel_exp_tuple  = ()

# ===============================
# RENDER HELPERS
# ===============================
def render_qe_strip(tokens, expansions):
    parts = ""
    for t in tokens:
        parts += f"<span class='qe-token'>{t}</span>"
    if expansions:
        parts += "<span class='qe-sep'>＋</span>"
        for e in expansions:
            parts += f"<span class='qe-exp'>{e}</span>"
    st.markdown(
        f"<div class='qe-strip'><span class='qe-label'>Query</span>{parts}</div>",
        unsafe_allow_html=True
    )


def render_results_page(results, tokens, expansions, model_key):
    total  = len(results)
    cur_pg = st.session_state.pages.get(model_key, 1)
    # Pastikan cur_pg tidak melebihi n_pages jika jumlah hasil berubah
    n_pages = max(1, -(-total // PER_PAGE))
    if cur_pg > n_pages:
        cur_pg = 1
        st.session_state.pages[model_key] = 1

    if total == 0:
        st.markdown(
            "<div style='padding:24px 0;color:#999;font-size:12px;text-align:center;'>"
            "Tidak ada hasil ditemukan.</div>",
            unsafe_allow_html=True
        )
        return

    meta  = next(m for m in TAB_META if m["key"] == model_key)
    start = (cur_pg - 1) * PER_PAGE
    end   = min(start + PER_PAGE, total)
    page_r = results[start:end]

    # Result header
    st.markdown(
        f"<div class='result-header'>"
        f"<span class='result-count'><b>{total}</b> hasil"
        f" &nbsp;·&nbsp; halaman {cur_pg} dari {n_pages}</span>"
        f"<span class='model-pill' style='background:{meta['color']};'>{meta['label']}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Build result items HTML
    items_html = "<div>"
    for r in page_r:
        row  = docs.iloc[r["doc_id"]]
        abst = str(row.get("abstract") or row.get("abstrak") or "").strip()
        prev = abst[:300] + ("…" if len(abst) > 300 else "")

        h_title = highlight_text(r["title"], tokens, expansions)
        h_abst  = highlight_text(prev, tokens, expansions)

        fac_str  = str(row.get("faculty", "") or "")
        dept_str = str(row.get("department", "") or "")
        dept_html = (f"<span style='color:#ddd'>·</span><span>{dept_str}</span>"
                     if dept_str else "")

        abst_html = (f"<div class='result-abstract'>{h_abst}</div>" if abst else "")

        items_html += f"""
<div class="result-item">
  <span class="result-rank">{r['rank']}.</span>
  <span class="result-body">
    <a class="result-title" href="{r['url']}" target="_blank">{h_title}</a>
    <div class="result-meta">
      <span class="result-score">{r['score']:.4f}</span>
      <span style="color:#ddd">|</span>
      <span>{fac_str}</span>
      {dept_html}
    </div>
    {abst_html}
  </span>
</div>"""
    items_html += "</div>"
    st.markdown(items_html, unsafe_allow_html=True)

    # ---- PAGINATION ----
    if n_pages > 1:
        # Tampilkan max 9 tombol halaman + prev/next = 11 kolom
        MAX_PG_BTNS = 9
        half = MAX_PG_BTNS // 2

        if n_pages <= MAX_PG_BTNS:
            page_range = list(range(1, n_pages + 1))
        else:
            start_pg = max(1, min(cur_pg - half, n_pages - MAX_PG_BTNS + 1))
            end_pg   = min(n_pages, start_pg + MAX_PG_BTNS - 1)
            page_range = list(range(start_pg, end_pg + 1))

        n_cols  = len(page_range) + 2  # +2 untuk prev & next
        pg_cols = st.columns([0.5] + [0.4] * len(page_range) + [0.5])

        # Prev
        with pg_cols[0]:
            if cur_pg > 1:
                if st.button("←", key=f"prev_{model_key}"):
                    st.session_state.pages[model_key] = cur_pg - 1
                    st.rerun()

        # Page numbers
        for ci, pg in enumerate(page_range, start=1):
            with pg_cols[ci]:
                label = f"**{pg}**" if pg == cur_pg else str(pg)
                if st.button(label, key=f"pg_{model_key}_{pg}"):
                    st.session_state.pages[model_key] = pg
                    st.rerun()

        # Next
        with pg_cols[-1]:
            if cur_pg < n_pages:
                if st.button("→", key=f"next_{model_key}"):
                    st.session_state.pages[model_key] = cur_pg + 1
                    st.rerun()


# ===============================
# MAIN CONTENT
# ===============================
if st.session_state.triggered and query.strip():

    # QE
    qe_raw     = expand_query_hybrid(query)
    tokens_raw = qe_raw["token_query"]
    exp_raw    = qe_raw["hasil_expansi"]

    # Inisialisasi pilihan ekspansi (sekali saja per query)
    if exp_raw and not st.session_state.last_exp:
        st.session_state.sel_exp  = exp_raw.copy()
        st.session_state.last_exp = exp_raw.copy()

    # ---- QE Checkbox row ----
    sel_exp = []
    if exp_raw:
        # Label + satu checkbox per ekspansi + sisa ruang
        qe_cols = st.columns([0.7] + [0.8] * len(exp_raw) + [5])
        with qe_cols[0]:
            st.markdown(
                f"<div style='font-size:10px;font-weight:700;color:{C_MID};"
                f"text-transform:uppercase;letter-spacing:.5px;padding-top:5px;'>"
                f"Ekspansi</div>",
                unsafe_allow_html=True
            )
        for i, exp in enumerate(exp_raw):
            with qe_cols[i + 1]:
                if st.checkbox(exp, value=(exp in st.session_state.sel_exp), key=f"qe_{exp}"):
                    sel_exp.append(exp)
        st.session_state.sel_exp = sel_exp
    else:
        sel_exp = []

    # Reset halaman ke 1 jika pilihan ekspansi berubah
    sel_exp_tuple = tuple(sorted(sel_exp))
    if sel_exp_tuple != st.session_state.last_sel_exp_tuple:
        st.session_state.pages               = {m["key"]: 1 for m in TAB_META}
        st.session_state.last_sel_exp_tuple  = sel_exp_tuple

    # ---- Jalankan semua model (cache by key) ----
    cache_key = (query, sel_exp_tuple, faculty or "", department or "")
    if st.session_state.cached.get("_key") != cache_key:
        with st.spinner("Menjalankan semua model..."):
            r_bm25_qe, t2, e2 = search_bm25_qe(
                query, selected_expansions=sel_exp, fakultas=faculty, prodi=department)
            r_base,    t3, e3 = search_base(
                query, fakultas=faculty, prodi=department)
            r_base_qe, t4, e4 = search_base_qe(
                query, selected_expansions=sel_exp, fakultas=faculty, prodi=department)
            r_ft,      t5, e5 = search_ft(
                query, fakultas=faculty, prodi=department)
            r_ft_qe,   t6, e6 = search_final(
                query, selected_expansions=sel_exp, fakultas=faculty, prodi=department)

        st.session_state.cached = {
            "_key":    cache_key,
            "bm25_qe": (r_bm25_qe, t2, e2),
            "base":    (r_base,    t3, e3),
            "base_qe": (r_base_qe, t4, e4),
            "ft":      (r_ft,      t5, e5),
            "ft_qe":   (r_ft_qe,   t6, e6),
        }

    CR = st.session_state.cached

    # ---- TABS ----
    tab_widgets = st.tabs([m["label"] for m in TAB_META])

    for tab_w, meta in zip(tab_widgets, TAB_META):
        key = meta["key"]
        results, tokens, expansions = CR[key]

        with tab_w:
            if meta["qe"] and (tokens or expansions):
                render_qe_strip(tokens, expansions)
            render_results_page(results, tokens, expansions, key)

# ---- EMPTY STATE ----
elif not st.session_state.triggered:
    st.markdown(
        f"<div style='text-align:center;padding:52px 0;color:#bbb;'>"
        f"<div style='font-size:28px;margin-bottom:8px;letter-spacing:4px;'>◎</div>"
        f"<div style='font-size:12px;'>Masukkan kata kunci dan klik "
        f"<strong style='color:#666'>Cari</strong> untuk mulai</div>"
        f"</div>",
        unsafe_allow_html=True
    )
else:
    st.warning("Masukkan kata kunci terlebih dahulu.")