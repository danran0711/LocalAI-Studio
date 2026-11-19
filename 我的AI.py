# -*- coding: utf-8 -*-
"""
本地AI大模型（Flask 后端）v2
========================================
运行：python 我的AI.py     浏览器打开 http://127.0.0.1:5000
环境变量：AI_HOST AI_PORT AI_EMB AI_LLM AI_NO_WEB

注意：faiss 的 C++ 写盘函数在 Windows 上打不开含中文的绝对路径，
      索引统一用「切到索引目录 + 相对路径」的方式保存/读取，见 _save_vectorstore。
"""
import os
import re
import sys
import json
import glob
import time
import shutil
import datetime
import importlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "faiss_index")
STATE_FILE = os.path.join(SCRIPT_DIR, "kb_state.json")

HOST = os.environ.get("AI_HOST", "127.0.0.1")
PORT = int(os.environ.get("AI_PORT", "5000"))
INDEX_FILE = os.environ.get("AI_INDEX", "index.html")
EMB_MODEL = os.environ.get("AI_EMB", "bge-m3")
NO_WEB = os.environ.get("AI_NO_WEB", "") not in ("", "0", "false", "False")

DOC_EXTS = [".pdf", ".docx", ".txt", ".md", ".csv"]
LLM_CANDIDATES = ["qwen2.5:7b", "qwen2.5:3b", "qwen2.5", "llama3.1:8b", "gemma2:9b"]

SIM_THRESHOLD = 0.50   # 归一化向量下的余弦相似度阈值（低于它认为是无关内容）
TOP_K = 4
WEB_MAX_CHARS = 1800
LOCAL_MAX_CHARS = 2600
WEB_RESULTS = 4

# 联网检索的状态，供提示词区分"确实没有信息"和"搜索坏了"
WEB_OK, WEB_EMPTY, WEB_ERROR, WEB_OFF, WEB_SKIP = "ok", "empty", "error", "off", "skipped"
STATUS_TEXT = {
    WEB_OK: "搜索成功，有可用内容",
    WEB_EMPTY: "搜索完成，但没有找到相关结果",
    WEB_ERROR: "搜索失败，本次拿不到任何网络内容",
    WEB_OFF: "联网搜索已关闭",
    WEB_SKIP: "本次无需联网",
}

# 兜底话术（必须原样输出，不加工）
FALLBACK_REALTIME = ("抱歉，当前网络搜索服务似乎遇到了一点小问题，暂时无法获取最新的实时信息。"
                     "请稍后再试。")
FALLBACK_REALTIME_OFF = ("当前没有开启联网搜索，我拿不到最新的实时信息。"
                         "如果要查天气、新闻这类时效内容，开一下联网我再帮你查。")

# 明确属于"实时信息"的问题：搜索不可用时直接兜底，不交给模型自由发挥
REALTIME_RE = re.compile(
    r"天气|气温|多少度|降雨|下雨|台风|空气质量|雾霾|"
    r"新闻|头条|热搜|时事|最新消息|最新进展|"
    r"股价|股票行情|汇率|金价|油价|比特币|"
    r"比分|赛果|赛况|开奖|彩票|"
    r"航班|高铁|路况|限行|疫情|地震|"
    r"实时|刚刚发生"
)

LINE = "=" * 52


def log(msg=""):
    print(msg, flush=True)


def die(msg, hint=None):
    log()
    log("  [错误] " + msg)
    if hint:
        log("         " + hint)
    log()
    sys.exit(1)


# ---------------------------------------------------------------- 依赖
DEPS = [
    ("langchain_community", "langchain-community"),
    ("langchain_ollama", "langchain-ollama"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("langchain_core", "langchain-core"),
    ("faiss", "faiss-cpu"),
    ("pdfplumber", "pdfplumber"),
    ("docx", "python-docx"),
    #.doc不支持
    ("requests", "requests"),
    ("flask", "flask"),
]


def ensure_deps():
    missing = []
    for mod, pkg in DEPS:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    # 联网搜索：优先新的 ddgs 包，其次老的 duckduckgo-search
    if not NO_WEB:
        try:
            importlib.import_module("ddgs")
        except Exception:
            try:
                importlib.import_module("duckduckgo_search")
            except Exception:
                missing.append("ddgs")
    if not missing:
        return
    pkgs = sorted(set(missing))
    log(f"  缺少 {len(pkgs)} 个依赖，正在安装：{' '.join(pkgs)}")
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *pkgs]
    try:
        subprocess.check_call(cmd)
    except Exception:
        log("  普通安装失败，尝试 --user 安装 ...")
        try:
            subprocess.check_call(cmd + ["--user"])
        except Exception as e:
            die("依赖安装失败", f"请手动执行：pip install {' '.join(pkgs)}  ({e})")


# ---------------------------------------------------------------- Ollama
def ollama_models():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20)
    except FileNotFoundError:
        die("没找到 ollama 命令", "请先安装 Ollama：https://ollama.com/download")
    if r.returncode != 0:
        die("Ollama 没有运行", "请打开 Ollama 或在命令行执行 ollama serve")
    models = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        name = line.split()[0]
        if name:
            models.append(name)
    return models


def looks_like_embedding(name):
    low = name.lower()
    return any(k in low for k in ("embed", "bge", "gte-", "e5-", "minilm"))


def resolve_model(models, want):
    if want and want in models:
        return want
    if want:
        base = want.split(":")[0]
        for m in models:
            if m.split(":")[0] == base:
                return m
    return None


def pick_default_llm(models):
    want = os.environ.get("AI_LLM", "")
    if want:
        got = resolve_model(models, want)
        if got:
            return got
        log(f"  [提示] 指定的模型 {want} 不存在，自动挑选")
    for c in LLM_CANDIDATES:
        got = resolve_model(models, c)
        if got:
            return got
    for m in models:
        if not looks_like_embedding(m):
            return m
    return None


# ---------------------------------------------------------------- 文件读取
def clean_text(text):
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out, blank = [], 0
    for ln in lines:
        if ln.strip():
            out.append(ln.strip())
            blank = 0
        else:
            blank += 1
            if blank == 1:
                out.append("")
    return "\n".join(out).strip()


def decode_bytes(raw):
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_docx(path):
    from docx import Document
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return clean_text("\n".join(parts))


# ---------------------------------------------------------------- PDF 解析
# 同一个 PDF，不同解析引擎的结果可能天差地别：pdfplumber 遇到 CID 子集字体、
# 特殊编码时会吐出一堆乱码；PyMuPDF（pdf2docx 底层用的就是它）往往能正确还原。
# 所以先跑 pdfplumber，质量不合格再换 PyMuPDF，最后才是"疑似扫描件"。
def _char_readable(ch):
    o = ord(ch)
    return (
        ch in "\n\t "
        or 0x20 <= o <= 0x7E                  # ASCII 可打印
        or 0x4E00 <= o <= 0x9FFF              # 常用汉字
        or 0x3400 <= o <= 0x4DBF              # 扩展 A
        or 0x3000 <= o <= 0x303F              # 中文标点
        or 0xFF00 <= o <= 0xFFEF              # 全角字符
        or 0x2010 <= o <= 0x2027              # 常见英文标点
        or 0x00B0 <= o <= 0x00FF              # 度/乘号等
    )


def text_quality(text):
    """0~1，越高越干净。乱码的典型特征是私用区字符(U+E000+)和替换符(U+FFFD)。"""
    s = (text or "").strip()
    if not s:
        return 0.0
    total = len(s)
    readable = sum(1 for ch in s if _char_readable(ch))
    broken = sum(1 for ch in s if ch == "\ufffd" or 0xE000 <= ord(ch) <= 0xF8FF)
    return max(0.0, (readable - broken * 2) / total)


def _pdf_text_pdfplumber(path):
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _pdf_text_pymupdf(path):
    import pymupdf
    parts = []
    with pymupdf.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text", sort=True))
    return "\n".join(parts)


def _sibling_docx(pdf_path):
    """PDF 旁边有没有同名的、不比 PDF 旧的 .docx（比如手动转好的版本）。"""
    cand = os.path.splitext(pdf_path)[0] + ".docx"
    if not os.path.exists(cand):
        return None
    try:
        if os.path.getmtime(cand) + 1 < os.path.getmtime(pdf_path):
            return None          # docx 比 PDF 旧，可能不是同一版
    except OSError:
        return None
    return cand


# ---------------------------------------------------------------- OCR
# 扫描件没有文字层，只能靠 OCR。后端优先级：
#   rapidocr（onnxruntime，CPU 就能跑，稳）→ paddleocr（装了就用，但部分显卡上会算错）
# 每个后端的结果都会过一遍质量分，乱码会被自动否决并换下一个。
OCR_ENABLED = os.environ.get("AI_OCR", "1") not in ("0", "false", "False")
OCR_DPI = int(os.environ.get("AI_OCR_DPI", "200"))
OCR_MAX_PAGES = int(os.environ.get("AI_OCR_MAX_PAGES", "30"))
_ocr_engines_cache = {}


def _load_rapidocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _read_rapidocr(engine, image):
    result, _ = engine(image)
    return [item[1] for item in (result or [])]


def _load_paddleocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=False)


def _read_paddleocr(engine, image):
    res = engine.predict(image)[0]
    texts = res.get("rec_texts") if hasattr(res, "get") else None
    return list(texts or [])


def _available_ocr_engines():
    import importlib.util
    engines = []
    if importlib.util.find_spec("rapidocr_onnxruntime"):
        engines.append(("rapidocr", _load_rapidocr, _read_rapidocr))
    if importlib.util.find_spec("paddleocr"):
        engines.append(("paddleocr", _load_paddleocr, _read_paddleocr))
    # AI_OCR_ENGINE=paddle 可以强制优先用 PaddleOCR（显卡驱动修好之后用得上）
    pref = os.environ.get("AI_OCR_ENGINE", "").strip().lower()
    if pref.startswith("paddle"):
        engines.sort(key=lambda e: 0 if e[0] == "paddleocr" else 1)
    elif pref.startswith("rapid"):
        engines.sort(key=lambda e: 0 if e[0] == "rapidocr" else 1)
    return engines


def _render_pdf_pages(path, dpi, max_pages):
    """把 PDF 每页渲染成 RGB 数组，交给 OCR。"""
    import numpy as np
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                img = np.repeat(img, 3, axis=2)
            pages.append(img)
    return pages


def ocr_pdf(path):
    """整份 PDF 走 OCR；返回纯文本，失败返回 None。"""
    if not OCR_ENABLED:
        return None
    name = os.path.basename(path)
    engines = _available_ocr_engines()
    if not engines:
        log("  [警告] 没有可用的 OCR 引擎，装一个即可：pip install rapidocr-onnxruntime")
        return None
    try:
        pages = _render_pdf_pages(path, OCR_DPI, OCR_MAX_PAGES)
    except Exception as e:
        log(f"  [警告] {name} 渲染成图片失败：{e}")
        return None
    if not pages:
        return None

    log(f"  [OCR] {name}：{len(pages)} 页 @{OCR_DPI}dpi，开始识别（每页约 1-10 秒）")
    for engine_name, loader, reader in engines:
        try:
            engine = _ocr_engines_cache.get(engine_name) or loader()
            _ocr_engines_cache[engine_name] = engine
        except Exception as e:
            log(f"  [警告] {engine_name} 初始化失败：{e}")
            continue

        page_texts, failed = [], False
        for idx, image in enumerate(pages, 1):
            try:
                lines = reader(engine, image)
            except Exception as e:
                log(f"  [警告] {engine_name} 第 {idx} 页识别失败：{e}")
                failed = True
                break
            page_texts.append("\n".join(lines))
            if len(pages) > 3:
                log(f"         · {idx}/{len(pages)} 页")
        if failed:
            continue

        joined = "\n".join(page_texts)
        quality = text_quality(joined)
        if joined.strip() and quality >= 0.5:
            log(f"  [OCR] {engine_name} 完成，质量 {quality:.2f}")
            return clean_text(joined)
        log(f"  [警告] {engine_name} 识别结果质量过低（{quality:.2f}），换下一个引擎")

    log(f"  [警告] {name} OCR 没能识别出可用文字")
    return None


def read_pdf(path):
    """PDF → 文本：同名 docx → 双引擎解析 → OCR。"""
    name = os.path.basename(path)

    # 1) 同名 docx 优先（本地转换过 / 手动用在线工具转好的）
    sib = _sibling_docx(path)
    if sib:
        try:
            text = _read_docx(sib)
            if text and text_quality(text) >= 0.7:
                log(f"  [提示] {name} 旁边有同名 docx，直接读 {os.path.basename(sib)}")
                return text
        except Exception as e:
            log(f"  [警告] 读取 {os.path.basename(sib)} 失败：{e}")

    # 2) 依次尝试两个解析引擎，取第一个质量合格的
    tried = []
    best = ("", 0.0, "")
    for engine, fn in (("pdfplumber", _pdf_text_pdfplumber), ("pymupdf", _pdf_text_pymupdf)):
        try:
            text = fn(path)
        except Exception as e:
            log(f"  [警告] {name} 用 {engine} 解析失败：{e}")
            continue
        quality = text_quality(text)
        tried.append(f"{engine}={quality:.2f}/{len(text.strip())}字")
        if quality > best[1]:
            best = (text, quality, engine)
        if quality >= 0.85 and len(text.strip()) >= 20:
            if engine != "pdfplumber":
                log(f"  [提示] {name} pdfplumber 结果偏差较大，已改用 {engine}")
            return clean_text(text)

    # 3) 文字层缺失或质量太差 → OCR
    text, quality, engine = best
    if len(text.strip()) < 20 or quality < 0.85:
        if len(text.strip()) < 20:
            log(f"  [提示] {name} 几乎没有文字层（疑似扫描件），改走 OCR")
        ocr_text = ocr_pdf(path)
        if ocr_text and text_quality(ocr_text) >= max(quality, 0.5):
            return ocr_text

    if len(text.strip()) < 20:
        log(f"  [警告] {name} 提取不到文字，OCR 也没成功")
        return None
    if quality < 0.85:
        log(f"  [警告] {name} 解析质量偏低（{'，'.join(tried)}），结果可能不完整")
    return clean_text(text)


def read_file(path):
    """返回纯文本；读不了返回 None（下次启动会重试）。"""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return read_pdf(path)
        if ext == ".docx":
            return _read_docx(path)
        if ext in (".txt", ".md", ".csv"):
            with open(path, "rb") as f:
                return clean_text(decode_bytes(f.read()))
    except Exception as e:
        log(f"  [警告] 读取 {os.path.basename(path)} 失败：{e}")
        return None
    log(f"  [警告] 不支持的格式：{os.path.basename(path)}")
    return None


# ---------------------------------------------------------------- 知识库
def scan_docs():
    seen, out = set(), []
    for ext in DOC_EXTS:
        for pattern in ("*" + ext, "*" + ext.upper()):
            for p in glob.glob(os.path.join(SCRIPT_DIR, pattern)):
                if not os.path.isfile(p):
                    continue
                key = os.path.normcase(os.path.realpath(p))
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
    return sorted(out)


def file_sig(path):
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def index_files_ok():
    return (os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
            and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.pkl")))


def decide(state, sigs):
    """返回 (动作, 原因)，动作 ∈ {'none','load','rebuild','clear'}"""
    cur = set(sigs)
    known = set(state.get("files", {})) | set(state.get("failed", {}))
    if not cur:
        if known or os.path.isdir(FAISS_INDEX_DIR):
            return "clear", "资料已全部移除"
        return "none", ""
    if state.get("embed") != EMB_MODEL:
        return "rebuild", f"embedding 模型变更（{state.get('embed')} → {EMB_MODEL}）"
    if not index_files_ok():
        return "rebuild", "索引不存在或残缺"
    if cur - known:
        return "rebuild", "发现新资料：" + "、".join(sorted(cur - known))
    if known - cur:
        return "rebuild", "资料被删除：" + "、".join(sorted(known - cur))
    for name in cur:
        old = state.get("files", {}).get(name) or state.get("failed", {}).get(name)
        if old != sigs[name]:
            return "rebuild", f"{name} 已修改"
    if state.get("files") == {}:
        return "rebuild", "上次没有成功入库的资料"
    return "load", ""


def _save_vectorstore(vs):
    """存盘。faiss 的 C++ 写盘用的是窄字符 fopen，Windows 上打不开含中文的路径
    （会报 "could not open ... No such file or directory"），所以切到索引目录、
    用纯 ASCII 的相对路径 "./index.faiss" 来写，绕开编码问题。"""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(FAISS_INDEX_DIR)
        vs.save_local(".")
    finally:
        os.chdir(cwd)


def build_index(embeddings, docs_paths, state, sigs, reason):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.documents import Document

    log(f"  重建向量库（{reason}）")
    if os.path.isdir(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    raw_docs, ok, failed = [], {}, {}
    for p in docs_paths:
        name = os.path.basename(p)
        text = read_file(p)
        if not text or not text.strip():
            failed[name] = sigs[name]
            continue
        mtime = os.path.getmtime(p)
        stamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        raw_docs.append(Document(page_content=text, metadata={"source": name, "mtime": stamp}))
        ok[name] = sigs[name]
        log(f"  读取资料：{name}（{len(text)} 字）")

    vs = None
    if raw_docs:
        chunks = splitter.split_documents(raw_docs)
        log(f"  切分 {len(chunks)} 段，正在向量化（首次较慢）...")
        try:
            vs = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
        except Exception as e:
            die(f"向量化失败：{e}",
                f"确认 Ollama 里有 {EMB_MODEL}：ollama pull {EMB_MODEL}")
        _save_vectorstore(vs)
    else:
        log("  没有可入库的资料（放入 pdf/docx/txt/md/csv 后重启即可）")

    save_state({"version": 2, "embed": EMB_MODEL, "files": ok, "failed": failed})
    return vs


def load_index(embeddings):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    cwd = os.getcwd()
    try:
        os.chdir(FAISS_INDEX_DIR)
        return FAISS.load_local(".", embeddings,
                                allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.COSINE)
    finally:
        os.chdir(cwd)


def ensure_kb(embeddings):
    docs = scan_docs()
    sigs = {os.path.basename(p): file_sig(p) for p in docs}
    state = load_state()
    action, reason = decide(state, sigs)

    if action == "none":
        log("  知识库为空（可把资料放到脚本同目录）")
        return None
    if action == "clear":
        log(f"  清空向量库（{reason}）")
        if os.path.isdir(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)
        save_state({"version": 2, "embed": EMB_MODEL, "files": {}, "failed": {}})
        return None
    if action == "rebuild":
        vs = build_index(embeddings, docs, state, sigs, reason)
        if vs is not None:
            log(f"  知识库就绪（{getattr(vs.index, 'ntotal', '?')} 段）")
        return vs

    log("  加载已有向量库...")
    try:
        vs = load_index(embeddings)
        total = getattr(getattr(vs, "index", None), "ntotal", "?")
        log(f"  知识库就绪（{total} 段）")
        return vs
    except Exception as e:
        log(f"  [警告] 向量库加载失败（{e}），改为重建")
        return build_index(embeddings, docs, state, sigs, "加载失败")


# ---------------------------------------------------------------- 检索 / 联网
def retrieve(vs, question, k=TOP_K):
    if vs is None:
        return [], ""
    if GREETING_RE.match(question.strip()) and len(question.strip()) <= 10:
        return [], ""
    try:
        pairs = vs.similarity_search_with_score(question, k=k)
    except Exception as e:
        log(f"  [警告] 本地检索失败：{e}")
        return [], ""
    hits = []
    for doc, dist in pairs:
        # 归一化向量上的平方 L2 距离 → 余弦相似度
        sim = max(0.0, 1.0 - float(dist) / 2.0)
        if sim >= SIM_THRESHOLD:
            hits.append((sim, doc))
    if not hits:
        return [], ""
    hits.sort(key=lambda x: -x[0])
    ctx, used, sources = [], 0, []
    for sim, doc in hits:
        text = doc.page_content.strip()
        if used + len(text) > LOCAL_MAX_CHARS:
            text = text[: max(0, LOCAL_MAX_CHARS - used)]
        if not text:
            break
        src = doc.metadata.get("source", "未知")
        stamp = doc.metadata.get("mtime", "")
        ctx.append(f"[{src} | {stamp}] {text}")
        used += len(text)
        if src not in [s[0] for s in sources]:
            sources.append((src, stamp))
        if used >= LOCAL_MAX_CHARS:
            break
    return "\n\n".join(ctx), "、".join(f"{s}（{t}）" if t else s for s, t in sources)


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
GREETING_RE = re.compile(r"^(你好|您好|hi|hello|嗨|在吗|谢谢|多谢|早上好|晚上好|哈喽)[!！。.~～\s]*$", re.I)
_search_cache = {}


def _strip_tags(html):
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
                .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">"))
    return re.sub(r"\s+", " ", text).strip()


def _bing_html(query, n):
    """直连 bing 结果页（国内可用，DDG 域名在国内多半不通）。"""
    import requests
    last = None
    for host in ("https://cn.bing.com/search", "https://www.bing.com/search"):
        try:
            r = requests.get(host, params={"q": query, "setlang": "zh-CN", "ensearch": "0"},
                             headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"},
                             timeout=10)
            if r.status_code != 200 or "b_algo" not in r.text:
                last = f"HTTP {r.status_code}"
                continue
            out = []
            for block in re.split(r'<li class="b_algo"', r.text)[1:]:
                m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
                if not m:
                    continue
                title = _strip_tags(m.group(2))
                p = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
                body = _strip_tags(p.group(1)) if p else ""
                if title:
                    out.append({"title": title, "body": body})
                if len(out) >= n:
                    break
            if out:
                return out
            last = "没有解析到结果"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"bing 抓取失败（{last}）")


def _ddgs_search(query, n):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS(timeout=8) as ddgs:
        res = list(ddgs.text(query, max_results=n))
    return [{"title": r.get("title") or "", "body": r.get("body") or ""} for r in res]


# ddgs 内部会按 bing→brave→duckduckgo→google→startpage 逐个重试，每次等满超时，
# 国内网络下能拖到 30 秒。这里给它一个硬性总预算，超时就直接换 bing 抓页。
_ddg_state = {"skip_until": 0.0}
DDG_BUDGET = 4.0          # ddgs 单次总预算（秒）
DDG_SKIP_SECONDS = 1800   # 连续超时后，本轮运行内先跳过它


def _call_with_timeout(fn, timeout, *args):
    """在守护线程里跑 fn，超时就放弃（线程自行结束，不阻塞退出）。"""
    import threading
    import queue
    box = queue.Queue()

    def worker():
        try:
            box.put(("ok", fn(*args)))
        except Exception as exc:  # noqa: BLE001 - 原样抛给调用方
            box.put(("err", exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        kind, value = box.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"超过 {timeout:.0f} 秒没有返回")
    if kind == "err":
        raise value
    return value


def web_search(question, n=WEB_RESULTS):
    """先 DDG（限时），失败自动切 bing 网页抓取；结果带 5 分钟缓存。

    返回 (给模型看的文本, 状态)。状态用于让提示词知道"到底是没有信息，还是搜索坏了"：
      ok       搜到内容
      empty    搜到了但没结果
      error    所有通道都失败
      off      用户关了联网
      skipped  寒暄类，无需搜索
    失败时的技术细节只写日志，不塞进提示词，避免模型复述报错信息。
    """
    if NO_WEB:
        return "", WEB_OFF
    if GREETING_RE.match(question.strip()):
        return "", WEB_SKIP

    key = question.strip()
    cached = _search_cache.get(key)
    if cached and datetime.datetime.now().timestamp() - cached[0] < 300:
        return cached[1], cached[2]

    errors, results = [], None
    for name, fn, budget in (("ddg", _ddgs_search, DDG_BUDGET), ("bing", _bing_html, 12)):
        if name == "ddg" and time.time() < _ddg_state["skip_until"]:
            errors.append("ddg: 刚超时过，本轮跳过")
            continue
        started = time.time()
        try:
            results = _call_with_timeout(fn, budget, question, n)
            if results:
                break
        except Exception as e:
            errors.append(f"{name}: {e}")
            if name == "ddg" and isinstance(e, TimeoutError):
                _ddg_state["skip_until"] = time.time() + DDG_SKIP_SECONDS
                log(f"  [提示] ddg 超过 {budget:.0f} 秒没响应，接下来半小时直接走 bing")
            else:
                log(f"  [提示] {name} 搜索不可用，换下一个通道：{e}")
        finally:
            spent = time.time() - started
            if spent > 3:
                log(f"  [提示] {name} 通道耗时 {spent:.1f}s")

    if not results:
        if errors:
            log("  [提示] 联网搜索全部通道失败")
            for e in errors:
                log(f"         · {e}")
            out = ("", WEB_ERROR)
        else:
            out = ("", WEB_EMPTY)
        _search_cache[key] = (datetime.datetime.now().timestamp(), out[0], out[1])
        return out

    lines, total = [], 0
    for r in results:
        body = (r.get("body") or "").strip()
        if not body:
            continue
        item = f"- {r.get('title', '').strip()}：{body}"
        if total + len(item) > WEB_MAX_CHARS:
            break
        lines.append(item)
        total += len(item)

    text = "\n".join(lines)
    out = (text, WEB_OK) if text else ("", WEB_EMPTY)
    _search_cache[key] = (datetime.datetime.now().timestamp(), out[0], out[1])
    return out


# ---------------------------------------------------------------- 模型调用
# 推理型模型（DeepSeek-R1 系列）会把思考过程一起吐在 content 里，
# 这里负责把 </think> 之前的部分切掉，只留最终答案。
REASONING_KEYS = ("deepseek-r1", "deepseek_r1", "deepseek-r1-distill", "r1-distill")
THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", re.I)
THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.I)
REASONING_EMPTY_HINT = ("（这个模型只输出了思考过程，没给出最终结论。"
                        "再把问题问得具体一点，或者在下拉框换成 qwen2.5 试试。）")


def is_reasoning_model(model_name):
    name = (model_name or "").lower()
    return any(k in name for k in REASONING_KEYS)


def strip_thinking(text):
    """切掉 </think> 及之前的思考内容；整段都是思考（没有闭合标签）时返回空串。"""
    if not text:
        return text
    m = THINK_CLOSE_RE.search(text)
    if m:
        return text[m.end():].lstrip("\n\r 　")
    if THINK_OPEN_RE.search(text):
        return ""
    return text


class ThinkFilter:
    """流式版本：在 </think> 出现前先把内容憋住，之后原样放行。

    只对 DeepSeek-R1 这类模型启用；普通模型遇到它会把第一个 token 直接放行，
    行为与原来完全一致。
    """

    def __init__(self, hold_limit=160):
        self.buf = ""
        self.released = False
        self.hold_limit = hold_limit

    def feed(self, piece):
        if self.released or not piece:
            return piece if self.released else ""
        self.buf += piece

        m = THINK_CLOSE_RE.search(self.buf)
        if m:
            rest = self.buf[m.end():]
            self.buf = ""
            self.released = True
            return rest.lstrip("\n\r 　")

        head = self.buf.lstrip("\n\r 　")
        # 内容不是以 "<" 开头 → 模型没在思考，立刻放行，不改变原有体验
        if head and not head.startswith("<"):
            self.buf = ""
            self.released = True
            return head
        # 憋太久还没有 think 标签 → 放弃拦截，原样吐出
        if len(self.buf) > self.hold_limit:
            out = self.buf
            self.buf = ""
            self.released = True
            return out
        return ""

    def finish(self):
        """流结束时调用；返回还残留的内容（纯思考就丢掉）。"""
        if self.released or not self.buf:
            return ""
        left = self.buf
        self.buf = ""
        self.released = True
        return "" if THINK_OPEN_RE.search(left) else left


def make_chat(model_name):
    from langchain_ollama import ChatOllama
    if is_reasoning_model(model_name):
        # 推理模型要先"想"再"答"，num_predict 太小会出现"全是思考、正文空白"
        return ChatOllama(model=model_name, temperature=0.6, num_ctx=8192, num_predict=3072)
    return ChatOllama(model=model_name, temperature=0.4, num_ctx=8192, num_predict=1200)


SYSTEM_PROMPT = """你是一个中文 AI 助手，回答准确、专业、自然。

【信息使用规则】
1. 联网检索正常时，以【网络信息】为主要依据；【本地资料】有相关补充就结合使用。两者冲突时采用时间更新的一方，并简单说明分歧。
2. 引用【本地资料】里的具体事实、数字、名称时，在句末自然标注来源，例如"（来源：报销制度.md）"。同一段最多标一次，不要每条都标。
3. 两边都没覆盖的通用知识，直接用你自己的知识完整回答。
4. 没有来源支撑的具体数字、日期、价格、人名、地点一律不编；确定不了就说"这一点我不确定"。

【检索状态的处理】先看【检索状态】这一行：
· 联网失败、无结果或已关闭，且本地资料也没有相关内容时：
  - 问的是实时信息（天气、新闻、股价汇率、赛事比分、航班路况、最新动态等）→ 只回复下面这一句，不加任何前后缀、不解释原因：
    抱歉，当前网络搜索服务似乎遇到了一点小问题，暂时无法获取最新的实时信息。请稍后再试。
  - 问的是通用知识（概念、原理、方法、写代码、写作、翻译、常识等）→ 当作没联网这回事，直接用你自己的知识回答，不要提"搜索失败""没有资料"。
· 联网失败但本地资料有内容 → 用本地资料回答，并说明来自哪个文件。
· 检索正常 → 正常作答。

【表达要求】
- 第一句话就是答案，不要复述问题，不要"好的，我来为你……"这种开场。
- 禁止"根据您提供的信息""从提供的资料来看""作为一个AI语言模型"这类套话。
- 不要输出【来源：xxx】标记或"引用1"之类编号，也不要罗列检索过程。
- 简洁优先：能三句说清就不写五句，只有分步骤操作时才用列表。
"""


def build_prompt(question, web_info, local_ctx, web_status=WEB_OK):
    """把检索状态一起交给模型：它必须知道"没搜到"和"搜索坏了"是两回事。"""
    has_local = bool((local_ctx or "").strip())
    status = (f"联网检索：{STATUS_TEXT.get(web_status, web_status)}｜"
              f"本地资料：{'检索到相关内容' if has_local else '无相关内容'}")
    parts = [
        SYSTEM_PROMPT.strip(),
        "",
        "【检索状态】" + status,
        "",
        "【网络信息】",
        (web_info or "").strip() or "（无）",
        "",
        "【本地资料】",
        (local_ctx or "").strip() or "（无）",
        "",
        "【问题】" + question.strip(),
        "",
        "回答：",
    ]
    return "\n".join(parts)


def clean_answer(text):
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"`{3}.*?`{3}", lambda m: m.group(0).strip("`"), text, flags=re.S)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def ask_llm(prompt, model_name):
    from langchain_core.messages import HumanMessage
    chat = make_chat(model_name)
    msg = [HumanMessage(content=prompt)]
    out = ""
    try:
        for chunk in chat.stream(msg):
            piece = getattr(chunk, "content", None)
            if piece:
                out += piece
    except Exception as e:
        log(f"  [警告] 流式调用失败，回退整段调用：{e}")
        try:
            resp = chat.invoke(msg)
            out = resp.content or ""
        except Exception as e2:
            return f"（模型调用失败：{e2}）"

    if is_reasoning_model(model_name):
        out = strip_thinking(out)
        if not out.strip():
            log("  [提示] 模型只输出了思考过程，没有最终答案")
            return REASONING_EMPTY_HINT
    return clean_answer(out)


def stream_llm(prompt, model_name):
    from langchain_core.messages import HumanMessage
    chat = make_chat(model_name)
    reasoning = is_reasoning_model(model_name)
    filt = ThinkFilter() if reasoning else None
    emitted = 0

    for chunk in chat.stream([HumanMessage(content=prompt)]):
        piece = getattr(chunk, "content", None)
        if not piece:
            continue
        if filt is not None:
            piece = filt.feed(piece)
        if piece:
            emitted += len(piece)
            yield piece

    if filt is not None:
        tail = filt.finish()
        if tail:
            emitted += len(tail)
            yield tail
        if emitted == 0:
            log("  [提示] 模型只输出了思考过程，没有最终答案")
            yield REASONING_EMPTY_HINT


def _realtime_fallback(web_status):
    return FALLBACK_REALTIME_OFF if web_status == WEB_OFF else FALLBACK_REALTIME


def _short_circuit(question, local_ctx, web_status):
    """搜索不可用 + 本地也没资料 + 问的是实时信息 → 直接给兜底话术，不浪费一次推理。"""
    if (local_ctx or "").strip():
        return None
    if web_status not in (WEB_ERROR, WEB_EMPTY, WEB_OFF):
        return None
    if REALTIME_RE.search(question):
        return _realtime_fallback(web_status)
    return None


def answer_question(question, model_name):
    """返回 (answer, used_local, sources, web_status)"""
    local_ctx, sources = retrieve(VS, question)
    web_info, web_status = web_search(question)
    early = _short_circuit(question, local_ctx, web_status)
    if early:
        log("  [兜底] 实时类问题但检索不可用，直接返回提示语")
        return early, False, "", web_status
    answer = ask_llm(build_prompt(question, web_info, local_ctx, web_status), model_name)
    return answer, bool(local_ctx), sources, web_status


# ---------------------------------------------------------------- HTTP
ensure_deps()

from flask import Flask, request, jsonify, Response, send_from_directory  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

log(LINE)
log("  本地AI大模型启动中")
log(LINE)

MODELS = ollama_models()
log(f"  Ollama 正常，可用模型：{', '.join(MODELS) if MODELS else '（空）'}")
DEFAULT_LLM = pick_default_llm(MODELS)
if not DEFAULT_LLM:
    die("没有可用的对话模型", "先执行：ollama pull qwen2.5:3b")
if not resolve_model(MODELS, EMB_MODEL):
    die(f"找不到 embedding 模型 {EMB_MODEL}", f"先执行：ollama pull {EMB_MODEL}")
log(f"  对话模型：{DEFAULT_LLM}    向量模型：{EMB_MODEL}")

EMBEDDINGS = OllamaEmbeddings(model=EMB_MODEL)
VS = ensure_kb(EMBEDDINGS)

app = Flask(__name__, static_folder=None)


@app.get("/")
def index_page():
    return send_from_directory(SCRIPT_DIR, INDEX_FILE)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": DEFAULT_LLM,
        "embed": EMB_MODEL,
        "models": MODELS,
        "kb_docs": len(load_state().get("files", {})),
        "web": not NO_WEB,
    })


# ---------------------------------------------------------------- 模型列表
OLLAMA_BASE = os.environ.get("AI_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TAGS_URL = OLLAMA_BASE + "/api/tags"
_tags_cache = {"ts": 0.0, "names": []}


def live_model_names(force=False):
    """实时向 Ollama 要 /api/tags（30 秒缓存）；失败就退回启动时的快照。"""
    now = time.time()
    if not force and _tags_cache["names"] and now - _tags_cache["ts"] < 30:
        return _tags_cache["names"]
    try:
        import requests
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        resp.raise_for_status()
        names = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
        if names:
            _tags_cache.update(ts=now, names=names)
            return names
    except Exception as e:
        log(f"  [提示] 读取 Ollama 模型列表失败，暂用启动时的快照：{e}")
    return _tags_cache["names"] or MODELS


def chat_model_names(names=None):
    return [n for n in (names if names is not None else live_model_names())
            if not looks_like_embedding(n)]


@app.get("/api/models")
def api_models():
    """前端下拉框的数据源：只把能对话的模型放进 models，向量模型单独列出。"""
    names = live_model_names(force=request.args.get("refresh") == "1")
    chats = chat_model_names(names)
    default = DEFAULT_LLM if DEFAULT_LLM in chats else (chats[0] if chats else DEFAULT_LLM)
    return jsonify({
        "models": chats,
        "embedding": [n for n in names if looks_like_embedding(n)],
        "all": names,
        "default": default,
    })


def _prepare(payload):
    """解析提问 + 前端选中的模型，模型不可用时回落到默认模型。"""
    q = (payload.get("message") or payload.get("q") or "").strip()
    want = (payload.get("model") or "").strip()
    if want:
        names = live_model_names()
        got = want if want in names else resolve_model(names, want)
        if got and not looks_like_embedding(got):
            return q, got
        log(f"  [提示] 前端指定的模型 {want} 不可用，改用 {DEFAULT_LLM}")
    return q, DEFAULT_LLM


@app.post("/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    q, model = _prepare(data)
    if not q:
        return jsonify({"reply": "请输入问题。", "error": "empty"}), 400
    try:
        answer, used_local, sources, web_status = answer_question(q, model)
    except Exception as e:
        log(f"  [错误] 处理失败：{e}")
        return jsonify({"reply": f"（服务出错：{e}）", "error": str(e)}), 500
    if sources:
        answer += f"\n\n（本地参考：{sources}）"
    return jsonify({
        "reply": answer,
        "used_web": web_status == WEB_OK,
        "web_status": web_status,
        "used_local": used_local,
        "sources": sources,
        "model_used": model,
    })


def _sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.post("/chat/stream")
def chat_stream():
    data = request.get_json(silent=True) or {}
    q, model = _prepare(data)
    if not q:
        return jsonify({"error": "empty"}), 400

    def gen():
        try:
            local_ctx, sources = retrieve(VS, q)
            yield _sse({"type": "meta", "used_local": bool(local_ctx),
                        "sources": sources, "model": model})
            web_info, web_status = web_search(q)
            early = _short_circuit(q, local_ctx, web_status)
            if early:
                log("  [兜底] 实时类问题但检索不可用，直接返回提示语")
                yield _sse({"type": "token", "text": early})
                yield _sse({"type": "done", "web_status": web_status})
                return
            for piece in stream_llm(build_prompt(q, web_info, local_ctx, web_status), model):
                yield _sse({"type": "token", "text": piece})
            yield _sse({"type": "done", "web_status": web_status})
        except Exception as e:
            log(f"  [错误] 流式处理失败：{e}")
            yield _sse({"type": "error", "message": str(e)})

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


if __name__ == "__main__":
    log(LINE)
    log(f"  服务已启动：http://127.0.0.1:{PORT}")
    if HOST not in ("127.0.0.1", "localhost"):
        log(f"  警告：正在监听 {HOST}，同一局域网的人都能访问")
    log("  按 Ctrl+C 停止")
    log(LINE)

    # 后台预热模型，避免第一次提问多等 8 秒（AI_WARMUP=0 可关闭）
    if os.environ.get("AI_WARMUP", "1") not in ("0", "false", "False"):
        def _warmup():
            from langchain_core.messages import HumanMessage
            try:
                make_chat(DEFAULT_LLM).invoke([HumanMessage(content="你好")])
                log("  模型已预热")
            except Exception as e:
                log(f"  [提示] 预热失败（不影响使用）：{e}")
        import threading
        threading.Thread(target=_warmup, daemon=True).start()

    app.run(host=HOST, port=PORT, threaded=True, debug=False)
