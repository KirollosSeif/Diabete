# streamlit_prod/chatbot.py
from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# PDF -> testo + TF-IDF
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    from PyPDF2 import PdfReader  # fallback

from sklearn.feature_extraction.text import TfidfVectorizer  # scikit-learn già nel progetto
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from openai import OpenAI

# === PATH E ENV ===
ROOT = Path(__file__).resolve().parents[1]     # root del progetto (repo)
load_dotenv(ROOT / ".env")

# Provider e chiavi
PROVIDER = os.getenv("CHAT_PROVIDER", "openai").lower()  # 'openai' o 'deepseek'

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# DeepSeek (opzionale)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Dati/Documento
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "CS-PANORAMA-DIABETE-LANCIO-DEF.pdf"

# === CLIENT FACTORY ===
def _make_client(provider: str | None = None):
    prov = (provider or PROVIDER).lower()
    if prov == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY mancante nel .env")
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), OPENAI_MODEL, "openai"
    if prov == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY mancante nel .env")
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL), DEEPSEEK_MODEL, "deepseek"
    raise RuntimeError(f"Provider non supportato: {prov}")

_client, _model, _prov = _make_client()

# === INDICE PDF (TF-IDF) ===
_vectorizer: Optional[TfidfVectorizer] = None
_doc_matrix = None
_chunks: List[str] = []

def _pdf_to_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    text = text.replace("\r", " ").replace("\t", " ")
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= max_chars:
            joined = " ".join(cur)
            chunks.append(joined)
            back = joined[-overlap:] if overlap > 0 else ""
            cur = [back] if back else []
            cur_len = len(back)
    if cur:
        chunks.append(" ".join(cur))
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    return chunks

def build_pdf_index(force: bool = False) -> Tuple[bool, str]:
    """Costruisce (o ricostruisce) l'indice TF-IDF del PDF."""
    global _vectorizer, _doc_matrix, _chunks
    if _vectorizer is not None and not force:
        return True, "Indice già pronto."

    raw = _pdf_to_text(PDF_PATH)
    if not raw.strip():
        _vectorizer, _doc_matrix, _chunks = None, None, []
        return False, f"PDF non trovato o vuoto: {PDF_PATH}"

    _chunks = _chunk_text(raw, max_chars=900, overlap=150)
    if not _chunks:
        _vectorizer, _doc_matrix = None, None
        return False, "Nessun contenuto indicizzabile nel PDF."

    _vectorizer = TfidfVectorizer(stop_words=None)
    _doc_matrix = _vectorizer.fit_transform(_chunks)
    return True, f"Indicizzate {len(_chunks)} sezioni dal PDF."

def _retrieve(query: str, k: int = 4) -> List[str]:
    """Ritorna i top-k chunk più simili alla query."""
    if _vectorizer is None or _doc_matrix is None or not _chunks:
        ok, _ = build_pdf_index()
        if not ok:
            return []
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _doc_matrix)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [_chunks[i] for i in idx if sims[i] > 0.01]

def _llm_chat(messages: List[Dict[str, str]], temperature: float = 0.6, max_tokens: int = 650) -> str:
    """Chiamata uniforme al provider selezionato."""
    resp = _client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return resp.choices[0].message.content

# === API PUBBLICA ===
def answer_with_rag(user_question: str, site_faq: str = "") -> str:
    """
    - Recupera i migliori estratti dal PDF e li passa come contesto.
    - Aggiunge un breve 'site_faq' (testo) per spiegare scopo e uso del sito.
    - Non fornisce diagnosi: tono prudente.
    """
    contexts = _retrieve(user_question, k=4)
    context_block = "\n\n".join([f"- {c}" for c in contexts]) if contexts else "Nessun estratto utile trovato."

    system = (
        "Sei un assistente prudente e chiaro. Non fornisci diagnosi e inviti a consultare un medico quando serve.\n"
        "Usa il contesto per rispondere in modo sintetico e pratico. Se non trovi risposte nel contesto, dillo. \n"
        " Non inventare risposte.\n"
        "Quando l'utente è soddisfatto della risposta o ti saluta, rispondi con 'Felice di aiutarti!'.\n"
    )
    if site_faq:
        system += f"\n\n[FAQ Sito]\n{site_faq}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Domanda: {user_question}\n\n[Contesto dal documento]\n{context_block}"}
    ]
    try:
        return _llm_chat(messages)
    except Exception as e:
        # Fallback minimale (offline) se l'API non è disponibile
        off = "Modalità offline: rispondo solo in base agli estratti del documento.\n\n"
        if contexts:
            return off + "\n\n".join(contexts[:2])
        return f"⚠️ Errore nel contattare il modello: {e}"

# costruisci l'indice una volta all'import (best-effort)
build_pdf_index(force=True)

# === CLI DI TEST ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="", help="Fai una domanda di test")
    parser.add_argument("--provider", type=str, default="", help="Forza provider: openai | deepseek")
    args = parser.parse_args()

    if args.provider:
        # Override provider a runtime (comodo per test)
        PROVIDER = args.provider.lower()
        _client, _model, _prov = _make_client(PROVIDER)

    if args.test:
        faq = (
            "Questo sito offre un form per una stima del rischio (0/1/2) e una sezione contatti per trovare ospedali nel proprio comune. "
            "Il risultato non è una diagnosi clinica."
        )
        print(f"Q: {args.test}")
        print("A:", answer_with_rag(args.test, faq))
    else:
        print(f"Provider attivo: {_prov} | Modello: {_model}")
        ok, msg = build_pdf_index(force=False)
        print(msg)
