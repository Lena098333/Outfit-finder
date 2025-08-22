""" Second-hand Outfit Finder — Streamlit app

Funkcja: wklejasz link do obrazka (np. z Pinteresta) lub wgrywasz zdjęcie. Aplikacja:

1. próbuje rozpoznać elementy garderoby zero-shot (CLIP)


2. wyciąga dominujące kolory


3. generuje gotowe linki do wyszukiwania na Vinted / OLX / Allegro Lokalnie (2. ręka)



Instalacja (Python 3.9+):

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit torch open_clip_torch Pillow requests numpy

Uruchomienie:

streamlit run app.py

Zmień nazwę tego pliku na app.py (albo wskaż własną ścieżkę przy uruchamianiu).

Uwaga: pierwsze uruchomienie może pobrać model (kilkadziesiąt MB). """

import io import math import os from typing import List, Tuple

import numpy as np import requests from PIL import Image

import streamlit as st

Optional: torch + open_clip may be heavy to import; do it lazily and handle errors nicely.

try: import torch import open_clip _CLIP_AVAILABLE = True except Exception as e: _CLIP_AVAILABLE = False _CLIP_IMPORT_ERROR = e

APP_TITLE = "Second‑hand Outfit Finder"

A focused set of garment labels (Polish) for zero‑shot classification with CLIP.

GARMENT_LABELS = [ "kurtka skórzana", "ramoneska", "trench", "płaszcz wełniany", "marynarka oversize", "kurtka jeansowa", "bluza oversize", "sweter wełniany", "golf", "koszula biała", "t-shirt basic", "top crop", "kamizelka pikowana", "kamizelka garniturowa", "spodnie jeansy mom fit", "spodnie jeansy straight", "spodnie wide leg", "spodnie cargo", "spodnie garniturowe", "spódnica midi", "spódnica mini", "sukienka slip", "sukienka koszulowa", "sneakersy", "trampki", "botki skórzane", "kozaki", "mokasyny", "baleriny", "klapki", "torebka na ramię", "torebka listonoszka", "plecak", "czapka beanie", "czapka z daszkiem", "apaszka" ]

Simple color words in Polish for query enrichment.

COLOR_WORDS_PL = [ "czarny", "biały", "beżowy", "szary", "granatowy", "brązowy", "zielony", "czerwony", "niebieski", "żółty", "różowy", "fioletowy" ]

def download_image_from_url(url: str) -> Image.Image: """Download image and return PIL Image.""" headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36" } r = requests.get(url, headers=headers, timeout=20) r.raise_for_status() img = Image.open(io.BytesIO(r.content)).convert("RGB") return img

def resize_short_side(img: Image.Image, short_side: int = 512) -> Image.Image: w, h = img.size if min(w, h) <= short_side: return img if w < h: new_w = short_side new_h = int(h * short_side / w) else: new_h = short_side new_w = int(w * short_side / h) return img.resize((new_w, new_h), Image.LANCZOS)

def extract_dominant_colors(img: Image.Image, k: int = 4) -> List[Tuple[int, int, int]]: """A fast, dependency‑light color extraction using quantization to 16×16×16 bins.""" small = img.copy() small.thumbnail((200, 200)) arr = np.array(small) # Quantize to 16 levels per channel q = (arr // 16).astype(np.int32) bins = q[:, :, 0] * 256 + q[:, :, 1] * 16 + q[:, :, 2] hist = np.bincount(bins.flatten(), minlength=4096) top_idx = hist.argsort()[::-1][:k] colors = [] for idx in top_idx: r = ((idx // 256) % 16) * 16 + 8 g = ((idx // 16) % 16) * 16 + 8 b = (idx % 16) * 16 + 8 colors.append((int(r), int(g), int(b))) return colors

def rgb_to_css(rgb: Tuple[int, int, int]) -> str: return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

def nearest_color_words(colors: List[Tuple[int, int, int]], max_words: int = 3) -> List[str]: # naive mapping by hue ranges words = [] for r, g, b in colors: arr = np.array([r, g, b], dtype=np.float32) # Convert to HSV-ish simple rules maxc = arr.max(); minc = arr.min() if maxc == minc: # grayish if maxc < 60: words.append("czarny") elif maxc > 200: words.append("biały") else: words.append("szary") continue # Heuristic hue r_, g_, b_ = arr / 255.0 mx = max(r_, g_, b_); mn = min(r_, g_, b_) if mx == r_: h = (60 * ((g_ - b_) / (mx - mn)) + 360) % 360 elif mx == g_: h = 60 * ((b_ - r_) / (mx - mn)) + 120 else: h = 60 * ((r_ - g_) / (mx - mn)) + 240 if 330 <= h or h < 15: words.append("czerwony") elif 15 <= h < 45: words.append("żółty") elif 45 <= h < 75: words.append("beżowy") elif 75 <= h < 170: words.append("zielony") elif 170 <= h < 255: words.append("niebieski") elif 255 <= h < 290: words.append("fioletowy") else: words.append("różowy") # Keep unique and at most max_words out = [] for w in words: if w not in out: out.append(w) if len(out) >= max_words: break return out

@st.cache_resource(show_spinner=False) def load_clip_model(): """Load OpenCLIP model + preprocess.""" if not _CLIP_AVAILABLE: raise RuntimeError(f"open_clip/torch niedostępne: {_CLIP_IMPORT_ERROR}") model, _, preprocess = open_clip.create_model_and_transforms( "ViT-B-32", pretrained="laion2b_s34b_b79k" ) model.eval() device = "cuda" if torch.cuda.is_available() else "cpu" model.to(device) tokenizer = open_clip.get_tokenizer("ViT-B-32") return model, preprocess, tokenizer, device

def clip_zero_shot_labels(img: Image.Image, labels: List[str], top_k: int = 6): if not _CLIP_AVAILABLE: return [] model, preprocess, tokenizer, device = load_clip_model() with torch.no_grad(): image = preprocess(img).unsqueeze(0).to(device) # Polish prompt templates; we try a couple for more robust scores templates = [ "zdjęcie przedstawiające {}", "ubranie: {}", "{} na osobie", ] # Build text tokens texts = [] for lab in labels: for t in templates: texts.append(t.format(lab)) text_tokens = tokenizer(texts).to(device) image_features = model.encode_image(image) text_features = model.encode_text(text_tokens) # Normalize image_features = image_features / image_features.norm(dim=-1, keepdim=True) text_features = text_features / text_features.norm(dim=-1, keepdim=True) # Compute similarities per template, then aggregate per label (max) sims = (image_features @ text_features.T).squeeze(0).float().cpu().numpy() sims = sims.reshape(len(labels), len(templates)) scores = sims.max(axis=1)  # max over templates top_idx = scores.argsort()[::-1][:top_k] results = [(labels[i], float(scores[i])) for i in top_idx] return results

def make_market_links(query: str) -> List[Tuple[str, str]]: """Return list of (market_name, url) pairs for second‑hand platforms in PL.""" q = requests.utils.quote(query) links = [ ("Vinted", f"https://www.vinted.pl/catalog?search_text={q}"), ("OLX", f"https://www.olx.pl/oferty/q-{q}/?search%5Bfilter_float_price%3Afrom%5D=&search%5Bfilter_float_price%3Ato%5D="), ("Allegro Lokalnie", f"https://allegrolokalnie.pl/oferty/q/{q}"), ("Allegro (używane)", f"https://allegro.pl/listing?string={q}&stan=u%C5%BCywane"), ] return links

---------------- UI ----------------

st.set_page_config(page_title=APP_TITLE, page_icon="🧥") st.title("🧥 Second‑hand Outfit Finder") st.caption("Wklej link do zdjęcia (np. z Pinteresta) albo wgraj obraz. Dostaniesz linki do wyszukań na Vinted/OLX/Allegro.")

col1, col2 = st.columns([2, 1]) with col1: url = st.text_input("Link do obrazka (bezpośredni URL do zdjęcia)") uploaded = st.file_uploader("…lub wgraj plik JPG/PNG", type=["jpg", "jpeg", "png"])
with col2: topk = st.slider("Ile elementów garderoby?", 3, 10, 6)

img = None error = None if uploaded is not None: try: img = Image.open(uploaded).convert("RGB") except Exception as e: error = f"Nie udało się otworzyć pliku: {e}" elif url: try: img = download_image_from_url(url) except Exception as e: error = f"Nie udało się pobrać obrazu: {e}"

if error: st.error(error)

if img is not None: img_disp = resize_short_side(img, 768) st.image(img_disp, caption="Podgląd zdjęcia", use_column_width=True)

# Colors
colors = extract_dominant_colors(img_disp, k=4)
color_words = nearest_color_words(colors, max_words=3)

with st.expander("Dominujące kolory (heurystycznie)"):
    cols = st.columns(len(colors))
    for i, c in enumerate(colors):
        with cols[i]:
            st.markdown(f"<div style='width:100%;height:48px;border-radius:8px;border:1px solid #ddd;background:{rgb_to_css(c)}'></div>", unsafe_allow_html=True)
            st.caption(rgb_to_css(c))
    st.write("Słowa‑klucze kolorów:", ", ".join(color_words) if color_words else "—")

# Garments via CLIP
detected = []
if _CLIP_AVAILABLE:
    with st.spinner("Analiza obrazu (CLIP)…"):
        detected = clip_zero_shot_labels(img_disp, GARMENT_LABELS, top_k=topk)
else:
    st.warning("Brak bibliotek open_clip/torch — pomijam automatyczne rozpoznanie. Użyj pola poniżej.")

auto_labels = [lab for lab, _ in detected]

if detected:
    st.subheader("Rozpoznane elementy (heurystycznie)")
    for lab, score in detected:
        st.write(f"• **{lab}** (pewność ~ {score:.2f})")

user_extra = st.text_input("Dodaj własne słowa‑klucze (np. 'oversize', 'z wysokim stanem', marka itp.)")

# Build final query
tokens = []
tokens.extend(auto_labels)
tokens.extend(color_words)
if user_extra.strip():
    tokens.append(user_extra.strip())
tokens = [t for t in tokens if t]
# Deduplicate while keeping order
seen = set(); final_tokens = []
for t in tokens:
    low = t.lower()
    if low not in seen:
        seen.add(low)
        final_tokens.append(t)
final_query = " ".join(final_tokens) if final_tokens else "ubranie vintage"

st.subheader("Proponowane wyszukanie")
st.code(final_query)

links = make_market_links(final_query)
st.subheader("Linki do platform 2. ręki")
for name, link in links:
    st.markdown(f"- [{name}]({link})")

else: st.info("Wklej link lub wgraj obraz, żeby zacząć.")

st.markdown("---") st.caption( "Wskazówka: jeśli wyników jest za mało, uprość frazę (np. usuń kolor). Jeśli jest ich za dużo, dodaj szczegóły (krój, materiał, marka)." )

