# 🚀 ANALIZZATORE GOOGLE REVIEWS - VERSIONE FINALE FUNZIONANTE
# Fix completo estrazione recensioni con task_get

import streamlit as st
import pandas as pd
from openai import OpenAI
import re
import json
import time
import warnings
import io
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import requests
import base64
from difflib import get_close_matches
warnings.filterwarnings('ignore')

# 🎨 CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="🚀 Analizzatore Google Reviews Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌍 LOCATION CODES
LOCATION_CODES_ITALY = {
    'roma': 1027864, 'milano': 1028595, 'napoli': 1028550, 'torino': 1028762,
    'palermo': 1028643, 'genova': 1028507, 'bologna': 1028397, 'firenze': 1028504,
    'bari': 1028371, 'catania': 1028442, 'venezia': 1028773, 'verona': 1028774,
    'padova': 1028635, 'trieste': 1028761, 'trento': 1028760, 'bolzano': 1028398,
    'udine': 1028769, 'vicenza': 1028777, 'treviso': 1028766, 'brescia': 1028413,
    'bergamo': 1028387, 'monza': 1028593, 'como': 1028459, 'varese': 1028772,
    'pavia': 1028651, 'mantova': 1028571, 'cremona': 1028475, 'lecco': 1028548,
    'lodi': 1028557, 'novara': 1028628, 'alessandria': 1028341, 'asti': 1028365,
    'cuneo': 1028480, 'biella': 1028388, 'vercelli': 1028775, 'aosta': 1028350,
    'parma': 1028647, 'modena': 1028583, 'reggio emilia': 1028701,
    'ferrara': 1028501, 'ravenna': 1028696, 'rimini': 1028709, 'forli': 1028505,
    'cesena': 1028449, 'piacenza': 1028659, 'perugia': 1028655, 'terni': 1028758,
    'ancona': 1028346, 'pesaro': 1028656, 'macerata': 1028563, 'latina': 1028543,
    'frosinone': 1028506, 'viterbo': 1028780, 'rieti': 1028708, 'pescara': 1028657,
    'chieti': 1028453, 'pisa': 1028662, 'livorno': 1028555, 'arezzo': 1028357,
    'siena': 1028738, 'grosseto': 1028513, 'pistoia': 1028663, 'prato': 1028673,
    'lucca': 1028560, 'massa': 1028577, 'salerno': 1028719, 'foggia': 1028503,
    'taranto': 1028755, 'brindisi': 1028414, 'lecce': 1028547, 'potenza': 1028671,
    'matera': 1028578, 'cosenza': 1028473, 'catanzaro': 1028443,
    'reggio calabria': 1028702, 'messina': 1028580, 'siracusa': 1028741,
    'trapani': 1028759, 'agrigento': 1028338, 'caltanissetta': 1028428,
    'enna': 1028496, 'ragusa': 1028693, 'sassari': 1028728, 'cagliari': 1028424,
    'nuoro': 1028629, 'oristano': 1028633, 'italia': 2380, 'italy': 2380
}

# 🎯 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4285F4, #34A853, #FBBC05, #EA4335);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #34A853 0%, #7CB342 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .business-card {
        background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .review-example {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .positive-review { border-left: 4px solid #34A853; }
    .negative-review { border-left: 4px solid #EA4335; }
    .owner-response {
        background: #E8F0FE;
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        border-left: 3px solid #4285F4;
    }
    .frequency-badge {
        background: #4285F4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 Analizzatore Google Reviews Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Sistema Completo Funzionante</h3>
    <p>• Ricerca business con 6 strategie adattive</p>
    <p>• Estrazione recensioni corretta via task_get</p>
    <p>• Clustering ML avanzato</p>
    <p>• Analisi AI con GPT-4</p>
    <p>• Export Excel professionale</p>
</div>
""", unsafe_allow_html=True)

# 🔧 HELPER
def normalizza_nome_citta(nome_citta):
    if not nome_citta:
        return None
    nome_clean = nome_citta.lower().strip()
    nome_clean = re.sub(r'\s+', ' ', nome_clean)
    if nome_clean in LOCATION_CODES_ITALY:
        return nome_clean
    return None

# 🔧 CLASSE DATAFORSEO
class DataForSEOClient:
    
    def __init__(self, username, password, debug=False):
        self.username = username
        self.password = password
        self.base_url = "https://api.dataforseo.com/v3"
        self.debug = debug
        self._location_cache = {}
    
    def _log(self, message, level="info"):
        if self.debug:
            if level == "info":
                st.info(f"ℹ️ {message}")
            elif level == "success":
                st.success(f"✅ {message}")
            elif level == "warning":
                st.warning(f"⚠️ {message}")
            elif level == "error":
                st.error(f"❌ {message}")
    
    def _get_auth_token(self):
        credentials = f"{self.username}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()
    
    def _make_request(self, endpoint, data=None, method="POST"):
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self._get_auth_token()}'
        }
        
        self._log(f"API: {method} {endpoint}")
        
        try:
            if method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method == "GET":
                if data:
                    response = requests.get(url, headers=headers, params=data, timeout=30)
                else:
                    response = requests.get(url, headers=headers, timeout=30)
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('status_code') == 20000:
                return result
            else:
                error_msg = result.get('status_message', 'Unknown')
                raise Exception(f"API Error: {error_msg}")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection Error: {str(e)}")
    
    def get_location_code(self, location_name):
        if location_name in self._location_cache:
            return self._location_cache[location_name]
        
        self._log(f"🔍 Location: '{location_name}'")
        
        nome_normalizzato = normalizza_nome_citta(location_name)
        
        if nome_normalizzato and nome_normalizzato in LOCATION_CODES_ITALY:
            code = LOCATION_CODES_ITALY[nome_normalizzato]
            self._log(f"✅ Database: {nome_normalizzato.title()} ({code})", "success")
            self._location_cache[location_name] = code
            return code
        
        self._log(f"🌐 API search...")
        
        try:
            endpoint = "business_data/google/locations"
            
            search_queries = [
                location_name,
                f"{location_name}, Italia",
                f"{location_name}, Italy"
            ]
            
            for search_query in search_queries:
                self._log(f"Try: '{search_query}'")
                
                params = {'location_name': search_query}
                result = self._make_request(endpoint, params, method="GET")
                
                tasks = result.get('tasks', [])
                if tasks and tasks[0].get('result'):
                    locations = tasks[0]['result']
                    
                    if locations:
                        italian_locs = [
                            loc for loc in locations
                            if 'italy' in loc.get('location_name', '').lower() or
                               'italia' in loc.get('location_name', '').lower()
                        ]
                        
                        if italian_locs:
                            location_code = italian_locs[0].get('location_code')
                            location_full = italian_locs[0].get('location_name')
                            
                            self._log(f"✅ API: {location_full} ({location_code})", "success")
                            
                            self._location_cache[location_name] = location_code
                            return location_code
        
        except Exception as e:
            self._log(f"⚠️ API error: {e}", "warning")
        
        self._log(f"⚠️ Fallback: Italia", "warning")
        return 2380
    
    def search_business(self, query, location):
        self._log(f"=== SEARCH BUSINESS ===")
        self._log(f"Query: '{query}'")
        self._log(f"Location: '{location}'")
        
        query_clean = self._clean_query(query)
        
        is_full_address = self._is_full_address(location)
        
        if is_full_address:
            self._log("📍 Full address")
            return self._search_by_address(query, query_clean, location)
        else:
            self._log("🏙️ City")
            return self._search_by_city(query, query_clean, location)
    
    def _is_full_address(self, location):
        indicators = ['via', 'viale', 'piazza', 'corso', 'largo', ',']
        return any(ind in location.lower() for ind in indicators)
    
    def _search_by_address(self, query_original, query_clean, address):
        endpoint = "business_data/google/my_business_info/live"
        
        strategies = [
            {"keyword": f"{query_original} {address}"},
            {"keyword": f"{query_clean} {address}"},
            {"keyword": query_original, "location_name": address}
        ]
        
        for idx, strategy in enumerate(strategies, 1):
            self._log(f"📍 Strategy {idx}/3")
            
            payload = [{**strategy, "language_code": "it"}]
            
            try:
                result = self._make_request(endpoint, payload)
                items = self._extract_items(result)
                
                if items:
                    self._log(f"✅ Found!", "success")
                    return {'items': items}
            
            except Exception as e:
                self._log(f"Strategy {idx} failed", "warning")
                continue
        
        raise Exception(f"No results with address")
    
    def _search_by_city(self, query_original, query_clean, city):
        location_code = self.get_location_code(city)
        
        if not location_code:
            raise Exception(f"No location code for '{city}'")
        
        endpoint = "business_data/google/my_business_info/live"
        
        strategies = [
            {"keyword": f"{query_original} {city}", "location_code": location_code},
            {"keyword": f"{query_clean} {city}", "location_code": location_code},
            {"keyword": f"{query_original}, {city}", "location_code": location_code},
            {"keyword": query_original, "location_code": location_code},
            {"keyword": query_clean, "location_code": location_code},
            {"keyword": f"{query_original} {city} italia", "location_code": location_code}
        ]
        
        seen = set()
        unique = []
        for s in strategies:
            kw = s.get('keyword', '')
            if kw not in seen:
                seen.add(kw)
                unique.append(s)
        
        for idx, strategy in enumerate(unique, 1):
            self._log(f"🏙️ Strategy {idx}/{len(unique)}: '{strategy['keyword']}'")
            
            payload = [{**strategy, "language_code": "it"}]
            
            try:
                result = self._make_request(endpoint, payload)
                items = self._extract_items(result)
                
                if items:
                    self._log(f"✅ Found!", "success")
                    return {'items': items}
            
            except Exception as e:
                self._log(f"Strategy {idx} failed", "warning")
                continue
        
        raise Exception(
            f"❌ No results for '{query_original}' in {city}\n\n"
            f"💡 Try:\n"
            f"• Exact name from Google Maps\n"
            f"• Add details\n"
            f"• Use full address"
        )
    
    def _extract_items(self, result):
        try:
            tasks = result.get('tasks', [])
            if not tasks:
                return None
            
            task = tasks[0]
            
            if task.get('status_code') != 20000:
                error_msg = task.get('status_message', '')
                if "No Search Results" in error_msg:
                    return None
                raise Exception(error_msg)
            
            task_result = task.get('result')
            if not task_result:
                return None
            
            items = task_result[0].get('items', [])
            return items if items else None
        
        except Exception as e:
            raise
    
    def _clean_query(self, query):
        legal = ['srl', 's.r.l.', 'spa', 's.p.a.', 'snc', 's.n.c.',
                'unipersonale', 'società', 'azienda', 'impresa', 'ditta']
        
        query_lower = query.lower()
        for form in legal:
            query_lower = re.sub(rf'\b{form}\b', '', query_lower, flags=re.IGNORECASE)
        
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()
        
        return query_clean
    
    def get_reviews(self, place_id, limit=100):
    """ESTRAZIONE RECENSIONI - FIX DEFINITIVO"""
    
    self._log(f"=== GET REVIEWS ===")
    self._log(f"Place ID: {place_id}")
    self._log(f"Limit: {limit}")
    
    # STEP 1: Crea task
    endpoint_post = "business_data/google/reviews/task_post"
    
    payload = [{
        "place_id": place_id,
        "language_code": "it",
        "depth": min(limit, 500),
        "sort_by": "newest"
    }]
    
    self._log("📤 Creating task...")
    result = self._make_request(endpoint_post, payload, method="POST")
    
    tasks = result.get('tasks', [])
    if not tasks:
        raise Exception("No task created")
    
    task_id = tasks[0].get('id')
    self._log(f"✅ Task created: {task_id}", "success")
    
    # STEP 2: ASPETTA PRIMA DI CONTROLLARE (importante!)
    initial_wait = 5
    self._log(f"⏳ Initial wait {initial_wait}s before checking...")
    time.sleep(initial_wait)
    
    # STEP 3: Polling con task_get
    self._log("🔄 Starting polling...")
    
    max_attempts = 60
    wait_time = 2
    
    for attempt in range(max_attempts):
        time.sleep(wait_time)
        
        endpoint_get = f"business_data/google/reviews/task_get/{task_id}"
        
        try:
            get_result = self._make_request(endpoint_get, data=None, method="GET")
            
            if get_result.get('status_code') == 20000:
                get_tasks = get_result.get('tasks', [])
                
                if get_tasks:
                    task_status = get_tasks[0]
                    
                    # Completato con successo
                    if task_status.get('status_code') == 20000:
                        result_data = task_status.get('result')
                        
                        if result_data and len(result_data) > 0:
                            items = result_data[0].get('items', [])
                            
                            if items:
                                elapsed = initial_wait + (attempt + 1) * wait_time
                                self._log(f"✅ {len(items)} reviews in {elapsed}s!", "success")
                                return result_data[0]
                            else:
                                self._log("⚠️ No reviews found", "warning")
                                return {'items': []}
                    
                    # In processing (40000)
                    elif task_status.get('status_code') == 40000:
                        if self.debug and attempt % 10 == 0:
                            self._log(f"⏳ Processing... ({attempt+1}/{max_attempts})")
                        continue
                    
                    # Task con errore
                    else:
                        error_msg = task_status.get('status_message', 'Unknown error')
                        
                        # Se "Task Not Found" nei primi tentativi, continua
                        if "Task Not Found" in error_msg and attempt < 20:
                            if self.debug and attempt % 5 == 0:
                                self._log(f"⏳ Task not ready yet... ({attempt+1}/20)")
                            continue
                        
                        # Altri errori o timeout su "Task Not Found"
                        raise Exception(f"Task error: {error_msg}")
        
        except Exception as e:
            error_str = str(e)
            
            # Se "Task Not Found" nei primi 20 tentativi, continua (task non ancora pronto)
            if "Task Not Found" in error_str and attempt < 20:
                if self.debug and attempt % 5 == 0:
                    self._log(f"⏳ Waiting for task to be ready... ({attempt+1}/20)")
                continue
            
            # Se errore task dopo 20 tentativi, solleva
            if "Task error:" in error_str:
                raise
            
            # Altri errori di connessione
            if self.debug and attempt % 10 == 0:
                self._log(f"⚠️ Attempt {attempt+1}: {error_str[:100]}", "warning")
            continue
    
    # Timeout
    total_time = initial_wait + (max_attempts * wait_time)
    raise Exception(
        f"❌ Timeout after {total_time}s\n\n"
        f"💡 Possible causes:\n"
        f"• DataForSEO processing queue overload\n"
        f"• Place ID has no accessible reviews\n"
        f"• Temporary API issue\n\n"
        f"Try again in 2-3 minutes."
    )

# 🔧 PROCESSING
@st.cache_data
def get_stopwords():
    return set([
        "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su", "per",
        "tra", "fra", "un", "una", "uno", "e", "ma", "anche", "come", "che", "non",
        "più", "meno", "molto", "poco", "tutto", "tutti", "tutte", "questo", "questa",
        "questi", "queste", "quello", "quella", "quelli", "quelle", "sono", "è", "ho",
        "hai", "ha", "hanno", "essere", "avere", "fare", "dire", "andare", "del", "della",
        "dei", "delle", "dal", "dalla", "dai", "dalle", "nel", "nella", "nei", "nelle",
        "sul", "sulla", "sui", "sulle", "al", "alla", "ai", "alle", "ottimo",
        "buono", "buona", "bene", "male", "servizio", "prodotto", "azienda", "sempre",
        "google", "maps", "recensione", "recensioni", "stelle", "mese", "anno"
    ])

def pulisci_testo(testo):
    if not testo:
        return ""
    
    stopwords = get_stopwords()
    testo = str(testo).lower()
    
    testo = re.sub(r'\d{1,2}\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}', '', testo)
    testo = re.sub(r'\d+\s+(giorn[oi]|settiman[ae]|mes[ie]|ann[oi])\s+fa', '', testo)
    testo = re.sub(r'[1-5]\s*stelle?', '', testo)
    testo = re.sub(r'[^\w\s]', ' ', testo)
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    
    parole = testo.split()
    parole_filtrate = [p for p in parole if p not in stopwords and len(p) > 2]
    return " ".join(parole_filtrate)

def processa_recensioni_dataforseo(items_api):
    recensioni = []
    
    for item in items_api:
        review_text = item.get('review_text') or item.get('text') or item.get('snippet', '')
        
        if not review_text:
            continue
        
        rating_obj = item.get('rating', {})
        if isinstance(rating_obj, dict):
            rating = rating_obj.get('value', 0)
        else:
            rating = rating_obj or 0
        
        timestamp = item.get('timestamp')
        review_date = None
        if timestamp:
            try:
                review_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            except:
                pass
        
        owner_response = None
        responses = item.get('responses', [])
        if responses:
            owner_response = responses[0].get('text', '')
        
        recensione = {
            'testo': review_text,
            'testo_pulito': pulisci_testo(review_text),
            'rating': int(rating) if rating else 0,
            'data': review_date,
            'autore': item.get('author_name') or item.get('author', 'Anonimo'),
            'risposta_owner': owner_response,
            'ha_foto': bool(item.get('images')),
            'link': item.get('url', '#'),
            'review_id': item.get('review_id', '')
        }
        
        recensioni.append(recensione)
    
    return recensioni

def clusterizza_recensioni(recensioni_data, n_clusters=None):
    if len(recensioni_data) < 5:
        return recensioni_data, []
    
    testi = [r['testo_pulito'] for r in recensioni_data if r.get('testo_pulito')]
    
    if not testi or len(testi) < 3:
        return recensioni_data, []
    
    vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8, token_pattern=r'\b[a-zA-Z]{3,}\b')
    
    try:
        X = vectorizer.fit_transform(testi)
    except:
        return recensioni_data, []
    
    if n_clusters is None:
        n_clusters = min(8, max(3, len(recensioni_data) // 10))
    
    n_clusters = min(n_clusters, len(testi))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    idx = 0
    for rec in recensioni_data:
        if rec.get('testo_pulito'):
            rec['cluster'] = int(cluster_labels[idx])
            idx += 1
    
    feature_names = vectorizer.get_feature_names_out()
    cluster_topics = []
    
    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices if len(feature_names[idx]) > 3]
        
        cluster_reviews = [r for r in recensioni_data if r.get('cluster') == i]
        
        if cluster_reviews:
            cluster_info = {
                'id': i,
                'parole_chiave': top_words[:5],
                'n_recensioni': len(cluster_reviews),
                'percentuale': (len(cluster_reviews) / len(recensioni_data)) * 100,
                'rating_medio': np.mean([r['rating'] for r in cluster_reviews if r.get('rating')]),
                'recensioni': cluster_reviews[:3]
            }
            cluster_topics.append(cluster_info)
    
    cluster_topics.sort(key=lambda x: x['n_recensioni'], reverse=True)
    
    return recensioni_data, cluster_topics

def analizza_risposte_owner(recensioni_data):
    recensioni_con_risposta = [r for r in recensioni_data if r.get('risposta_owner')]
    
    if not recensioni_data:
        return {}
    
    tasso_risposta = (len(recensioni_con_risposta) / len(recensioni_data)) * 100
    
    risposte_per_rating = {}
    for rating in range(1, 6):
        recensioni_rating = [r for r in recensioni_data if r.get('rating') == rating]
        con_risposta = [r for r in recensioni_rating if r.get('risposta_owner')]
        if recensioni_rating:
            risposte_per_rating[rating] = {
                'totali': len(recensioni_rating),
                'con_risposta': len(con_risposta),
                'percentuale': (len(con_risposta) / len(recensioni_rating)) * 100
            }
    
    return {
        'tasso_risposta': tasso_risposta,
        'n_risposte': len(recensioni_con_risposta),
        'risposte_per_rating': risposte_per_rating,
        'esempi_risposte': recensioni_con_risposta[:5]
    }

def analizza_trend_temporale(recensioni_data):
    if not recensioni_data:
        return {}
    
    recensioni_con_data = [r for r in recensioni_data if r.get('data')]
    
    if not recensioni_con_data:
        return {}
    
    trend_mensile = {}
    for rec in recensioni_con_data:
        try:
            data = datetime.strptime(rec['data'], "%Y-%m-%d")
            mese_anno = data.strftime("%Y-%m")
            
            if mese_anno not in trend_mensile:
                trend_mensile[mese_anno] = {'count': 0, 'rating_sum': 0, 'ratings': []}
            
            trend_mensile[mese_anno]['count'] += 1
            trend_mensile[mese_anno]['rating_sum'] += rec['rating']
            trend_mensile[mese_anno]['ratings'].append(rec['rating'])
        except:
            continue
    
    for mese in trend_mensile:
        trend_mensile[mese]['rating_medio'] = trend_mensile[mese]['rating_sum'] / trend_mensile[mese]['count']
    
    return dict(sorted(trend_mensile.items()))

def analizza_frequenza_temi(risultati, recensioni_data):
    frequenze = {'punti_forza': {}, 'punti_debolezza': {}}
    
    for punto in risultati.get('punti_forza', []):
        count = 0
        esempi = []
        ids_usati = set()
        
        parole = [p for p in punto.lower().split() if len(p) > 3][:3]
        
        for rec in recensioni_data:
            if rec.get('rating') and rec['rating'] >= 4:
                rec_id = f"{rec.get('autore', '')}_{rec.get('data', '')}"
                
                if rec_id not in ids_usati:
                    testo_lower = rec.get('testo_pulito', '').lower()
                    matches = sum(1 for p in parole if p in testo_lower)
                    
                    if matches >= min(2, len(parole)):
                        count += 1
                        ids_usati.add(rec_id)
                        
                        if len(esempi) < 2:
                            esempi.append(rec)
        
        if count > 0:
            recensioni_positive = [r for r in recensioni_data if r.get('rating') and r['rating'] >= 4]
            frequenze['punti_forza'][punto] = {
                'count': count,
                'percentuale': (count / len(recensioni_positive)) * 100 if recensioni_positive else 0,
                'esempi': esempi
            }
    
    for punto in risultati.get('punti_debolezza', []):
        count = 0
        esempi = []
        ids_usati = set()
        
        parole = [p for p in punto.lower().split() if len(p) > 3][:3]
        
        for rec in recensioni_data:
            if rec.get('rating') and rec['rating'] <= 2:
                rec_id = f"{rec.get('autore', '')}_{rec.get('data', '')}"
                
                if rec_id not in ids_usati:
                    testo_lower = rec.get('testo_pulito', '').lower()
                    matches = sum(1 for p in parole if p in testo_lower)
                    
                    if matches >= min(2, len(parole)):
                        count += 1
                        ids_usati.add(rec_id)
                        
                        if len(esempi) < 2:
                            esempi.append(rec)
        
        if count > 0:
            recensioni_negative = [r for r in recensioni_data if r.get('rating') and r['rating'] <= 2]
            frequenze['punti_debolezza'][punto] = {
                'count': count,
                'percentuale': (count / len(recensioni_negative)) * 100 if recensioni_negative else 0,
                'esempi': esempi
            }
    
    frequenze['punti_forza'] = dict(sorted(frequenze['punti_forza'].items(), key=lambda x: x[1]['count'], reverse=True))
    frequenze['punti_debolezza'] = dict(sorted(frequenze['punti_debolezza'].items(), key=lambda x: x[1]['count'], reverse=True))
    
    return frequenze

def analizza_blocchi_con_ai(blocchi, client, progress_bar, status_text):
    risultati = {
        "punti_forza": [], "punti_debolezza": [], "leve_marketing": [],
        "parole_chiave": [], "suggerimenti_local_seo": [], "suggerimenti_reputation": [],
        "suggerimenti_google_ads": [], "suggerimenti_cro": [], "suggerimenti_risposte": [],
        "sentiment_distribution": {"positivo": 0, "neutro": 0, "negativo": 0}
    }

    for i, blocco in enumerate(blocchi):
        status_text.text(f"🤖 AI {i+1}/{len(blocchi)}...")

        prompt = f"""
Analizza recensioni:
{blocco}

JSON:
{{
    "punti_forza": ["punto"],
    "punti_debolezza": ["problema"],
    "leve_marketing": ["leva"],
    "parole_chiave": ["keyword"],
    "suggerimenti_local_seo": ["seo"],
    "suggerimenti_reputation": ["rep"],
    "suggerimenti_google_ads": ["ads"],
    "suggerimenti_cro": ["cro"],
    "suggerimenti_risposte": ["template"],
    "sentiment_counts": {{"positivo": N, "neutro": N, "negativo": N}}
}}
"""

        for tentativo in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000
                )

                content = response.choices[0].message.content
                content_cleaned = re.sub(r"```json\n?|```", "", content).strip()
                dati = json.loads(content_cleaned)

                for chiave in risultati:
                    if chiave in dati:
                        if chiave == "sentiment_counts":
                            for sent_type in ['positivo', 'neutro', 'negativo']:
                                if sent_type in dati['sentiment_counts']:
                                    risultati['sentiment_distribution'][sent_type] += dati['sentiment_counts'][sent_type]
                        else:
                            nuovi = [elem for elem in dati[chiave] if elem not in risultati[chiave]]
                            risultati[chiave].extend(nuovi)
                break
            except:
                if tentativo < 2:
                    time.sleep(2)

        progress_bar.progress((i + 1) / len(blocchi))

    for chiave in risultati:
        if chiave != 'sentiment_distribution':
            risultati[chiave] = list(dict.fromkeys(risultati[chiave]))
    
    return risultati

def mostra_esempi_recensioni(tema, esempi, tipo="positivo"):
    if not esempi:
        return
    
    st.markdown(f"**Esempi:**")
    
    for esempio in esempi[:2]:
        rating_stars = "⭐" * esempio.get('rating', 3)
        testo_breve = esempio.get('testo', '')[:200]
        if len(esempio.get('testo', '')) > 200:
            testo_breve += "..."
        
        css_class = "positive-review" if tipo == "positivo" else "negative-review"
        
        risposta_html = ""
        if esempio.get('risposta_owner'):
            risposta_html = f"""
            <div class="owner-response">
                <strong>💬 Risposta:</strong>
                <p>{esempio['risposta_owner'][:150]}...</p>
            </div>
            """
        
        st.markdown(f"""
        <div class="review-example {css_class}">
            <div><strong>{rating_stars}</strong> - {esempio.get('autore', 'Anonimo')} • {esempio.get('data', 'N/A')}</div>
            <div>{testo_breve}</div>
            {risposta_html}
        </div>
        """, unsafe_allow_html=True)

def crea_excel_download(recensioni_data, risultati, clusters, frequenze, analisi_owner, trend, business_info):
    output = io.BytesIO()
    
    df_recensioni = pd.DataFrame([{
        'Testo': r.get('testo', ''),
        'Rating': r.get('rating', 0),
        'Data': r.get('data', ''),
        'Autore': r.get('autore', ''),
        'Risposta Owner': 'Sì' if r.get('risposta_owner') else 'No',
        'Link': r.get('link', '')
    } for r in recensioni_data])
    
    df_business = pd.DataFrame([business_info])
    
    df_clusters = pd.DataFrame([{
        'Cluster': c['id'],
        'Tematiche': ', '.join(c['parole_chiave']),
        'N. Recensioni': c['n_recensioni'],
        'Percentuale': f"{c['percentuale']:.1f}%"
    } for c in clusters])
    
    df_forza = pd.DataFrame([{
        'Punto Forza': p,
        'Frequenza': d['count'],
        'Percentuale': f"{d['percentuale']:.1f}%"
    } for p, d in frequenze['punti_forza'].items()])
    
    df_debolezza = pd.DataFrame([{
        'Punto Debolezza': p,
        'Frequenza': d['count'],
        'Percentuale': f"{d['percentuale']:.1f}%"
    } for p, d in frequenze['punti_debolezza'].items()])
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not df_business.empty:
            df_business.to_excel(writer, sheet_name='Business Info', index=False)
        df_recensioni.to_excel(writer, sheet_name='Recensioni', index=False)
        if not df_clusters.empty:
            df_clusters.to_excel(writer, sheet_name='Clusters', index=False)
        if not df_forza.empty:
            df_forza.to_excel(writer, sheet_name='Punti Forza', index=False)
        if not df_debolezza.empty:
            df_debolezza.to_excel(writer, sheet_name='Punti Debolezza', index=False)
    
    return output.getvalue()

# 🎮 MAIN
def main():
    with st.sidebar:
        st.markdown("## 🔧 Config")
        
        st.markdown("### 🔑 Keys")
        api_key_openai = st.text_input("OpenAI", type="password")
        dataforseo_username = st.text_input("DataForSEO User")
        dataforseo_password = st.text_input("DataForSEO Pass", type="password")
        
        st.markdown("---")
        st.markdown("### 🏢 Business")
        
        nome_attivita = st.text_input("Nome", placeholder="Es: Moca Interactive")
        
        if nome_attivita and len(nome_attivita) < 5:
            st.warning("⚠️ Troppo corto")
        
        location = st.text_input("Città/Indirizzo", placeholder="Es: Treviso")
        
        if location:
            if ',' in location or any(x in location.lower() for x in ['via', 'viale']):
                st.info("📍 Indirizzo")
            else:
                normalized = normalizza_nome_citta(location)
                if normalized:
                    st.success(f"✅ {normalized.title()}")
                else:
                    st.info(f"🌐 API search")
        
        max_reviews = st.slider("Recensioni", 50, 500, 100, 50)
        n_clusters = st.slider("Cluster", 3, 15, 8)
        
        st.markdown("---")
        debug_mode = st.checkbox("🐛 Debug", value=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Analisi")
        
        if st.button("🔍 Avvia", type="primary", use_container_width=True):
            
            if not all([api_key_openai, dataforseo_username, dataforseo_password, nome_attivita, location]):
                st.error("❌ Compila tutto")
                return
            
            try:
                client_openai = OpenAI(api_key=api_key_openai)
                client_dataforseo = DataForSEOClient(dataforseo_username, dataforseo_password, debug=debug_mode)
                
                # FASE 1
                st.markdown("### 🔍 Search")
                business_result = client_dataforseo.search_business(nome_attivita, location)
                
                if not business_result or not business_result.get('items'):
                    st.error("❌ Not found")
                    return
                
                business = business_result['items'][0]
                place_id = business.get('place_id') or business.get('cid', '')
                
                business_info = {
                    'place_id': place_id,
                    'nome': business.get('title', nome_attivita),
                    'indirizzo': business.get('address', location),
                    'rating_medio': business.get('rating', {}).get('value', 0) if isinstance(business.get('rating'), dict) else business.get('rating', 0),
                    'n_recensioni': business.get('rating', {}).get('votes_count', 0) if isinstance(business.get('rating'), dict) else business.get('reviews_count', 0),
                    'categoria': business.get('category', 'N/A')
                }
                
                st.markdown(f"""
                <div class="business-card">
                    <h3>🏢 {business_info['nome']}</h3>
                    <p>📍 {business_info['indirizzo']}</p>
                    <p>⭐ {business_info['rating_medio']}/5 ({business_info['n_recensioni']} recensioni)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # FASE 2
                st.markdown("### 📥 Reviews")
                reviews_result = client_dataforseo.get_reviews(place_id, max_reviews)
                
                if not reviews_result or not reviews_result.get('items'):
                    st.error("❌ No reviews")
                    return
                
                recensioni_data = processa_recensioni_dataforseo(reviews_result['items'])
                
                st.success(f"✅ {len(recensioni_data)}")
                
                rating_medio = np.mean([r['rating'] for r in recensioni_data if r['rating']]) if recensioni_data else 0
                n_con_risposta = len([r for r in recensioni_data if r.get('risposta_owner')])
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("⭐", f"{rating_medio:.1f}")
                with col_s2:
                    st.metric("💬", n_con_risposta)
                with col_s3:
                    st.metric("📊", f"{(n_con_risposta/len(recensioni_data)*100):.0f}%")
                
                # FASE 3
                st.markdown("### 🎨 Clustering")
                recensioni_data, clusters = clusterizza_recensioni(recensioni_data, n_clusters)
                st.success(f"✅ {len(clusters)}")
                
                # FASE 4-5
                st.markdown("### 💬 Owner")
                analisi_owner = analizza_risposte_owner(recensioni_data)
                
                st.markdown("### 📈 Trend")
                trend_temporale = analizza_trend_temporale(recensioni_data)
                
                # FASE 6
                st.markdown("### 📝 Prep")
                recensioni_pulite = [r['testo_pulito'] for r in recensioni_data if r.get('testo_pulito')]
                testo_completo = " ".join(recensioni_pulite)
                parole = testo_completo.split()
                blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
                st.info(f"{len(blocchi)} blocchi")
                
                # FASE 7
                st.markdown("### 🤖 AI")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                risultati = analizza_blocchi_con_ai(blocchi, client_openai, progress_bar, status_text)
                
                # FASE 8
                st.markdown("### 📊 Freq")
                frequenze = analizza_frequenza_temi(risultati, recensioni_data)
                
                st.markdown('<div class="success-box"><h3>🎉 Done!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI
                st.markdown("## 📊 Results")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("📝", len(recensioni_data))
                with col_m2:
                    st.metric("💪", len(risultati.get('punti_forza', [])))
                with col_m3:
                    st.metric("⚠️", len(risultati.get('punti_debolezza', [])))
                with col_m4:
                    st.metric("🎯", len(clusters))
                
                tab1, tab2, tab3 = st.tabs(["💪", "⚠️", "🎨"])
                
                with tab1:
                    if frequenze['punti_forza']:
                        for punto, dati in list(frequenze['punti_forza'].items())[:10]:
                            st.markdown(f"""
                            **{punto}**
                            <span class="frequency-badge">{dati['percentuale']:.1f}%</span>
                            """, unsafe_allow_html=True)
                            if dati['esempi']:
                                with st.expander("Esempi"):
                                    mostra_esempi_recensioni(punto, dati['esempi'], "positivo")
                
                with tab2:
                    if frequenze['punti_debolezza']:
                        for punto, dati in list(frequenze['punti_debolezza'].items())[:10]:
                            st.markdown(f"""
                            **{punto}**
                            <span class="frequency-badge" style="background: #EA4335;">{dati['percentuale']:.1f}%</span>
                            """, unsafe_allow_html=True)
                            if dati['esempi']:
                                with st.expander("Esempi"):
                                    mostra_esempi_recensioni(punto, dati['esempi'], "negativo")
                
                with tab3:
                    for cluster in clusters:
                        with st.expander(f"{cluster['id']+1}: {', '.join(cluster['parole_chiave'][:3])}"):
                            st.write(f"**N:** {cluster['n_recensioni']} ({cluster['percentuale']:.1f}%)")
                            st.write(f"**⭐:** {cluster['rating_medio']:.1f}")
                
                # DOWNLOAD
                st.markdown("## 📥 Export")
                
                excel_data = crea_excel_download(
                    recensioni_data, risultati, clusters, 
                    frequenze, analisi_owner, trend_temporale, business_info
                )
                
                st.download_button(
                    "📊 Excel",
                    excel_data,
                    f"Reviews_{business_info['nome'].replace(' ', '_')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ {str(e)}")
                if debug_mode:
                    st.exception(e)
    
    with col2:
        st.markdown("## 📋 Info")
        st.markdown("""
        ### ✅ Sistema:
        • 6 strategie search
        • task_get corretto
        • 100+ città
        • Indirizzo OK
        
        ### ⏱️ Tempi:
        50: ~4min
        100: ~7min
        200+: ~12min
        """)

if __name__ == "__main__":
    main()
