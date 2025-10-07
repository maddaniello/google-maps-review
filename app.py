# 🚀 ANALIZZATORE GOOGLE REVIEWS - VERSIONE CON GESTIONE CODA
# Sistema di priorità e attesa svuotamento coda completo

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
from datetime import datetime
import requests
import base64
warnings.filterwarnings('ignore')

# 🎨 CONFIGURAZIONE
st.set_page_config(
    page_title="🚀 Analizzatore Google Reviews Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌍 DATABASE LOCATION CODES
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

# 🎯 CSS STYLING
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
    .warning-box {
        background: linear-gradient(135deg, #FBBC05 0%, #F57C00 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .business-card {
        background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .queue-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .queue-empty { background: #E8F5E9; color: #2E7D32; }
    .queue-busy { background: #FFF3E0; color: #E65100; }
    .queue-full { background: #FFEBEE; color: #C62828; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 Analizzatore Google Reviews Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Sistema con Gestione Coda Avanzata</h3>
    <p>✅ Controllo coda prima dell'estrazione</p>
    <p>✅ Attesa automatica svuotamento</p>
    <p>✅ Gestione status code 40602 (In Queue)</p>
    <p>✅ Polling intelligente fino a 15 minuti</p>
    <p>✅ Priorità alta opzionale (costo extra)</p>
</div>
""", unsafe_allow_html=True)

# 🔧 HELPER FUNCTIONS
def normalizza_nome_citta(nome_citta):
    if not nome_citta:
        return None
    nome_clean = nome_citta.lower().strip()
    nome_clean = re.sub(r'\s+', ' ', nome_clean)
    return nome_clean if nome_clean in LOCATION_CODES_ITALY else None

# 🔧 CLASSE DATAFORSEO CON GESTIONE CODA COMPLETA
class DataForSEOClient:
    
    def __init__(self, username, password, debug=False):
        self.username = username
        self.password = password
        self.base_url = "https://api.dataforseo.com/v3"
        self.debug = debug
        self._location_cache = {}
    
    def _log(self, message, level="info"):
        """Log messaggi con livelli"""
        if self.debug:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌"
            }.get(level, "ℹ️")
            
            st.write(f"`[{timestamp}]` {prefix} {message}")
    
    def _get_auth_token(self):
        credentials = f"{self.username}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()
    
    def _make_request(self, endpoint, data=None, method="POST"):
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self._get_auth_token()}'
        }
        
        self._log(f"API: {method} /{endpoint}")
        
        try:
            if method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method == "GET":
                response = requests.get(url, headers=headers, params=data, timeout=30)
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('status_code') == 20000:
                return result
            else:
                error_msg = result.get('status_message', 'Unknown error')
                raise Exception(f"API Error {result.get('status_code')}: {error_msg}")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection Error: {str(e)}")
    
    def get_tasks_ready(self):
        """
        Ottiene lista task pronti/in coda
        Usa endpoint tasks_ready per vedere cosa c'è in coda
        """
        endpoint = "business_data/google/reviews/tasks_ready"
        
        try:
            result = self._make_request(endpoint, data=None, method="GET")
            
            if result.get('status_code') == 20000:
                tasks = result.get('tasks', [])
                if tasks and len(tasks) > 0:
                    task_result = tasks[0].get('result', [])
                    return task_result
                return []
            return []
        
        except Exception as e:
            self._log(f"Errore controllo coda: {str(e)}", "warning")
            return []
    
    def wait_for_queue_clear(self, max_wait_minutes=5):
        """
        Attende che la coda si svuoti COMPLETAMENTE
        Controlla tasks_ready ogni 10 secondi
        """
        max_attempts = max_wait_minutes * 6  # 10s * 6 = 1 minuto
        
        self._log("🔍 Controllo coda tasks_ready...", "info")
        
        for attempt in range(max_attempts):
            tasks_ready = self.get_tasks_ready()
            n_tasks = len(tasks_ready)
            
            if n_tasks == 0:
                self._log("✅ Coda vuota! Posso procedere.", "success")
                return True
            
            # Mostra status
            if attempt == 0 or attempt % 6 == 0:  # Ogni minuto
                minutes_elapsed = attempt / 6
                st.warning(f"⏳ {n_tasks} task in coda. Attesa: {minutes_elapsed:.1f}/{max_wait_minutes} min")
                self._log(f"⏳ {n_tasks} task in coda, attendo...", "warning")
            
            time.sleep(10)
        
        # Timeout
        tasks_ready = self.get_tasks_ready()
        n_tasks = len(tasks_ready)
        
        st.error(f"⏱️ Timeout dopo {max_wait_minutes} minuti. {n_tasks} task ancora in coda.")
        
        return False
    
    def get_location_code(self, location_name):
        if location_name in self._location_cache:
            return self._location_cache[location_name]
        
        self._log(f"🔍 Ricerca location: '{location_name}'")
        
        # Check database locale
        nome_normalizzato = normalizza_nome_citta(location_name)
        if nome_normalizzato and nome_normalizzato in LOCATION_CODES_ITALY:
            code = LOCATION_CODES_ITALY[nome_normalizzato]
            self._log(f"✅ Trovato: {nome_normalizzato.title()} ({code})", "success")
            self._location_cache[location_name] = code
            return code
        
        # Fallback API
        self._log("🌐 Ricerca via API...")
        try:
            endpoint = "business_data/google/locations"
            search_queries = [location_name, f"{location_name}, Italia", f"{location_name}, Italy"]
            
            for query in search_queries:
                params = {'location_name': query}
                result = self._make_request(endpoint, params, method="GET")
                
                tasks = result.get('tasks', [])
                if tasks and tasks[0].get('result'):
                    locations = tasks[0]['result']
                    
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
            self._log(f"⚠️ Errore API: {e}", "warning")
        
        self._log("⚠️ Fallback: Italia (2380)", "warning")
        return 2380
    
    def search_business(self, query, location):
        """Cerca business - implementazione precedente"""
        self._log("=" * 60)
        self._log("🔍 RICERCA BUSINESS")
        self._log(f"Query: '{query}'")
        self._log(f"Location: '{location}'")
        
        query_clean = self._clean_query(query)
        is_full_address = self._is_full_address(location)
        
        if is_full_address:
            self._log("📍 Modalità: indirizzo completo")
            return self._search_by_address(query, query_clean, location)
        else:
            self._log("🏙️ Modalità: città")
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
            self._log(f"📍 Strategia {idx}/{len(strategies)}")
            
            payload = [{**strategy, "language_code": "it"}]
            
            try:
                result = self._make_request(endpoint, payload)
                items = self._extract_items(result)
                
                if items:
                    self._log(f"✅ Business trovato!", "success")
                    return {'items': items}
            
            except:
                continue
        
        raise Exception(f"Business non trovato con indirizzo '{address}'")
    
    def _search_by_city(self, query_original, query_clean, city):
        location_code = self.get_location_code(city)
        
        if not location_code:
            raise Exception(f"Location code non trovato per '{city}'")
        
        endpoint = "business_data/google/my_business_info/live"
        
        strategies = [
            {"keyword": f"{query_original} {city}", "location_code": location_code},
            {"keyword": f"{query_clean} {city}", "location_code": location_code},
            {"keyword": query_original, "location_code": location_code},
        ]
        
        for idx, strategy in enumerate(strategies, 1):
            self._log(f"🏙️ Strategia {idx}/{len(strategies)}: '{strategy['keyword']}'")
            
            payload = [{**strategy, "language_code": "it"}]
            
            try:
                result = self._make_request(endpoint, payload)
                items = self._extract_items(result)
                
                if items:
                    self._log(f"✅ Business trovato!", "success")
                    return {'items': items, 'location_code': location_code}
            
            except:
                continue
        
        raise Exception(f"Business non trovato per '{query_original}' in {city}")
    
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
        
        except:
            raise
    
    def _clean_query(self, query):
        legal_forms = ['srl', 's.r.l.', 'spa', 's.p.a.', 'snc', 's.n.c.',
                      'unipersonale', 'società', 'azienda', 'impresa', 'ditta']
        
        query_lower = query.lower()
        for form in legal_forms:
            query_lower = re.sub(rf'\b{form}\b', '', query_lower, flags=re.IGNORECASE)
        
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()
        
        return query_clean
    
    def get_reviews(self, place_id, location_code, limit=100, use_priority=False):
        """
        ESTRAZIONE RECENSIONI CON GESTIONE CODA COMPLETA
        
        Args:
            place_id: ID del business
            location_code: Codice location
            limit: Numero recensioni
            use_priority: Se True usa priorità alta (COSTO EXTRA)
        """
        self._log("=" * 60)
        self._log("📥 ESTRAZIONE RECENSIONI CON GESTIONE CODA")
        self._log(f"Place ID: {place_id}")
        self._log(f"Location Code: {location_code}")
        self._log(f"Limit: {limit}")
        self._log(f"Priority: {'ALTA (extra cost)' if use_priority else 'NORMALE'}")
        
        # STEP 0: CONTROLLA CODA E ATTENDI SE NECESSARIO
        st.info("🔍 Controllo coda DataForSEO...")
        tasks_ready = self.get_tasks_ready()
        n_tasks = len(tasks_ready)
        
        if n_tasks > 0:
            st.warning(f"⚠️ Trovati {n_tasks} task in coda. Attendo svuotamento...")
            self._log(f"⚠️ {n_tasks} task in coda, attendo svuotamento...", "warning")
            
            # Attendi svuotamento
            queue_cleared = self.wait_for_queue_clear(max_wait_minutes=5)
            
            if not queue_cleared:
                raise Exception(
                    f"⏱️ Timeout: coda ancora occupata dopo 5 minuti.\n\n"
                    f"💡 Riprova tra 10 minuti oppure usa priorità alta (costo extra)."
                )
            
            # Attesa extra di sicurezza
            self._log("⏳ Attesa extra 10s per sicurezza...", "info")
            time.sleep(10)
        else:
            st.success("✅ Coda vuota, procedo immediatamente!")
            self._log("✅ Coda vuota, procedo!", "success")
        
        # STEP 1: Crea task recensioni
        endpoint_post = "business_data/google/reviews/task_post"
        
        payload = [{
            "place_id": place_id,
            "location_code": location_code,
            "language_code": "it",
            "depth": min(limit, 500),
            "sort_by": "newest"
        }]
        
        # Aggiungi priorità se richiesto
        if use_priority:
            payload[0]["priority"] = 2
            self._log("⚡ Usando priorità ALTA (costo extra)", "warning")
        
        self._log("📤 Creazione task recensioni...")
        
        try:
            result = self._make_request(endpoint_post, payload, method="POST")
            
            tasks = result.get('tasks', [])
            if not tasks:
                raise Exception("Nessun task creato")
            
            task = tasks[0]
            task_id = task.get('id')
            task_status = task.get('status_code')
            
            if task_status == 20100:
                self._log(f"✅ Task creato: {task_id}", "success")
            elif task_status in [40501, 40502]:
                error_msg = task.get('status_message', 'Parametri invalidi')
                raise Exception(f"Errore parametri: {error_msg}")
            else:
                self._log(f"⚠️ Status {task_status}: {task_id}", "warning")
        
        except Exception as e:
            raise Exception(f"Impossibile creare task: {str(e)}")
        
        # STEP 2: ATTESA INIZIALE (15 secondi come da documentazione)
        initial_wait = 15
        self._log(f"⏳ Attesa iniziale obbligatoria: {initial_wait}s...", "info")
        
        progress_placeholder = st.empty()
        for i in range(initial_wait):
            progress_placeholder.progress(
                (i + 1) / initial_wait, 
                text=f"⏳ Attesa tecnica: {i+1}/{initial_wait}s"
            )
            time.sleep(1)
        progress_placeholder.empty()
        
        # STEP 3: POLLING CON GESTIONE 40602
        self._log("🔄 Inizio polling risultati...", "info")
        
        max_attempts = 180  # 15 minuti
        wait_intervals = [5, 7, 10, 15, 20]  # Exponential backoff
        attempt = 0
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        while attempt < max_attempts:
            attempt += 1
            
            # Calcola intervallo
            interval_idx = min(attempt // 10, len(wait_intervals) - 1)
            wait_time = wait_intervals[interval_idx]
            
            time.sleep(wait_time)
            
            # Update UI
            elapsed_minutes = (attempt * wait_time + initial_wait) / 60
            progress = min(attempt / max_attempts, 0.99)
            progress_placeholder.progress(
                progress, 
                text=f"⏳ Polling: {attempt}/{max_attempts} (~{elapsed_minutes:.1f} min)"
            )
            
            try:
                # Check task
                endpoint_get = f"business_data/google/reviews/task_get/{task_id}"
                result_get = self._make_request(endpoint_get, method="GET")
                
                tasks_result = result_get.get('tasks', [])
                if not tasks_result:
                    self._log("⏳ Task non ancora disponibile")
                    continue
                
                task_data = tasks_result[0]
                current_status = task_data.get('status_code')
                status_msg = task_data.get('status_message', '')
                
                # Gestione status code
                if current_status == 20000:
                    # SUCCESS
                    result_data = task_data.get('result')
                    
                    if result_data and len(result_data) > 0:
                        items = result_data[0].get('items', [])
                        
                        if items:
                            progress_placeholder.empty()
                            status_placeholder.empty()
                            elapsed_total = (attempt * wait_time + initial_wait) / 60
                            self._log(f"✅ {len(items)} recensioni in ~{elapsed_total:.1f} min!", "success")
                            return result_data[0]
                        else:
                            progress_placeholder.empty()
                            status_placeholder.empty()
                            self._log("⚠️ Nessuna recensione trovata", "warning")
                            return {'items': []}
                    else:
                        continue
                
                elif current_status in [40100, 40200, 40300, 40602]:
                    # IN ELABORAZIONE / IN QUEUE
                    # 40602 = "Task In Queue" (non documentato ma esistente)
                    status_placeholder.text(f"⏳ {status_msg}")
                    if self.debug and attempt % 10 == 0:
                        self._log(f"⏳ Status {current_status}: {status_msg}")
                    continue
                
                elif current_status == 40000:
                    raise Exception(f"Task non trovato: {task_id}")
                
                elif current_status == 40400:
                    raise Exception(f"Task fallito: {status_msg}")
                
                else:
                    # Status non riconosciuto
                    self._log(f"⚠️ Status {current_status}: {status_msg}", "warning")
                    continue
            
            except Exception as e:
                error_str = str(e)
                
                if "Task fallito" in error_str or "Task non trovato" in error_str:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    raise
                
                if self.debug and attempt % 10 == 0:
                    self._log(f"⚠️ Errore: {error_str[:100]}", "warning")
                continue
        
        # TIMEOUT
        progress_placeholder.empty()
        status_placeholder.empty()
        
        raise Exception(
            f"⏱️ Timeout dopo 15 minuti\n\n"
            f"Il task potrebbe ancora essere in elaborazione.\n\n"
            f"💡 Opzioni:\n"
            f"• Riprova tra 10 minuti\n"
            f"• Usa priorità alta (costo extra)\n"
            f"• Riduci numero recensioni"
        )

# [RESTO DEL CODICE IDENTICO - processing functions, clustering, AI, etc.]
# Per brevità ometto le funzioni già corrette precedentemente
# Includi tutte le altre funzioni dal codice precedente

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
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S +00:00")
                review_date = dt.strftime("%Y-%m-%d")
            except:
                pass
        
        owner_response = item.get('owner_answer')
        
        recensione = {
            'testo': review_text,
            'testo_pulito': pulisci_testo(review_text),
            'rating': int(rating) if rating else 0,
            'data': review_date,
            'autore': item.get('profile_name', 'Anonimo'),
            'risposta_owner': owner_response,
            'ha_foto': bool(item.get('images')),
            'link': item.get('review_url', '#'),
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
    
    vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=2,
        max_df=0.8,
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
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

def analizza_frequenza_temi(risultati, recensioni_data):
    frequenze = {
        'punti_forza': {},
        'punti_debolezza': {}
    }
    
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
    
    frequenze['punti_forza'] = dict(sorted(
        frequenze['punti_forza'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    ))
    
    frequenze['punti_debolezza'] = dict(sorted(
        frequenze['punti_debolezza'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    ))
    
    return frequenze

def analizza_blocchi_con_ai(blocchi, client, progress_bar, status_text):
    risultati = {
        "punti_forza": [],
        "punti_debolezza": [],
        "leve_marketing": [],
        "parole_chiave": [],
        "suggerimenti_local_seo": [],
        "suggerimenti_reputation": [],
        "suggerimenti_google_ads": [],
        "suggerimenti_cro": [],
        "suggerimenti_risposte": [],
        "sentiment_distribution": {
            "positivo": 0,
            "neutro": 0,
            "negativo": 0
        }
    }

    for i, blocco in enumerate(blocchi):
        status_text.text(f"🤖 Analisi AI {i+1}/{len(blocchi)}...")

        prompt = f"""
Analizza queste recensioni Google:

{blocco}

Rispondi SOLO con JSON valido:
{{
    "punti_forza": ["punto forza 1", "punto forza 2"],
    "punti_debolezza": ["problema 1", "problema 2"],
    "leve_marketing": ["leva 1", "leva 2"],
    "parole_chiave": ["keyword1", "keyword2"],
    "suggerimenti_local_seo": ["seo 1"],
    "suggerimenti_reputation": ["rep 1"],
    "suggerimenti_google_ads": ["ads 1"],
    "suggerimenti_cro": ["cro 1"],
    "suggerimenti_risposte": ["template risposta 1"],
    "sentiment_counts": {{"positivo": 10, "neutro": 5, "negativo": 2}}
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

def crea_excel_download(recensioni_data, risultati, clusters, frequenze, business_info):
    output = io.BytesIO()
    
    df_business = pd.DataFrame([business_info])
    
    df_recensioni = pd.DataFrame([{
        'Testo': r.get('testo', ''),
        'Rating': r.get('rating', 0),
        'Data': r.get('data', ''),
        'Autore': r.get('autore', ''),
        'Risposta Owner': 'Sì' if r.get('risposta_owner') else 'No',
        'Link': r.get('link', '')
    } for r in recensioni_data])
    
    df_clusters = pd.DataFrame([{
        'Cluster': c['id'],
        'Tematiche': ', '.join(c['parole_chiave']),
        'N. Recensioni': c['n_recensioni'],
        'Percentuale': f"{c['percentuale']:.1f}%"
    } for c in clusters]) if clusters else pd.DataFrame()
    
    df_forza = pd.DataFrame([{
        'Punto Forza': p,
        'Frequenza': d['count'],
        'Percentuale': f"{d['percentuale']:.1f}%"
    } for p, d in frequenze['punti_forza'].items()]) if frequenze['punti_forza'] else pd.DataFrame()
    
    df_debolezza = pd.DataFrame([{
        'Punto Debolezza': p,
        'Frequenza': d['count'],
        'Percentuale': f"{d['percentuale']:.1f}%"
    } for p, d in frequenze['punti_debolezza'].items()]) if frequenze['punti_debolezza'] else pd.DataFrame()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not df_business.empty:
            df_business.to_excel(writer, sheet_name='Business Info', index=False)
        if not df_recensioni.empty:
            df_recensioni.to_excel(writer, sheet_name='Recensioni', index=False)
        if not df_clusters.empty:
            df_clusters.to_excel(writer, sheet_name='Clusters', index=False)
        if not df_forza.empty:
            df_forza.to_excel(writer, sheet_name='Punti Forza', index=False)
        if not df_debolezza.empty:
            df_debolezza.to_excel(writer, sheet_name='Punti Debolezza', index=False)
    
    return output.getvalue()

# 🎮 MAIN APPLICATION
def main():
    with st.sidebar:
        st.markdown("## 🔧 Configurazione")
        
        st.markdown("### 🔑 API Keys")
        api_key_openai = st.text_input("OpenAI API Key", type="password")
        dataforseo_username = st.text_input("DataForSEO Username")
        dataforseo_password = st.text_input("DataForSEO Password", type="password")
        
        st.markdown("---")
        st.markdown("### 🏢 Dati Business")
        
        nome_attivita = st.text_input("Nome Attività", placeholder="Es: Ristorante Da Mario")
        location = st.text_input("Città o Indirizzo", placeholder="Es: Milano")
        
        max_reviews = st.slider("Max Recensioni", 50, 500, 100, 50)
        n_clusters = st.slider("Numero Cluster", 3, 15, 8)
        
        st.markdown("---")
        st.markdown("### ⚡ Opzioni Avanzate")
        
        use_priority = st.checkbox(
            "🚀 Priorità Alta (costo extra)", 
            value=False,
            help="Usa priorità 2 per task più veloce. ATTENZIONE: costi maggiorati!"
        )
        
        debug_mode = st.checkbox("🐛 Debug Mode", value=False)
        
        st.markdown("---")
        
        if use_priority:
            st.warning("""
            ⚠️ **PRIORITÀ ALTA ATTIVA**
            
            Costi maggiorati secondo pricing DataForSEO.
            Il task NON salta la coda ma viene elaborato più velocemente.
            """)
        
        st.info("""
        **📊 Sistema Coda:**
        
        • Controlla coda PRIMA dell'estrazione
        • Attende automaticamente se occupata  
        • Polling fino a 15 minuti
        • Gestisce status 40602 (In Queue)
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Avvia Analisi")
        
        if st.button("🔍 Analizza Recensioni", type="primary", use_container_width=True):
            
            if not all([api_key_openai, dataforseo_username, dataforseo_password, nome_attivita, location]):
                st.error("❌ Compila tutti i campi obbligatori")
                return
            
            try:
                # Init clients
                client_openai = OpenAI(api_key=api_key_openai)
                client_dataforseo = DataForSEOClient(dataforseo_username, dataforseo_password, debug=debug_mode)
                
                # FASE 1: Ricerca Business
                st.markdown("### 🔍 Ricerca Business")
                with st.spinner("Ricerca in corso..."):
                    business_result = client_dataforseo.search_business(nome_attivita, location)
                
                if not business_result or not business_result.get('items'):
                    st.error("❌ Attività non trovata")
                    return
                
                business = business_result['items'][0]
                place_id = business.get('place_id') or business.get('cid', '')
                location_code = business_result.get('location_code')
                
                if not location_code:
                    location_code = LOCATION_CODES_ITALY.get(normalizza_nome_citta(location), 2380)
                
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
                    <p>🏷️ {business_info['categoria']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # FASE 2: Estrazione Recensioni CON GESTIONE CODA
                st.markdown("### 📥 Estrazione Recensioni")
                
                reviews_result = client_dataforseo.get_reviews(
                    place_id, 
                    location_code, 
                    max_reviews,
                    use_priority=use_priority
                )
                
                if not reviews_result or not reviews_result.get('items'):
                    st.error("❌ Nessuna recensione trovata")
                    return
                
                recensioni_data = processa_recensioni_dataforseo(reviews_result['items'])
                
                st.success(f"✅ Estratte {len(recensioni_data)} recensioni")
                
                # Metriche
                rating_medio = np.mean([r['rating'] for r in recensioni_data if r['rating']]) if recensioni_data else 0
                n_con_risposta = len([r for r in recensioni_data if r.get('risposta_owner')])
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("⭐ Rating Medio", f"{rating_medio:.1f}")
                with col_s2:
                    st.metric("💬 Con Risposta", n_con_risposta)
                with col_s3:
                    st.metric("📊 Tasso Risposta", f"{(n_con_risposta/len(recensioni_data)*100):.0f}%")
                
                # FASE 3: Clustering
                st.markdown("### 🎨 Clustering ML")
                with st.spinner("Clustering..."):
                    recensioni_data, clusters = clusterizza_recensioni(recensioni_data, n_clusters)
                st.success(f"✅ {len(clusters)} cluster identificati")
                
                # FASE 4: AI
                st.markdown("### 📝 Preparazione AI")
                recensioni_pulite = [r['testo_pulito'] for r in recensioni_data if r.get('testo_pulito')]
                testo_completo = " ".join(recensioni_pulite)
                parole = testo_completo.split()
                blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
                st.info(f"📊 {len(blocchi)} blocchi")
                
                st.markdown("### 🤖 Analisi AI")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                risultati = analizza_blocchi_con_ai(blocchi, client_openai, progress_bar, status_text)
                
                progress_bar.empty()
                status_text.empty()
                
                # FASE 5: Frequenze
                st.markdown("### 📊 Analisi Frequenze")
                with st.spinner("Calcolo..."):
                    frequenze = analizza_frequenza_temi(risultati, recensioni_data)
                
                st.markdown('<div class="success-box"><h3>🎉 Completato!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI (identici a prima)
                st.markdown("## 📊 Risultati")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("📝 Recensioni", len(recensioni_data))
                with col_m2:
                    st.metric("💪 Punti Forza", len(risultati.get('punti_forza', [])))
                with col_m3:
                    st.metric("⚠️ Criticità", len(risultati.get('punti_debolezza', [])))
                with col_m4:
                    st.metric("🎯 Cluster", len(clusters))
                
                tab1, tab2, tab3 = st.tabs(["💪 Forza", "⚠️ Debolezza", "🎨 Cluster"])
                
                with tab1:
                    if frequenze['punti_forza']:
                        for punto, dati in list(frequenze['punti_forza'].items())[:10]:
                            st.markdown(f"**{punto}** `{dati['count']} ({dati['percentuale']:.1f}%)`")
                            st.markdown("---")
                
                with tab2:
                    if frequenze['punti_debolezza']:
                        for punto, dati in list(frequenze['punti_debolezza'].items())[:10]:
                            st.markdown(f"**{punto}** `{dati['count']} ({dati['percentuale']:.1f}%)`")
                            st.markdown("---")
                
                with tab3:
                    for cluster in clusters:
                        with st.expander(f"🎯 Cluster {cluster['id']+1}"):
                            st.write(f"**Recensioni:** {cluster['n_recensioni']}")
                            st.write(f"**Keywords:** {', '.join(cluster['parole_chiave'])}")
                
                # DOWNLOAD
                st.markdown("## 📥 Download")
                
                excel_data = crea_excel_download(
                    recensioni_data, risultati, clusters, 
                    frequenze, business_info
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Reviews_{business_info['nome'].replace(' ', '_')}_{timestamp}.xlsx"
                
                st.download_button(
                    "📊 Scarica Excel",
                    excel_data,
                    filename,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Errore")
                st.exception(e)
    
    with col2:
        st.markdown("## 📋 Guida")
        
        st.markdown("""
        ### ✅ Setup
        
        1. Inserisci API keys
        2. Nome attività esatto
        3. Città o indirizzo
        4. Avvia analisi
        
        ---
        
        ### ⏱️ Tempi
        
        - Business: 2-5s
        - **Attesa coda: 0-5min**
        - Reviews: 2-10min
        - AI: 2-4min
        - **Totale: 5-20min**
        
        ---
        
        ### 🔧 Coda DataForSEO
        
        **Cosa fa l'app:**
        • Controlla coda PRIMA
        • Attende svuotamento
        • Crea task solo se vuota
        • Polling 15 min max
        
        **Status code:**
        • 40602 = In Queue
        • 40100/40200 = Processing
        • 20000 = Done
        
        ---
        
        ### 💡 Tips
        
        ✅ **Coda vuota** → ~5 min totali  
        ⏳ **Coda occupata** → +5 min attesa  
        ⚡ **Priorità alta** → elaborazione più veloce (costo extra)
        """)

if __name__ == "__main__":
    main()
