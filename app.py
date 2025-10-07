# 🚀 ANALIZZATORE GOOGLE REVIEWS - STREAMLIT APP
# Versione completa con DataForSEO API Integration

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
warnings.filterwarnings('ignore')

# 🎨 CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="🚀 Analizzatore Google Reviews Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎯 CSS PERSONALIZZATO
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
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4285F4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
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
        background: linear-gradient(135deg, #EA4335 0%, #D32F2F 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .cluster-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4285F4;
        margin: 1rem 0;
    }
    
    .review-example {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    
    .positive-review {
        border-left: 4px solid #34A853;
    }
    
    .negative-review {
        border-left: 4px solid #EA4335;
    }
    
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
    
    .business-card {
        background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 HEADER PRINCIPALE
st.markdown('<h1 class="main-header">🚀 Analizzatore Google Reviews Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
    <h3>🎯 Cosa fa questa App?</h3>
    <p>• Cerca automaticamente attività su Google Maps tramite DataForSEO</p>
    <p>• Estrae recensioni con rating, date, autori e risposte del proprietario</p>
    <p>• Clusterizza le recensioni per tematiche comuni</p>
    <p>• Analizza la frequenza e il peso di ogni punto critico</p>
    <p>• Mostra esempi reali di recensioni positive/negative</p>
    <p>• Analizza le risposte del proprietario e il tasso di risposta</p>
    <p>• Genera strategie di Digital Marketing personalizzate</p>
    <p>• Fornisce suggerimenti specifici per Local SEO, Reputation Management e CRO</p>
</div>
""", unsafe_allow_html=True)

# 🔧 CLASSE DATAFORSEO CLIENT
class DataForSEOClient:
    """Client per interagire con le API DataForSEO"""
    
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.base_url = "https://api.dataforseo.com/v3"
    
    def _make_request(self, endpoint, data):
        """Effettua una richiesta POST alle API DataForSEO"""
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self._get_auth_token()}'
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Errore API DataForSEO: {str(e)}")
    
    def _get_auth_token(self):
        """Genera il token di autenticazione Basic"""
        import base64
        credentials = f"{self.username}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()
    
    def search_business(self, query, location):
        """Cerca un'attività su Google Maps"""
        endpoint = "business_data/google/my_business_info/live"
        
        # Prepara la richiesta
        payload = [{
            "location_name": location,
            "keyword": query,
            "language_code": "it"
        }]
        
        result = self._make_request(endpoint, payload)
        
        if result.get('status_code') == 20000:
            tasks = result.get('tasks', [])
            if tasks and tasks[0].get('result'):
                return tasks[0]['result']
        
        return None
    
    def get_reviews(self, place_id, limit=100):
        """Estrae recensioni di un'attività"""
        endpoint = "business_data/google/reviews/task_post"
        
        # Crea task per estrarre recensioni
        payload = [{
            "place_id": place_id,
            "language_code": "it",
            "depth": min(limit, 500),  # Max 500 per richiesta
            "sort_by": "newest"
        }]
        
        # Invia task
        result = self._make_request(endpoint, payload)
        
        if result.get('status_code') != 20000:
            raise Exception(f"Errore creazione task: {result.get('status_message')}")
        
        # Ottieni task_id
        tasks = result.get('tasks', [])
        if not tasks:
            raise Exception("Nessun task creato")
        
        task_id = tasks[0].get('id')
        
        # Attendi completamento task (max 30 secondi)
        for _ in range(30):
            time.sleep(1)
            
            # Controlla stato task
            check_endpoint = f"business_data/google/reviews/task_get/{task_id}"
            check_result = self._make_request(check_endpoint, [])
            
            if check_result.get('status_code') == 20000:
                check_tasks = check_result.get('tasks', [])
                if check_tasks and check_tasks[0].get('result'):
                    return check_tasks[0]['result']
        
        raise Exception("Timeout durante l'estrazione delle recensioni")

# 🔧 FUNZIONI BACKEND

@st.cache_data
def get_stopwords():
    return set([
        "il", "lo", "la", "i", "gli", "le", "di", "a", "da", "in", "con", "su", "per", 
        "tra", "fra", "un", "una", "uno", "e", "ma", "anche", "come", "che", "non", 
        "più", "meno", "molto", "poco", "tutto", "tutti", "tutte", "questo", "questa", 
        "questi", "queste", "quello", "quella", "quelli", "quelle", "sono", "è", "ho", 
        "hai", "ha", "hanno", "essere", "avere", "fare", "dire", "andare", "del", "della",
        "dei", "delle", "dal", "dalla", "dai", "dalle", "nel", "nella", "nei", "nelle",
        "sul", "sulla", "sui", "sulle", "al", "alla", "ai", "alle", "ho", "ottimo",
        "buono", "buona", "bene", "male", "servizio", "prodotto", "azienda", "sempre",
        "google", "maps", "recensione", "recensioni", "stelle", "mese", "mesi", "anno",
        "settimana", "giorno", "giorni", "fa", "ago"
    ])

def pulisci_testo(testo):
    """Pulizia avanzata del testo per Google Reviews"""
    if not testo:
        return ""
    
    stopwords_italiane = get_stopwords()
    testo = str(testo).lower()
    
    # Rimozione date e riferimenti temporali
    testo = re.sub(r'\d{1,2}\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}', '', testo)
    testo = re.sub(r'(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}', '', testo)
    testo = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', testo)
    
    # Rimozione riferimenti temporali comuni
    testo = re.sub(r'\d+\s+(giorn[oi]|settiman[ae]|mes[ie]|ann[oi])\s+fa', '', testo)
    testo = re.sub(r'(ieri|oggi|domani)', '', testo)
    
    # Rimozione stelle e rating
    testo = re.sub(r'[1-5]\s*stelle?', '', testo)
    testo = re.sub(r'rating:?\s*\d', '', testo)
    
    # Pulizia generale
    testo = re.sub(r'[^\w\s]', ' ', testo)
    testo = re.sub(r'\d+', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    
    parole = testo.split()
    parole_filtrate = [parola for parola in parole if parola not in stopwords_italiane and len(parola) > 2]
    return " ".join(parole_filtrate)

def cerca_attivita_google(client, nome_attivita, location, progress_bar, status_text):
    """Cerca l'attività su Google Maps tramite DataForSEO"""
    status_text.text(f"🔍 Ricerca attività: {nome_attivita}...")
    progress_bar.progress(0.1)
    
    try:
        # Chiamata API per cercare l'attività
        result = client.search_business(nome_attivita, location)
        progress_bar.progress(0.5)
        
        if not result or not result.get('items'):
            return None, "Nessuna attività trovata con questi criteri"
        
        # Prendi il primo risultato (più rilevante)
        business = result['items'][0]
        progress_bar.progress(1.0)
        
        return business, None
        
    except Exception as e:
        return None, f"Errore durante la ricerca: {str(e)}"

def estrai_recensioni_google(client, place_id, max_reviews, progress_bar, status_text):
    """Estrae recensioni da Google Maps tramite DataForSEO"""
    recensioni_data = []
    
    try:
        status_text.text(f"📥 Estrazione recensioni in corso...")
        progress_bar.progress(0.1)
        
        # Chiamata API per ottenere recensioni
        result = client.get_reviews(place_id, max_reviews)
        progress_bar.progress(0.5)
        
        if not result or not result.get('items'):
            return [], "Nessuna recensione trovata per questa attività"
        
        # Processa le recensioni
        for item in result['items'][:max_reviews]:
            # Estrai dati recensione
            review_text = item.get('review_text', '')
            if not review_text:
                continue
            
            # Estrai risposta owner se presente
            owner_response = None
            if item.get('responses'):
                responses = item['responses']
                if responses and len(responses) > 0:
                    owner_response = responses[0].get('text', '')
            
            # Estrai data
            review_date = None
            if item.get('timestamp'):
                try:
                    review_date = datetime.fromtimestamp(item['timestamp']).strftime("%Y-%m-%d")
                except:
                    review_date = None
            
            # Estrai rating
            rating = item.get('rating', {}).get('value', 0)
            
            # Verifica presenza foto
            has_images = bool(item.get('images'))
            
            recensione = {
                'testo': review_text,
                'testo_pulito': pulisci_testo(review_text),
                'rating': rating,
                'data': review_date,
                'autore': item.get('author_name', 'Anonimo'),
                'risposta_owner': owner_response,
                'ha_foto': has_images,
                'link': item.get('url', f"https://maps.google.com/?cid={place_id}"),
                'review_id': item.get('review_id', '')
            }
            
            recensioni_data.append(recensione)
        
        progress_bar.progress(1.0)
        return recensioni_data, None
        
    except Exception as e:
        return [], f"Errore durante l'estrazione: {str(e)}"

def clusterizza_recensioni(recensioni_data, n_clusters=None):
    """Clusterizza le recensioni per tematiche"""
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
    
    # Assicurati che n_clusters non superi il numero di campioni
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
    recensioni_usate = set()
    
    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        top_words = [w for w in top_words if len(w) > 3 and not w.isdigit()]
        
        cluster_reviews = [r for r in recensioni_data if r.get('cluster') == i]
        
        esempi_cluster = []
        for rec in cluster_reviews:
            rec_id = f"{rec.get('autore', '')}_{rec.get('data', '')}"
            if rec_id not in recensioni_usate:
                esempi_cluster.append(rec)
                recensioni_usate.add(rec_id)
                if len(esempi_cluster) >= 3:
                    break
        
        if cluster_reviews:
            cluster_info = {
                'id': i,
                'parole_chiave': top_words[:5],
                'n_recensioni': len(cluster_reviews),
                'percentuale': (len(cluster_reviews) / len(recensioni_data)) * 100,
                'rating_medio': np.mean([r['rating'] for r in cluster_reviews if r.get('rating')]),
                'recensioni': esempi_cluster
            }
            cluster_topics.append(cluster_info)
    
    cluster_topics.sort(key=lambda x: x['n_recensioni'], reverse=True)
    
    return recensioni_data, cluster_topics

def analizza_risposte_owner(recensioni_data):
    """Analizza le risposte del proprietario"""
    recensioni_con_risposta = [r for r in recensioni_data if r.get('risposta_owner')]
    
    if not recensioni_data:
        return {}
    
    tasso_risposta = (len(recensioni_con_risposta) / len(recensioni_data)) * 100
    
    # Analizza per rating
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
    """Analizza l'andamento temporale delle recensioni"""
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
                trend_mensile[mese_anno] = {
                    'count': 0,
                    'rating_sum': 0,
                    'ratings': []
                }
            
            trend_mensile[mese_anno]['count'] += 1
            trend_mensile[mese_anno]['rating_sum'] += rec['rating']
            trend_mensile[mese_anno]['ratings'].append(rec['rating'])
        except:
            continue
    
    for mese in trend_mensile:
        trend_mensile[mese]['rating_medio'] = trend_mensile[mese]['rating_sum'] / trend_mensile[mese]['count']
    
    trend_ordinato = dict(sorted(trend_mensile.items()))
    
    return trend_ordinato

def analizza_frequenza_temi(risultati, recensioni_data):
    """Analizza la frequenza dei temi identificati nelle recensioni"""
    frequenze = {
        'punti_forza': {},
        'punti_debolezza': {}
    }
    
    # Calcola frequenze per punti di forza
    for punto in risultati.get('punti_forza', []):
        count = 0
        esempi = []
        recensioni_ids_usate = set()
        
        parole_punto = [p for p in punto.lower().split() if len(p) > 3][:3]
        
        for rec in recensioni_data:
            if rec.get('rating') and rec['rating'] >= 4:
                rec_id = f"{rec.get('autore', '')}_{rec.get('data', '')}"
                
                if rec_id not in recensioni_ids_usate:
                    testo_lower = rec.get('testo_pulito', '').lower()
                    matches = sum(1 for parola in parole_punto if parola in testo_lower)
                    
                    if matches >= min(2, len(parole_punto)):
                        count += 1
                        recensioni_ids_usate.add(rec_id)
                        
                        if len(esempi) < 2 and any(parola in testo_lower for parola in parole_punto):
                            esempi.append(rec)
        
        if count > 0:
            recensioni_positive = [r for r in recensioni_data if r.get('rating') and r['rating'] >= 4]
            frequenze['punti_forza'][punto] = {
                'count': count,
                'percentuale': (count / len(recensioni_positive)) * 100 if recensioni_positive else 0,
                'esempi': esempi
            }
    
    # Calcola frequenze per punti di debolezza
    for punto in risultati.get('punti_debolezza', []):
        count = 0
        esempi = []
        recensioni_ids_usate = set()
        
        parole_punto = [p for p in punto.lower().split() if len(p) > 3][:3]
        
        for rec in recensioni_data:
            if rec.get('rating') and rec['rating'] <= 2:
                rec_id = f"{rec.get('autore', '')}_{rec.get('data', '')}"
                
                if rec_id not in recensioni_ids_usate:
                    testo_lower = rec.get('testo_pulito', '').lower()
                    matches = sum(1 for parola in parole_punto if parola in testo_lower)
                    
                    if matches >= min(2, len(parole_punto)):
                        count += 1
                        recensioni_ids_usate.add(rec_id)
                        
                        if len(esempi) < 2 and any(parola in testo_lower for parola in parole_punto):
                            esempi.append(rec)
        
        if count > 0:
            recensioni_negative = [r for r in recensioni_data if r.get('rating') and r['rating'] <= 2]
            frequenze['punti_debolezza'][punto] = {
                'count': count,
                'percentuale': (count / len(recensioni_negative)) * 100 if recensioni_negative else 0,
                'esempi': esempi
            }
    
    frequenze['punti_forza'] = dict(sorted(frequenze['punti_forza'].items(), 
                                          key=lambda x: x[1]['count'], reverse=True))
    frequenze['punti_debolezza'] = dict(sorted(frequenze['punti_debolezza'].items(), 
                                              key=lambda x: x[1]['count'], reverse=True))
    
    return frequenze

def analizza_blocchi_avanzata_con_sentiment(blocchi, client, progress_bar, status_text):
    """Analisi AI con sentiment analysis per Google Reviews"""
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
        "sentiment_distribution": {"positivo": 0, "neutro": 0, "negativo": 0}
    }

    for i, blocco in enumerate(blocchi):
        status_text.text(f"🤖 Analizzando blocco {i+1}/{len(blocchi)} con AI...")

        prompt = f"""
        Analizza le seguenti recensioni Google di un'attività locale e fornisci insights strategici dettagliati:

        RECENSIONI GOOGLE:
        {blocco}

        Rispondi SOLO in formato JSON valido con queste chiavi:
        {{
            "punti_forza": ["punto specifico 1", "punto specifico 2", ...],
            "punti_debolezza": ["problema specifico 1", "problema specifico 2", ...],
            "leve_marketing": ["leva concreta 1", "leva concreta 2", ...],
            "parole_chiave": ["termine rilevante 1", "termine rilevante 2", ...],
            "suggerimenti_local_seo": ["suggerimento Local SEO 1", "suggerimento Local SEO 2", ...],
            "suggerimenti_reputation": ["strategia reputation management 1", "strategia 2", ...],
            "suggerimenti_google_ads": ["strategia Google Ads locale 1", "strategia 2", ...],
            "suggerimenti_cro": ["ottimizzazione scheda GMB 1", "ottimizzazione 2", ...],
            "suggerimenti_risposte": ["template risposta situazione 1", "template 2", ...],
            "sentiment_counts": {{"positivo": N, "neutro": N, "negativo": N}}
        }}

        LINEE GUIDA SPECIFICHE PER GOOGLE REVIEWS:
        - Estrai punti MOLTO SPECIFICI legati all'esperienza locale (location, parcheggio, orari, etc.)
        - I punti di forza devono essere elementi concreti lodati dai clienti
        - I punti di debolezza devono essere problemi specifici
        - Local SEO: ottimizzazioni specifiche per ricerca locale e Google Maps
        - Reputation: strategie per gestire recensioni negative e aumentare quelle positive
        - Google Ads: campagne local per aumentare visite e conversioni
        - CRO: ottimizzazioni della scheda Google My Business
        - Risposte: template per rispondere a diverse tipologie di recensioni
        - IGNORA date, mesi, giorni, riferimenti temporali
        - Conta il sentiment delle recensioni

        Fornisci suggerimenti PRATICI, SPECIFICI e IMPLEMENTABILI per business locali.
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
                            nuovi_elementi = [elem for elem in dati[chiave] if elem not in risultati[chiave]]
                            risultati[chiave].extend(nuovi_elementi)

                break
                
            except json.JSONDecodeError:
                if tentativo < 2:
                    time.sleep(2)
            except Exception:
                if tentativo < 2:
                    time.sleep(5)

        progress_bar.progress((i + 1) / len(blocchi))

    for chiave in risultati:
        if chiave != 'sentiment_distribution':
            risultati[chiave] = list(dict.fromkeys(risultati[chiave]))
    
    termini_da_escludere = ['google', 'maps', 'recensione', 'stelle', 'mese', 'anno']
    risultati['parole_chiave'] = [
        parola for parola in risultati['parole_chiave'] 
        if parola.lower() not in termini_da_escludere and not any(t in parola.lower() for t in termini_da_escludere)
    ]
    
    return risultati

def mostra_esempi_recensioni_google(tema, esempi, tipo="positivo"):
    """Mostra esempi di recensioni Google con risposte owner"""
    if not esempi:
        return
    
    st.markdown(f"**Esempi di recensioni:**")
    
    esempi_mostrati = set()
    
    for esempio in esempi[:2]:
        esempio_id = f"{esempio.get('autore', '')}_{esempio.get('data', '')}"
        
        if esempio_id not in esempi_mostrati:
            esempi_mostrati.add(esempio_id)
            
            rating_stars = "⭐" * esempio.get('rating', 3)
            
            testo_breve = esempio.get('testo', '')[:200]
            if len(esempio.get('testo', '')) > 200:
                testo_breve += "..."
            
            css_class = "positive-review" if tipo == "positivo" else "negative-review"
            
            risposta_html = ""
            if esempio.get('risposta_owner'):
                risposta_breve = esempio['risposta_owner'][:150]
                if len(esempio['risposta_owner']) > 150:
                    risposta_breve += "..."
                risposta_html = f"""
                <div class="owner-response">
                    <strong>💬 Risposta del proprietario:</strong>
                    <p style="margin-top: 0.5rem; margin-bottom: 0;">{risposta_breve}</p>
                </div>
                """
            
            st.markdown(f"""
            <div class="review-example {css_class}">
                <div style="margin-bottom: 0.5rem;">
                    <strong>{rating_stars}</strong>
                    <small style="color: #666;"> - {esempio.get('autore', 'Anonimo')} • {esempio.get('data', 'N/A')}</small>
                    {' 📸' if esempio.get('ha_foto') else ''}
                </div>
                <div style="margin-bottom: 0.5rem; color: #333;">
                    {testo_breve}
                </div>
                {risposta_html}
                <a href="{esempio.get('link', '#')}" target="_blank" style="color: #4285F4; text-decoration: none; margin-top: 0.5rem; display: inline-block;">
                    🔗 Vedi su Google Maps →
                </a>
            </div>
            """, unsafe_allow_html=True)

def crea_excel_download_avanzato_google(recensioni_data, risultati, clusters, frequenze, analisi_owner, trend_temporale, business_info):
    """Crea file Excel avanzato per Google Reviews"""
    output = io.BytesIO()
    
    # DataFrame recensioni
    df_recensioni = pd.DataFrame([{
        'Testo Originale': r.get('testo', ''),
        'Rating': r.get('rating', 0),
        'Data': r.get('data', ''),
        'Autore': r.get('autore', ''),
        'Ha Risposta Owner': 'Sì' if r.get('risposta_owner') else 'No',
        'Risposta Owner': r.get('risposta_owner', ''),
        'Ha Foto': 'Sì' if r.get('ha_foto') else 'No',
        'Cluster': r.get('cluster', ''),
        'Link': r.get('link', '')
    } for r in recensioni_data])
    
    # DataFrame info business
    df_business = pd.DataFrame([business_info])
    
    # DataFrame clusters
    df_clusters = pd.DataFrame([{
        'Cluster ID': c['id'],
        'Tematica': ', '.join(c['parole_chiave']),
        'N. Recensioni': c['n_recensioni'],
        'Percentuale': f"{c['percentuale']:.1f}%",
        'Rating Medio': f"{c['rating_medio']:.1f}" if not np.isnan(c['rating_medio']) else 'N/A'
    } for c in clusters])
    
    # DataFrame analisi risposte owner
    if analisi_owner:
        df_owner = pd.DataFrame([{
            'Tasso Risposta Globale': f"{analisi_owner['tasso_risposta']:.1f}%",
            'Numero Risposte': analisi_owner['n_risposte']
        }])
        
        df_owner_per_rating = pd.DataFrame([{
            'Rating': rating,
            'Totali': dati['totali'],
            'Con Risposta': dati['con_risposta'],
            'Tasso Risposta': f"{dati['percentuale']:.1f}%"
        } for rating, dati in analisi_owner.get('risposte_per_rating', {}).items()])
    else:
        df_owner = pd.DataFrame()
        df_owner_per_rating = pd.DataFrame()
    
    # DataFrame trend temporale
    if trend_temporale:
        df_trend = pd.DataFrame([{
            'Mese': mese,
            'N. Recensioni': dati['count'],
            'Rating Medio': f"{dati['rating_medio']:.1f}"
        } for mese, dati in trend_temporale.items()])
    else:
        df_trend = pd.DataFrame()
    
    # DataFrame frequenze
    df_forza = pd.DataFrame([{
        'Punto di Forza': punto,
        'Frequenza': dati['count'],
        'Percentuale': f"{dati['percentuale']:.1f}%"
    } for punto, dati in frequenze['punti_forza'].items()])
    
    df_debolezza = pd.DataFrame([{
        'Punto di Debolezza': punto,
        'Frequenza': dati['count'],
        'Percentuale': f"{dati['percentuale']:.1f}%"
    } for punto, dati in frequenze['punti_debolezza'].items()])
    
    # DataFrame strategie
    df_strategie = pd.DataFrame({
        'Canale': ['Local SEO', 'Reputation Management', 'Google Ads', 'CRO Scheda GMB', 'Template Risposte'],
        'Suggerimenti': [
            ' | '.join(risultati.get('suggerimenti_local_seo', [])),
            ' | '.join(risultati.get('suggerimenti_reputation', [])),
            ' | '.join(risultati.get('suggerimenti_google_ads', [])),
            ' | '.join(risultati.get('suggerimenti_cro', [])),
            ' | '.join(risultati.get('suggerimenti_risposte', []))
        ]
    })
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not df_business.empty:
            df_business.to_excel(writer, sheet_name='Info Business', index=False)
        df_recensioni.to_excel(writer, sheet_name='Recensioni', index=False)
        if not df_clusters.empty:
            df_clusters.to_excel(writer, sheet_name='Clusters Tematici', index=False)
        if not df_owner.empty:
            df_owner.to_excel(writer, sheet_name='Analisi Risposte Owner', index=False)
        if not df_owner_per_rating.empty:
            df_owner_per_rating.to_excel(writer, sheet_name='Risposte per Rating', index=False)
        if not df_trend.empty:
            df_trend.to_excel(writer, sheet_name='Trend Temporale', index=False)
        if not df_forza.empty:
            df_forza.to_excel(writer, sheet_name='Punti Forza Frequenza', index=False)
        if not df_debolezza.empty:
            df_debolezza.to_excel(writer, sheet_name='Punti Debolezza Frequenza', index=False)
        df_strategie.to_excel(writer, sheet_name='Strategie Digital', index=False)
    
    return output.getvalue()

# 🎮 INTERFACCIA PRINCIPALE
def main():
    # SIDEBAR PER INPUT
    with st.sidebar:
        st.markdown("## 🔧 Configurazione")
        
        # API Keys
        st.markdown("### 🔑 API Keys")
        api_key_openai = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Inserisci la tua API Key di OpenAI"
        )
        
        dataforseo_username = st.text_input(
            "DataForSEO Username",
            help="Username o email del tuo account DataForSEO"
        )
        
        dataforseo_password = st.text_input(
            "DataForSEO Password",
            type="password",
            help="Password del tuo account DataForSEO"
        )
        
        st.markdown("---")
        
        # Dati attività
        st.markdown("### 🏢 Dati Attività")
        
        nome_attivita = st.text_input(
            "🏪 Nome Attività",
            placeholder="Es: Ristorante Da Mario",
            help="Nome esatto dell'attività su Google Maps"
        )
        
        location = st.text_input(
            "📍 Città/Indirizzo",
            placeholder="Es: Milano, Italia",
            help="Città o indirizzo dell'attività"
        )
        
        # Numero recensioni
        max_reviews = st.slider(
            "📝 Numero recensioni target",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Più recensioni = analisi più accurata"
        )
        
        # Numero clusters
        n_clusters = st.slider(
            "🎯 Numero di cluster tematici",
            min_value=3,
            max_value=15,
            value=8,
            help="Numero di tematiche in cui raggruppare le recensioni"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Suggerimenti")
        st.info(
            "• Inizia con 50-100 recensioni\n"
            "• 6-10 cluster sono ideali\n"
            "• Il nome deve corrispondere a Google Maps\n"
            "• Tempo stimato: 3-8 minuti"
        )

    # AREA PRINCIPALE
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Inizia l'Analisi")
        
        if st.button("🔍 Avvia Analisi Google Reviews", type="primary", use_container_width=True):
            # Validazione input
            if not api_key_openai:
                st.error("❌ Inserisci la tua OpenAI API Key")
                return
            
            if not dataforseo_username or not dataforseo_password:
                st.error("❌ Inserisci le credenziali DataForSEO")
                return
            
            if not nome_attivita:
                st.error("❌ Inserisci il nome dell'attività")
                return
            
            if not location:
                st.error("❌ Inserisci la città/indirizzo")
                return
            
            # Inizializza clients
            try:
                client_openai = OpenAI(api_key=api_key_openai)
                client_dataforseo = DataForSEOClient(dataforseo_username, dataforseo_password)
            except Exception as e:
                st.error(f"❌ Errore inizializzazione: {e}")
                return
            
            # Container per risultati
            results_container = st.container()
            
            with results_container:
                # FASE 1: Ricerca Attività
                st.markdown("### 🔍 Fase 1: Ricerca Attività su Google Maps")
                progress_bar_1 = st.progress(0)
                status_text_1 = st.empty()
                
                business_info, error = cerca_attivita_google(
                    client_dataforseo,
                    nome_attivita,
                    location,
                    progress_bar_1,
                    status_text_1
                )
                
                if error:
                    st.error(f"❌ {error}")
                    return
                
                if not business_info:
                    st.error("❌ Attività non trovata")
                    return
                
                # Estrai info business
                place_id = business_info.get('place_id', '')
                business_data = {
                    'place_id': place_id,
                    'nome': business_info.get('title', nome_attivita),
                    'indirizzo': business_info.get('address', location),
                    'rating_medio': business_info.get('rating', {}).get('value', 0),
                    'n_recensioni': business_info.get('rating', {}).get('votes_count', 0),
                    'categoria': business_info.get('category', 'N/A'),
                    'telefono': business_info.get('phone', 'N/A'),
                    'sito_web': business_info.get('url', 'N/A')
                }
                
                status_text_1.text("✅ Attività trovata!")
                
                # Mostra card business
                st.markdown(f"""
                <div class="business-card">
                    <h3>🏢 {business_data['nome']}</h3>
                    <p><strong>📍 Indirizzo:</strong> {business_data['indirizzo']}</p>
                    <p><strong>⭐ Rating:</strong> {business_data['rating_medio']}/5 ({business_data['n_recensioni']} recensioni)</p>
                    <p><strong>🏷️ Categoria:</strong> {business_data['categoria']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # FASE 2: Estrazione Recensioni
                st.markdown("### 📥 Fase 2: Estrazione Recensioni")
                progress_bar_2 = st.progress(0)
                status_text_2 = st.empty()
                
                recensioni_data, error = estrai_recensioni_google(
                    client_dataforseo,
                    place_id,
                    max_reviews,
                    progress_bar_2,
                    status_text_2
                )
                
                if error:
                    st.error(f"❌ {error}")
                    return
                
                if not recensioni_data:
                    st.error("❌ Nessuna recensione estratta")
                    return
                
                # Statistiche recensioni
                n_con_rating = len([r for r in recensioni_data if r.get('rating')])
                rating_medio = np.mean([r['rating'] for r in recensioni_data if r.get('rating')]) if n_con_rating > 0 else 0
                n_con_risposta = len([r for r in recensioni_data if r.get('risposta_owner')])
                tasso_risposta = (n_con_risposta / len(recensioni_data)) * 100 if recensioni_data else 0
                
                st.success(f"✅ Estratte {len(recensioni_data)} recensioni!")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("⭐ Rating Medio", f"{rating_medio:.1f}")
                with col_stat2:
                    st.metric("💬 Con Risposta", f"{n_con_risposta}")
                with col_stat3:
                    st.metric("📊 Tasso Risposta", f"{tasso_risposta:.1f}%")
                
                # FASE 3: Clustering
                st.markdown("### 🎨 Fase 3: Clustering Tematico")
                with st.spinner("Clustering in corso..."):
                    recensioni_data, clusters = clusterizza_recensioni(recensioni_data, n_clusters)
                
                st.success(f"✅ Identificati {len(clusters)} cluster tematici!")
                
                # FASE 4: Analisi Risposte Owner
                st.markdown("### 💬 Fase 4: Analisi Risposte Owner")
                with st.spinner("Analisi risposte in corso..."):
                    analisi_owner = analizza_risposte_owner(recensioni_data)
                
                # FASE 5: Trend Temporale
                st.markdown("### 📈 Fase 5: Analisi Trend Temporale")
                with st.spinner("Analisi trend in corso..."):
                    trend_temporale = analizza_trend_temporale(recensioni_data)
                
                # FASE 6: Preparazione AI
                st.markdown("### 📝 Fase 6: Preparazione Analisi AI")
                recensioni_pulite = [r['testo_pulito'] for r in recensioni_data if r.get('testo_pulito')]
                testo_completo = " ".join(recensioni_pulite)
                parole = testo_completo.split()
                blocchi = [' '.join(parole[i:i+8000]) for i in range(0, len(parole), 8000)]
                st.info(f"📊 Creati {len(blocchi)} blocchi per l'analisi AI")
                
                # FASE 7: Analisi AI
                st.markdown("### 🤖 Fase 7: Analisi AI con Sentiment")
                progress_bar_3 = st.progress(0)
                status_text_3 = st.empty()
                
                with st.spinner("Analisi AI in corso..."):
                    risultati = analizza_blocchi_avanzata_con_sentiment(
                        blocchi, client_openai, progress_bar_3, status_text_3
                    )
                
                # FASE 8: Analisi Frequenze
                st.markdown("### 📊 Fase 8: Analisi Frequenze")
                with st.spinner("Analisi frequenze in corso..."):
                    frequenze = analizza_frequenza_temi(risultati, recensioni_data)
                
                st.markdown('<div class="success-box"><h3>🎉 Analisi Completata!</h3></div>', unsafe_allow_html=True)
                
                # RISULTATI
                st.markdown("## 📊 Risultati Analisi Completa")
                
                # Metriche principali
                col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
                
                with col_m1:
                    st.metric("📝 Recensioni", len(recensioni_data))
                with col_m2:
                    st.metric("⭐ Rating", f"{rating_medio:.1f}")
                with col_m3:
                    st.metric("💪 Forze", len(risultati.get('punti_forza', [])))
                with col_m4:
                    st.metric("⚠️ Criticità", len(risultati.get('punti_debolezza', [])))
                with col_m5:
                    st.metric("🎯 Cluster", len(clusters))
                with col_m6:
                    st.metric("💬 Tasso Risp.", f"{tasso_risposta:.0f}%")
                
                # Distribuzione sentiment
                if risultati.get('sentiment_distribution', {}).get('positivo', 0) > 0:
                    st.markdown("### 😊 Distribuzione Sentiment")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    total_sentiment = sum(risultati['sentiment_distribution'].values())
                    
                    with col_s1:
                        perc = (risultati['sentiment_distribution']['positivo'] / total_sentiment * 100) if total_sentiment > 0 else 0
                        st.metric("😊 Positivo", f"{perc:.1f}%")
                    with col_s2:
                        perc = (risultati['sentiment_distribution']['neutro'] / total_sentiment * 100) if total_sentiment > 0 else 0
                        st.metric("😐 Neutro", f"{perc:.1f}%")
                    with col_s3:
                        perc = (risultati['sentiment_distribution']['negativo'] / total_sentiment * 100) if total_sentiment > 0 else 0
                        st.metric("😞 Negativo", f"{perc:.1f}%")
                
                # Tabs risultati
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                    "💪 Punti Forza",
                    "⚠️ Criticità",
                    "🎨 Cluster",
                    "💬 Risposte Owner",
                    "📈 Trend",
                    "🎯 Leve Marketing",
                    "📊 Strategie Digital",
                    "🔍 Keywords"
                ])
                
                with tab1:
                    st.markdown("### 💪 Punti di Forza")
                    if frequenze['punti_forza']:
                        for punto, dati in list(frequenze['punti_forza'].items())[:10]:
                            st.markdown(f"""
                            **{punto}**
                            <span class="frequency-badge">{dati['percentuale']:.1f}% delle recensioni positive</span>
                            """, unsafe_allow_html=True)
                            if dati['esempi']:
                                with st.expander(f"Vedi esempi ({len(dati['esempi'])})"):
                                    mostra_esempi_recensioni_google(punto, dati['esempi'], "positivo")
                    else:
                        for i, punto in enumerate(risultati.get('punti_forza', [])[:10], 1):
                            st.markdown(f"**{i}.** {punto}")
                
                with tab2:
                    st.markdown("### ⚠️ Punti di Debolezza")
                    if frequenze['punti_debolezza']:
                        for punto, dati in list(frequenze['punti_debolezza'].items())[:10]:
                            st.markdown(f"""
                            **{punto}**
                            <span class="frequency-badge" style="background: #EA4335;">{dati['percentuale']:.1f}% delle recensioni negative</span>
                            """, unsafe_allow_html=True)
                            if dati['esempi']:
                                with st.expander(f"Vedi esempi ({len(dati['esempi'])})"):
                                    mostra_esempi_recensioni_google(punto, dati['esempi'], "negativo")
                    else:
                        for i, punto in enumerate(risultati.get('punti_debolezza', [])[:10], 1):
                            st.markdown(f"**{i}.** {punto}")
                
                with tab3:
                    st.markdown("### 🎨 Cluster Tematici")
                    for cluster in clusters:
                        with st.expander(f"Cluster {cluster['id'] + 1}: {', '.join(cluster['parole_chiave'][:3])} ({cluster['percentuale']:.1f}%)"):
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                st.metric("Recensioni", cluster['n_recensioni'])
                                st.metric("Rating Medio", f"{cluster['rating_medio']:.1f} ⭐")
                            with col_c2:
                                st.markdown("**Tematiche:**")
                                for parola in cluster['parole_chiave']:
                                    st.markdown(f"• {parola}")
                            st.markdown("**Esempi:**")
                            for rec in cluster['recensioni'][:2]:
                                st.markdown(f"> {'⭐' * rec.get('rating', 3)} {rec.get('testo', '')[:150]}...")
                
                with tab4:
                    st.markdown("### 💬 Analisi Risposte")
                    if analisi_owner:
                        col_o1, col_o2 = st.columns(2)
                        with col_o1:
                            st.metric("Tasso Risposta", f"{analisi_owner['tasso_risposta']:.1f}%")
                        with col_o2:
                            st.metric("Risposte Totali", analisi_owner['n_risposte'])
                        
                        st.markdown("#### Tasso per Rating")
                        for rating in range(5, 0, -1):
                            if rating in analisi_owner.get('risposte_per_rating', {}):
                                dati = analisi_owner['risposte_per_rating'][rating]
                                st.markdown(f"**{'⭐' * rating}** - {dati['con_risposta']}/{dati['totali']} ({dati['percentuale']:.1f}%)")
                
                with tab5:
                    st.markdown("### 📈 Trend Temporale")
                    if trend_temporale:
                        df_trend = pd.DataFrame([
                            {'Mese': m, 'Recensioni': d['count'], 'Rating': d['rating_medio']}
                            for m, d in list(trend_temporale.items())[-12:]
                        ])
                        st.line_chart(df_trend.set_index('Mese'))
                
                with tab6:
                    st.markdown("### 🎯 Leve Marketing")
                    for i, leva in enumerate(risultati.get('leve_marketing', [])[:10], 1):
                        st.markdown(f"**{i}.** {leva}")
                
                with tab7:
                    st.markdown("### 📊 Strategie Digital")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown("#### 🌐 Local SEO")
                        for sug in risultati.get('suggerimenti_local_seo', [])[:5]:
                            st.markdown(f"• {sug}")
                        st.markdown("#### 📢 Google Ads")
                        for sug in risultati.get('suggerimenti_google_ads', [])[:5]:
                            st.markdown(f"• {sug}")
                    with col_d2:
                        st.markdown("#### ⭐ Reputation")
                        for sug in risultati.get('suggerimenti_reputation', [])[:5]:
                            st.markdown(f"• {sug}")
                        st.markdown("#### 🔄 CRO GMB")
                        for sug in risultati.get('suggerimenti_cro', [])[:5]:
                            st.markdown(f"• {sug}")
                
                with tab8:
                    st.markdown("### 🔍 Parole Chiave")
                    cols = st.columns(3)
                    for i, parola in enumerate(risultati.get('parole_chiave', [])[:15]):
                        with cols[i % 3]:
                            st.markdown(f"🔸 **{parola}**")
                
                # DOWNLOAD
                st.markdown("## 📥 Download Report")
                
                excel_data = crea_excel_download_avanzato_google(
                    recensioni_data,
                    risultati,
                    clusters,
                    frequenze,
                    analisi_owner,
                    trend_temporale,
                    business_data
                )
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📊 Scarica Report Excel",
                        data=excel_data,
                        file_name=f"GoogleReviews_{business_data['nome'].replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
                
                with col_dl2:
                    json_report = {
                        'business_info': business_data,
                        'metadata': {
                            'n_recensioni': len(recensioni_data),
                            'rating_medio': float(rating_medio),
                            'tasso_risposta': float(tasso_risposta)
                        },
                        'insights': risultati
                    }
                    
                    st.download_button(
                        label="💾 Scarica Report JSON",
                        data=json.dumps(json_report, indent=2, ensure_ascii=False),
                        file_name=f"GoogleReviews_{business_data['nome'].replace(' ', '_')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    with col2:
        st.markdown("## 📋 Guida Rapida")
        st.markdown("""
        ### 🎯 Funzionalità:
        - 🔍 Ricerca automatica attività
        - 💬 Analisi risposte owner
        - 📈 Trend temporale
        - ⭐ Rating distribution
        - 🎯 Local SEO insights
        
        ### 🚀 Come Usare:
        1. Inserisci API Keys
        2. Nome attività esatto
        3. Location precisa
        4. Avvia analisi
        5. Esplora risultati
        6. Scarica report
        
        ### ⏱️ Tempo:
        • 50 rec: ~2-3 min
        • 100 rec: ~4-5 min
        • 200+ rec: ~8-10 min
        """)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚡ Best Practices</h4>
            <p>• Nome esatto da Google Maps</p>
            <p>• Città precisa per risultati</p>
            <p>• Analizza competitor</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Google Reviews Analyzer PRO v1.0 - Powered by DataForSEO & OpenAI</p>
        <p>Sviluppato per Local Business Marketing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
