import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Función para calcular el Índice de Fuerza Relativa (RSI)
def compute_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función para calcular el Oscilador Estocástico
def compute_stochastic_oscillator(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    return k

# Función para tomar decisiones de trading basadas en RSI y Estocástico

def trading_decision(rsi, k, date):
    if rsi.loc[date] < 30 and k.loc[date] < 20:
        return 'Comprar'
    elif rsi.loc[date] > 70 and k.loc[date] > 80:
        return 'Vender'
    else:
        return 'Esperar'
    
def decision_rsi(rsi, date):
    if rsi.loc[date] < 30:
        return 'Comprar'
    elif rsi.loc[date] > 70:
        return 'Vender'
    else:
        return 'Esperar'

def decision_stochastic(k, date):
    if k.loc[date] < 20:
        return 'Comprar'
    elif k.loc[date] > 80:
        return 'Vender'
    else:
        return 'Esperar'

# Titulo y descripción    
st.title('Análisis técnico de la soja')
st.markdown("""
Si el RSI es menor que 30 y el Estocástico es menor que 20, entonces es una señal de compra. <br>
Si el RSI es mayor que 70 y el Estocástico es mayor que 80, entonces es una señal de venta. <br>
De lo contrario, se mantiene en hold.
""", unsafe_allow_html=True)


# Streamlit widgets para recibir inputs
start_date = st.sidebar.date_input('Fecha de inicio', pd.to_datetime('2023-01-01'))
start_date_minus_30 = start_date - pd.Timedelta(days=150)
end_date = st.sidebar.date_input('Fecha de fin', pd.to_datetime('today'))
ticker = st.sidebar.text_input('Ticker', 'ZS=F')
decision_date = st.sidebar.date_input('Fecha de decisión', (pd.to_datetime('today') - pd.Timedelta(days=1)))
decision_date = decision_date.strftime("%Y-%m-%d")
# decision_date = (pd.to_datetime('today') - pd.Timedelta(days=1)).strftime("%Y-%m-%d")



# Cargar datos usando yfinance (o usar df si ya está pre-cargado)
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

df = load_data(ticker, start_date_minus_30, end_date)

################
# Procesamiento de datos y nuevas variables

# Calculamos la media móvil simple de 30 días para el precio de cierre.
df['SMA_30'] = df['Close'].rolling(window=30).mean()

# Calculamos la media móvil simple de 100 días para el precio de cierre.
df['SMA_100'] = df['Close'].rolling(window=100).mean()

# Agregamos una columna que indica el cruce
df['Crossover'] = np.where(df['SMA_30'] > df['SMA_100'], 'Alcista', 'Bajista')

# Encontramos los índices donde ocurren los cruces
crossover_indices = df['Crossover'].ne(df['Crossover'].shift())

# Filtramos el DataFrame para mostrar solo las fechas con cruces
crossover_df = df[crossover_indices]

# Calcular RSI y Estocástico
df['RSI'] = compute_rsi(df['Close'])
df['Estocastico'] = compute_stochastic_oscillator(df)

df = df[df.index >= pd.to_datetime(start_date)]

# Asumiendo que has añadido las columnas 'SMA_30', 'SMA_100', etc. a tu DataFrame 'df'

# Crear figuras con Matplotlib para mostrar en Streamlit
fig, ax = plt.subplots()
ax.plot(df['Close'], label='Precio de Cierre')
ax.plot(df['SMA_30'], label='Media Móvil 30 Días')
ax.plot(df['SMA_100'], label='Media Móvil 100 Días')

# Añadimos los puntos de intersección
for date, row in crossover_df.iterrows():
    ax.plot(date, row['SMA_30'], 'ro' if row['Crossover'] == 'Alcista' else 'go')  # 'ro' = red circle, 'go' = green circle


ax.set_title('Análisis técnico de los precios de la soja')
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio de cierre')
ax.legend()

# Mostrar gráficos con Streamlit
st.pyplot(fig)

# Plot
fig2, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

# Precio de cierre y RSI
axes[0].plot(df['Close'], label='Precio de Cierre', color='blue')
axes[0].set_title('Precio de Cierre y Índice de Fuerza Relativa (RSI)')
axes[0].legend()

# Añadir RSI al gráfico de precios de cierre
ax2 = axes[0].twinx()
ax2.plot(df['RSI'], label='RSI', color='green')
ax2.axhline(70, color='red', linestyle='--', linewidth=1)
ax2.axhline(30, color='red', linestyle='--', linewidth=1)
ax2.legend()

# Oscilador Estocástico
axes[1].plot(df['Estocastico'], label='%Estocastico', color='orange')
axes[1].axhline(80, color='red', linestyle='--', linewidth=1)
axes[1].axhline(20, color='red', linestyle='--', linewidth=1)
axes[1].set_title('Oscilador Estocástico')
axes[1].legend()

st.pyplot(fig2)


# Tomar la decisión basada en la fecha de decisión.
  
if decision_date in df.index:
    decision_ambos = trading_decision(df['RSI'], df['Estocastico'], decision_date)
    decision_rsi_val = decision_rsi(df['RSI'], decision_date)
    decision_stochastic_val = decision_stochastic(df['Estocastico'], decision_date)
else:
    decision_ambos = 'Fecha fuera de rango'
    decision_rsi_val = 'Fecha fuera de rango'
    decision_stochastic_val = 'Fecha fuera de rango'


st.info(f'Decisión basada en RSI para {decision_date}: {decision_rsi_val}')
st.info(f'Decisión basada en Estocástico para {decision_date}: {decision_stochastic_val}')
st.info(f'Decisión basada en ambos para {decision_date}: {decision_ambos}')

######################################
# Análisis de noticias

# Extraer el link relacionado a soja de economies.com
import requests
from bs4 import BeautifulSoup

base_url = 'https://www.economies.com'
url = 'https://www.economies.com/commodities/analysis'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, headers=headers)

links = []

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    for link in soup.find_all('a', href=True):
        if 'soja' in link.text.lower() or 'soybean' in link.text.lower() or 'soja' in link['href'].lower() or 'soybean' in link['href'].lower():
            full_link = base_url + link['href']
            links.append(full_link)
else:
    print(f'Error accessing page: {response.status_code}')


# Ahora que tenemos los links, podemos acceder a cada noticia y extraer el contenido
url_noticia = links[0]
response_noticia = requests.get(url_noticia, headers=headers)
image_url = None
h1_texts = []
p_texts = []

if response_noticia.status_code == 200:
    soup_noticia = BeautifulSoup(response_noticia.content, 'html.parser')

    # Buscamos todos los elementos h1 en el contenido de la respuesta
    for h1 in soup_noticia.find_all('h1'):
        # Añadimos el texto de cada elemento h1 a la lista
        h1_texts.append(h1.get_text())
    
    # Buscamos todos los elementos p1 en el contenido de la respuesta
    for p in soup_noticia.find_all('p'):
        # Añadimos el texto de cada elemento h1 a la lista
        p_texts.append(p.get_text())
    
   

    # Buscamos todos los elementos <a> y luego buscamos un elemento <img> dentro de ellos
    for a_tag in soup_noticia.find_all('a', attrs={"@click": True}):
        img_tag = a_tag.find('img')
        if img_tag and 'src' in img_tag.attrs:
            image_url = img_tag['src']
            break  # Suponiendo que solo necesitas la primera imagen que coincida

else:
    print(f'Error accessing page: {response_noticia.status_code}')

# Limpiamos el contenido de la noticia
title = h1_texts[0].strip()
texto = [p.strip() for p in p_texts if p.strip()]
texto = ' '.join(texto)

# Mostramos la noticia
st.subheader(f'Noticia: {title}')
st.image(image_url, caption='Análisis tecnico de la soja', use_column_width=True)
st.write(f'{texto}')
st.write(f'Link: {url_noticia}')

# st.title('Esto es title')
# st.header('Esto es header')
# st.subheader('Esto es subheader')
# st.write('Esto es write')
# st.markdown('Esto es markdown')
# st.success('Esto es success')
# st.info('Esto es info')





