import pandas as pd
import numpy as np
import streamlit as st

# Ruta del archivo Excel
archivo_excel = 'Rendimientos_bonos_internacionales_USD.xlsx'

# Crear un diccionario vacío para almacenar los DataFrames
dfs = {}

# Cargar el archivo Excel
xls = pd.ExcelFile(archivo_excel)

# Iterar sobre cada hoja del archivo
for hoja in xls.sheet_names:
    # Leer cada hoja y almacenarla en el diccionario
    dfs[hoja] = pd.read_excel(xls, hoja, header=0)


# Se detectaron errores en las fechas
# - 31 de septiembre
# - Fecha con parentesis
# - Fecha que escribieron 023 en vez de 2023

for nombre, df in dfs.items():
    if nombre == 'Vencimientos':
        continue
    else:
        try:
            # Intentar convertir la columna 'Fecha' a datetime
            df['Fecha'] = pd.to_datetime(df['Fecha'], format= 'mixed', dayfirst=True)
        except Exception as e:
            # Imprimir el nombre de la hoja y el error si algo sale mal
            print(f"Error al procesar la hoja '{nombre}': {e}")


# Inicializar el DataFrame final con el primer DataFrame del diccionario
df_final = None
df_vencimientos = dfs['Vencimientos']
dfs.pop('Vencimientos')

# Iterar sobre cada DataFrame y unirlos
for nombre, df in dfs.items():
    if df_final is None:
        df_final = df
    else:
        # Unir utilizando la columna 'Fecha' como referencia y haciendo una unión 'outer'
        df_final = pd.merge(df_final, df, on='Fecha', how='outer', suffixes=('', f'_{nombre}'))


df_final.set_index('Fecha', inplace=True)
df_final.sort_index(inplace=True)

# Eliminar filas que no tienen al menos 3 valores no nulos
df_limpio = df_final.dropna(thresh=3)


# Fecha que buscamos analizar de los bonos
fecha_a_analizar = st.sidebar.date_input('Fecha de análisis', df_limpio.index[-1])
fecha_a_analizar = pd.Timestamp(fecha_a_analizar)


# Extraer la fila correspondiente 
fila_analisis = df_limpio.loc[fecha_a_analizar]

# Transponer la fila para convertirla en una columna
df_analisis = fila_analisis.transpose().to_frame()

# Renombramos la columna de rendimiento
df_analisis.columns = ['Yield']

# Extraer los vencimientos
df_analisis['Maturity'] = pd.to_datetime(df_vencimientos['Fecha de vencimiento'], dayfirst= True) - pd.to_datetime(fecha_a_analizar, dayfirst= True)

# Calculamos el maturity en años (restamos las fechas y dividimos por 360)
diferencia = pd.to_datetime(df_vencimientos['Fecha de vencimiento'], dayfirst=True) - pd.to_datetime(fecha_a_analizar, dayfirst=True)
diferencia_anos = diferencia.dt.days / 360
df_analisis['Maturity'] = diferencia_anos.values

# Convertir el rendimiento a porcentaje
df_analisis['Yield'] = df_analisis['Yield'] / 100


#############################################
# Aplicamos el modelo de Nelson-Siegel
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import matplotlib.ticker as mtick

dd = df_analisis.copy()
dd.sort_values('Maturity', inplace=True)
df = dd.copy()

sf = df.copy()
sf = sf.dropna()
sf1 = sf.copy()
sf1['Y'] = round(sf['Yield']*100,4)
sf = sf.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.4%}'})


β0 = 0.01
β1 = 0.01
β2 = 0.01
λ = 1.00


df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))

df['Residual'] =  (df['Yield'] - df['NS'])**2
df22 = df[['Maturity','Yield','NS','Residual']]  


def myval(c):
    df = dd.copy()
    df['NS'] =(c[0])+(c[1]*((1-np.exp(-df['Maturity']/c[3]))/(df['Maturity']/c[3])))+(c[2]*((((1-np.exp(-df['Maturity']/c[3]))/(df['Maturity']/c[3])))-(np.exp(-df['Maturity']/c[3]))))
    df['Residual'] =  (df['Yield'] - df['NS'])**2
    val = np.sum(df['Residual'])
    print("[β0, β1, β2, λ]=",c,", SUM:", val)
    return(val)
    
c = fmin(myval, [0.01, 0.00, -0.01, 1.0])


β0 = c[0]
β1 = c[1]
β2 = c[2]
λ = c[3]


df = df.copy()
df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100,4)
sf5['N'] = round(sf4['NS']*100,4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.2%}', 'NS': '{:,.2%}'})
M0 = 0.00
M1 = 3.50
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import matplotlib.ticker as mtick
fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("USD international bonds",fontsize=fontsize)
fig.patch.set_facecolor('white')
X = sf5["Maturity"]
Y = sf5["Y"]
x = sf5["Maturity"]
y = sf5["N"]
plt.plot(x, y, color="orange", label="Svensson model")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturity (years)',fontsize=fontsize)
plt.ylabel('Yield (%)',fontsize=fontsize)
plt.legend(loc="lower right")
plt.grid()


st.title('Yield Curve')
st.pyplot(fig)



