import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy import stats
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def cross_validation_rmse(model,X_train,y_train):
            scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv = 12, scoring = "neg_mean_squared_error"))
            mean = np.mean(scores)
            print("Mean CV score (rmse): ",mean)
            
def rmse(y_pred, y_test): 
            rmse_ = np.sqrt(mean_squared_error(y_pred,y_test))
            print("rmse: ", rmse_)
            
def real_vs_pred(y_train,y_pred):
            fig = plt.figure(figsize=(12,12))
            fig, ax = plt.subplots()
            ax.scatter(y_train, y_pred,color = "blue",edgecolor = 'black')
            ax.plot([y_train.min(),y_train.max()], [y_train.min(), y_train.max()], 'k--',lw=0.2)
            ax.set_xlabel('Real')
            ax.set_ylabel('Predicción')
            plt.suptitle("Gráfico de dispersión real frente a predicción",size=14)
            plt.show()

def r_cuadrado(y_test,y_pred):
        r2=r2_score(y_test, y_pred)
        print("R-cuadrado: {:.2f}".format(r2))


def distribucion(data):
  asimetria = stats.skew(data)
  curtosis = stats.kurtosis(data)
  x = np.linspace(data.min(), data.max(), 100)
  y = stats.norm(data.mean(), data.std()).pdf(x)
  fig, ax1 = plt.subplots(figsize=(5, 3))
  ax1.hist(data, density=True, bins=20, color='blue', alpha=0.7, edgecolor='black')
  ax1.plot(x, y, color='red', linestyle='--', linewidth=2, label='Distribución normal')
  ax1.set_xlabel('Valores')
  ax1.set_ylabel('Densidad')
  ax1.set_title('Distribución de los datos')
  ax1.legend()
  plt.grid(True)
  print(f"Asimetría: {asimetria}")
  print(f"Curtosis: {curtosis}")
  plt.tight_layout()
  plt.show()

def aplicar_logaritmos(df):
    df_copy = df.copy()
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values > 10 and all(df[col] > 0):
            df_copy[col] = np.log1p(df[col])
    return df_copy
  
def boxplot_por_categoria(dataframe, columna_numerica, columna_categorica):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=columna_categorica, y=columna_numerica, data=dataframe, palette="Set3")
    plt.title(f'Boxplot de {columna_numerica} por {columna_categorica}', fontsize=16)
    plt.xlabel(columna_categorica, fontsize=14)
    plt.ylabel(columna_numerica, fontsize=14)
    plt.show()

