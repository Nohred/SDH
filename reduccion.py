import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# Ruta del archivo CSV
names = ('features_12V_12H', 'features_6V_12H','features_12V_16H','features_6V_16H')
for i in range(0,4):
    
    ruta = 'C:\\Personal Local\\Recuperacion\\Escuela\\5to Semestre\\Machine Learning\\convolucional\\'+names[i]+'.csv'
    
    # Leer el archivo CSV
    tabla = pd.read_csv(ruta)
    
    # División de los datos en X (características) e Y (etiquetas)
    X = tabla.iloc[:, :-1].values  # Todas las columnas excepto la última
    Y = tabla.iloc[:, -1].values    # Solo la última columna
    
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ---------------- PCA (Análisis de Componentes Principales) ----------------
    pca = PCA(n_components=0.90)  # Retener el 90% de la varianza
    X_pca = pca.fit_transform(X_scaled)
    
    # Guardar resultados de PCA
    pca_results = pd.DataFrame(X_pca)
    pca_results['Y'] = Y
    pca_results.to_csv(names[i]+'_pca.csv', index=False)
    
    # ---------------- LDA (Análisis Discriminante Lineal) ----------------
    lda = LDA(n_components=6)  # Ajustar según el número de clases
    X_lda = lda.fit_transform(X_scaled, Y)
    
    # Guardar resultados de LDA
    lda_results = pd.DataFrame(X_lda)
    lda_results['Y'] = Y
    lda_results.to_csv(names[i]+'_lda.csv', index=False)
    
    # ---------------- ICA (Análisis de Componentes Independientes) ----------------
    ica = FastICA(n_components=6, random_state=1)
    X_ica = ica.fit_transform(X_scaled)
    
    # Guardar resultados de ICA
    ica_results = pd.DataFrame(X_ica)
    ica_results['Y'] = Y
    ica_results.to_csv(names[i]+'_ica.csv', index=False)
    
    # Resultados
    print(f'\nDimensiones originales: {X.shape}')
    print(f'Dimensiones después de PCA: {X_pca.shape}')
    print(f'Dimensiones después de LDA: {X_lda.shape}')
    print(f'Dimensiones después de ICA: {X_ica.shape}')