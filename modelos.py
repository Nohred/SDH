import warnings
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Ignorar todos los warnings
warnings.filterwarnings("ignore")
# Ruta del archivo CSV

names = ('features_12V_12H', 'features_6V_12H','features_12V_16H','features_6V_16H')
redcts = ('pca','lda','ica') 


for i in names:
    for j in redcts:
        ruta = i+'_'+j+'.csv'
        print('############################################')
        print('--------------'+i+j+'----------------')
        print('############################################')
        
        # Leer el archivo CSV
        tabla = pd.read_csv(ruta)
        
        # División de los datos en X (características) e Y (etiquetas)
        X = tabla.iloc[:, :-1].values  # Todas las columnas excepto la última
        Y = tabla.iloc[:, -1].values    # Solo la última columna
        
        # Definición de métricas personalizadas para validación cruzada
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted')
        }
        
        
        # ---------------- ÁRBOL DE DECISIÓN ----------------
        # Crear un clasificador de árbol de decisión
        clf_dt = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=5,
            min_samples_split=4,
            random_state=42
        )
        scores_tree = cross_validate(clf_dt, X, Y, cv=5, scoring=scoring)
        
        # ---------------- BOSQUE ALEATORIO ----------------
        # Crear un clasificador RandomForest
        clf_rf = RandomForestClassifier(
            n_estimators=100,  # Número de árboles
            criterion='gini',
            max_depth=5,
            min_samples_split=4,
            random_state=42
        )
        scores_rf = cross_validate(clf_rf, X, Y, cv=5, scoring=scoring)
        
        # ---------------- ADABOOST ----------------
        # Crear un clasificador AdaBoost
        stomp = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=3,
            min_samples_split=4,
            random_state=42
        )
        
        clf_ab = AdaBoostClassifier(
            estimator = stomp,
            n_estimators=50,  # Número de clasificadores base
            learning_rate=0.3,
            random_state=42
        )
        scores_adaboost = cross_validate(clf_ab, X, Y, cv=5, scoring=scoring)
        
        # Archivo Utilizado
        
        
        # Resultados Promedios
        print('Resultados Promedios:')
        print(f'Árbol de Decisión - Accuracy: {np.mean(scores_tree["test_accuracy"]):.2f}, Precision: {np.mean(scores_tree["test_precision"]):.2f}, Recall: {np.mean(scores_tree["test_recall"]):.2f}')
        print(f'Bosque Aleatorio - Accuracy: {np.mean(scores_rf["test_accuracy"]):.2f}, Precision: {np.mean(scores_rf["test_precision"]):.2f}, Recall: {np.mean(scores_rf["test_recall"]):.2f}')
        print(f'AdaBoost - Accuracy: {np.mean(scores_adaboost["test_accuracy"]):.2f}, Precision: {np.mean(scores_adaboost["test_precision"]):.2f}, Recall: {np.mean(scores_adaboost["test_recall"]):.2f}')

# Función para generar y mostrar la matriz de confusión para cada modelo
def plot_confusion_matrix(model, X, Y):
    from sklearn.model_selection import cross_val_predict
    Y_pred = cross_val_predict(model, X, Y, cv=5)
    C_matrix = confusion_matrix(Y, Y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=C_matrix)
    disp.plot()
    plt.show()


tabla = pd.read_csv(r'features_12V_12H_lda.csv')
        
# División de los datos en X (características) e Y (etiquetas)
X = tabla.iloc[:, :-1].values  # Todas las columnas excepto la última
Y = tabla.iloc[:, -1].values    # Solo la última columna

# Mostrar las matrices de confusión del mejor modelo
print("\nMatriz de Confusión - Best Model")
plot_confusion_matrix(clf_rf, X, Y)


