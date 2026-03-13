# Alumna: Acevedo Medina Andrea Montserrat
# Grupo: 6AM2

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def clasificador_humano_v2(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Versión mejorada del clasificador humano.

    Parámetros:
    -----------
    bill_length_mm : float
    bill_depth_mm : float
    flipper_length_mm : float
    body_mass_g : float

    Retorna:
    --------
    str : 'Adelie', 'Chinstrap', o 'Gentoo'

    DOCUMENTA TUS CAMBIOS:
    En la gráfica scatterplot longitud de pico, se ve que el 44 separa a la Adelie y a la Chinstrap.
    Para distinguirlas de Gentoo se identificó que los Gentoo tienen aletas mayores a 207 mm.
    Algunos Chinstrap también pueden tener aletas largas, pero se distinguen por la profundidad del pico,
    que suele ser mayor a 17.5 mm.
    """

    if flipper_length_mm < 206.5:
        if bill_length_mm < 43.4:
            return "Adelie"
        else:
            return "Chinstrap"
    else:
        if bill_depth_mm < 17.6:
            return "Gentoo"
        else:
            return "Chinstrap"


def recibir_csv(archivo_csv):
    """
    Lee el CSV y limpia valores faltantes
    """
    df = pd.read_csv(archivo_csv)

    df = df[['bill_length_mm',
             'bill_depth_mm',
             'flipper_length_mm',
             'body_mass_g',
             'species']]

    df = df.dropna().reset_index(drop=True)

    return df


def split(df):
    """
    Divide dataset en entrenamiento y prueba
    """

    X = df[['bill_length_mm',
            'bill_depth_mm',
            'flipper_length_mm',
            'body_mass_g']]

    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y )

    return X_train, X_test, y_train, y_test


def clasificador_maquina(X_train, X_test, y_train):
    """
    Entrena el árbol de decisión y genera predicciones
    """

    modelo_ml = DecisionTreeClassifier(random_state=42, max_depth=4)
    modelo_ml.fit(X_train, y_train)
    predicciones_ml = modelo_ml.predict(X_test)

    return predicciones_ml


def predicciones_humanas(X_test):
    """
    Aplica el clasificador humano fila por fila
    """

    preds = []

    for _, fila in X_test.iterrows():

        pred = clasificador_humano_v2(
            fila['bill_length_mm'],
            fila['bill_depth_mm'],
            fila['flipper_length_mm'],
            fila['body_mass_g']
        )

        preds.append(pred)

    return preds


def crear_csv(pred_ml, pred_humano):

    df_resultado = pd.DataFrame({
        "prediccion_maquina": pred_ml,
        "prediccion_humano": pred_humano
    })

    df_resultado.to_csv("predicciones.csv", index=False)
    print("Archivo predicciones.csv generado correctamente")


def main():

    if len(sys.argv) < 2:
        print("Uso: python main.py penguins.csv")
        sys.exit()

    archivo = sys.argv[1]
    df = recibir_csv(archivo)

    X_train, X_test, y_train, y_test = split(df)
    pred_ml = clasificador_maquina(X_train, X_test, y_train)
    pred_humano = predicciones_humanas(X_test)
    crear_csv(pred_ml, pred_humano)


if __name__ == "__main__":
    main()