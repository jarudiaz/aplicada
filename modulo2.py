import pandas as pd
from afinn import Afinn

# Inicializa el objeto AFINN
afinn = Afinn()

# Carga de dataset
data = pd.read_csv('test_data.csv')


# Funcion para obtener los puntajes según las ecuaciones (8), (9), (10)
"""Se separan las palabras que contiene el tweet para poder iterar sobre las mismas, 
asignar el score para cada una de ellas y calcular el score de cada tweet"""
def afinn_sentiment_scores(tweet):
    words = tweet.split()
    pos_score = 0
    neg_score = 0
    for word in words:
        score = afinn.score(word)
        if score > 0:
            pos_score += score  # Ecuación (9)
        elif score < 0:
            neg_score += -score  # Ecuación (10)
    return pos_score, neg_score

# Aplica la función a cada tweet y agrega las columnas TweetPos y TweetNeg
data['TweetPos'], data['TweetNeg'] = zip(*data['sentence'].apply(afinn_sentiment_scores))

# Guarda el dataset modificado
data.to_csv('dataset_con_afinn_puntajes.csv', index=False)

# Mostra las primeras filas del nuevo dataset
print(data)