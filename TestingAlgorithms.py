'''
Esse script foi utilizado para  testar 2 algoritmos, "Logistic Regression"
e " K Nearest Neighbor". Consiste em :
    1 - Importar o conjunto de dados para aplicar a tarefa;
    2 - Utilizar o método "bag of words" para deixar o conjunto de dados
    própria para aplicar o algoritmo;
    3- Limpar alguns atributos do conjunto de dados;
    4- Salvar a o conjunto de dados por segurança e para análises futuras;
    5- Testar os algoritmos e
    6- Apresentar o resultado do teste em forma de gráfico.

Biblioteca(s) utilizada(s) :
    Pandas = biblioteca de Data Science utilizada na manipualação de arquivos em csv e xls.
    SkLearn = Biblioteca utilizada para a apliacação de  Machine Learning;
    MatplotLib = Biblioteca para a programação de gráficos.

'''

import  pandas as pd
from sklearn.model_selection import  cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

'''
PRÉ PROCESSAMENTO

'''
# Importa o conjunto de dados.
df = pd.read_csv("Youtube02-KatyPerry.csv",index_col="AUTHOR",parse_dates=["DATE"])

# objeto da classe  CountVectorizer, esta tranaforma o conjunto de dados.
# ou parte dele em un bag of words .
count_vect = CountVectorizer()

# Faz o "encaixe" dos dados contidos na coluna "CONTENT" do conjunto de dados.
X_count= count_vect.fit_transform(df.CONTENT)

# objeto da classe  TfidfTransformer (term frequency–inverse document frequency), esta limpa o "bag of words",
# tirando as palavras mais "comuns" e e sem muita relevância.
tfidf_transformer = TfidfTransformer()

#Faz o "encaixe" dos da matriz contida no "bag of words".
X_tfidf = tfidf_transformer.fit_transform(X_count)


# Salva o "bag of words"  em um arquivo csv
d = pd.DataFrame(X_count.A,columns=count_vect.get_feature_names())
d.to_csv("newD.csv")

'''

TESTE DE ALGORITMOS PARA A TAREFA DE CLASSIFICAÇÃO

'''
# Número de iterações que  determina o  estratégia de separaçao do cross-validation splitting strategyy
cross_validation_range = list(range(2,50))

# Lista que armazena os valores de cada iteração do cross-validation
result_LR = []

# Objeto da classe Logistic Regression, que é responsável por aplicar o algortimo de logist regression
#na matriz formada anteriormente.
lr = LogisticRegression()

#Laço que itera o cross validation-scores com os valores determinados anteriormente,
#utlizando a precisão como métrica e armazena o resultado na lista.
for i in cross_validation_range:
    scores = cross_val_score(lr, X_tfidf,df.CLASS, cv =i, scoring='accuracy')
    result_LR.append(scores.mean())


# Número de iterações que  determina o  estratégia de separaçao do cross-validation splitting strategyy
k_range = list(range(1,50))

# Lista que armazena os valores de cada iteração do cross-validation
k_scores = []


#Laço que itera o número de vizinhos  com os valores determinados anteriormente,
#utlizando a precisão como métrica e armazena o resultado na lista.
for i in k_range:
    knn = KNeighborsClassifier ( n_neighbors=i )
    scores = cross_val_score(knn, X_tfidf,df.CLASS, cv =20, scoring='accuracy')
    k_scores.append(scores.mean())


'''

VIZUALIZAÇÃO DO RESULTADO DO TESTE

'''

#Plota o resultado da execução do logistic regression
plt.plot(cross_validation_range,result_LR,color='red',marker = '.',markersize = 7,label="Logistic Regression")

#Plota o resultado da execução do k nearest neighbor
plt.plot(k_range,k_scores,color='blue',marker = '.',markersize = 7,label="K-Neighbors")

# Coloca legenda no gráfico
plt.xlabel('Iteration/n neighbors')
plt.ylabel('Accuracy')

#Habilita a legenda
plt.legend()

#Mostra o gráfico
plt.show()
