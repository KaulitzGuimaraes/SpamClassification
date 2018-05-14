'''
Script para  transformar conjunto de dados em bag of words, aplicar o algoritmo e vizualizar
resultado da classificação.

Segue os mesmos passos do script de teste, com a diferença de não ter necessidade de aplicar
cross validation.

Bibliotecas usadas :
 Pandas = biblioteca de Data Science utilizada na manipualação de arquivos em csv e xls.
 SkLearn = Biblioteca utilizada para a apliacação de  Machine Learning;
 Numpy = Biblioteca para trabalhar com opereções matemáticas com mais eficiências
'''

import  pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv("merged.csv",index_col="ARTIST",parse_dates=["DATE"])
lr = LogisticRegression()
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


# Nota : 1 = Eminem, 2 = LMFA) conj. = conjunto de dados

'''
Separa em três partes o conjunto de dados :
1 - training_data = conj. a ser treinado
2 - predict_data1 = conj. a ser testado
3 - predict_data2 = onj. a ser testado
'''
traing_data,predict_data1 = [x for _, x in df.groupby(df.index == 'EMINEM')]
traing_data,predict_data2 = [x for _, x in df.groupby(df.index == 'LMFAO')]

# Processo de tranformar o conj. para treinar  em bag of words
count_bag_train_c = count_vect.fit_transform(traing_data.CONTENT)
count_bag_train_t = tfidf_transformer.fit_transform(count_bag_train_c)

# Processo de tranformar o conj. para testar   em bag of words
count_vect2 = CountVectorizer(vocabulary=count_vect.get_feature_names())
count_bag_pred1 = count_vect2.fit_transform(predict_data1.CONTENT)
count_bag_pred2 = count_vect2.fit_transform(predict_data2.CONTENT)
count_bag_pred1 = tfidf_transformer.fit_transform(count_bag_pred1)
count_bag_pred2 = tfidf_transformer.fit_transform(count_bag_pred2)

#Treina o algoritmo com o conj. para treinar e o vetor com os rótulos
#contidos na coluna "CLASS" do conj. inicial
lr.fit(count_bag_train_t,traing_data.CLASS)

#Faz o teste do algortimos com os conj. 1 e 2
y_pred1 = lr.predict(count_bag_pred1)
y_pred2 = lr.predict(count_bag_pred2)
y_pred1 = list(y_pred1)
y_pred2 = list(y_pred2)

# Calcula e mostra a porcentagem da precisão de cada con. de teste,
# utliza a classe metrics.
print("Eminem : ",(metrics.accuracy_score(predict_data1.CLASS,y_pred1)* 100),'%' )
print("LMFAO : " , (metrics.accuracy_score(predict_data2.CLASS,y_pred2)* 100),'%' )

# Mostra um gráfico de barras com os resultados, onde y corresponde aos artistas e x ao
# número de spam/não spam de cada um deles.
artists =  [ "Eminem","LMFAO"]
xpos = np.arange( len( artists ) )
plt.xticks( xpos , artists ) # replace the elemenst in the legend
plt.bar( xpos[0]-0.2 , [y_pred1.count(1)] ,width = 0.4, color ='black' , label =("Spam (" + "Eminem : " + str(y_pred1.count(1)) + ")"))
plt.bar( xpos[0]+0.2 , [y_pred1.count(0)],width = 0.4, label=("Not spam (" + "Eminem : " + str(y_pred1.count(0)) + ")"))
plt.bar( xpos[1]-0.2 , [y_pred2.count(1)] ,width = 0.4 , label =("Spam (" + "LMFAO : " + str(y_pred2.count(1)) + ")"))
plt.bar( xpos[1]+0.2 , [y_pred2.count(0)],width = 0.4, label=("Not spam (" + "LMFAO : " + str(y_pred2.count(0)) + ")"))
plt.ylabel("Classification")
plt.title('Artists')
plt.legend(loc = 'lower left')
#plt.show()