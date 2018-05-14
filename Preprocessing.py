'''
Pré processamento da massa de  dados

Biblioteca(s) utilizada(s) :
    Pandas = biblioteca de Data Science utilizada na manipualação de arquivos em csv e xls.
'''

#biblioteca(s) importada(s )
import pandas as pd

'''
Classe de pré processamento dos dados :
    Consiste em
     1 - Ler as 5 conjuntos de dados que serão utilizadas na mineração dos dados;
     2 - fazer uma limpeza de cada base, preenchendo os dados faltantes na coluna "DATE"
     com o preenchimento "back-forward";
     3 - Adicionar a coluna "ARTIST", contendo o nome artista a qual cada amostra pertence e 
     4 - Concatenar todos os conjunto de dados  em um só.
'''
class Pre_processing ():

    '''
    Método responsável por ler os conjuntos de dados
    utilizando a operação "read_csv" da biblioteca pandas.

    :parameter self
    :return void
    '''
    def import_files(self):
        self.df1 = pd.read_csv ( "Youtube01-Psy.csv" )
        self.df2 = pd.read_csv ( "Youtube02-KatyPerry.csv" )
        self.df3 = pd.read_csv ( "Youtube05-Shakira.csv" )
        self.df4 = pd.read_csv ( "Youtube04-Eminem.csv" )
        self.df5 = pd.read_csv ( "Youtube03-LMFAO.csv" )
    '''
    Método responsável por concatenar todos os conjuntos de dados,
    formando um só. Consiste em :
        1- Criar uma lista com o nome do artista juntamente 
        de seu respectivo conjunto de dados;
        2- Chamar a operação que insere a coluna "ARTIST";
        3- Chamar a operação que concatena os conjuntos de dados e
        coloca o novo conjunto de dados  na variável "new_csv";
        4- Cria um novo arquivo csv contendo os conjuntos concatenados
        utlizando a operação "to_csv" da biblioteca pandas.
        
        :parameter : self
        :return void
    '''
    def concatenate_files(self):
        list_files = [
            [ "PSY" , self.df1 ] ,
            [ "KATY PERRY" , self.df2 ] ,
            [ "SHAKIRA" , self.df3 ] ,
            [ "EMINEM" , self.df4 ] ,
            [ "LMFAO" , self.df5 ] ,
        ]

        self.insert_artist_column ( list_files )

        new_csv = pd.concat ( [ file[ 1 ] for file in list_files ] )
        n_list = self.delete_no_useful_data ( new_csv )
        new_csv.to_csv ( "merged.csv" )

    '''
    Método responsável por fazer uma limpeza dos dados na coluna "DATE" utilzando a operação 
    "fillna" da classe pandas, utilzando o método "bfill".
    :parameter self, list_files
    :return void
    '''

    def fill_missing_data(self , list_files):

        list_files[ "DATE" ].fillna ( method="bfill" , inplace=True )

    '''
    Método responsável por fazer uma limpeza de dados, removendo a coluna "COMMENT_ID" 
    utilzando a operação "drop" pandas.
    :parameter self, database
    :return database
    '''

    def delete_no_useful_data(self , database):
        database.drop ( 'COMMENT_ID' , axis=1 , inplace=True )
        return database
    '''
     Método responsável por inserir a coluna "ARTIST" no conjunto de dados. Consiste em :
        Para cada conjunto de dados :
        1- Cria a coluna vazia associando os valores ao tipo "nan"( null na biblioteca pandas";
        2- Preenche essa coluna com o nome do artista utilizando a operação "fill_missing_data",
        da própria classe;
    :parameter self, list_files
    :return void
    '''
    def insert_artist_column(self , list_files):

        list_files = list ( list_files )
        if (len ( list_files ) == 0):
            return
        else:

            list_files[ -1 ][ 1 ][ "ARTIST" ] = pd.np.nan
            self.fill_missing_data ( list_files[ -1 ][ 1 ] )
            list_files[ -1 ][ 1 ].fillna ( list_files[ -1 ][ 0 ] , inplace=True )
            list_files.pop ( -1 )
            self.insert_artist_column ( list_files )

p = Pre_processing()
p.import_files()
p.concatenate_files()