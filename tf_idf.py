# Paula Beatriz Louback Jardim

# ------------------------------------------------------
### Enunciado ###

# Sua tarefa será gerar a matriz termo-documento usando TF-IDF por meio da aplicação das
# fórmulas TF-IDF na matriz termo-documento criada com a utilização do algoritmo Bag of Words.
# Sobre o Corpus que recuperamos anteriormente.

# ------------------------------------------------------
# Realizando os imports:
#!python -m spacy download en_core_web_sm

from bs4 import BeautifulSoup
from requests import get
from spacy import load
import numpy as np
import string
from sklearn.preprocessing import normalize

# ------------------------------------------------------
# Guardando as referências para todos os artigos em uma lista:

artigos = ["https://aliz.ai/en/blog/natural-language-processing-a-short-introduction-to-get-you-started/",
           "https://medium.com/nlplanet/two-minutes-nlp-python-regular-expressions-cheatsheet-d880e95bb468",
           "https://hbr.org/2022/04/the-power-of-natural-language-processing",
           "https://www.activestate.com/blog/how-to-do-text-summarization-with-python/",
           "https://towardsdatascience.com/multilingual-nlp-get-started-with-the-paws-x-dataset-in-5-minutes-or-less-45a70921d709"]

# ------------------------------------------------------
# Esse código percorre todos os documentos, criando uma lista com as sentenças de cada documento
# e ao final, une todas essas listas em uma matriz.
# Ao mesmo tempo, o código abaixo realiza a mesma atividade descrita acima para o vocabulário de cada documento,
# criando uma lista de palavras para cada um e juntando essas listas no final. 

matriz_palavras = [[], [], [], [], []]
matriz_sentencas = []

i = 0
for site in artigos:
  sents_list = []
  document_words = set()

  r = get(site)
  r = r.content

  soup = BeautifulSoup(r, 'html.parser')
  text = soup.find_all('p')
  nlp = load("en_core_web_sm")

  for paragraph in text:
    content = paragraph.get_text()
    sentences = nlp(content).sents

    for sent in sentences:
      sent = sent.text.strip(string.punctuation)
      sent = sent.strip(string.digits)
      sent = sent.strip('\n')
      sents_list.append(sent)
      words = sent.split(" ")
      
      for word in words:
        word = word.strip(string.punctuation)
        word = word.strip(string.digits)
        word = word.strip('\n')
        document_words.add(word)

  matriz_sentencas.append(sents_list)
  for w in document_words:
    matriz_palavras[i].append(w)

  i += 1

# ------------------------------------------------------
# O trecho abaixo cria um Bag of words unindo todas as palavras de todos os documentos.
# Neste trabalho estou considerando uppercases e lowcases como caracteres diferentes.

corpus = set()

for lists in matriz_palavras:
  for word in lists:
    corpus.add(word)

# ------------------------------------------------------
# Logo abaixo está a criação da header da matriz-termo.

header = []

for each in corpus:
  if each != "":
    header.append(each)

header = sorted(header)

# ------------------------------------------------------
# Agora, vamos criar a matriz-termo.
# Essa matriz está sendo criada em um dicionário, onde cada sentença é uma key e o valor de cada key é uma lista
# na qual os elementos correspondem a quantidade de vezes que cada termo da header aparece nessa sentença.

dict_matriz= {}
dict_matriz["Sentenças"] = header[1:]

for doc in matriz_sentencas:
  for sent in doc:
    values_list = []
    termos = sent.split(" ")
    for palavra in dict_matriz["Sentenças"]:
      contador = 0
      for cada_plv in termos:
        if cada_plv == palavra:
          contador += 1
      values_list.append(contador)
    dict_matriz[sent] = values_list

# ------------------------------------------------------
# No trecho abaixo, estou criando uma lista que armazena a quantidade de sentenças em que cada termo aparece.
# Essa lista vai ser usada na sequência para calcular o IDF.

qnt_sent = np.zeros(len(dict_matriz["Sentenças"]), dtype=float)
qnt_sent = list(qnt_sent)

for sent in dict_matriz:
  if sent != "Sentenças":
    index = 0
    while index < len(dict_matriz[sent]):
      valor = int(dict_matriz[sent][index])
      if valor != 0:
        qnt_sent[index] += 1  
      index += 1

# ------------------------------------------------------
# Agora, vou realizar o cálculo do TF-IDF para todos os termos em cada sentença
# e então substituir esses valores no dicionário onde está minha matriz.

for sent in dict_matriz:
  if sent != "Sentenças":
    termos = sent.split(' ')
    qnt_termos = len(termos) # quantidade de termos na sentença atual

    tfidf = np.zeros(len(dict_matriz["Sentenças"]), dtype=float)
    index = 0
    while index < len(dict_matriz[sent]):
      valor = dict_matriz[sent][index]
      if qnt_sent[index] != 0:
        calculo = (valor/qnt_termos) * np.log((len(dict_matriz) - 1)/qnt_sent[index])
        tfidf[index] = calculo
      index += 1
    tfidf_list = normalize([tfidf]) # normaliza os valores para ficar entre 0 e 1
    dict_matriz[sent] = list(tfidf_list[0])

# ------------------------------------------------------
# Abaixo estou setando os valores pra no máximo 3 casas decimais

for sent in dict_matriz:
  if sent != "Sentenças":
    index = 0
    while index < len(dict_matriz[sent]):
      valor = dict_matriz[sent][index]
      new_valor = float(str(valor)[:5])
      dict_matriz[sent][index] = new_valor
      index += 1

# ------------------------------------------------------
# Como o Colab limita o output por conta do tamanho do dicionario, abaixo estou printando apenas as 50 primeiras linhas da matriz.

loop = 0
for sent in dict_matriz:
    if loop <= 50:
      print(dict_matriz[sent])
    loop += 1