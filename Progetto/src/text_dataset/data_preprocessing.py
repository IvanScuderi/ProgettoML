import nltk.corpus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from src.path_folders import DS_FOLDER_TEXT, DRIVE_FOLDER
from sklearn.feature_extraction.text import TfidfVectorizer

# CARICAMENTO DEL DATASET

df = pd.read_excel(DS_FOLDER_TEXT)

# ELIMINO I DUPLICATI

df = df.drop_duplicates(subset=['OriginalTweet'])

print(f"Mostro il dataset di testi caricato:\n{df.head(20)}\n")

# EFFETTUO OPERAZIONI DI PULITURA DEI RECORD:
# ESSENDO UN DATASET DI TWEET DEVO CONSIDERARE ELEMENTI TIPICI DI QUESTO GENERE DI MSG QUALI HASHTAG, USERNAME, LINK, ECC..

print('Effettuo operazioni di pulitura dei tweet: rimozione hashtag, username, link,...')

def clean_tweet(text):
    text = text.lower()                                                         # PONGO TUTTO MINUSCOLO
    text = re.sub(r'[^\x20-\x7F]+', " ", text)                                  # SOSTITUISCO CON SPAZIO I CARATTERI SPECIALI
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', "", text)     # SOSTITUISCO CON VUOTO I LINK HTTP/S
    text = re.sub(r'[A-Za-z0-9]+@[A-Za-z]*\.?[A-Za-z0-9]*', "", text)           # SOSTITUISCO CON VUOTO GLI INDIRIZZI MAIL
    text = re.sub(r'@([A-Za-z0-9_]+)', "", text)                                # SOSTITUISCO CON VUOTO GLI USERNAME
    text = re.sub(r'-', " ", text)                                              # SOSTITUISCO CON SPAZIO IL CARATTERE '-', SOLITAMENTE DIVISORIO
    #text = "".join([char for char in text if char not in string.punctuation])   # RIMUOVO LA PUNTEGGIATUTA
    text = re.sub(r'[^\w\d\s]+', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub('[0-9]+', '', text)                                           # RIMUOVO EVENTUALI NUMERI
    # A CAUSA DI UN PROBLEMA NON NOTO SPESSO I TWEET RIPULITI PRESENTAVANO SEQUENZE LUNGHE DI 'xdxd' PER QUESTO MOTIVO HO AGGIUNTO QUESTA ULTERIORE REGEX
    text = re.sub('(xd)+', '', text)
    return text

df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: clean_tweet(x))

print(f"Dataset di testi una volta terminata la fase di pulitura dei tweet:\n{df.head(20)}")

# UNA VOLTA TERMINATA LA PULITURA CI POTREBBERO ESSERE RECORD RIMASTI TOTALMENTE VUOTI: VADO AD ELIMINARE TALI RECORD

df = df.replace(r'^\s*$', np.NaN, regex=True)
noneCount = (df['OriginalTweet'].isna()).sum()

print(f'Numero di record completamente vuoti dopo le operazioni di pulitura: {noneCount}\n')
print(str(df.info())+'\n')

df = df.dropna(subset=['OriginalTweet'])

print('Elimino tali record vuoti:\n')
print(str(df.info())+'\n')

# VADO AD EFFETTUARE LA TOKENIZZAZIONE DEI RECORD DEL DATASET E CONTEMPORANEAMENTE ELIMINO LE STOPWORDS

print('Effettuo operazioni di tokenizzazione e rimozione delle stopwords dai record del dataset!')

def tokenize_remove_stopwords(text):
    tokens = word_tokenize(text)
    stpw = stopwords.words('english')
    words = [w for w in tokens if w not in stpw]
    return words

df['CleanTweet'] = df['OriginalTweet'].apply(tokenize_remove_stopwords)

print(f'Record del dataset una volta effettuate le operazioni di tokenizzazione e rimozione delle stopwords:\n{df["CleanTweet"].head(20)}')

# PROBLEMA: A CAUSA DELL'ELIMINAZIONE DELLE STOPWORDS VI POTREBBERO ESSERE RECORD CHE SONO RIMASTI SOSTANZIALMENTE SENZA ALCUN ELEMENTO

print(f'Numero di record rimasti vuoti dopo le operazioni di tokenizzazione e rimozione delle stopwords: {(df["CleanTweet"].str.len()==0).sum()}')

df[df['CleanTweet'].str.len() == 0] = np.NaN
df = df.dropna(subset=['CleanTweet'])

print(f'Dimensione dataset in seguito alla rimozione: {len(df)}\n')

# VADO AD APPLICARE LEMMATIZZAZIONE

print("Effettuo l'operazione di lemmatizzazione dei singoli token dei record del dataset!")

wn = nltk.wordnet.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['LemmatizedTweet'] = df['CleanTweet'].apply(lemmatizer)

print(f"Record del dataset una volta effettuata l'operazione di lemmatizzazione:\n{df['LemmatizedTweet'].head(20)}")
print(f"Numero di record rimasti vuoti dopo l'operazione di lemmatizzazione: {(df['LemmatizedTweet'].str.len()==0).sum()}\n")

# VADO A VISUALIZZARE WORDCLOUD PER LE VARIE LABEL

def token_to_string(text):
    text = " ".join(word for word in text)
    return text

def generate_frequency(words):
    list_label = words.split(" ")
    set_label = set(list_label)
    dict_label = dict.fromkeys(set_label, 0.)
    tot = 0.
    for w in list_label:
        dict_label[w] += 1
        tot += 1
    for k in dict_label:
        dict_label[k] = dict_label[k] / tot
    return dict_label

df['LemmatizedString'] = df["LemmatizedTweet"].apply(token_to_string)

clouds = []
wc = WordCloud(background_color="white", max_words=200)
for l in df['Sentiment'].unique():
    df_label = df[df['Sentiment'] == l]
    word_label = " ".join(word for word in df_label['LemmatizedString'])
    frequency = generate_frequency(word_label)
    cloud = wc.generate_from_frequencies(frequency)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(f'{l} Label WordCloud')
    plt.show()

# DALLE WORDCLOUD HO NOTATO LA MASSICCIA PRESENZA DI PAROLE QUALI 'coronavirus' E 'covid' NEI RECORD DI TUTTE LE CLASSI
# PER QUESTO MOTIVO VADO AD ELIMINARE TALI PAROLE TRATTANDOLE COME FOSSERO STOPWORDS:

print('Vado a rimuovere i termini "covid" e "coronavirus" che vengono utilizzati frequentemente nei record a prescindere dalla label!')

def remove_useless_words(text):
    useless = ['coronavirus', 'covid']
    words = [w for w in text if w not in useless]
    return words

df['LemmatizedTweet'] = df['LemmatizedTweet'].apply(remove_useless_words)

print(f'Record del dataset una volta effettuate la rimozione dei termini poco utili:\n{df["LemmatizedTweet"].head(20)}')
print(f'Numero di record rimasti vuoti dopo le operazioni di rimozione dei termini poco utili: {(df["LemmatizedTweet"].str.len()==0).sum()}')

df[df['LemmatizedTweet'].str.len() == 0] = np.NaN
df = df.dropna(subset=['LemmatizedTweet'])

print(f'Dimensione dataset in seguito alla rimozione: {len(df)}\n')

df['LemmatizedString'] = df["LemmatizedTweet"].apply(token_to_string)

# VADO AD APPLICARE STEMMING PER RIDURRE IL NUMERO DI TOKEN

print("Effettuo l'operazione di stemming sui record del dataset!")

sm = nltk.PorterStemmer()

def stemming(text):
    text = [sm.stem(word) for word in text]
    return text

df['StemmedTweet'] = df['LemmatizedTweet'].apply(stemming)
df['StemmedString'] = df['StemmedTweet'].apply(token_to_string)

print(f"Record del dataset una volta effettuata l'operazione di stemming:\n{df['StemmedTweet'].head(20)}")
print(f"Numero di record rimasti vuoti dopo l'operazione di stemming: {(df['StemmedTweet'].str.len()==0).sum()}\n")

# VADO A PLOTTARE LA NUMEROSITA' DELLE CLASSI DEL DATASET DI TESTI DOPO LE OPERAZIONI DI PULITURA E RIMOZIONE DI RECORD VUOTI

gb = df.groupby('Sentiment').count()
totale = len(df)

plt.figure(figsize=(15, 15))
plt.title("Numerosità Classi Dataset Testi: Totale "+str(totale)+" Elementi")
graph = sns.barplot(x=gb.index, y=gb['OriginalTweet'].values)
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.3, gb['OriginalTweet'].values[i], ha="center")
    i += 1
plt.show()

# SALVO IL DATAFRAME CONTENENTE I DATI SU CUI E' STATA EFFETTUATA PULITURA E UNA PARTE DEL PREPROCESSING

exists = os.path.isfile(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'))
if not exists:
    with open(os.path.join(DRIVE_FOLDER, 'text_dataframe_preproc'), 'wb') as hist_file:
        pickle.dump(df, hist_file)

# UTILIZZO UN TFIFDVECTORIZER PER ANDARE AD OTTENERE IL VOCABOLARIO COMPLESSIVO DEL DATASET
v = TfidfVectorizer()
voc_len = len(v.fit(df['StemmedString']).vocabulary_)
print(f'Numerosità vocabolario del dateset (insieme dei termini che compongono i record): {voc_len}')

