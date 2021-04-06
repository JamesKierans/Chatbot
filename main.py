# Assignment 1 - Question Answering Chatbot
# COMP 3074
# James Kierans
import nltk
import string
import random
import warnings
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

# nltk.download('punkt') # Need to comment out as only needed first time is run, then permanently downloaded
# nltk.download('wordnet') # Ditto
from sklearn.metrics import jaccard_score

# So that stop word warnings do not appear
warnings.filterwarnings('ignore')

# Reading in of the data
f = open('data/CW_Data.csv', 'r', encoding='utf-8', errors='ignore')
Content = f.read().lower()  # Including .lower so that all read in is in lower case text to remove casing complications later on
# Removal of punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# Lemmatiser
lemmatiser = WordNetLemmatizer()

# csv file Tokenisation
Sentence_Tokens = nltk.sent_tokenize(Content)


# Processing of tokens
def Tokens(text):
    return [lemmatiser.lemmatize(token) for token in word_tokenize(text.lower().translate(remove_punct_dict))]


# Token Matrix creation and transformed
def Vectorize(Sentence_Tokens):
    Vectorizer = CountVectorizer(tokenizer=Tokens, stop_words=stopwords.words('english'))
    return Vectorizer.fit_transform(Sentence_Tokens)


# Intent 3 - Question Answering
def Question_Answering(s):
    Sentence_Tokens.append(s)
    x = Vectorize(Sentence_Tokens)
    # Works out the similarity between the transposed set of data and the original set of data
    y = x[-1]
    # cos_sim = dot(y,x) / (norm(y) * norm(x))
    Cos_Sim = cosine_similarity(y, x)
    values_1D = Cos_Sim.flatten()
    values_1D.sort()
    Minimum_Freq_Value = values_1D[-2]
    index = Cos_Sim.argsort()[0][-2]
    if Minimum_Freq_Value == 0:
        Answer = "Sorry I am not able to answer this question at the moment"
        return Answer
    else:
        Answer = Sentence_Tokens[index]
        return Answer


punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
Hello_Synonyms_Responses = ("Hey there", "Hi! I hope you are well!", "hi!", "hiya")
How_Are_You_Responses = ("I am good thanks!", "Not so great :(", "Couldn't be better! :)")

# Main While Loop
stop = False
while not stop:
    name = input("Hey there! I am your chatbot for the day. Can I take your name please?\n")
    if name == "STOP":
        stop = True
    else:
        print("Welcome", name, "!\n")
        print("If at any point you want to change your name, just say!")
        print("If you want to stop the program, just say stop!")
        finish = False
        while not finish:
            query = input("What is your query for today?\n")
            query = query.lower()
            # define all the possible punctuation there is
            # for loop which checks for punctuation and removes it from the string
            no_punct = ""
            for char in query:
                if char not in punctuations:
                    no_punct = no_punct + char
            # redefining our punct to query to match the if statements
            query = no_punct
            if query in ["change my name", "i would like to change my name"]:
                name = input("So you wanted to change your name! That's cool, what would you like it to be now?\n")
                print("Okay ", name, "from now on this is your name! If you would like to change again, just say so!\n")
            elif query == "stop":
                finish = True
                stop = True
            elif query in ["what is my name", "tell me my name please", "whats my name"]:
                print("Your name is", name)
            elif query in ["hello", "hey", "Hi There", "hi", "heya", "hiya"]:
                print(random.choice(Hello_Synonyms_Responses))
            elif query in ["how are you", "how you doing", "how is you"]:
                print(random.choice(How_Are_You_Responses))
            else:
                print("Lets get that query sorted then")
                answers = Question_Answering(query).split("\n")
                Sentence_Tokens.remove(query)
                answer = answers[randint(0, len(answers) - 1)]
                output = answer.split(',', 2)

                try:
                    print(output[2])
                except:
                    print("Sorry I am not able to answer this question at the moment")
