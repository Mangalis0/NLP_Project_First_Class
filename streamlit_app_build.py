#streamlit dependencies

import streamlit as st
import joblib, os

## data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from nlppreprocess import NLP # pip install nlppreprocess
#import en_core_web_sm
from nltk import pos_tag

import seaborn as sns
import re

from nlppreprocess import NLP
nlp = NLP()

def cleaner(line):

    # Removes RT, url and trailing white spaces
    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 

    # Removes puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", line.lower()) 

    # Removes stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet) # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]
    # https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52

    # tokenisation
    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point
    # and the twitter data is not complicated
    tweet = tweet.split() 

    # POS 
    pos = pos_tag(tweet)


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if po[0].lower() in ['n', 'r', 'v', 'a'] else word for word, po in pos])
    # tweet = ' '.join([lemmatizer.lemmatize(word, 'v') for word in tweet])

    return tweet


##reading in the raw data and its cleaner

vectorizer = open('resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

#@st.cache
data = pd.read_csv('resources/train.csv')

def main():
    """Tweets classifier App"""

    st.title('Tweets  Unclassified  :)')

    from PIL import Image
    image = Image.open('resources/imgs/Tweeter.png')

    st.image(image, caption='Which Tweet are you?', use_column_width=True)

    st.subheader('Climate Change Belief Analysis: Based on Tweets')
    

    ##creating a sidebar for selection purposes


    pages = ['Information', 'Visuals', 'Make Prediction', 'Contact App Developers']

    selection = st.sidebar.radio('Go to....', pages)

    #st.sidebar.image(image, caption='Which Tweet are you?', use_column_width=True)



    ##information page

    if selection == 'Information':
        st.info('General Information')
        st.write('Explorers Explore and.....boooom EXPLODE!!!!!!!!!!!')
        st.markdown(""" We have deployed Machine Learning models that are able to classify 
        whether or not a person believes in climate change, based on their novel tweet data. 
        Like any data lovers, these are robust solutions to that can provide access to a 
        broad base of consumer sentiment, spanning multiple demographic and geographic categories. 
        So, do you have a Twitter API and ready to scrap? or just have some tweets off the top of your head? 
        Do explore the rest of this app's buttons.
        """)


        raw = st.checkbox('See raw data')
        if raw:
            st.dataframe(data.head(25))

    ## Charts page

    if selection == 'Visuals':
        st.info('The following are some of the charts that we have created from the raw data. Some of the text is too long and may cut off, feel free to right click on the chart and either save it or open it in a new window to see it properly.')


       # Number of Messages Per Sentiment
        st.write('Distribution of the sentiments')
        # Labeling the target
        data['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in data['sentiment']]
        
        # checking the distribution
        st.write('The numerical proportion of the sentiments')
        values = data['sentiment'].value_counts()/data.shape[0]
        labels = (data['sentiment'].value_counts()/data.shape[0]).index
        colors = ['lightgreen', 'blue', 'purple', 'lightsteelblue']
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)
        st.pyplot()
        
        # checking the distribution
        sns.countplot(x='sentiment' ,data = data, palette='PRGn')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        plt.title('Number of Messages Per Sentiment')
        st.pyplot()

        # Popular Tags
        st.write('Popular tags found in the tweets')
        data['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in data.message]
        sns.countplot(y="users", hue="sentiment", data=data,
                    order=data.users.value_counts().iloc[:20].index, palette='PRGn') 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        # Tweet lengths
        st.write('The length of the sentiments')
        st.write('The average Length of Messages in all Sentiments is 100 which is of no surprise as tweets have a limit of 140 characters.')

        # Repeated tags
        
        # Generating Counts of users
        st.write("Analysis of hashtags in the messages")
        counts = data[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending=False)
        values = [sum(np.array(counts['message']) == 1)/len(counts['message']), sum(np.array(counts['message']) != 1)/len(counts['message'])]
        labels = ['First Time Tags', 'Repeated Tags']
        colors = ['lightsteelblue', "purple"]
        plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0), colors=colors)
        st.pyplot()

        # Popular hashtags
        st.write("The Amount of popular hashtags")
        repeated_tags_rate = round(sum(np.array(counts['message']) > 1)*100/len(counts['message']), 1)
        print(f"{repeated_tags_rate} percent of the data are from repeated tags")
        sns.countplot(y="users", hue="sentiment", data=data, palette='PRGn',
              order=data.users.value_counts().iloc[:20].index) 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()

        st.markdown("Now that we've had a look at the tweets themselves as well as the users, we now analyse the hastags:")

        # Generating graphs for the tags
        st.write('Analysis of most popular tags, sorted by populariy')
        # Analysis of most popular tags, sorted by populariy
        sns.countplot(x="users", data=data[data['sentiment'] == 'Positive'],
                    order=data[data['sentiment'] == 'Positive'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 Positive Tags')
        plt.xticks(rotation=85)
        st.pyplot()

        # Analysis of most popular tags, sorted by populariy
        st.write("Analysis of most popular tags, sorted by populariy")
        sns.countplot(x="users", data=data[data['sentiment'] == 'Negative'],
                    order=data[data['sentiment'] == 'Negative'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 Negative Tags')
        plt.xticks(rotation=85)
        st.pyplot()


        st.write("Analysis of most popular tags, sorted by populariy")
        # Analysis of most popular tags, sorted by populariy
        sns.countplot(x="users", data=data[data['sentiment'] == 'News'],
                    order=data[data['sentiment'] == 'News'].users.value_counts().iloc[:20].index) 

        plt.xlabel('User')
        plt.ylabel('Number of Tags')
        plt.title('Top 20 News Tags')
        plt.xticks(rotation=85)
        st.pyplot()

    ## prediction page

    if selection == 'Make Prediction':

        st.info('Make Predictions of your Tweet(s) using our ML Model')

        data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

        source_selection = st.selectbox('What to classify?', data_source)

        # Load Our Models
        def load_prediction_models(model_file):
            loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
            return loaded_models

        # Getting the predictions
        def get_keys(val,my_dict):
            for key,value in my_dict.items():
                if val == value:
                    return key


        if source_selection == 'Single text':
            ### SINGLE TWEET CLASSIFICATION ###
            st.subheader('Single tweet classification')

            input_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
            all_ml_models = ["LR","NB","RFOREST","DECISION_TREE"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')

            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(input_text))
                text1 = cleaner(input_text) ###passing the text through the 'cleaner' function
                vect_text = tweet_cv.transform([text1]).toarray()
                if model_choice == 'LR':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RFOREST':
                    predictor = load_prediction_models("resources/RFOREST_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'NB':
                    predictor = load_prediction_models("resources/NB_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'DECISION_TREE':
                    predictor = load_prediction_models("resources/DTrees_model.pkl")
                    prediction = predictor.predict(vect_text)
				# st.write(prediction)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as:: {}".format(final_result))

        if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
            st.subheader('Dataset tweet classification')

            all_ml_models = ["LR","NB","RFOREST","SupportVectorMachine", "MLR", "LDA"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')


            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)

            #X = text_input.drop(columns='tweetid', axis = 1, inplace = True)   

            uploaded_dataset = st.checkbox('See uploaded dataset')
            if uploaded_dataset:
                st.dataframe(text_input.head(25))

            col = st.text_area('Enter column to classify')

            #col_list = list(text_input[col])

            #low_col[item.lower() for item in tweet]
            #X = text_input[col]

            #col_class = text_input[col]
            
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(text_input))
                X1 = text_input[col].apply(cleaner) ###passing the text through the 'cleaner' function
                vect_text = tweet_cv.transform([X1]).toarray()
                if model_choice == 'LR':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RFOREST':
                    predictor = load_prediction_models("resources/Random_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'NB':
                    predictor = load_prediction_models("resources/NB_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SupportVectorMachine':
                    predictor = load_prediction_models("resources/svm_model.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'MLR':
                    predictor = load_prediction_models("resources/mlr_model.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'SupportVectorMachine':
                    predictor = load_prediction_models("resources/simple_lda_model.pkl")
                    prediction = predictor.predict(vect_text)

                
				# st.write(prediction)
                text_input['sentiment'] = prediction
                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweets Categorized as:: {}".format(final_result))

                
                csv = text_input.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

                st.markdown(href, unsafe_allow_html=True)


    ##contact page
    if selection == 'Contact App Developers':

        st.info('Contact details in case you any query or would like to know more of our designs:')
        st.write('Kea: Lefifikea@gmail.com')
        st.write('Noxolo: Kheswanl925@gmail.com')
        st.write('Sam: makhoba808@gmail.com')
        st.write('Neli: cenygal@gmail.com')
        st.write('Khathu: netsiandakhathutshelo2@gmail.com')
        st.write('Ife: ifeadeoni@gmail.com')

        # Footer 
        image = Image.open('resources/imgs/EDSA_logo.png')

        st.image(image, caption='Team-SS4-Johannesbrug', use_column_width=True)

if __name__ == '__main__':
	main()


