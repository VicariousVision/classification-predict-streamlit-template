"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from pathlib import Path
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vectorizer_4.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer CB5")
	st.subheader("Climate change tweet classification predict")

	
#########################################################################################################################################
################################################ THIS IS THE INFO PAGE ##################################################################

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Prediction","About"]
	selection = st.sidebar.selectbox("Navigation", options)

	# Building out the "Information" page
	if selection == "Home":
		image = Image.open('resources\imgs\EDSA_logo.png')
		st.image(image)

		# You can read a markdown file from supporting resources folder
		st.markdown(""" Consumers gravitate toward companies that are built around lessening one’s environmental impact or carbon footprint. Companies who offer products and services that are environmentally friendly and sustainable, in line with consumer values and ideals. The objective of this project is to determine how people perceive climate change and whether or not they believe it is a real threat based on their tweets. This would add to the companies market research efforts in gauging how their product/service may be received. The task of this predict is to create a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data. Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories, thus increasing their insights and informing future marketing strategies""")
		st.info('''

The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected.
''')

		st.subheader('Findings')

		image = Image.open('resources\imgs\sentiment.png')
		st.image(image,caption='Percentage of tweets in the data and their corresponding sentiment')


		st.markdown(""" We started off looking at the distribution of sentiment within the data and found that most people support the belief that climate change is man made""")



		image = Image.open('resources\imgs\most_words.png')
		st.image(image,caption='Most occurrences of words in the data')

		st.markdown(""" In our search through the data we found that certain words appear often in different sentiment classes. For example those who dont believe climate change is even real often use words like "fake","scam" and those who do believe often have two main things that can be associated with with their tweets, that being the precence of web links (which we have changed to 'url-web'""")
		
		st.markdown("""Through exploratory data analysis we have found that certain words appear in the sentiments more often than others. For example those leaning to the "non man made" opinion often tweet tweets that cointain words that can be link to conspiratorial speech whilst those on the "pro man made" side often have tweets that contain words like "scientist" and more often have links to no twitter sites, we can view these as arguments made by those individuals with evidence to back up their claims. We also noted that those same individuals have, in general, longer tweet lengths which strangthens the arguemnt that they use facts to back up their claims as opposed to the individuals of the "non man made" opinion """)
		image = Image.open('resources/imgs/anti_count.png')
		st.image(image,caption='Word occurances of people linked to the ''anti'' sentiment')

		image = Image.open('resources/imgs/pro_count.png')
		st.image(image,caption='Word occurances of people linked to the ''pro'' sentiment')


		st.markdown('''   After performing data wrangling we were able to come up with a model that is able to predict which category of sentiment a tweet will fall into based on the words in it. To test the model please click on the drop down navigation page on the left where you can input tweets and the model will tell you what it will be classified as.            ''')
		st.markdown(' From the directive given by the client, we have build an through this site, have deployed a service that can be used by marketing departments to guage the sentiment of their target customers. For futher development a service can be set up to integrate with the companies internal systems and databases to collect information from various sources that can be used to better the marketing strategy and improve the companies public image by positioning itself as a foward thinking company that is able to meet the expectations from the consumer and government regulations regarding climate change. ')





#########################################################################################################################################
################################################ THIS IS THE PREDICT PAGE ###############################################################

	# Building out the predication page
	if selection == "Prediction":
		st.markdown("""This process requires the user to input text
						(ideally a tweet relating to climate change), and will
						classify it according to whether or not they believe in
						climate change.Below you will find information about the data source
						and a brief data description. You can have a look at word clouds and
						other general EDA on the EDA page, and make your predictions on the
						prediction page that you can navigate to in the sidebar.""")
		st.markdown(""" 
						- 2(News): the tweet links to factual news about climate change
		""")
		st.markdown(""" 
						- 1(Pro): the tweet supports the belief of man-made climate change
		""")
		st.markdown(""" 
						- 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change
		""")
		st.markdown(""" 
						- -1(Anti): the tweet does not believe in man-made climate change

		""")

		st.info("Enter a sample tweet or your thoughts on climate change and the model will determine if you are Pro/Neutral/Anti Clmate Change or if you are posting something from a news source")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):



			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/SVC_linear_CB5.pkl"),"rb"))
			prediction = predictor.predict(vect_text)


			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
		
		
		st.info("If you'd like to test accuracy of the model, Copy and past one of the tweets in the list below into the classification prompt and check the corresponding sentiment that the model gives us")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	if selection == "About":
		st.write("""
		Team members:

    Kamogelo Potjane Makhuloane

    Mapula Tlhompho Victoria Maponya

    Njabulo Mkhwanazi

    Ozzey Padayachee

    Thapelo Nkhumishe

""")
		st.markdown("We are member of the Data Mechanics company, a data science services provider ")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
