from tkinter import Image

import sumy

import plotly.express as px
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import nlp
import sns as sns
import spacy as spacy
# Import all the required livraries..
import streamlit as st
import pandas as pd
import nltk
import feedparser
import cufflinks
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

nltk.download('punkt')
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


class RSSFeed():
    global ndf
    feedurl = ""


    def __init__(self, paramrssurl):
        self.feedurl = paramrssurl
        self.parse()

    def parse(self):
        global ndf
        ndf = pd.DataFrame(columns=['title', 'link', 'decription', 'published', 'content'])
        thefeed = feedparser.parse(self.feedurl)
        published = thefeed.feed.get("description", "")
        for thefeedentry in thefeed.entries:
            title = thefeedentry.get("title", "")
            link = thefeedentry.get("link", "")
            decr = thefeedentry.get("description", "")
            published = thefeedentry.get("published", "")
            content = ""
            if thefeedentry.get("content"):
                content = thefeedentry.get("content")[0].get("value", "")
            ndf = ndf.concat([{'title': title, 'link': link, 'decription': decr, 'published': published}],
                                 ignore_index=True)
        return ndf


# url_link = "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"
# rss_feed = RSSFeed(url_link)
#
# st.write(rss_feed.ndf)
# df_n = rss_feed.ndf['title']  # Extracting the 'title' column as the corpus for bigram extraction


# Beautiful Soup Code
@st.cache
def full_text(my_url):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    url = my_url
    article = requests.get(url)
    articles = BeautifulSoup(article.content, 'html.parser')
    articles_body = articles.findAll('body')
    p_blocks = articles_body[0].findAll('p')
    p_blocks_df = pd.DataFrame(columns=['element_name', 'parent_hierarchy', 'element_text', 'element_text_Count'])
    for i in range(0, len(p_blocks)):
        parents_list = []
        for parent in p_blocks[i].parents:
            Parent_id = ''
            try:
                Parent_id = parent['id']
            except:
                pass
            parents_list.append(parent.name + 'id: ' + Parent_id)
        parent_element_list = ['' if (x == 'None' or x is None) else x for x in parents_list]
        parent_element_list.reverse()
        parent_hierarchy = ' -> '.join(parent_element_list)
        p_blocks_df = p_blocks_df.append({"element_name": p_blocks[i].name
                                             , "parent_hierarchy": parent_hierarchy
                                             , "element_text": p_blocks[i].text
                                             , "element_text_Count": len(str(p_blocks[i].text))}
                                         , ignore_index=True
                                         , sort=False)
    if len(p_blocks_df) > 0:
        p_blocks_df_groupby_parent_hierarchy = p_blocks_df.groupby(by=['parent_hierarchy'])
        p_blocks_df_groupby_parent_hierarchy_sum = p_blocks_df_groupby_parent_hierarchy[['element_text_Count']].sum()
        p_blocks_df_groupby_parent_hierarchy_sum.reset_index(inplace=True)
    maxid = p_blocks_df_groupby_parent_hierarchy_sum.loc[
        p_blocks_df_groupby_parent_hierarchy_sum['element_text_Count'].idxmax()
        , 'parent_hierarchy']
    merge_text = '\n'.join(p_blocks_df.loc[p_blocks_df['parent_hierarchy'] == maxid, 'element_text'].to_list())
    return merge_text


# Matplt lib
@st.cache
def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')
    return ReviewText


# Get top words
@st.cache
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


@st.cache
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


@st.cache
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


@st.cache
def sentiment_vader(text, sid):
    ss = sid.polarity_scores(text)
    ss.pop('compound')
    return max(ss, key=ss.get)


@st.cache
def sentiment_textblob(text):
    x = TextBlob(text).sentiment.polarity

    if x < 0:
        return 'neg'
    elif x == 0:
        return 'neu'
    else:
        return 'pos'


# @st.cache(suppress_st_warning=True)
def plot_sentiment_barchart(text, method='TextBlob'):
    if method == 'TextBlob':
        sentiment = text.map(lambda x: sentiment_textblob(x))
    elif method == 'Vader':
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        sentiment = text.map(lambda x: sentiment_vader(x, sid=sid))
    else:
        raise ValueError('Textblob or Vader')

    plt.bar(sentiment.value_counts().index,
            sentiment.value_counts(), color=['cyan', 'red', 'green', 'black'], edgecolor='yellow')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


# Entity recognition
def plot_most_common_named_entity_barchart(text, entity="PERSON"):
    nlp = spacy.load("en_core_web_sm")

    def _get_ner(text, ent):
        doc = nlp(text)
        return [X.text for X in doc.ents if X.label_ == ent]

    entity_filtered = text.apply(lambda x: _get_ner(x, entity))
    entity_filtered = [i for x in entity_filtered for i in x]

    counter = Counter(entity_filtered)
    x, y = map(list, zip(*counter.most_common(10)))
    sns.barplot(y, x).set_title(entity)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


# Parts of Speech Tagging
def plot_parts_of_speach_barchart(text):
    nltk.download('averaged_perceptron_tagger')

    def _get_pos(text):
        pos = nltk.pos_tag(word_tokenize(text))
        pos = list(map(list, zip(*pos)))[1]
        return pos

    tags = text.apply(lambda x: _get_pos(x))
    tags = [x for l in tags for x in l]
    counter = Counter(tags)
    x, y = list(map(list, zip(*counter.most_common(7))))

    sns.barplot(x=y, y=x)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


# st.set_page_config(layout="wide")
st.title('News Articles Analysis -NLP App')
st.header("""
This app displays the news articles appeared in the top News Publications!
""")

st.sidebar.header('Please select the news org from the dropdown list')
lnews = ["NY Times", "LA Times", "CNN", "Washington Post", "USA Today"]
s_news = st.sidebar.selectbox('News', lnews)
st.sidebar.header('Please select the Function')
lnlp = ["Intro", "Snapshot", "Unigrams", "Bigrams", "Trigrams", "WordCloud", "Text Stat", "Topic Modeling",
        "Entity Extraction", "Sentiment Analysis TextBlob", "Sentiment Analysis-Vader", "Text Summarization",
        "Parts of Speech"]
s_nlp = st.sidebar.selectbox('Functions', lnlp)


def load_data(news, nlp):


    if news == "NY Times":
        # st.write(news)
        # st.write(nlp)
        if nlp == "Intro":
            # st.write("this is intro")
            # image1 = Image.open(r'C:\BPC_DOCS\IUB\Projects_Medium\New-York-Times-logo-500x281.jpg')
            # st.write(" ")
            # st.image(image1, width=300)
            # st.write(" ")
            # st.write(" ")
            # st.write(" ")
            # image = Image.open(r'C:\BPC_DOCS\IUB\Projects_Medium\nytimes-building-ap-img.jpg')
            # st.image(image, width=700)
            # st.write(" ")
            # st.write(" ")
            # st.write(" ")
            st.markdown("""
           The New York Times (NYT or NY Times) is an American daily newspaper based in New York City with a worldwide readership.Founded in 1851, the Times has since won 130 Pulitzer Prizes (the most of any newspaper),and has long been regarded within the industry as a national "newspaper of record". It is ranked 18th in the world by circulation and 3rd in the U.S.
           The paper is owned by The New York Times Company, which is publicly traded. It has been governed by the Sulzberger family since 1896, through a dual-class share structure after its shares became publicly traded.A. G. Sulzberger and his father, Arthur Ochs Sulzberger Jr.—the paper's publisher and the company's chairman, respectively—are the fourth and fifth generation of the family to head the paper.
           Since the mid-1970s, The New York Times has expanded its layout and organization, adding special weekly sections on various topics supplementing the regular news, editorials, sports, and features. Since 2008,the Times has been organized into the following sections: News, Editorials/Opinions-Columns/Op-Ed, New York (metropolitan), Business, Sports, Arts, Science, Styles, Home, Travel, and other features.[15] On Sundays, the Times is supplemented by the Sunday Review (formerly the Week in Review),The New York Times Book Review, The New York Times Magazine,and T: The New York Times Style Magazine.
           The Times stayed with the broadsheet full-page set-up and an eight-column format for several years after most papers switched to six,and was one of the last newspapers to adopt color photography, especially on the front page.The paper's motto, "All the News That's Fit to Print", appears in the upper left-hand corner of the front page.
           """)

        df = pd.DataFrame(columns=['title', 'link', 'description', 'published', 'content'])
        url_link = "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"
        RSSFeed(url_link)
        df = ndf

        # st.header('Display the dataframe')
        # st.dataframe(df)
        pd.set_option('display.max_rows', df.shape[0] + 1)
        df.reset_index(inplace=True, drop=True)

        for ind in df.index:
            # print(df['title'][ind], df['link'][ind], df['content'][ind])
            url = df['link'][ind]
            # print(url)
            text = full_text(url)
            df['content'][ind] = text
        # st.write(df['title'])
        # Build the corpus.
        corpus = []
        for ind in df.index:
            # corpus = df['content'][ind]
            corpus.append(df['title'][ind])
        # print(corpus)
        df = df.dropna()
        X_train1 = df['title']

        if nlp == "Snapshot":
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.subheader('Display the dataframe')
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.dataframe(df)
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.markdown("""<style>
                             .big2-font {
                              font-size:30px !important;
                               }
                               </style>""", unsafe_allow_html=True)
            st.markdown('<p class="big2-font">The no of articles :</p>', unsafe_allow_html=True)
            st.write(df.shape[0])
            st.write(" ")
            st.write("The Url Link ")
            for index, row in df.iterrows():
                st.write(row['link'])

            if nlp == "WordCloud":
                st.markdown("""<style>.big1-font {font-size:20px !important;}</style>
                                       """, unsafe_allow_html=True)
                st.write(' ')
                st.markdown('<p class="big1-font">WordCloud</p>', unsafe_allow_html=True)
                st.write(' ')
                st.markdown(
                    '<p class="big1-font">Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text. The larger the word in the visual the more common the word was in the document(s).</p>',
                    unsafe_allow_html=True)
                st.write(' ')
                st.write(' ')
                long_string = ','.join(list(X_train1.values))
                # Create a WordCloud object
                wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                                      contour_color='steelblue')
                # Generate a word cloud
                wordcloud.generate(long_string)
                # Visualize the word cloud
                plt.figure(figsize=(20, 10))
                plt.imshow(wordcloud)
                st.image(wordcloud.to_array(), width=700)
                st.write("Word Cloud")
                # Generate word cloud
                long_string = ','.join(list(X_train1.values))
                wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='salmon',
                                      colormap='Pastel1',
                                      collocations=False, stopwords=STOPWORDS).generate(long_string)
                # Visualize the word cloud
                plt.figure(figsize=(20, 10))
                plt.imshow(wordcloud)
                st.image(wordcloud.to_array(), width=700)
                st.write("Word Cloud")
                wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='black',
                                      colormap='Set2',
                                      collocations=False, stopwords=STOPWORDS).generate(long_string)
                # Visualize the word cloud
                plt.figure(figsize=(20, 10))
                plt.imshow(wordcloud)
                st.image(wordcloud.to_array(), width=700)

            df_n = df
            df_n['title'] = preprocess(df['title'])

            if nlp == "Unigrams":
                st.markdown("""<style>.big1-font {font-size:20px !important;}
                                                                              </style>
                                                                              """, unsafe_allow_html=True)
                st.write(' ')
                st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
                st.write(' ')
                st.markdown(
                    '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
                    unsafe_allow_html=True)
                st.write(' ')
                st.write(' ')

                common_words = get_top_n_words(df_n['title'], 10)
                # for word, freq in common_words:
                #    (word, freq)
                df2 = pd.DataFrame(common_words, columns=['Words', 'Count'])
                st.table(df2)
                # st.bar_chart(df2["Words"])
                with st.echo(code_location='below'):
                    # import plotly.express as px

                    fig = px.scatter(
                        x=df2["Words"],
                        y=df2["Count"],
                        color=df2["Count"],
                    )
                    fig.update_layout(
                        xaxis_title="Words",
                        yaxis_title="Count",
                    )

                    # st.write(fig)
                    st.plotly_chart(fig)

            if nlp == "Bigrams":
                st.markdown("""<style>.big1-font {font-size:20px !important;}
                                                                             </style>
                                                                             """, unsafe_allow_html=True)
                st.write(' ')
                st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
                st.write(' ')
                st.markdown(
                    '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
                    unsafe_allow_html=True)
                st.write(' ')
                st.write(' ')
                common_words = get_top_n_bigram(df_n['title'], 10)
                df4 = pd.DataFrame(common_words, columns=['Bigrams', 'Count'])
                st.table(df4)
                fig = px.bar(df4, x='Bigrams', y='Count', color='Count', height=500)
                st.plotly_chart(fig)

            if nlp == 'Trigrams':
                st.markdown("""
                                                                             <style>
                                                                             .big1-font {
                                                                                 font-size:20px !important;
                                                                             }
                                                                             </style>
                                                                             """, unsafe_allow_html=True)
                st.write(' ')
                st.markdown('<p class="big1-font">N Grams</p>', unsafe_allow_html=True)
                st.write(' ')
                st.markdown(
                    '<p class="big1-font">N-grams of texts are extensively used in text mining and natural language processing tasks. They are basically a set of co-occuring words within a given window and when computing the n-grams you typically move one word forward (although you can move X words forward in more advanced scenarios). When N=1, this is referred to as unigrams and this is essentially the individual words in a sentence. When N=2, this is called bigrams and when N=3 this is called trigrams. When N>3 this is usually referred to as four grams or five grams and so on.</p>',
                    unsafe_allow_html=True)
                st.write(' ')
                st.write(' ')
                common_words = get_top_n_trigram(df_n['title'], 10)
                df6 = pd.DataFrame(common_words, columns=['Trigrams', 'Count'])
                st.table(df6)
                fig = px.scatter(
                    x=df6["Trigrams"],
                    y=df6["Count"],
                    color=df6["Count"],
                )
                fig.update_layout(
                    xaxis_title="Trigrams",
                    yaxis_title="Count",
                )

                # st.write(fig)
                st.plotly_chart(fig)

            if nlp == "Sentiment Analysis TextBlob":
                st.markdown("""
                                      <style>
                                      .big1-font {
                                          font-size:20px !important;
                                      }
                                      </style>
                                      """, unsafe_allow_html=True)
                t_word = "The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective."
                st.write(' ')
                st.markdown('<p class="big1-font">TextBlob Sentiment Analyzer</p>', unsafe_allow_html=True)
                st.write(' ')
                st.markdown(
                    '<p class="big1-font">The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0]</p>',
                    unsafe_allow_html=True)
                st.write(' ')
                st.write(' ')
                plot_sentiment_barchart(df['title'], method='TextBlob')

            if nlp == "Sentiment Analysis-Vader":
                st.markdown("""
                                                    <style>
                                                    .big1-font {
                                                        font-size:20px !important;
                                                    }
                                                    </style>
                                                    """, unsafe_allow_html=True)
                st.write(' ')
                st.markdown('<p class="big1-font">Vader Sentiment Analyzer</p>', unsafe_allow_html=True)
                st.write(' ')
                st.markdown(
                    '<p class="big1-font">VADER ( Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. ... VADER sentimental analysis relies on a dictionary that maps lexical features to emotion intensities known as sentiment scores.</p>',
                    unsafe_allow_html=True)
                st.write(' ')
                st.write(' ')
                plot_sentiment_barchart(df['title'], method='Vader')

            if nlp == "Entity Extraction":
                st.write(" ")
                st.write(" ")
                st.header("Entity Extraction")
                st.write(" ")
                # st.write(" ")
                st.subheader(
                    "Named entity recognition is an information extraction method in which entities that are present in the text are classified into predefined entity types like “Person”,” Place”,” Organization”, etc.By using NER we can get great insights about the types of entities present in the given text dataset.")
                st.write(" ")
                st.write(" ")
                plot_most_common_named_entity_barchart(df['title'], entity="PERSON")

            if nlp == "Topic Modeling":
                st.write(" ")
                st.write(" ")
                st.header("Topic Modeling")
                st.write(" ")
                # st.write(" ")
                st.subheader(
                    "Topic modeling is a method for unsupervised classification of documents, similar to clustering on numeric data, which finds some natural groups of items (topics) even when we’re not sure what we’re looking for.Topic modeling provides methods for automatically organizing, understanding, searching, and summarizing large electronic archives.")
                st.write(" ")
                st.write(" ")

                from sklearn.feature_extraction.text import CountVectorizer
                import warnings

                warnings.simplefilter("ignore", DeprecationWarning)
                # Load the LDA model from sk-learn
                from sklearn.decomposition import LatentDirichletAllocation as LDA


                # Helper function
                def print_topics(model, count_vectorizer, n_top_words):
                    words = count_vectorizer.get_feature_names()
                    for topic_idx, topic in enumerate(model.components_):
                        # print("\nTopic #%d:" % topic_idx)
                        w_stl = (" ".join([words[i]
                                           for i in topic.argsort()[:-n_top_words - 1:-1]]))
                        st.write(w_stl)


                # Tweak the two parameters below
                number_topics = 10
                number_words = 6
                X_train1 = df['content']
                # Initialise the count vectorizer with the English stop words
                count_vectorizer = CountVectorizer(stop_words='english')
                # Fit and transform the processed titles
                count_data = count_vectorizer.fit_transform(X_train1)
                # Create and fit the LDA model
                lda = LDA(n_components=number_topics, n_jobs=-1)
                lda.fit(count_data)
                # Print the topics found by the LDA model
                st.write("Topics:")
                print_topics(lda, count_vectorizer, number_words)

            if nlp == "Text Summarization":
                st.write(" ")
                st.write(" ")
                st.header("Text Summarization")
                st.write(" ")
                # st.write(" ")
                st.subheader(
                    "Text summarization refers to the technique of shortening long pieces of text. The intention is to create a coherent and fluent summary having only the main points outlined in the document.Automatic text summarization is a common problem in machine learning and natural language processing (NLP).")
                st.write(" ")
                st.write(" ")
                from sumy.summarizers.lex_rank import LexRankSummarizer

                # dfs = df['content']
                for index, row in df.iterrows():
                    parser = PlaintextParser.from_string(row['content'], Tokenizer("english"))
                    # Using LexRank
                    summarizer = LexRankSummarizer()
                    # Summarize the document with 4 sentences
                    summary = summarizer(parser.document, 3)
                    st.write("Summarized Document")
                    st.write(" ")
                    st.write(row['title'])
                    st.write(" ")
                    for sentence in summary:
                        # st.write("Summarized Document")
                        # st.write(row['title'])
                        st.write(sentence)

            if nlp == "Parts of Speech":
                st.markdown("""
                             <style>
                             .big1-font {
                                 font-size:20px !important;
                             }
                             </style>
                             """, unsafe_allow_html=True)
                st.write(" ")
                st.markdown('<p class="big1-font">Noun (NN)- Joseph, London, table, cat, teacher, pen, city</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<p class="big1-font">Verb (VB)- read, speak, run, eat, play, live, walk, have, like, are, is</p>',
                    unsafe_allow_html=True)
                st.markdown('<p class="big1-font">Adjective(JJ)- beautiful, happy, sad, young, fun, three</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<p class="big1-font">Adverb(RB)- slowly, quietly, very, always, never, too, well, tomorrow</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<p class="big1-font">Preposition (IN)- at, on, in, from, with, near, between, about, under</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<p class="big1-font">Conjunction (CC)- and, or, but, because, so, yet, unless, since, if</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<p class="big1-font">Pronoun(PRP)- I, you, we, they, he, she, it, me, us, them, him, her,this</p>',
                    unsafe_allow_html=True)
                st.markdown('<p class="big1-font">Interjection (INT)- Ouch! Wow! Great! Help! Oh! Hey! Hi!</p>',
                            unsafe_allow_html=True)
                plot_parts_of_speach_barchart(df['content'])

            if nlp == "Text Stat":
                import inspect
                import textstat

                st.markdown("""
                  <style>
                  .big-font {
                      font-size:30px !important;
                  }
                  </style>
                  """, unsafe_allow_html=True)

                textstat.set_lang("en")
                text = df['content'][2]
                funcs = ["textstat." + inspect.getmembers(textstat, predicate=inspect.ismethod)[i][0] for i in
                         range(1, 28)]
                st.write(" ")
                st.markdown(
                    '<p class="big-font">Textstat is an easy to use library to calculate statistics from text. It helps determine readability, complexity, and grade level.</p>',
                    unsafe_allow_html=True)
                st.write(" ")
                for elem in funcs:
                    method = eval(elem)
                    textstat.set_lang("en")
                    w_1 = (elem.split(".")[1])
                    st.write(w_1)
                    st.write(method(text))
                    st.write(" ")

load_data(s_news, s_nlp)
