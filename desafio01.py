import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


feedback_data = [
    "O produto é incrível, mas poderia ter mais opções de cores.",
    "A entrega foi rápida, estou muito satisfeito!",
    "Não gostei da qualidade do produto, esperava mais.",

]


def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    return ' '.join(words)

# Análise de Sentimento
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

# Transformação dos dados de feedback
transformed_data = []
for feedback in feedback_data:
    processed_text = preprocess_text(feedback)
    sentiment = analyze_sentiment(processed_text)
    transformed_data.append({'feedback': feedback, 'processed_text': processed_text, 'sentiment': sentiment})

# Visualização de Sentimentos
sentiment_scores = [item['sentiment']['compound'] for item in transformed_data]
plt.hist(sentiment_scores, bins=10, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Scores')
plt.show()

# Nuvem de Palavras
all_processed_text = ' '.join([item['processed_text'] for item in transformed_data])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_processed_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Processed Text')
plt.show()

# Criando um DataFrame e salvando em CSV
df = pd.DataFrame(transformed_data)
df.to_csv('feedback_analysis.csv', index=False)
