import requests
from bs4 import BeautifulSoup
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Data Sourcing
def scrape_pratham():
    url = "https://www.pratham.org"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract relevant information
    # Note: You may need to adjust the selectors based on the actual structure of pratham.org
    articles = soup.find_all('article')
    
    data = []
    for article in articles:
        title = article.find('h2').text.strip()
        content = article.find('div', class_='content').text.strip()
        data.append({'title': title, 'content': content})
    
    # Save to CSV
    with open('pratham_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'content'])
        writer.writeheader()
        writer.writerows(data)

# Knowledge Base
def process_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_data = []

    with open('pratham_data.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tokens = word_tokenize(row['content'].lower())
            lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
            keywords = [word for word in lemmatized if word not in stop_words]
            
            processed_data.append({
                'title': row['title'],
                'content': row['content'],
                'keywords': ' '.join(keywords)
            })

    with open('processed_pratham_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'content', 'keywords'])
        writer.writeheader()
        writer.writerows(processed_data)

#  Q&A Bot
class PrathamBot:
    def __init__(self):
        model_name = "distilbert-base-cased-distilled-squad"
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
        self.knowledge_base = self.load_knowledge_base()

    def load_knowledge_base(self):
        knowledge_base = ""
        with open('processed_pratham_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                knowledge_base += f"{row['title']}\n{row['content']}\n\n"
        return knowledge_base

    def answer_question(self, question):
        result = self.nlp(question=question, context=self.knowledge_base)
        return result['answer']

# Evaluation
def evaluate_bot():
    bot = PrathamBot()
    questions = [
        "What is Pratham's mission?",
        "When was Pratham founded?",
        "What programs does Pratham offer?",
        "How does Pratham measure its impact?",
        "What is Pratham's approach to education?"
    ]

    print("Pratham.org Q&A Bot Evaluation")
    print("==============================")
    for question in questions:
        answer = bot.answer_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()

    print("Note: Please record a video of this output on your local machine.")

if __name__ == "__main__":
    scrape_pratham()
    process_data()
    evaluate_bot()
