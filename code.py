import requests
from bs4 import BeautifulSoup
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

#  Data Sourcing
def scrape_pratham():
    url = "https://www.pratham.org"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract relevant information
    # Adjust these selectors based on the actual structure of pratham.org
    articles = soup.find_all('div', class_='content-wrapper')
    
    data = []
    for article in articles:
        title = article.find('h2').text.strip() if article.find('h2') else "No Title"
        content = article.find('div', class_='field-item').text.strip() if article.find('div', class_='field-item') else "No Content"
        data.append({'title': title, 'content': content})
    
    # Save to CSV
    with open('pratham_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'content'])
        writer.writeheader()
        writer.writerows(data)

#  Knowledge Base
def process_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

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
        writer.writerows(data)

# Q&A Bot
class PrathamBot:
    def __init__(self):
        self.model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        self.model = BertForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.knowledge_base = self.load_knowledge_base()

    def load_knowledge_base(self):
        knowledge_base = ""
        with open('processed_pratham_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                knowledge_base += f"{row['title']}\n{row['content']}\n\n"
        return knowledge_base

    def answer_question(self, question):
        inputs = self.tokenizer.encode_plus(question, self.knowledge_base, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.model(**inputs, return_dict=False)

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer

# Checker
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
    print("Scraping Pratham.org...")
    scrape_pratham()
    print("Processing data...")
    process_data()
    print("Evaluating bot...")
    evaluate_bot()
