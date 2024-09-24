import requests
from bs4 import BeautifulSoup
import mysql.connector
import configparser
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from datetime import datetime, timedelta

# Load configurations from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Function to extract articles
def get_articles():
    url = "https://www.nature.com/subjects/oncology"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    for article in soup.find_all('article'):
        # Extract title, author, date, abstract
        title = article.find('h3').text if article.find('h3') else 'No title available'
        
        author_tag = article.find('p', class_='author')
        author = author_tag.text if author_tag else 'No author available'
        
        date_tag = article.find('time')
        date = date_tag.get('datetime') if date_tag else 'No date available'
        
        abstract_tag = article.find('div', class_='abstract')
        abstract = abstract_tag.text if abstract_tag else 'No abstract available'
        
        articles.append((title, author, date, abstract))

    return articles

# Function to store data in MySQL
def store_in_mysql(articles):
    connection = mysql.connector.connect(
        host=config['mysql']['host'],
        user=config['mysql']['user'],
        password=config['mysql']['password'],
        database=config['mysql']['database']
    )
    cursor = connection.cursor()
    for article in articles:
        cursor.execute(
            "INSERT INTO oncology_articles (title, author, pub_date, abstract) VALUES (%s, %s, %s, %s)",
            article
        )
    connection.commit()
    cursor.close()
    connection.close()

# Function to connect and store in Milvus
def store_in_milvus(titles):
    connections.connect(alias="default", host=config['milvus']['host'], port=config['milvus']['port'])

    # Load pre-trained sentence embedding model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(titles)

    if not utility.has_collection("oncology_titles"):
        create_milvus_collection()

    collection = Collection("oncology_titles")
    collection.insert([titles, embeddings])

def create_milvus_collection():
    # Define schema and create collection in Milvus
    pass

def search_in_milvus(query):
    # Connect to Milvus
    connections.connect(alias="default", host=config['milvus']['host'], port=config['milvus']['port'])

    # Load the pre-trained model for query embedding
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Create an embedding for the query
    query_embedding = model.encode([query])

    # Define search parameters directly as a dictionary
    search_params = {
        "metric_type": "IP",  # Using Inner Product for similarity search
        "params": {"nprobe": 10},  # Number of probes
    }

    # Perform the search in the Milvus collection
    collection = Collection("oncology_titles")
    results = collection.search(query_embedding, search_params, limit=5)

    # Process and return the results
    articles = []
    for result in results:
        for hit in result:
            articles.append(hit.entity)  # Assuming 'hit.entity' contains the relevant data

    return articles

# Function to search based on natural language queries
def search_journals(query):
    if "last week" in query:
        last_week = datetime.now() - timedelta(days=7)
        return search_by_date(last_week)
    else:
        return search_in_milvus(query)

# Function to search articles by date in MySQL
def search_by_date(date):
    connection = mysql.connector.connect(
        host=config['mysql']['host'],
        user=config['mysql']['user'],
        password=config['mysql']['password'],
        database=config['mysql']['database']
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM oncology_articles WHERE pub_date > %s", (date,))
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return results

# Main execution
if __name__ == "__main__":
    articles = get_articles()
    print(articles)

    store_in_mysql(articles)  # This will insert articles but won't print anything

    # When searching
    found_articles = search_journals("some query")
    print(found_articles)
