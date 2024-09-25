import requests
from bs4 import BeautifulSoup
import mysql.connector
import configparser
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from datetime import datetime, timedelta

config = configparser.ConfigParser()
config.read('config.ini')

def get_articles():
    """
    Scrape oncology articles from the Nature website.
    Returns a list of tuples containing title, author, date, and abstract.
    """
    url = "https://www.nature.com/subjects/oncology"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    for article in soup.find_all('article'):
        title = article.find('h3').text if article.find('h3') else 'No title available'
        author_tag = article.find('p', class_='author')
        author = author_tag.text if author_tag else 'No author available'
        date_tag = article.find('time')
        date = date_tag.get('datetime') if date_tag else 'No date available'
        abstract_tag = article.find('div', class_='abstract')
        abstract = abstract_tag.text if abstract_tag else 'No abstract available'

        articles.append((title, author, date, abstract))

    return articles

def store_in_mysql(articles):
    """
    Connects to MySQL database and inserts extracted articles.
    Takes a list of article tuples (title, author, date, abstract) as input.
    """
    connection = mysql.connector.connect(
        host=config['mysql']['host'],
        user=config['mysql']['user'],
        password=config['mysql']['password'],
        database=config['mysql']['database']
    )
    cursor = connection.cursor()
    for article in articles:
        cursor.execute(
            "INSERT INTO oncology_data (title, author, pub_date, abstract) VALUES (%s, %s, %s, %s)",
            article
        )
    connection.commit()
    cursor.close()
    connection.close()

def store_in_milvus(titles):
    """
    Connects to Milvus VectorDB and stores article titles as vectors using embeddings.
    Titles are encoded using a pre-trained SentenceTransformer model.
    """
    connections.connect(alias="default", host=config['milvus']['host'], port=config['milvus']['port'])
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(titles)

    if not utility.has_collection("oncology_titles"):
        create_milvus_collection()

    collection = Collection("oncology_titles")
    collection.insert([titles, embeddings])

def create_milvus_collection():
    """
    Defines and creates a new collection in Milvus for storing oncology titles.
    (This function can be customized further based on schema.)
    """
    pass

def search_in_milvus(query):
    """
    Performs a similarity search on the stored article titles in Milvus based on an input query.
    Returns a list of similar articles.
    """
    connections.connect(alias="default", host=config['milvus']['host'], port=config['milvus']['port'])
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }

    collection = Collection("oncology_titles")
    results = collection.search(query_embedding, search_params, limit=5)

    articles = []
    for result in results:
        for hit in result:
            articles.append(hit.entity)

    return articles

def search_journals(query):
    """
    Handles search requests. If the query contains "last week," it performs a date-based search.
    Otherwise, it uses Milvus for vector search.
    """
    if "last week" in query:
        last_week = datetime.now() - timedelta(days=7)
        return search_by_date(last_week)
    else:
        return search_in_milvus(query)
    
def search_by_date(date):
    """
    Queries MySQL to find articles published after a given date.
    Returns articles published after the specified date.
    """
    connection = mysql.connector.connect(
        host=config['mysql']['host'],
        user=config['mysql']['user'],
        password=config['mysql']['password'],
        database=config['mysql']['database']
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM oncology_data WHERE pub_date > %s", (date,))
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return results


if __name__ == "__main__":
    articles = get_articles()
    store_in_mysql(articles)

    titles = [article[0] for article in articles]
    store_in_milvus(titles)

    found_articles = search_journals("Give me the journal those are published last week")
    print(found_articles)
