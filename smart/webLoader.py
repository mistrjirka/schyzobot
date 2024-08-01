
import bs4
from typing import List
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import html2text
from langchain_core.documents import Document

class MarkdownWebLoader:
    def __init__(self, url):
        self.url = url

    def load(self):
        # Download the webpage
        print("Downloading page")
        response = requests.get(self.url)
        print("Page downloaded")
        
        # Parse the HTML
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        #print("HTML parsed")

        # Convert HTML to markdown
        article_content = soup.find('article')
        main_content = soup.find('main')
        if article_content:
            html = str(article_content)
        if main_content:
            html = str(main_content)
        else:
            html = str(soup)  # Fallback to the entire content if <main> is not found

        
        #print("html: ", soup)
        markdown = html2text.html2text(html)
        #print("markdown: ", markdown)
        document = Document(page_content=markdown, metadata={"source": self.url})
        return [document]


class TimeoutException(Exception):
    print("Timeout")

def load_page_with_timeout(url, timeout, results, index):
    def target():
        try:
            print("Loading page")
            loader = MarkdownWebLoader(url) 
            data = loader.load()
            
            print("Page loaded")
            results[index] = data
        except Exception as e:
            print("Error loading page" + str(e))
            results[index] = None
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        results[index] = None
        raise TimeoutException(f"Loading page {url} timed out.")
    else:
        print(f"Page {url} loaded")

def load_websites(search_results: List[dict], timeout=10, max_concurrent_loads=5):
    urlstosearch = filter(lambda x: "link" in x, search_results)
    urls = [result['link'] for result in urlstosearch]
    print("loading content")
    
    results = [None] * len(urls)
    threads = []
    for i in range(0, len(urls), max_concurrent_loads):
        for j in range(max_concurrent_loads):
            if i + j < len(urls):
                thread = threading.Thread(target=load_page_with_timeout, args=(urls[i + j], timeout, results, i + j))
                threads.append(thread)
                thread.start()
        for thread in threads:
            thread.join()
    print("results: ", results)
    predocs = [result for result in results if result is not None]
    print("predocs: ", predocs)
    docs = []
    for doc in predocs:
        if doc is not None:
            docs += [item for item in doc if item is not None]
    print("docs: ", docs)
    return docs

def load_and_split_websites(search_results: List[dict], timeout=5, max_concurrent_loads=5):
    #print(search_results)
    docs = load_websites(search_results, timeout, max_concurrent_loads)
    
    
    print("---------------")
    print(f"{len(docs)} documents loaded")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, add_start_index=True
    )
    
    splited_docs = text_splitter.split_documents(docs)
    
    print(f"{len(splited_docs)} documents split")
    
    return splited_docs
