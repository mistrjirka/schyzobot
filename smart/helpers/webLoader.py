
import bs4
from typing import List
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import html2text
import textstat
from langchain_core.documents import Document
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException as SeleniumTimeoutException
import time

class TimeoutException(Exception):
    pass

class MarkdownWebLoader:
    def __init__(self, urls):
        self.urls = urls
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")

    def render_page(self, url):
        driver = webdriver.Chrome(options=self.chrome_options)
        try:
            driver.get(url)
            # Adjust the following line to wait for an element that indicates the page is fully loaded
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            # Scroll down to load more content
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # Wait to load page
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            html = driver.page_source
        except SeleniumTimeoutException:
            html = None
            print(f"Timeout while waiting for page to load: {url}")
        except Exception as e:
            html = None
            print(f"An error occurred: {e}")
        finally:
            
            driver.quit()
        return html

    def load(self, url):
        try:
            print(f"Downloading page: {url}")
            html = self.render_page(url)
            if html is None:
                raise Exception("Failed to render page with Selenium")
            print("Page rendered and downloaded")

            soup = bs4.BeautifulSoup(html, 'html.parser')

            article_content = soup.find('article')
            main_content = soup.find('main')
            if article_content:
                html = str(article_content)
            elif main_content:
                html = str(main_content)
            else:
                html = str(soup)  # Fallback to the entire content if <main> is not found

            markdown = html2text.html2text(html)
            document = Document(page_content=markdown, metadata={"source": url})
            return [document]
        except Exception as e:
            print(f"Error loading page {url}: {e}")
            return None

    def load_page_with_timeout(self, url, timeout, results, index):
        def target():
            results[index] = self.load(url)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            results[index] = None
            raise TimeoutException(f"Loading page {url} timed out.")
        else:
            print(f"Page {url} loaded")

    def load_multiple(self, timeout=30, max_concurrent_loads=10):
        results = [None] * len(self.urls)
        threads = []

        for i in range(0, len(self.urls), max_concurrent_loads):
            current_threads = []
            for j in range(max_concurrent_loads):
                if i + j < len(self.urls):
                    thread = threading.Thread(target=self.load_page_with_timeout, args=(self.urls[i + j], timeout, results, i + j))
                    threads.append(thread)
                    current_threads.append(thread)
                    thread.start()
            for thread in current_threads:
                thread.join()

        print("Results:", results)
        docs = [doc for result in results if result is not None for doc in result]
        print("Docs:", docs)
        return docs

def load_websites(search_results: List[dict], timeout=60, max_concurrent_loads=10):
    urlstosearch = filter(lambda x: "link" in x, search_results)
    urls = [result['link'] for result in urlstosearch]
    print("loading content")
    loader = MarkdownWebLoader(urls)
    docs = loader.load_multiple(timeout, max_concurrent_loads)
    return docs


def filter_and_sort_by_readability(documents: List[Document], min_score=35) -> List[Document]:
    scored_docs = []
    print("Scoring documents by readability")
    for doc in documents:
        readability_score = textstat.flesch_reading_ease(doc.page_content)
        #print(f"Readability score for: {readability_score} {doc.page_content[:250]} ")
        if min_score <= readability_score:
            doc.metadata["readability_score"] = readability_score
            scored_docs.append(doc)
    
    # Sort documents by readability score
    scored_docs.sort(key=lambda doc: doc.metadata["readability_score"], reverse=True)
    return scored_docs

def load_and_split_websites(search_results: List[dict], timeout=50, max_concurrent_loads=10, max_docs=150):
    docs = load_websites(search_results, timeout, max_concurrent_loads)
    
    print("---------------")
    print(f"{len(docs)} documents loaded")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, add_start_index=True
    )

    splited_docs = text_splitter.split_documents(docs)
    print(f"{len(splited_docs)} documents split")
    filtered_and_sorted_docs = filter_and_sort_by_readability(splited_docs)
    filtered_and_sorted_docs = filtered_and_sorted_docs[:max_docs]

    print(f"{len(filtered_and_sorted_docs)} documents split")
    
    return filtered_and_sorted_docs