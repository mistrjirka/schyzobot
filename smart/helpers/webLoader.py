
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
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("window-size=1920,1080")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--incognito")        
        self.chrome_options.add_argument("--log-level=3")  # Suppress logs
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])  # Exclude logging to prevent file creation
         

    def render_page(self, url):
        driver = webdriver.Chrome(options=self.chrome_options)
        try:
            print(f"loading {url}")
            driver.get(url)
            # Adjust the following line to wait for an element that indicates the page is fully loaded
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            # Scroll down to load more content
            last_height = driver.execute_script("return document.body.scrollHeight")
            max_tries = 5
            while True and max_tries > 0:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)  # Wait to load page
                print("going down")
                max_tries -= 1
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
            #print(f"Downloading page: {url}")
            html = self.render_page(url)
            if html is None:
                raise Exception("Failed to render page with Selenium")
            #print("Page rendered and downloaded")

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

    def load_multiple(self, timeout=20, max_concurrent_loads=20):
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

        docs = [doc for result in results if result is not None for doc in result]
        return docs

def load_websites(search_results: List[dict], timeout=30, max_concurrent_loads=20):
    urlstosearch = filter(lambda x: "link" in x, search_results)
    urls = [result['link'] for result in urlstosearch]
    print("loading content")
    loader = MarkdownWebLoader(urls)
    docs = loader.load_multiple(timeout, max_concurrent_loads)
    return docs


def filter_and_sort_by_readability(documents: List[Document], min_score=35, min_number_of_docs: int = 10) -> List[Document]:
    scored_docs = []
    print("Scoring documents by readability")
    i = 0
    for doc in documents:
        #adding progress bar
        if i % 10 == 0:
            print(f"Scoring document {i}/{len(documents)}", end="\r")
        
        readability_score = textstat.flesch_reading_ease(doc.page_content)

        doc.metadata["readability_score"] = readability_score
        scored_docs.append(doc)
        i += 1
    print()
    # Sort documents by readability score
    scored_docs.sort(key=lambda doc: doc.metadata["readability_score"], reverse=True)
    
    scored_docs = [doc for doc in scored_docs if doc.metadata["readability_score"] >= min_score]
    if len(scored_docs) < min_number_of_docs:
        print(f"Warning: only {len(scored_docs)} documents with readability score >= {min_score} found")
        scored_docs = scored_docs[:min_number_of_docs]
    return scored_docs

def load_and_split_websites(search_results: List[dict], timeout=30, max_concurrent_loads=4, max_docs=150):
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
