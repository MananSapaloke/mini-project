import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
from unstructured.partition.html import partition_html
from unstructured.staging.base import elements_to_json


# ---------------- Helper Functions ----------------
def get_resource_type(url):
    """Determines the type of a resource based on its URL extension."""
    if url.endswith('.pdf'):
        return 'pdf'
    if url.endswith('.docx'):
        return 'docx'
    if url.endswith('.doc'):
        return 'doc'
    return 'html'


def is_valid_url(url):
    """Checks if a URL is a valid, non-fragment HTTP/HTTPS URL that we want to crawl."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and \
           parsed.scheme in ['http', 'https'] and not parsed.fragment


def crawl_website(start_url, max_depth):
    """
    Crawls a website starting from a given URL to a maximum depth.
    Returns a list of discovered resources.
    """
    urls_to_visit = [(start_url, 0)]
    visited_urls = set()
    crawled_resources = []
    
    print(f"\nStarting crawl from: {start_url} (Max Depth: {max_depth})")
    
    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)

        if current_url in visited_urls or current_depth > max_depth:
            continue

        try:
            visited_urls.add(current_url)
            print(f"  -> Depth {current_depth}: Crawling {current_url}")

            headers = {'User-Agent': 'MyProjectCrawler/1.0'}
            response = requests.get(current_url, timeout=10, headers=headers)
            response.raise_for_status()

            resource_info = {
                'source_url': current_url,
                'type': get_resource_type(current_url),
                'depth': current_depth,
            }
            crawled_resources.append(resource_info)

            # If it's HTML and depth not exceeded, extract links
            if resource_info['type'] == 'html' and current_depth < max_depth:
                soup = BeautifulSoup(response.content, 'html.parser')
                for link_tag in soup.find_all('a', href=True):
                    href = link_tag['href']
                    absolute_url = urljoin(current_url, href)

                    if is_valid_url(absolute_url):
                        if urlparse(absolute_url).netloc == urlparse(start_url).netloc:
                            if absolute_url not in visited_urls:
                                urls_to_visit.append((absolute_url, current_depth + 1))

        except requests.exceptions.RequestException as e:
            print(f"    [!] Error crawling {current_url}: {e}")
        except Exception as e:
            print(f"    [!] Unexpected error at {current_url}: {e}")
            
    return crawled_resources


def process_with_unstructured(resources, results_dir="results"):
    """
    Processes HTML resources with unstructured and saves results to JSON.
    """
    os.makedirs(results_dir, exist_ok=True)

    for i, res in enumerate(resources, 1):
        if res['type'] != 'html':
            continue

        url = res['source_url']
        try:
            print(f"\n[Unstructured] Processing {url}")

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            html_content = response.text
            tmp_file = os.path.join(results_dir, f"page_{i}.html")

            # Save the raw HTML temporarily
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            start_time = time.time()

            elements = partition_html(filename=tmp_file)

            output_file = os.path.join(results_dir, f"page_{i}-output.json")
            elements_to_json(elements=elements, filename=output_file)

            elapsed = time.time() - start_time
            print(f"  ✔ Saved parsed content to {output_file} ({elapsed:.2f}s)")

            # Optional: remove raw HTML after parsing
            os.remove(tmp_file)

        except Exception as e:
            print(f"    [!] Error processing {url}: {e}")


# ---------------- Main Script ----------------
if __name__ == "__main__":
    # Ask user for input website
    start_url = input("Enter the website URL to crawl: ").strip()

    # Ask crawl depth
    while True:
        try:
            depth_input = input("Enter the maximum crawl depth (e.g., 2): ")
            max_depth = int(depth_input)
            if max_depth < 0:
                print("Please enter a non-negative number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # Run crawler
    all_resources = crawl_website(start_url, max_depth)

    # Save crawl metadata
    output_filename = 'crawled_resources.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_resources, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*40)
    print(f"Crawling finished. Found {len(all_resources)} resources.")
    print(f"Metadata saved to '{output_filename}'")
    print("="*40)

    # Process with unstructured
    process_with_unstructured(all_resources, results_dir="results")
    print("\nAll done! Parsed docs are in the 'results' folder.")
