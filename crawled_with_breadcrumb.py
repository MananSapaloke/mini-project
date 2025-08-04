import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

# --- (Helper Functions) ---
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
    """Checks if a URL is a valid, non-fragment HTTP/HTTPS URL."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and \
           parsed.scheme in ['http', 'https'] and not parsed.fragment

def generate_breadcrumb(url, base_url):
    """
    Generates a breadcrumb-style location path from the base_url to the current url.
    Example: Home → Catalogue → Category → Books → Romance
    """
    path = urlparse(url).path.strip('/')
    parts = path.split('/')
    breadcrumb = ' → '.join([
        p.replace('-', ' ').replace('_', ' ').title()
        for p in parts if p
    ])
    return "Home" if not breadcrumb else f"Home → {breadcrumb}"

# --- (Main Crawler Function) ---
def crawl_website(start_url, max_depth):
    """
    Crawls a website starting from a given URL to a maximum depth.
    It saves the results in the specified JSON format.
    """
    urls_to_visit = [(start_url, 0)]
    visited_urls = set()
    crawled_resources = []

    print(f"Starting crawl from: {start_url} (Max Depth: {max_depth})")

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

            if resource_info['type'] == 'html':
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract page title
                title_tag = soup.find('title')
                resource_info['title'] = title_tag.text.strip() if title_tag else "No title available"

                # Generate breadcrumb
                resource_info['location'] = generate_breadcrumb(current_url, start_url)

                # Extract and queue next-level URLs
                if current_depth < max_depth:
                    for link_tag in soup.find_all('a', href=True):
                        href = link_tag['href']
                        absolute_url = urljoin(current_url, href)

                        if is_valid_url(absolute_url):
                            if urlparse(absolute_url).netloc == urlparse(start_url).netloc:
                                if absolute_url not in visited_urls:
                                    urls_to_visit.append((absolute_url, current_depth + 1))

            crawled_resources.append(resource_info)

        except requests.exceptions.RequestException as e:
            print(f"    [!] Error crawling {current_url}: {e}")
        except Exception as e:
            print(f"    [!] An unexpected error occurred at {current_url}: {e}")

    return crawled_resources

# --- Script Entry Point ---
if __name__ == "__main__":
    initial_url = "http://books.toscrape.com/"

    # --- Get crawl depth from user input ---
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

    all_resources = crawl_website(initial_url, max_depth)

    # Save to JSON
    output_filename = 'crawled_resources.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_resources, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 40)
    print(f"Crawling finished. Found {len(all_resources)} resources.")
    print(f"Results saved to '{output_filename}'")
    print("=" * 40)
