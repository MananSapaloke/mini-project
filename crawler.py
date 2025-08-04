import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

# --- (No changes to these helper functions) ---
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
    """
    Checks if a URL is a valid, non-fragment HTTP/HTTPS URL that we want to crawl.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and \
           parsed.scheme in ['http', 'https'] and not parsed.fragment

# --- (No changes to the main crawl_website function) ---
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
            crawled_resources.append(resource_info)

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
            print(f"    [!] An unexpected error occurred at {current_url}: {e}")
            
    return crawled_resources

# --- This is where the script execution starts (MODIFIED SECTION) ---
if __name__ == "__main__":
    initial_url = "http://books.toscrape.com/"
    
    # --- NEW: Get crawl depth from user input ---
    while True:
        try:
            # Ask the user to enter the depth
            depth_input = input("Enter the maximum crawl depth (e.g., 2): ")
            # Convert the input string to an integer
            max_depth = int(depth_input)
            # Ensure the number is not negative
            if max_depth < 0:
                print("Please enter a non-negative number.")
                continue
            # If input is valid, break the loop
            break
        except ValueError:
            # This block runs if the user enters text instead of a number
            print("Invalid input. Please enter a whole number.")

    # Run the crawler with the user-defined depth
    all_resources = crawl_website(initial_url, max_depth)
    
    # Save the Output to a JSON file
    output_filename = 'crawled_resources.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_resources, f, indent=4, ensure_ascii=False)
    
    # Print a summary
    print("\n" + "="*40)
    print(f"Crawling finished. Found {len(all_resources)} resources.")
    print(f"Results saved to '{output_filename}'")
    print("="*40)