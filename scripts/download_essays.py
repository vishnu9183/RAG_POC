"""
Download Paul Graham essays for the RAG demo.

This script:
1. Fetches the essay index from paulgraham.com
2. Downloads each essay as a text file
3. Saves them to data/paul_graham/

The essays are used as the knowledge base for the RAG comparison.
"""
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm


# Paul Graham's essay archive
BASE_URL = "http://www.paulgraham.com"
ARTICLES_URL = f"{BASE_URL}/articles.html"

# Output directory
OUTPUT_DIR = Path("data/paul_graham")


def get_essay_links():
    """Get all essay URLs from the articles page."""
    print("Fetching essay index...")
    
    response = requests.get(ARTICLES_URL)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all links that look like essays
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Essays are .html files in the root
        if href.endswith(".html") and not href.startswith("http"):
            links.append(f"{BASE_URL}/{href}")
    
    # Remove duplicates and sort
    links = sorted(set(links))
    print(f"Found {len(links)} essays")
    
    return links


def clean_text(html_content: str) -> str:
    """Extract and clean text from HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    
    return text


def download_essay(url: str, output_dir: Path) -> bool:
    """Download a single essay and save as text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Get essay name from URL
        name = url.split("/")[-1].replace(".html", "")
        
        # Clean and extract text
        text = clean_text(response.text)
        
        # Skip if too short (probably not an essay)
        if len(text) < 500:
            return False
        
        # Save to file
        output_path = output_dir / f"{name}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    """Download all Paul Graham essays."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get essay links
    links = get_essay_links()
    
    # Download each essay
    print("Downloading essays...")
    success_count = 0
    
    for url in tqdm(links):
        if download_essay(url, OUTPUT_DIR):
            success_count += 1
        # Be nice to the server
        time.sleep(0.5)
    
    print(f"\nDownloaded {success_count} essays to {OUTPUT_DIR}")
    
    # List what we got
    essays = list(OUTPUT_DIR.glob("*.txt"))
    print(f"Total files: {len(essays)}")
    
    # Show some stats
    total_chars = sum(f.stat().st_size for f in essays)
    print(f"Total size: {total_chars / 1024:.1f} KB")
    print(f"Approx tokens: {total_chars // 4:,}")


if __name__ == "__main__":
    main()
