import os
import requests
import csv
import pandas as pd
from bs4 import BeautifulSoup
import asyncio

def _extract_best_table(tables) -> Optional[pd.DataFrame]:
    import pandas as pd
    import io
    valid_tables = []
    for t in tables:
        classes = t.get('class', [])
        classes_str = " ".join(classes).lower()
        # Exclude infoboxes, navboxes, sidebars, TOCs, etc.
        if any(term in classes_str for term in ['infobox', 'navbox', 'sidebar', 'metadata', 'navigation', 'toc', 'nomobile', 'navbox-inner']):
            continue
        
        try:
            df_list = pd.read_html(io.StringIO(str(t)))
            if df_list:
                df = df_list[0]
                if df.shape[1] >= 2 and df.shape[0] >= 2:
                    valid_tables.append((df, df.shape[0] * df.shape[1]))
        except Exception:
            continue
            
    if valid_tables:
        # Pick the largest table (rows * cols)
        valid_tables.sort(key=lambda x: x[1], reverse=True)
        return valid_tables[0][0]
        
    # Fallback to the first readable table
    for t in tables:
        try:
            df_list = pd.read_html(io.StringIO(str(t)))
            if df_list and df_list[0].shape[0] >= 1:
                return df_list[0]
        except Exception:
            continue
            
    return None

from typing import Optional

async def scrape_url(url: str, output_csv_path: str) -> dict:
    """
    Scrapes tabular data from a URL. If Playwright is available and works, uses it to load JS content.
    Otherwise, falls back to standard HTTP requests and BeautifulSoup.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Try using BeautifulSoup first (quicker, doesn't require browser binaries)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        tables = soup.find_all('table')
        
        if tables:
            df = _extract_best_table(tables)
            if df is not None:
                df.to_csv(output_csv_path, index=False)
                return {
                    "success": True,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "method": "BeautifulSoup Table Extraction"
                }
    except Exception as e:
        print(f"BeautifulSoup scraping failed/no tables: {e}")

    # Playwright dynamic scraper fallback
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=15000)
            content = await page.content()
            await browser.close()
            
            soup = BeautifulSoup(content, 'html.parser')
            tables = soup.find_all('table')
            if tables:
                df = _extract_best_table(tables)
                if df is not None:
                    df.to_csv(output_csv_path, index=False)
                    return {
                        "success": True,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "method": "Playwright JS Table Extraction"
                    }
    except Exception as e:
        print(f"Playwright scraping failed: {e}")

    # Standard fallback: Parse lists/paragraphs if no tables found at all
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        data = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            text = tag.get_text().strip()
            if text and len(text) > 10:
                data.append({"type": tag.name, "content": text})
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_csv_path, index=False)
            return {
                "success": True,
                "rows": len(df),
                "columns": list(df.columns),
                "method": "BeautifulSoup Text Compilation"
            }
    except Exception as e:
        return {"success": False, "error": f"All scraping methods failed: {str(e)}"}
        
    return {"success": False, "error": "No table or meaningful text content found to extract."}

