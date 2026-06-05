import os
import json
import re
import sys
import asyncio
import threading
import httpx
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from bs4 import BeautifulSoup
import anthropic

# Load environment variables from a local .env file (e.g. SCRAPERAPI_KEY).
load_dotenv()

app = FastAPI(
    title="AI Product Scraper API",
    description="Scrape product data from any e-commerce website using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic API key, loaded from the environment / .env file.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Add it to your .env file "
        "(ANTHROPIC_API_KEY=your_key) and restart the server."
    )

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ScraperAPI key, loaded from the environment / .env file.
SCRAPERAPI_KEY = os.environ.get("SCRAPERAPI_KEY")


class ScrapeRequest(BaseModel):
    url: HttpUrl
    max_products: int = 10  # limit how many products to return


class Product(BaseModel):
    name: str | None = None
    price: str | None = None
    description: str | None = None
    image_url: str | None = None
    product_url: str | None = None
    category: str | None = None
    rating: str | None = None
    availability: str | None = None
    sku: str | None = None
    brand: str | None = None
    extra: dict | None = None  # any other fields Claude finds


class ScrapeResponse(BaseModel):
    url: str
    total_products: int
    products: list[Product]


async def _render(url: str) -> str:
    """Render a page in headless Chromium and return its fully-loaded HTML."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            # domcontentloaded is fast and reliable; networkidle can stall
            # forever on heavy sites (ads/trackers keep the network busy).
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # Give the network a brief window to settle so client-side data
            # loads, but don't fail if it never goes fully idle.
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            # Give late client-side renders a moment to populate the DOM.
            await page.wait_for_timeout(1500)
            return await page.content()
        finally:
            await browser.close()


def fetch_html(url: str) -> str:
    """Fetch fully-rendered HTML using a headless browser so JavaScript executes.

    Many modern e-commerce sites are JavaScript single-page apps whose product
    data is loaded client-side. A plain HTTP request returns only an empty shell,
    so we render the page in a real browser engine before extracting.

    On Windows, uvicorn installs the Selector event-loop policy globally, but
    Playwright must spawn the browser via subprocess_exec, which only works on a
    Proactor loop. We therefore run the render in a dedicated thread with its own
    explicitly-created Proactor loop, fully isolated from the server's loop.
    """
    box: dict = {}

    def worker():
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["html"] = loop.run_until_complete(_render(url))
        except BaseException as e:  # propagate to the caller thread
            box["err"] = e
        finally:
            loop.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    if "err" in box:
        raise box["err"]
    return box["html"]


def fetch_html_scraperapi(url: str) -> str:
    """Fetch fully-rendered HTML through ScraperAPI.

    ScraperAPI acts as a proxy that handles JavaScript rendering, rotating
    proxies and anti-bot bypass on its own servers, then returns the final
    HTML. This avoids running a local browser and is more robust against
    sites that block scrapers (e.g. Amazon).
    """
    if not SCRAPERAPI_KEY:
        raise RuntimeError(
            "SCRAPERAPI_KEY is not set. Add it to your .env file "
            "(SCRAPERAPI_KEY=your_key) and restart the server."
        )

    params = {
        "api_key": SCRAPERAPI_KEY,
        "url": url,
        "render": "true",  # let ScraperAPI execute the page's JavaScript
    }
    # ScraperAPI renders remotely and can be slow; allow a generous timeout.
    with httpx.Client(timeout=90) as http:
        resp = http.get("https://api.scraperapi.com/", params=params)
        resp.raise_for_status()
        return resp.text


def clean_html(html: str) -> str:
    """Strip scripts, styles, and boilerplate to reduce tokens."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"]):
        tag.decompose()

    # Get cleaned text with some structure preserved
    text = soup.get_text(separator="\n", strip=True)

    # Also extract all image src attributes (Claude needs these)
    img_tags = soup.find_all("img", src=True)
    img_urls = "\n".join(
        f"IMG: {img.get('src', '')} ALT: {img.get('alt', '')}"
        for img in img_tags[:100]  # cap at 100 images
    )

    # Extract all anchor hrefs that look like product links
    links = soup.find_all("a", href=True)
    product_links = "\n".join(
        f"LINK: {a.get('href', '')} TEXT: {a.get_text(strip=True)[:80]}"
        for a in links[:200]
    )

    combined = f"""
=== PAGE TEXT ===
{text[:6000]}

=== IMAGE URLS FOUND ===
{img_urls}

=== PAGE LINKS ===
{product_links[:3000]}
"""
    return combined


def extract_products_with_ai(cleaned_content: str, base_url: str, max_products: int) -> list[dict]:

    prompt = f"""You are a product data extraction expert. Analyze the following web page content and extract all products you can find.

Base URL of the page: {base_url}

For each product, extract as many of these fields as available:
- name: product name/title
- price: price including currency symbol
- description: product description or short text about the product
- image_url: full image URL (fix relative URLs using the base URL)
- product_url: full link to the product page (fix relative URLs using the base URL)
- category: product category if visible
- rating: rating/review score if available
- availability: in stock, out of stock, etc.
- sku: product ID or SKU if visible
- brand: brand name if visible
- extra: any other interesting fields as a key-value object

Rules:
- Fix all relative URLs to absolute using base URL: {base_url}
- Return ONLY a valid JSON array of product objects, no explanation, no markdown
- Maximum {max_products} products
- If a field is not found, omit it (don't include null values)
- Each product must at least have a name or price to be included

PAGE CONTENT:
{cleaned_content}

Return ONLY the JSON array:"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    # Strip markdown code fences if present
    response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)

    products = json.loads(response_text)
    return products if isinstance(products, list) else []


@app.post("/scrape", response_model=ScrapeResponse)
def scrape_products(request: ScrapeRequest):
    """
    Scrape products from any e-commerce website.

    - **url**: The full URL of the product listing page
    - **max_products**: Maximum number of products to return (default: 50)
    """
    url_str = str(request.url)

    try:
        html = fetch_html(url_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/render URL: {str(e)}")

    cleaned = clean_html(html)

    try:
        raw_products = extract_products_with_ai(cleaned, url_str, request.max_products)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction error: {str(e)}")

    # Validate and parse into Product models
    products = []
    for item in raw_products:
        try:
            products.append(Product(**item))
        except Exception:
            # Skip malformed product entries
            continue

    return ScrapeResponse(
        url=url_str,
        total_products=len(products),
        products=products
    )


@app.post("/scrape-scraperapi", response_model=ScrapeResponse)
def scrape_products_scraperapi(request: ScrapeRequest):
    """
    Scrape products from any e-commerce website using ScraperAPI.

    Identical to /scrape, but fetches the page through ScraperAPI (remote
    JavaScript rendering + anti-bot proxies) instead of a local browser.
    Requires SCRAPERAPI_KEY to be set in the environment / .env file.

    - **url**: The full URL of the product listing page
    - **max_products**: Maximum number of products to return (default: 10)
    """
    url_str = str(request.url)

    try:
        html = fetch_html_scraperapi(url_str)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"ScraperAPI error: {e.response.status_code} {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ScraperAPI fetch error: {str(e)}")

    cleaned = clean_html(html)

    try:
        raw_products = extract_products_with_ai(cleaned, url_str, request.max_products)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction error: {str(e)}")

    # Validate and parse into Product models
    products = []
    for item in raw_products:
        try:
            products.append(Product(**item))
        except Exception:
            # Skip malformed product entries
            continue

    return ScrapeResponse(
        url=url_str,
        total_products=len(products),
        products=products
    )


@app.get("/")
async def root():
    return {
        "message": "AI Product Scraper API",
        "usage": "POST /scrape with { url: 'https://example.com/products', max_products: 50 }",
        "routes": {
            "/scrape": "Render locally with Playwright (headless browser)",
            "/scrape-scraperapi": "Render remotely via ScraperAPI (needs SCRAPERAPI_KEY in .env)",
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
