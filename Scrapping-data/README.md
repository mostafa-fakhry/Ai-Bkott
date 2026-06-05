# AI Product Scraper API

A dynamic FastAPI service that scrapes product data from **any e-commerce website** using Claude AI. Just POST a URL and get back structured product JSON — no site-specific configuration needed.

---

## Features

- Works on any e-commerce site (no custom selectors needed)
- Extracts: name, price, description, image URL, product URL, category, rating, availability, SKU, brand
- Returns clean JSON list of products
- Auto-fixes relative URLs to absolute
- Configurable max product count

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run the server

```bash
uvicorn main:app --reload --port 8000
```

---

## Usage

### Endpoint

```
POST /scrape
```

### Request Body

```json
{
  "url": "https://books.toscrape.com",
  "max_products": 20
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | ✅ | — | Full URL of the product listing page |
| `max_products` | integer | ❌ | 50 | Max number of products to return |

### Example with curl

```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://books.toscrape.com", "max_products": 10}'
```

### Example with Python

```python
import requests

response = requests.post("http://localhost:8000/scrape", json={
    "url": "https://books.toscrape.com",
    "max_products": 10
})
data = response.json()

for product in data["products"]:
    print(product["name"], product.get("price"))
```

---

## Response Format

```json
{
  "url": "https://books.toscrape.com",
  "total_products": 10,
  "products": [
    {
      "name": "A Light in the Attic",
      "price": "£51.77",
      "description": "It's hard to imagine a world without A Light in the Attic...",
      "image_url": "https://books.toscrape.com/media/cache/2c/da/2cdad67c44b002e7ead0cc35693c0e8b.jpg",
      "product_url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
      "rating": "Three",
      "availability": "In stock"
    }
  ]
}
```

### Product Fields

| Field | Description |
|-------|-------------|
| `name` | Product name/title |
| `price` | Price with currency symbol |
| `description` | Product description or tagline |
| `image_url` | Absolute URL to product image |
| `product_url` | Absolute URL to product detail page |
| `category` | Product category |
| `rating` | Rating/review score |
| `availability` | Stock status |
| `sku` | Product ID or SKU |
| `brand` | Brand name |
| `extra` | Any other fields Claude finds |

---

## Interactive Docs

Visit `http://localhost:8000/docs` for the Swagger UI where you can test the API directly in your browser.

---

## Notes

- Some websites block scrapers (Cloudflare, bot detection). Results may vary.
- Amazon and similar large platforms have strong anti-scraping measures.
- Works best on standard e-commerce and product listing pages.
- Each request uses Anthropic API tokens (approx. $0.01–$0.05 per scrape depending on page size).
