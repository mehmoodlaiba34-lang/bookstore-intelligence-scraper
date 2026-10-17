#!/usr/bin/env python3
"""
The Bookstore Intelligence Mission - Web Scraper
================================================
Scrapes https://books.toscrape.com/ (sandbox site) with requests + BeautifulSoup,
builds a Pandas DataFrame, saves competitor_books.csv and answers every
analysis question from the assignment (incl. the CEO GBP 100 challenge and bonuses).

Usage
-----
    python scraper.py                 # BOSS LEVEL: scrape ALL catalogue pages (default)
    python scraper.py --pages 3       # scrape only the first 3 pages
    python scraper.py --pages 5       # BONUS 1

Outputs (in ./output by default, CSV in the current folder)
-----------------------------------------------------------
    competitor_books.csv     Title, Price, Rating, Availability, Product_URL
    analysis_results.md      Answers to the 7 questions, recommendation, CEO challenge
    rating_counts.png        Bar chart of books per star level (BONUS 4)
"""
import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_URL = "https://books.toscrape.com/"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
HEADERS = {"User-Agent": "Mozilla/5.0 (student-scraping-assignment)"}
BUDGET_GBP = 100


# --------------------------------------------------------------------------
# Mission 2 - capture one page
# --------------------------------------------------------------------------
def get_soup(url, session, retries=5):
    """Download a page and return a BeautifulSoup object (None if not found).
    Retries automatically on slow/failed connections."""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            response.encoding = "utf-8"      # avoids the 'Â£' encoding glitch on prices
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException as err:
            if attempt == retries:
                raise
            wait = attempt * 3
            print(f"  Connection problem ({type(err).__name__}). Retry {attempt}/{retries - 1} in {wait}s...")
            time.sleep(wait)


def parse_book(card, page_url):
    """Extract one book record from an <article class='product_pod'> element."""
    link = card.h3.a
    title = link["title"]                                   # full title (link text is truncated)
    price_text = card.find("p", class_="price_color").get_text(strip=True)
    price = float(re.sub(r"[^\d.]", "", price_text))        # '£51.77' -> 51.77
    rating_word = [c for c in card.find("p", class_="star-rating")["class"] if c != "star-rating"][0]
    rating = RATING_MAP[rating_word]                        # 'Three' -> 3
    availability = card.find("p", class_="instock availability").get_text(strip=True)
    url = urljoin(page_url, link["href"])
    return {"Title": title, "Price": price, "Rating": rating,
            "Availability": availability, "Product_URL": url}


def scrape_page(page_url, session):
    """Return (list_of_records, has_next_page) for one catalogue page."""
    soup = get_soup(page_url, session)
    if soup is None:
        return [], False
    records = [parse_book(card, page_url) for card in soup.find_all("article", class_="product_pod")]
    has_next = soup.find("li", class_="next") is not None
    return records, has_next


# --------------------------------------------------------------------------
# Mission 3 - scale the operation (one reusable loop, any number of pages)
# --------------------------------------------------------------------------
def scrape_catalogue(base_url=BASE_URL, max_pages=None, delay=0.3):
    """Scrape catalogue pages in a loop. max_pages=None -> all pages (Boss level)."""
    session = requests.Session()
    all_records, page = [], 1
    while max_pages is None or page <= max_pages:
        page_url = urljoin(base_url, f"catalogue/page-{page}.html")
        records, has_next = scrape_page(page_url, session)
        print(f"Page {page:>2}: {len(records)} books")
        all_records.extend(records)
        if not has_next:
            break
        page += 1
        time.sleep(delay)                # be polite to the server
    return pd.DataFrame(all_records, columns=["Title", "Price", "Rating", "Availability", "Product_URL"])


def clean(df):
    """Basic cleaning: strip text, enforce dtypes, drop duplicates."""
    df = df.copy()
    df["Title"] = df["Title"].str.strip()
    df["Availability"] = df["Availability"].str.strip()
    df["Price"] = df["Price"].astype(float)
    df["Rating"] = df["Rating"].astype(int)
    return df.drop_duplicates(subset="Product_URL").reset_index(drop=True)


# --------------------------------------------------------------------------
# Mission 4 - intelligence analysis
# --------------------------------------------------------------------------
def analyse(df):
    in_stock = df["Availability"].str.contains("In stock", case=False)
    five = df[df["Rating"] == 5]
    return {
        "most_expensive": df[df["Price"] == df["Price"].max()],
        "cheapest": df[df["Price"] == df["Price"].min()],
        "avg_price": df["Price"].mean(),
        "five_star_count": int((df["Rating"] == 5).sum()),
        "in_stock_pct": in_stock.mean() * 100,
        "top5": df.sort_values(["Rating", "Price"], ascending=[False, True]).head(5),
        "cheapest_five_star": five[five["Price"] == five["Price"].min()],
        "rating_counts": df["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0),
    }


# --------------------------------------------------------------------------
# Final challenge - GBP 100: maximise total stars (0/1 knapsack, solved exactly)
# --------------------------------------------------------------------------
def ceo_challenge(df, budget=BUDGET_GBP):
    """Exact dynamic-programming knapsack. Prices handled in pence (integers).
    Each book can be bought once. Ties on stars -> choose the cheaper basket."""
    cost = np.rint(df["Price"].values * 100).astype(int)
    value = df["Rating"].values.astype(int)
    cap = int(budget * 100)
    NEG = -10**9
    dp = np.full(cap + 1, NEG, dtype=np.int64)
    dp[0] = 0
    keep = np.zeros((len(df), cap + 1), dtype=bool)
    for i in range(len(df)):
        c, v = cost[i], value[i]
        if c > cap:
            continue
        cand = dp[:cap + 1 - c] + v                      # value if book i is added
        better = cand > dp[c:]
        keep[i, c:] = better
        dp[c:] = np.where(better, cand, dp[c:])
    best_value = dp.max()
    spent = int(np.where(dp == best_value)[0].min())     # cheapest way to reach best value
    chosen, c = [], spent
    for i in range(len(df) - 1, -1, -1):
        if keep[i, c]:
            chosen.append(i)
            c -= cost[i]
    basket = df.iloc[sorted(chosen)][["Title", "Price", "Rating"]].reset_index(drop=True)
    return basket, basket["Price"].sum(), int(basket["Rating"].sum())


# --------------------------------------------------------------------------
# Bonus 4 + report writing
# --------------------------------------------------------------------------
def rating_chart(counts, path):
    ax = counts.plot(kind="bar", color="#4C72B0", edgecolor="black", figsize=(7, 4))
    ax.set_title("Number of Books per Star Rating")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of books")
    ax.set_xticklabels([f"{i} star" for i in counts.index], rotation=0)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [f"£{v:.2f}" if c == "Price" else str(v).replace("|", "/") for c, v in zip(cols, r)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_recommendation(df, res):
    five = df[df["Rating"] == 5]
    median = df["Price"].median()
    cheap5 = five[five["Price"] <= median]
    return (
        f"I would prioritise the 5-star books that are priced at or below the median price (£{median:.2f}) and are in stock. "
        f"Evidence 1: {len(five)} of {len(df)} scraped books ({len(five) / len(df) * 100:.1f}%) are rated 5 stars, "
        f"so customer-approved titles are plentiful, and {res['in_stock_pct']:.1f}% of the catalogue is in stock, so they can be sourced reliably. "
        f"Evidence 2: 5-star books average £{five['Price'].mean():.2f} versus £{res['avg_price']:.2f} for the whole catalogue, "
        f"and {len(cheap5)} of them cost no more than the median, giving the best rating-per-pound for a new store."
    )


def write_report(df, res, basket, total_cost, total_stars, path):
    def one(frame):
        s = md_table(frame[["Title", "Price", "Rating"]])
        return s
    top_exp, cheap = res["most_expensive"], res["cheapest"]
    lines = [
        "# Analysis Results (generated by scraper.py)", "",
        f"Dataset: **{len(df)} books** scraped from books.toscrape.com", "",
        "## Mission 4 — Intelligence Analysis", "",
        "**1. Most expensive book**", one(top_exp), "",
        "**2. Cheapest book**", one(cheap), "",
        f"**3. Average book price:** £{res['avg_price']:.2f}", "",
        f"**4. Books with a 5-star rating:** {res['five_star_count']}", "",
        f"**5. Percentage in stock:** {res['in_stock_pct']:.2f}%", "",
        "**6. Top 5 highest-rated books** (ties on rating broken by lowest price)", one(res["top5"]), "",
        "**7. Cheapest 5-star book**", one(res["cheapest_five_star"]), "",
        "## Business Recommendation", "", build_recommendation(df, res), "",
        f"## Final Challenge — CEO has £{BUDGET_GBP}", "",
        md_table(basket), "",
        f"- **Number of books:** {len(basket)}",
        f"- **Total cost:** £{total_cost:.2f}",
        f"- **Total rating score:** {total_stars}", "",
        "Method: exact 0/1 knapsack (dynamic programming) — each book bought once, "
        "maximising total stars with total cost ≤ £100 (cheapest basket chosen on ties).", "",
        "## Bonus 3 — Books per star rating", "",
        md_table(res["rating_counts"].rename_axis("Rating").reset_index(name="Books")), "",
        "![Rating counts](rating_counts.png)", "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Bookstore Intelligence scraper")
    ap.add_argument("--pages", type=int, default=None, help="pages to scrape (default: all)")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--csv", default="competitor_books.csv")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    df = clean(scrape_catalogue(args.base_url, args.pages))
    print("\nFirst rows:\n", df.head())
    print("\nColumns:", list(df.columns), "| rows:", len(df))
    print(df.dtypes)

    df.to_csv(args.csv, index=False, encoding="utf-8")
    print(f"\nSaved {args.csv}")

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    res = analyse(df)
    basket, cost, stars = ceo_challenge(df)
    rating_chart(res["rating_counts"], out / "rating_counts.png")
    write_report(df, res, basket, cost, stars, out / "analysis_results.md")
    print(f"Saved {out / 'analysis_results.md'} and {out / 'rating_counts.png'}")
    print(f"CEO challenge: {len(basket)} books, £{cost:.2f}, {stars} stars")


if __name__ == "__main__":
    main()
