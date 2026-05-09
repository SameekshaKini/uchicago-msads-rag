import time
import json
import logging
import os
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
DOMAIN = "datascience.uchicago.edu"
PATH_PREFIX = "/education/masters-programs/ms-in-applied-data-science/"

CRAWL_DELAY = 1.2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OUT_DIR = "data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection":      "keep-alive",
}

SEED_URLS = [
    BASE_URL,
    BASE_URL + "in-person-program/",
    BASE_URL + "online-program/",
    BASE_URL + "course-progressions/",
    BASE_URL + "how-to-apply/",
    BASE_URL + "instructors-staff/",
    BASE_URL + "tuition-fees-aid/",
    BASE_URL + "career-outcomes/",
    BASE_URL + "faqs/",
    BASE_URL + "capstone-projects/",
    BASE_URL + "events-deadlines/",
    BASE_URL + "our-students/",
]


@dataclass
class PageChunk:
    url: str
    page_title: str
    section: str
    chunk_index: int
    text: str

@dataclass
class ScrapedPage:
    url: str
    page_title: str
    section: str
    raw_text: str
    chunks: list = field(default_factory=list)


def section_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    last = path.split("/")[-1]
    if not last or "ms-in-applied-data-science" in last:
        return "Overview"
    return last.replace("-", " ").title()


def html_to_clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "button"]):
        tag.decompose()

    noise_patterns = [
        "nav", "menu", "footer", "header", "sidebar", "cookie",
        "banner", "breadcrumb", "social", "share", "alert",
        "announcement", "skip-link", "site-header", "site-footer",
        "wp-admin-bar",
    ]
    for pattern in noise_patterns:
        for el in soup.find_all(class_=lambda c: c and pattern in " ".join(c).lower()):
            el.decompose()
        for el in soup.find_all(id=lambda i: i and pattern in i.lower()):
            el.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(id="main-content")
        or soup.find(class_="entry-content")
        or soup.find(class_="page-content")
        or soup.find("body")
        or soup
    )

    raw = main_content.get_text(separator="\n")

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if len(line) > 10:
            lines.append(line)

    return "\n\n".join(lines)


def chunk_text(text: str, url: str, title: str, section: str) -> list:
    words = text.split()
    if not words:
        return []
    chunks, i, idx = [], 0, 0
    while i < len(words):
        chunk_str = " ".join(words[i: i + CHUNK_SIZE])
        chunks.append(PageChunk(
            url=url, page_title=title, section=section,
            chunk_index=idx, text=chunk_str
        ))
        idx += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def fetch_page(url: str, session: requests.Session) -> tuple:
    try:
        response = session.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  x Failed: {url}  ({type(e).__name__}: {e})")
        return None, None

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        log.info(f"  Skipped (not HTML): {url}")
        return None, None

    raw_html = response.text
    soup = BeautifulSoup(raw_html, "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else section_from_url(url)
    title = title.split("|")[0].strip() if "|" in title else title

    section  = section_from_url(url)
    raw_text = html_to_clean_text(soup)

    if len(raw_text) < 100:
        log.info(f"  Skipped (too little content): {url}")
        return None, raw_html

    chunks = chunk_text(raw_text, url, title, section)
    log.info(f"  OK [{section:20s}]  {len(raw_text):6,} chars  ->  {len(chunks):3d} chunks")

    return ScrapedPage(url=url, page_title=title, section=section,
                       raw_text=raw_text, chunks=chunks), raw_html


def discover_links(html: str, base_url: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    found = []
    skip_ext = {".pdf", ".docx", ".xlsx", ".zip", ".png", ".jpg", ".gif"}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            continue

        full_url = urljoin(base_url, href)
        parsed   = urlparse(full_url)

        if parsed.netloc != DOMAIN:
            continue
        if not parsed.path.startswith(PATH_PREFIX):
            continue
        if any(parsed.path.lower().endswith(ext) for ext in skip_ext):
            continue

        clean_path = parsed.path.replace("%20", "").rstrip("/") + "/"
        clean = parsed._replace(path=clean_path, fragment="", query="").geturl()

        if clean not in found:
            found.append(clean)

    return found


def crawl() -> list:
    session = requests.Session()
    visited = set()
    queue = list(SEED_URLS)
    pages = []

    log.info(f"Starting crawl | {len(SEED_URLS)} seed URLs | domain: {DOMAIN}")
    log.info("-" * 70)

    while queue:
        url        = queue.pop(0)
        normalized = url.rstrip("/") + "/"

        if normalized in visited:
            continue
        visited.add(normalized)

        log.info(f"Fetching ({len(visited)}/{len(visited)+len(queue)}): {url}")

        page, raw_html = fetch_page(url, session)

        if page is not None:
            pages.append(page)

        if raw_html:
            new_links = discover_links(raw_html, url)
            added = 0
            for link in new_links:
                norm = link.rstrip("/") + "/"
                if norm not in visited and link not in queue:
                    queue.append(link)
                    added += 1
            if added:
                log.info(f"    - Discovered {added} new link(s)")

        time.sleep(CRAWL_DELAY)

    log.info("-" * 70)
    log.info(f"Done: {len(pages)} pages, {sum(len(p.chunks) for p in pages)} chunks")
    return pages


def save_results(pages: list) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    pages_path = os.path.join(OUT_DIR, "scraped_pages.json")
    with open(pages_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pages], f, indent=2, ensure_ascii=False)
    log.info(f"Saved - {pages_path}")

    all_chunks = [asdict(c) for p in pages for c in p.chunks]
    chunks_path = os.path.join(OUT_DIR, "all_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(all_chunks)} chunks -> {chunks_path}")

    txt_path = os.path.join(OUT_DIR, "raw_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n{'='*70}\n")
            f.write(f"SECTION : {page.section}\n")
            f.write(f"URL : {page.url}\n")
            f.write(f"TITLE : {page.page_title}\n")
            f.write(f"{'='*70}\n\n")
            f.write(page.raw_text)
            f.write("\n")
    log.info(f"Saved readable text - {txt_path}")


def print_summary(pages: list) -> None:
    print("\n" + "=" * 65)
    print(f"{'SECTION':<25} {'CHUNKS':>6}  {'CHARS':>8}")
    print("-" * 65)
    total_chunks = total_chars = 0
    for page in pages:
        c, ch = len(page.chunks), len(page.raw_text)
        total_chunks += c; total_chars += ch
        print(f"{page.section:<25} {c:>6}  {ch:>8,}")
    print("-" * 65)
    print(f"{'TOTAL':<25} {total_chunks:>6}  {total_chars:>8,}")
    print("=" * 65)
