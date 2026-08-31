#!/usr/bin/env python3
"""Crawl the public Search OS Korean knowledge base and preserve full article content.

This script is intended for an authorized Search OS operator. It discovers every
/ko/knowledge/* article, extracts the article body through the related-content
section, removes the repeated corporate footer, and exports Markdown, HTML,
JSON, CSV, and a ZIP archive with validation reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as to_markdown
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://searchos.io"
LISTING_URL = f"{BASE}/ko/knowledge"
EXPECTED_TOTAL = 238
EXPECTED_CATEGORIES = {"SEO": 125, "GEO·AI 검색": 61, "콘텐츠·전략": 52}
OUTPUT = Path("output")
ARTICLES_DIR = OUTPUT / "articles"
HTML_DIR = OUTPUT / "article_html"

CATEGORY_NAMES = tuple(EXPECTED_CATEGORIES)
FOOTER_MARKERS = (
    "사이트는 더 잘 읽히고",
    "콘텐츠는 더 명확해지고",
    "브랜드는 더 많은 질문 속에서 발견됩니다",
    "제품 소개서로 Search OS가 어떻게 동작하는지 먼저 확인해보세요.",
    "자동 검색 최적화 솔루션",
    "## GET STARTED",
    "## COMPANY",
    "## LEGAL",
)


@dataclass
class Article:
    number: int
    category: str
    title: str
    url: str
    slug: str
    markdown: str
    plain_text: str
    html: str
    sha256: str
    characters: int
    words: int
    headings: int
    links: int
    tables: int
    code_blocks: int


def session_with_retries() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    session.headers.update(
        {
            "User-Agent": "SearchOS-Authorized-Archive/1.0 (+https://searchos.io/ko/knowledge)",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Cache-Control": "no-cache",
        }
    )
    return session


def fetch(session: requests.Session, url: str, timeout: int = 45) -> str:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    text = response.text
    if len(text) < 300:
        raise RuntimeError(f"Suspiciously short response ({len(text)} chars): {url}")
    return text


def normalize_url(raw: str) -> str | None:
    url = urljoin(BASE, raw)
    parsed = urlparse(url)
    if parsed.netloc not in {"searchos.io", "www.searchos.io"}:
        return None
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if not re.fullmatch(r"/ko/knowledge/[A-Za-z0-9][A-Za-z0-9._~-]*", path):
        return None
    return urlunparse(("https", "searchos.io", path, "", "", ""))


def infer_category_from_card(anchor: Tag) -> str:
    node: Tag | None = anchor
    for _ in range(6):
        if node is None:
            break
        text = " ".join(node.get_text(" ", strip=True).split())
        for category in CATEGORY_NAMES:
            if category in text:
                return category
        node = node.parent if isinstance(node.parent, Tag) else None
    return ""


def discover_from_listing(html: str) -> tuple[list[str], dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    ordered: list[str] = []
    categories: dict[str, str] = {}

    for anchor in soup.select("a[href]"):
        url = normalize_url(anchor.get("href", ""))
        if not url or url in ordered:
            continue
        ordered.append(url)
        category = infer_category_from_card(anchor)
        if category:
            categories[url] = category

    # Next/React pages may serialize URLs in script data even when anchors hydrate later.
    raw_candidates = re.findall(r"(?:https?:\\?/\\?/searchos\\.io)?\\?/ko\\?/knowledge\\?/([A-Za-z0-9][A-Za-z0-9._~-]*)", html)
    for slug in raw_candidates:
        url = f"{BASE}/ko/knowledge/{slug}"
        if url not in ordered:
            ordered.append(url)

    return ordered, categories


def discover_sitemaps(session: requests.Session) -> list[str]:
    candidates = [
        f"{BASE}/sitemap.xml",
        f"{BASE}/sitemap_index.xml",
        f"{BASE}/sitemap-index.xml",
        f"{BASE}/sitemaps.xml",
    ]
    seen_xml: set[str] = set()
    found: list[str] = []

    def read_sitemap(url: str, depth: int = 0) -> None:
        if depth > 4 or url in seen_xml:
            return
        seen_xml.add(url)
        try:
            xml = fetch(session, url)
        except Exception:
            return
        soup = BeautifulSoup(xml, "xml")
        locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        for loc in locs:
            if loc.endswith(".xml") or "sitemap" in urlparse(loc).path.lower():
                read_sitemap(loc, depth + 1)
                continue
            normalized = normalize_url(loc)
            if normalized and normalized not in found:
                found.append(normalized)

    for candidate in candidates:
        read_sitemap(candidate)
    return found


def select_article_root(soup: BeautifulSoup, title: str) -> Tag:
    h1 = soup.find("h1")
    if not h1:
        raise RuntimeError("No H1 found")

    article = h1.find_parent("article")
    if article:
        return article

    candidates: list[Tag] = []
    node: Tag | None = h1
    while node is not None and node.name not in {"html", "body"}:
        text = " ".join(node.get_text(" ", strip=True).split())
        if title in text and "관련 콘텐츠" in text and len(text) >= 500:
            candidates.append(node)
        node = node.parent if isinstance(node.parent, Tag) else None
    if candidates:
        return candidates[0]  # smallest useful ancestor

    main = h1.find_parent("main") or soup.find("main")
    if main:
        return main
    return soup.body or soup


def detect_category(soup: BeautifulSoup, h1: Tag, fallback: str = "") -> str:
    # Prefer exact nearby text before H1.
    for previous in h1.find_all_previous(string=True, limit=30):
        text = " ".join(str(previous).split())
        if text in CATEGORY_NAMES:
            return text
    page_text = "\n".join(soup.stripped_strings)
    h1_pos = page_text.find(h1.get_text(" ", strip=True))
    prelude = page_text[max(0, h1_pos - 300):h1_pos]
    for category in CATEGORY_NAMES:
        if category in prelude:
            return category
    return fallback


def trim_markdown(markdown: str, title: str) -> str:
    markdown = markdown.replace("\r\n", "\n").replace("\u00a0", " ")
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    # Start at the article H1, discarding breadcrumb/navigation material.
    patterns = [rf"(?m)^#\s+{re.escape(title)}\s*$", rf"(?m)^#\s+{re.escape(title)}\b.*$"]
    start = None
    for pattern in patterns:
        match = re.search(pattern, markdown)
        if match:
            start = match.start()
            break
    if start is None:
        # markdownify may escape punctuation in a heading; use the first H1.
        match = re.search(r"(?m)^#\s+.+$", markdown)
        start = match.start() if match else 0
    markdown = markdown[start:]

    # Keep the complete related-content section; cut at the repeated CTA/footer.
    end_positions = [markdown.find(marker) for marker in FOOTER_MARKERS if markdown.find(marker) >= 0]
    if end_positions:
        markdown = markdown[: min(end_positions)]

    # Remove occasional trailing document-title echo and empty links/images.
    markdown = re.sub(r"(?m)^\[?Image\]?\s*$", "", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown + "\n"


def extract_article(number: int, url: str, html: str, fallback_category: str) -> Article:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if not h1:
        raise RuntimeError(f"No article H1: {url}")
    title = " ".join(h1.get_text(" ", strip=True).split())
    category = detect_category(soup, h1, fallback_category)
    if category not in CATEGORY_NAMES:
        raise RuntimeError(f"Unknown category {category!r}: {url}")

    root = select_article_root(soup, title)
    isolated = BeautifulSoup(str(root), "lxml")
    for tag in isolated.select("script, style, noscript, nav, footer, form, button, svg, canvas"):
        tag.decompose()
    for tag in isolated.select("[aria-hidden='true']"):
        tag.decompose()

    markdown = to_markdown(
        str(isolated),
        heading_style="ATX",
        bullets="-",
        strip=["span"],
        escape_asterisks=False,
        escape_underscores=False,
    )
    markdown = trim_markdown(markdown, title)

    content_soup = BeautifulSoup(str(isolated), "lxml")
    plain = "\n".join(s.strip() for s in content_soup.stripped_strings if s.strip())
    for marker in FOOTER_MARKERS:
        pos = plain.find(marker)
        if pos >= 0:
            plain = plain[:pos].rstrip()
    # Make the plain text start at the title rather than breadcrumbs.
    title_pos = plain.find(title)
    if title_pos >= 0:
        plain = plain[title_pos:]

    if len(markdown) < 400:
        raise RuntimeError(f"Extracted article is too short ({len(markdown)} chars): {url}")
    if "관련 콘텐츠" not in markdown:
        raise RuntimeError(f"Related-content boundary missing: {url}")
    if any(marker in markdown for marker in FOOTER_MARKERS):
        raise RuntimeError(f"Footer marker leaked into article: {url}")

    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return Article(
        number=number,
        category=category,
        title=title,
        url=url,
        slug=slug,
        markdown=markdown,
        plain_text=plain,
        html=str(isolated),
        sha256=digest,
        characters=len(markdown),
        words=len(re.findall(r"\S+", plain)),
        headings=len(re.findall(r"(?m)^#{1,6}\s+", markdown)),
        links=len(re.findall(r"\[[^\]]+\]\([^)]+\)", markdown)),
        tables=len(isolated.find_all("table")),
        code_blocks=len(re.findall(r"```", markdown)) // 2,
    )


def frontmatter(article: Article) -> str:
    def q(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    return (
        "---\n"
        f"number: {article.number}\n"
        f"category: {q(article.category)}\n"
        f"title: {q(article.title)}\n"
        f"url: {q(article.url)}\n"
        f"slug: {q(article.slug)}\n"
        f"sha256: {q(article.sha256)}\n"
        "---\n\n"
    )


def export(articles: list[Article], discovery: dict) -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    ARTICLES_DIR.mkdir(parents=True)
    HTML_DIR.mkdir(parents=True)

    for article in articles:
        md_path = ARTICLES_DIR / f"{article.number:03d}_{article.slug}.md"
        html_path = HTML_DIR / f"{article.number:03d}_{article.slug}.html"
        md_path.write_text(frontmatter(article) + article.markdown, encoding="utf-8")
        html_path.write_text(article.html, encoding="utf-8")

    manifest_fields = [
        "number", "category", "title", "url", "slug", "sha256", "characters",
        "words", "headings", "links", "tables", "code_blocks",
    ]
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=manifest_fields)
        writer.writeheader()
        for article in articles:
            data = asdict(article)
            writer.writerow({field: data[field] for field in manifest_fields})

    json_articles = []
    for article in articles:
        data = asdict(article)
        data.pop("html")
        json_articles.append(data)
    (OUTPUT / "articles.json").write_text(
        json.dumps(json_articles, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    category_counts = Counter(a.category for a in articles)
    report = {
        "source": LISTING_URL,
        "expected_total": EXPECTED_TOTAL,
        "actual_total": len(articles),
        "expected_categories": EXPECTED_CATEGORIES,
        "actual_categories": dict(category_counts),
        "unique_urls": len({a.url for a in articles}),
        "unique_titles": len({a.title for a in articles}),
        "total_characters": sum(a.characters for a in articles),
        "total_words": sum(a.words for a in articles),
        "total_tables": sum(a.tables for a in articles),
        "total_code_blocks": sum(a.code_blocks for a in articles),
        "discovery": discovery,
        "validation": {
            "total_matches": len(articles) == EXPECTED_TOTAL,
            "category_counts_match": dict(category_counts) == EXPECTED_CATEGORIES,
            "no_duplicate_urls": len({a.url for a in articles}) == len(articles),
            "no_duplicate_titles": len({a.title for a in articles}) == len(articles),
            "all_have_related_content": all("관련 콘텐츠" in a.markdown for a in articles),
            "footer_removed": all(not any(m in a.markdown for m in FOOTER_MARKERS) for a in articles),
        },
    }
    (OUTPUT / "crawl_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    combined = [
        "# Search OS SEO·GEO·AI 검색·콘텐츠 전략 전체 콘텐츠 아카이브",
        "",
        f"- 원문 목록: {LISTING_URL}",
        f"- 수록 게시물: {len(articles)}개",
        f"- 분류: SEO {category_counts.get('SEO', 0)}개 · GEO·AI 검색 {category_counts.get('GEO·AI 검색', 0)}개 · 콘텐츠·전략 {category_counts.get('콘텐츠·전략', 0)}개",
        "- 공통 사이트 CTA·회사·법적 고지 푸터는 제외하고, 각 게시물의 제목부터 관련 콘텐츠 섹션까지 보존했습니다.",
        "",
        "## 전체 목차",
        "",
    ]
    for article in articles:
        combined.append(f"{article.number}. [{article.title}]({article.url}) — {article.category}")
    for article in articles:
        combined.extend(
            [
                "",
                "\\newpage",
                "",
                f"<!-- ARTICLE {article.number:03d}: {article.url} -->",
                "",
                f"> **분류:** {article.category}  ",
                f"> **원문:** {article.url}",
                "",
                article.markdown.rstrip(),
            ]
        )
    (OUTPUT / "SearchOS_238_full_content.md").write_text(
        "\n".join(combined).rstrip() + "\n", encoding="utf-8"
    )

    shutil.make_archive("searchos_crawl_output", "zip", root_dir=OUTPUT)


def validate_or_fail(articles: list[Article]) -> None:
    errors: list[str] = []
    if len(articles) != EXPECTED_TOTAL:
        errors.append(f"Expected {EXPECTED_TOTAL} articles, got {len(articles)}")
    counts = Counter(a.category for a in articles)
    if dict(counts) != EXPECTED_CATEGORIES:
        errors.append(f"Category counts mismatch: {dict(counts)}")
    if len({a.url for a in articles}) != len(articles):
        errors.append("Duplicate article URLs")
    if len({a.title for a in articles}) != len(articles):
        errors.append("Duplicate article titles")
    if errors:
        raise RuntimeError("; ".join(errors))


def main() -> int:
    session = session_with_retries()
    listing_html = fetch(session, LISTING_URL)
    urls, listing_categories = discover_from_listing(listing_html)
    listing_count = len(urls)
    sitemap_urls: list[str] = []
    if listing_count != EXPECTED_TOTAL:
        sitemap_urls = discover_sitemaps(session)
        for url in sitemap_urls:
            if url not in urls:
                urls.append(url)

    discovery = {
        "listing_url_count": listing_count,
        "sitemap_url_count": len(sitemap_urls),
        "combined_url_count": len(urls),
    }
    print(json.dumps(discovery, ensure_ascii=False), flush=True)

    if len(urls) != EXPECTED_TOTAL:
        OUTPUT.mkdir(exist_ok=True)
        (OUTPUT / "discovered_urls.json").write_text(
            json.dumps(urls, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"URL discovery count mismatch: expected {EXPECTED_TOTAL}, got {len(urls)}")

    articles: list[Article] = []
    errors: list[dict[str, str | int]] = []
    for number, url in enumerate(urls, start=1):
        print(f"[{number:03d}/{len(urls)}] {url}", flush=True)
        try:
            html = fetch(session, url)
            article = extract_article(number, url, html, listing_categories.get(url, ""))
            articles.append(article)
            print(
                f"  OK {article.category} | {article.title} | {article.characters} chars | "
                f"{article.tables} tables | {article.code_blocks} code blocks",
                flush=True,
            )
        except Exception as exc:
            errors.append({"number": number, "url": url, "error": repr(exc)})
            print(f"  ERROR {exc!r}", file=sys.stderr, flush=True)
        time.sleep(0.15)

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export(articles, discovery)
    validate_or_fail(articles)
    print(f"SUCCESS: exported {len(articles)} complete articles", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
