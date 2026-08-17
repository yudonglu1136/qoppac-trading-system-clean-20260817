#!/usr/bin/env python3
"""Download qoppac.blogspot.com posts into a local SQLite/Markdown archive."""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BLOG_FEED = "https://qoppac.blogspot.com/feeds/posts/default"
USER_AGENT = "Mozilla/5.0 (compatible; qoppac-local-archive/1.0)"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_link_text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "pre", "blockquote"}:
            self.parts.append("\n")
        if tag == "a":
            attrs_dict = dict(attrs)
            self._current_href = attrs_dict.get("href")
            self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._current_href:
            text = clean_text("".join(self._current_link_text))
            self.links.append((text, self._current_href))
            self._current_href = None
            self._current_link_text = []
        if tag in {"p", "div", "li", "h1", "h2", "h3", "pre", "blockquote"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data)
        if self._current_href is not None:
            self._current_link_text.append(data)

    def text(self) -> str:
        return clean_text("".join(self.parts))


@dataclass(frozen=True)
class BlogComment:
    blogger_id: str
    author: str
    published: str
    updated: str
    html_content: str
    text_content: str


@dataclass(frozen=True)
class BlogPost:
    blogger_id: str
    title: str
    url: str
    published: str
    updated: str
    labels: list[str]
    html_content: str
    text_content: str
    links: list[tuple[str, str]]
    comment_count: int
    comments: list[BlogComment]


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:90] or "post"


def request_json(url: str, params: dict[str, str | int], retries: int = 3) -> dict:
    full_url = f"{url}?{urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(full_url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not fetch {full_url}: {last_error}")


def get_self_link(entry: dict) -> str:
    for link in entry.get("link", []):
        if link.get("rel") == "alternate":
            return link.get("href", "")
    return ""


def parse_entry(entry: dict) -> BlogPost:
    html_content = entry.get("content", {}).get("$t", "")
    extractor = TextExtractor()
    extractor.feed(html_content)
    return BlogPost(
        blogger_id=entry.get("id", {}).get("$t", ""),
        title=clean_text(entry.get("title", {}).get("$t", "")),
        url=get_self_link(entry),
        published=entry.get("published", {}).get("$t", ""),
        updated=entry.get("updated", {}).get("$t", ""),
        labels=[item.get("term", "") for item in entry.get("category", []) if item.get("term")],
        html_content=html_content,
        text_content=extractor.text(),
        links=extractor.links,
        comment_count=int(entry.get("thr$total", {}).get("$t", "0")),
        comments=[],
    )


def parse_comment(entry: dict) -> BlogComment:
    html_content = entry.get("content", {}).get("$t", "")
    extractor = TextExtractor()
    extractor.feed(html_content)
    authors = entry.get("author", [])
    author = ""
    if authors:
        author = clean_text(authors[0].get("name", {}).get("$t", ""))
    return BlogComment(
        blogger_id=entry.get("id", {}).get("$t", ""),
        author=author,
        published=entry.get("published", {}).get("$t", ""),
        updated=entry.get("updated", {}).get("$t", ""),
        html_content=html_content,
        text_content=extractor.text(),
    )


def post_numeric_id(blogger_id: str) -> str | None:
    match = re.search(r"\.post-(\d+)$", blogger_id)
    if match:
        return match.group(1)
    return None


def fetch_comments_for_post(post: BlogPost, page_size: int = 20) -> list[BlogComment]:
    post_id = post_numeric_id(post.blogger_id)
    if not post_id or post.comment_count == 0:
        return []
    comments_url = f"https://qoppac.blogspot.com/feeds/{post_id}/comments/default"
    comments: list[BlogComment] = []
    for start in range(1, post.comment_count + 1, page_size):
        payload = request_json(
            comments_url,
            {"alt": "json", "max-results": page_size, "start-index": start},
        )
        entries = payload.get("feed", {}).get("entry", [])
        comments.extend(parse_comment(entry) for entry in entries)
    comments.sort(key=lambda comment: comment.published)
    return comments


def fetch_posts(page_size: int = 20) -> list[BlogPost]:
    first = request_json(BLOG_FEED, {"alt": "json", "max-results": 1})
    total = int(first["feed"]["openSearch$totalResults"]["$t"])
    posts: list[BlogPost] = []
    for start in range(1, total + 1, page_size):
        payload = request_json(
            BLOG_FEED,
            {"alt": "json", "max-results": page_size, "start-index": start},
        )
        entries = payload.get("feed", {}).get("entry", [])
        posts.extend(parse_entry(entry) for entry in entries)
    posts = [
        BlogPost(
            blogger_id=post.blogger_id,
            title=post.title,
            url=post.url,
            published=post.published,
            updated=post.updated,
            labels=post.labels,
            html_content=post.html_content,
            text_content=post.text_content,
            links=post.links,
            comment_count=post.comment_count,
            comments=fetch_comments_for_post(post),
        )
        for post in posts
    ]
    posts.sort(key=lambda post: post.published)
    return posts


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;

        DROP TABLE IF EXISTS posts_fts;
        DROP TABLE IF EXISTS links;
        DROP TABLE IF EXISTS labels;
        DROP TABLE IF EXISTS comments;
        DROP TABLE IF EXISTS posts;

        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            blogger_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            published TEXT,
            updated TEXT,
            labels TEXT NOT NULL,
            labels_json TEXT NOT NULL,
            html_content TEXT NOT NULL,
            text_content TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS links (
            post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            link_text TEXT,
            url TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            blogger_id TEXT UNIQUE NOT NULL,
            author TEXT,
            published TEXT,
            updated TEXT,
            html_content TEXT NOT NULL,
            text_content TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS labels (
            post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            label TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(
            title,
            labels,
            text_content,
            content='posts',
            content_rowid='id'
        );
        """
    )


def write_db(posts: Iterable[BlogPost], db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for post in posts:
            word_count = len(re.findall(r"\w+", post.text_content))
            cur = conn.execute(
                """
                INSERT INTO posts (
                    blogger_id, title, url, published, updated, labels,
                    labels_json, html_content, text_content, word_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    post.blogger_id,
                    post.title,
                    post.url,
                    post.published,
                    post.updated,
                    " | ".join(post.labels),
                    json.dumps(post.labels, ensure_ascii=False),
                    post.html_content,
                    post.text_content,
                    word_count,
                    now,
                ),
            )
            post_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO links (post_id, link_text, url) VALUES (?, ?, ?)",
                [(post_id, text, url) for text, url in post.links],
            )
            conn.executemany(
                "INSERT INTO labels (post_id, label) VALUES (?, ?)",
                [(post_id, label) for label in post.labels],
            )
            conn.executemany(
                """
                INSERT INTO comments (
                    post_id, blogger_id, author, published, updated,
                    html_content, text_content
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        post_id,
                        comment.blogger_id,
                        comment.author,
                        comment.published,
                        comment.updated,
                        comment.html_content,
                        comment.text_content,
                    )
                    for comment in post.comments
                ],
            )
            conn.execute(
                "INSERT INTO posts_fts(rowid, title, labels, text_content) VALUES (?, ?, ?, ?)",
                (post_id, post.title, " ".join(post.labels), post.text_content),
            )
        conn.commit()
    finally:
        conn.close()


def write_markdown(posts: Iterable[BlogPost], markdown_dir: Path) -> None:
    markdown_dir.mkdir(parents=True, exist_ok=True)
    for old_file in markdown_dir.glob("*.md"):
        old_file.unlink()
    for post in posts:
        date = post.published[:10] or "undated"
        file_path = markdown_dir / f"{date}-{slugify(post.title)}.md"
        labels = ", ".join(post.labels)
        links_block = "\n".join(f"- [{text or url}]({url})" for text, url in post.links)
        comments_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"### {comment.author or 'Anonymous'} - {comment.published}",
                        "",
                        comment.text_content,
                    ]
                )
                for comment in post.comments
            ]
        )
        body = [
            f"# {post.title}",
            "",
            f"- URL: {post.url}",
            f"- Published: {post.published}",
            f"- Updated: {post.updated}",
            f"- Labels: {labels}",
            "",
            "## Text",
            "",
            post.text_content,
            "",
            "## Links",
            "",
            links_block,
            "",
            "## Comments",
            "",
            comments_block,
            "",
        ]
        file_path.write_text("\n".join(body), encoding="utf-8")


def write_jsonl(posts: Iterable[BlogPost], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for post in posts:
            handle.write(
                json.dumps(
                    {
                        "blogger_id": post.blogger_id,
                        "title": post.title,
                        "url": post.url,
                        "published": post.published,
                        "updated": post.updated,
                        "labels": post.labels,
                        "text_content": post.text_content,
                        "links": post.links,
                        "comments": [
                            {
                                "blogger_id": comment.blogger_id,
                                "author": comment.author,
                                "published": comment.published,
                                "updated": comment.updated,
                                "text_content": comment.text_content,
                            }
                            for comment in post.comments
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "blog",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    posts = fetch_posts()
    write_db(posts, output_dir / "qoppac_blog.sqlite")
    write_jsonl(posts, output_dir / "qoppac_posts.jsonl")
    write_markdown(posts, output_dir / "markdown")
    print(f"Archived {len(posts)} posts into {output_dir}")


if __name__ == "__main__":
    main()
