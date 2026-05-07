#!/usr/bin/env python3
"""Convert basic HTML files to Markdown without third-party dependencies."""

from __future__ import annotations

import argparse
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


BLOCK_TAGS = {"address", "article", "aside", "div", "footer", "header", "main", "nav", "section"}


class MarkdownHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.link_stack: list[str | None] = []
        self.list_stack: list[str] = []
        self.pre_depth = 0
        self.skip_depth = 0
        self.katex_depth = 0
        self.katex_display = False
        self.annotation_depth = 0
        self.annotation_parts: list[str] = []
        self.table_depth = 0
        self.table_rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.cell_tag_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class") or ""
        if self.katex_depth:
            self.katex_depth += 1
            if tag == "annotation" and attrs_dict.get("encoding") == "application/x-tex":
                self.annotation_depth = self.katex_depth
                self.annotation_parts = []
            return
        if tag == "span" and ("katex" in class_name.split() or "katex-display" in class_name.split()):
            self.katex_depth = 1
            self.katex_display = "katex-display" in class_name.split()
            self.annotation_depth = 0
            self.annotation_parts = []
            return
        if self.table_depth:
            self._handle_table_starttag(tag)
            return
        if tag == "table":
            self.table_depth = 1
            self.table_rows = []
            self.current_row = None
            self.current_cell = None
            self.cell_tag_depth = 0
            return
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in BLOCK_TAGS:
            self._blank_line()
        elif tag in {"p", "blockquote"}:
            self._blank_line()
            if tag == "blockquote":
                self._write("> ")
        elif re.fullmatch(r"h[1-6]", tag):
            self._blank_line()
            self._write("#" * int(tag[1]) + " ")
        elif tag == "br":
            self._write("\n")
        elif tag in {"strong", "b"}:
            self._write("**")
        elif tag in {"em", "i"}:
            self._write("*")
        elif tag == "code" and not self.pre_depth:
            self._write("`")
        elif tag == "pre":
            self.pre_depth += 1
            self._blank_line()
            self._write("```\n")
        elif tag == "a":
            self._write("[")
            self.link_stack.append(attrs_dict.get("href"))
        elif tag == "img":
            alt = attrs_dict.get("alt") or ""
            src = attrs_dict.get("src") or ""
            self._write(f"![{alt}]({src})")
        elif tag in {"ul", "ol"}:
            self.list_stack.append(tag)
            self._blank_line()
        elif tag == "li":
            marker = "1." if self.list_stack and self.list_stack[-1] == "ol" else "-"
            indent = "  " * max(len(self.list_stack) - 1, 0)
            self._write(f"\n{indent}{marker} ")

    def handle_endtag(self, tag: str) -> None:
        if self.katex_depth:
            if tag == "annotation" and self.annotation_depth == self.katex_depth:
                latex = unescape("".join(self.annotation_parts)).strip()
                if latex:
                    self._write_math(latex, self.katex_display)
                self.annotation_depth = 0
                self.annotation_parts = []
            self.katex_depth = max(0, self.katex_depth - 1)
            if not self.katex_depth:
                self.katex_display = False
            return
        if self.table_depth:
            self._handle_table_endtag(tag)
            return
        if tag in {"script", "style", "noscript"}:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in {"strong", "b"}:
            self._write("**")
        elif tag in {"em", "i"}:
            self._write("*")
        elif tag == "code" and not self.pre_depth:
            self._write("`")
        elif tag == "pre":
            self._write("\n```")
            self.pre_depth = max(0, self.pre_depth - 1)
            self._blank_line()
        elif tag == "a":
            href = self.link_stack.pop() if self.link_stack else None
            self._write(f"]({href})" if href else "]")
        elif tag in {"p", "blockquote"} or re.fullmatch(r"h[1-6]", tag) or tag in BLOCK_TAGS:
            self._blank_line()
        elif tag in {"ul", "ol"}:
            if self.list_stack:
                self.list_stack.pop()
            self._blank_line()

    def handle_data(self, data: str) -> None:
        if self.annotation_depth:
            self.annotation_parts.append(data)
            return
        if self.katex_depth:
            return
        if self.current_cell is not None:
            self.current_cell.append(re.sub(r"\s+", " ", unescape(data)))
            return
        if self.skip_depth:
            return
        text = data if self.pre_depth else re.sub(r"\s+", " ", data)
        self._write(unescape(text))

    def handle_entityref(self, name: str) -> None:
        self._write(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self._write(unescape(f"&#{name};"))

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    def _write(self, text: str) -> None:
        self.parts.append(text)

    def _blank_line(self) -> None:
        current = "".join(self.parts)
        if not current:
            return
        if current.endswith("\n\n"):
            return
        if current.endswith("\n"):
            self.parts.append("\n")
        else:
            self.parts.append("\n\n")

    def _write_math(self, latex: str, display: bool) -> None:
        if display:
            self._blank_line()
            self._write(f"$$\n{latex}\n$$")
            self._blank_line()
        else:
            self._write(f"${latex}$")

    def _handle_table_starttag(self, tag: str) -> None:
        if tag == "table":
            self.table_depth += 1
        elif tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []
            self.cell_tag_depth = 0
        elif self.current_cell is not None:
            self.cell_tag_depth += 1
            if tag == "br":
                self.current_cell.append("<br>")

    def _handle_table_endtag(self, tag: str) -> None:
        if tag == "table":
            self.table_depth = max(0, self.table_depth - 1)
            if not self.table_depth:
                self._write_table()
        elif tag == "tr":
            if self.current_row is not None:
                self.table_rows.append(self.current_row)
            self.current_row = None
        elif tag in {"td", "th"}:
            if self.current_row is not None and self.current_cell is not None:
                cell = re.sub(r"\s+", " ", "".join(self.current_cell)).strip()
                self.current_row.append(cell)
            self.current_cell = None
            self.cell_tag_depth = 0
        elif self.current_cell is not None:
            self.cell_tag_depth = max(0, self.cell_tag_depth - 1)

    def _write_table(self) -> None:
        rows = [row for row in self.table_rows if row]
        if not rows:
            return
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        escaped = [[cell.replace("|", "\\|") for cell in row] for row in normalized]
        self._blank_line()
        self._write("| " + " | ".join(escaped[0]) + " |\n")
        self._write("| " + " | ".join(["---"] * width) + " |\n")
        for row in escaped[1:]:
            self._write("| " + " | ".join(row) + " |\n")
        self._blank_line()


def convert_html(html: str) -> str:
    parser = MarkdownHTMLParser()
    parser.feed(html)
    parser.close()
    return parser.markdown()


def clean_chatgpt_markdown(markdown: str, fallback_title: str, *, styled: bool = False) -> str:
    lines = markdown.splitlines()
    title = next((line.strip() for line in lines if line.strip()), fallback_title)

    first_message = next(
        (index for index, line in enumerate(lines) if line.strip() == "#### You said:"),
        None,
    )
    if first_message is None:
        return markdown

    body = lines[first_message:]
    for index, line in enumerate(body):
        if "ChatGPT can make mistakes" in line:
            body = body[:index]
            break

    cleaned: list[str] = [f"# {title}", ""]
    skip_blank_after_marker = False
    index = 0
    while index < len(body):
        line = body[index].rstrip()
        stripped = line.strip()

        if stripped == "#### You said:":
            cleaned.extend(["## You", ""])
            skip_blank_after_marker = True
        elif stripped == "#### ChatGPT said:":
            cleaned.extend(["## ChatGPT", ""])
            skip_blank_after_marker = True
        elif stripped in {"Extended", "Share", "Copy", "Copied", "Regenerate"}:
            pass
        elif skip_blank_after_marker and not stripped:
            pass
        elif stripped in {"-", "1."}:
            next_index = index + 1
            while next_index < len(body) and not body[next_index].strip():
                next_index += 1
            if next_index < len(body):
                cleaned.append(f"{stripped} {body[next_index].strip()}")
                index = next_index
            else:
                cleaned.append(line)
        elif stripped == ">":
            next_index = index + 1
            while next_index < len(body) and not body[next_index].strip():
                next_index += 1
            if next_index < len(body):
                cleaned.append(f"> {body[next_index].strip()}")
                index = next_index
            else:
                cleaned.append(line)
        else:
            cleaned.append(line)
            skip_blank_after_marker = False
        index += 1

    text = "\n".join(cleaned)
    while "```\n\n```" in text:
        text = text.replace("```\n\n```", "```")
    text = re.sub(r"\$\$\s*\$\$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip() + "\n"
    if styled:
        return style_chatgpt_markdown(text)
    return text


def style_chatgpt_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    title = lines[0] if lines and lines[0].startswith("# ") else "# ChatGPT Conversation"
    body = lines[1:]
    messages: list[tuple[str, list[str]]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    for line in body:
        stripped = line.strip()
        if stripped in {"## You", "## ChatGPT"}:
            if current_role is not None:
                messages.append((current_role, trim_blank_lines(current_lines)))
            current_role = "user" if stripped == "## You" else "assistant"
            current_lines = []
        elif current_role is not None:
            current_lines.append(line)

    if current_role is not None:
        messages.append((current_role, trim_blank_lines(current_lines)))

    output = [
        title,
        "",
        '<p class="chat-note">此博客为我和 ChatGPT 的对话。</p>',
        "",
        '<div class="chat-thread" markdown="1">',
        "",
    ]
    for role, content in messages:
        class_name = "chat-message-user" if role == "user" else "chat-message-assistant"
        label = "You" if role == "user" else "ChatGPT"
        output.extend(
            [
                f'<section class="chat-message {class_name}" aria-label="{label}" markdown="1">',
                '<div class="chat-bubble" markdown="1">',
                "",
                *content,
                "",
                "</div>",
                "</section>",
                "",
            ]
        )
    output.extend(["</div>", "", '<p class="post-end">完</p>'])
    return "\n".join(output).strip() + "\n"


def trim_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def convert_file(source: Path, target: Path, *, chatgpt: bool = False, styled: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    markdown = convert_html(source.read_text(encoding="utf-8"))
    if chatgpt:
        markdown = clean_chatgpt_markdown(markdown, source.stem, styled=styled)
    target.write_text(markdown, encoding="utf-8", newline="\n")


def iter_html_files(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in {".html", ".htm"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert HTML files to Markdown.")
    parser.add_argument("--chatgpt", action="store_true", help="clean ChatGPT exported conversation pages")
    parser.add_argument("--styled", action="store_true", help="wrap ChatGPT messages for blog bubble styling")
    parser.add_argument("source", type=Path, help="HTML file or directory to convert")
    parser.add_argument("target", type=Path, help="Markdown file or output directory")
    args = parser.parse_args()
    if args.styled and not args.chatgpt:
        parser.error("--styled requires --chatgpt")

    if args.source.is_file():
        target = args.target
        if target.exists() and target.is_dir():
            target = target / f"{args.source.stem}.md"
        convert_file(args.source, target, chatgpt=args.chatgpt, styled=args.styled)
        return 0

    if args.source.is_dir():
        args.target.mkdir(parents=True, exist_ok=True)
        for source in iter_html_files(args.source):
            relative = source.relative_to(args.source).with_suffix(".md")
            convert_file(source, args.target / relative, chatgpt=args.chatgpt, styled=args.styled)
        return 0

    parser.error(f"source does not exist: {args.source}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
