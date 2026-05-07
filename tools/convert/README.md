# Conversion Tools

Use this directory for reusable conversion scripts. Keep source files and generated output in the ignored local workspace:

```text
.local/convert/input/
.local/convert/output/
.local/convert/tmp/
```

Convert one HTML file:

```powershell
python tools/convert/html_to_markdown.py .local/convert/input/page.html .local/convert/output/page.md
```

Convert a directory recursively:

```powershell
python tools/convert/html_to_markdown.py .local/convert/input .local/convert/output
```

Clean a ChatGPT exported conversation page:

```powershell
python tools/convert/html_to_markdown.py --chatgpt .local/convert/input/page.htm .local/convert/output/page.md
```

In `--chatgpt` mode, KaTeX formulas are converted to Markdown math (`$...$` or `$$...$$`), tables are converted to Markdown table syntax, and ChatGPT page chrome is removed.

Generate a blog-ready ChatGPT-style conversation layout:

```powershell
python tools/convert/html_to_markdown.py --chatgpt --styled .local/convert/input/page.htm .local/convert/output/page.md
```

The `--styled` option keeps the file as Markdown, but wraps each message in HTML blocks that use the site-wide chat bubble styles from `assets/main.scss`.

For public blog pages, use an English slug filename such as `rational-inference-of-brain-in-a-vat.md`, add Jekyll front matter, and copy only the generated body into the source page. Styled ChatGPT output automatically includes the note `此博客为我和 ChatGPT 的对话。` and the closing marker `完`.

The script reads and writes UTF-8. It does not modify input files.
