# Repository Guidelines

## Project Structure & Module Organization

This repository is a small GitHub Pages site built with Jekyll and the Minima theme. The site configuration lives in `_config.yml`. Top-level Markdown files are source pages: `index.md` is the homepage, while files such as `R3S.md` and `rational-inference-of-brain-in-a-vat.md` become standalone pages. Ruby dependencies are declared in `Gemfile` and locked in `Gemfile.lock`. The `_site/` directory is generated output and must not be edited or committed.

Custom site styling lives in `assets/main.scss`; shared head additions such as MathJax live in `_includes/head.html`. Reusable local conversion scripts live in `tools/convert/`. Keep source exports and generated drafts under `.local/convert/`, which is ignored by Git.

## Build, Test, and Development Commands

Run commands from the repository root:

```powershell
bundle _2.5.22_ install
bundle _2.5.22_ exec jekyll build
bundle _2.5.22_ exec jekyll serve --livereload
```

`bundle install` installs dependencies. `jekyll build` validates the site and writes `_site/`. `jekyll serve --livereload` starts local preview at `http://127.0.0.1:4000/`.

## Coding Style & Naming Conventions

Use Markdown for content pages. Prefer UTF-8 encoding, concise headings, and relative links between pages. Keep public page filenames descriptive, lowercase, and URL-friendly. For new posts, use Jekyll's `_posts/YYYY-MM-DD-title.md` pattern.

YAML files use two-space indentation. Use English slug filenames for public pages, for example `rational-inference-of-brain-in-a-vat.md`. For ChatGPT-derived articles, include a short note near the top stating that the post is based on a ChatGPT conversation.

## Testing Guidelines

There is no separate test framework. Treat a clean Jekyll build as the required check before committing:

```powershell
bundle _2.5.22_ exec jekyll build
```

Also preview locally and click changed links, especially after renaming Markdown files or adding navigation.

For converted ChatGPT pages, check that formulas render through MathJax, tables remain readable, and user messages align right while assistant messages align left.

## Commit & Pull Request Guidelines

The existing history uses short imperative commit messages, for example `Add initial GitHub Pages HTML template`. Continue that style: `Add mind page`, `Fix homepage links`, or `Update Jekyll config`.

Before every commit, run a privacy check. At minimum, inspect `git status --short --ignored`, search tracked content for emails, tokens, keys, local absolute paths, and ChatGPT session fields, and confirm ignored local exports under `.local/` are not staged. Do not commit if sensitive content is found; report the finding and remove or sanitize it first.

Pull requests should describe visible site changes, list build verification, and include screenshots for layout changes. Do not include `_site/`, local logs, or generated cache files.

## Agent-Specific Instructions

Do not overwrite user content in Markdown files without checking current contents first. Keep changes scoped, preserve UTF-8 Chinese text, and prefer GitHub Pages-compatible Jekyll features.
