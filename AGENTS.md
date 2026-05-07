# Repository Guidelines

## Project Structure & Module Organization

This repository is a small GitHub Pages site built with Jekyll and the Minima theme. The site configuration lives in `_config.yml`. Top-level Markdown files are source pages: `index.md` is the homepage, while files such as `R3S.md` and `mind.md` become standalone pages. Ruby dependencies are declared in `Gemfile` and locked in `Gemfile.lock`. The `_site/` directory is generated output and must not be edited or committed.

There are currently no dedicated test, asset, or layout directories. Add `_posts/` for dated blog posts, `assets/` for images/CSS/JS, and `_layouts/` or `_includes/` only when custom theme behavior is needed.

## Build, Test, and Development Commands

Run commands from the repository root:

```powershell
bundle _2.5.22_ install
bundle _2.5.22_ exec jekyll build
bundle _2.5.22_ exec jekyll serve --livereload
```

`bundle install` installs the GitHub Pages/Jekyll dependencies. `jekyll build` validates the site and writes `_site/`. `jekyll serve --livereload` starts local preview at `http://127.0.0.1:4000/`.

On this Windows setup, open a new terminal after Ruby installation so `ruby`, `gem`, and `bundle` are on `PATH`. If needed, temporarily prepend `C:\Ruby33-x64\bin`.

## Coding Style & Naming Conventions

Use Markdown for content pages. Prefer UTF-8 encoding, concise headings, and relative links between pages. Keep filenames descriptive and URL-friendly; existing files use simple names such as `index.md` and `mind.md`. For new posts, use Jekyll's `_posts/YYYY-MM-DD-title.md` pattern.

YAML files use two-space indentation. Keep `_config.yml` minimal and avoid adding unsupported GitHub Pages plugins.

## Testing Guidelines

There is no separate test framework. Treat a clean Jekyll build as the required check before committing:

```powershell
bundle _2.5.22_ exec jekyll build
```

Also preview locally and click changed links, especially after renaming Markdown files or adding navigation.

## Commit & Pull Request Guidelines

The existing history uses short imperative commit messages, for example `Add initial GitHub Pages HTML template`. Continue that style: `Add mind page`, `Fix homepage links`, or `Update Jekyll config`.

Pull requests should describe the visible site change, list build verification, and include screenshots for layout or theme changes. Do not include `_site/`, local logs, or generated cache files.

## Agent-Specific Instructions

Do not overwrite user content in Markdown files without checking current contents first. Keep changes scoped, preserve UTF-8 Chinese text, and prefer GitHub Pages-compatible Jekyll features.
