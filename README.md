# stikbuf.github.io

个人 GitHub Pages 博客，基于 Jekyll 和 Minima 主题构建。  
Personal GitHub Pages blog built with Jekyll and the Minima theme.

## 项目结构 / Project Structure

```text
.
├── _config.yml                         # Jekyll site configuration
├── index.md                            # Homepage / 首页
├── R3S.md                              # Blog page / 博客页面
├── rational-inference-of-brain-in-a-vat.md
│                                        # ChatGPT conversation article
├── assets/main.scss                    # Site styles and chat layout
├── _includes/
│   ├── head.html                       # MathJax and head override
│   ├── header.html                     # Navigation limited to blog pages
│   └── footer.html                     # Footer with home link
├── tools/convert/                      # Reusable conversion scripts
├── .local/convert/                     # Local-only conversion workspace
└── _site/                              # Generated site output
```

`_site/` 和 `.local/` 是本地生成或本地工作目录，不提交到仓库。  
`_site/` and `.local/` are local output/workspace directories and should not be committed.

## 本地预览 / Local Preview

```powershell
bundle _2.5.22_ install
bundle _2.5.22_ exec jekyll build
bundle _2.5.22_ exec jekyll serve --livereload
```

本地站点地址：`http://127.0.0.1:4000/`。  
The local site is served at `http://127.0.0.1:4000/`.

## 内容约定 / Content Notes

- 公开页面文件名使用英文 slug，例如 `rational-inference-of-brain-in-a-vat.md`。
- Public page filenames should use English slugs, such as `rational-inference-of-brain-in-a-vat.md`.
- ChatGPT 对话导出使用 `tools/convert/html_to_markdown.py` 转换。
- ChatGPT conversation exports are converted with `tools/convert/html_to_markdown.py`.
- 对话类博客正文开头注明“此博客为我和 ChatGPT 的对话。”，结尾加“完”。
- Conversation posts include a short ChatGPT note at the top and a closing `完`.
