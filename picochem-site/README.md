# picochem-site

Personal blog/portfolio for the picochem project. Built with [Astro](https://astro.build) and MDX.

## Develop

```bash
npm install
npm run dev
```

Then open `http://localhost:4321/`.

## Build

```bash
npm run build      # outputs to ./dist
npm run preview    # serve the production build locally
```

The output is fully static — drop the `dist/` folder on Vercel, Netlify, or Cloudflare Pages.

## Adding a post

Create a new MDX file in `src/content/blog/<slug>.mdx`:

```mdx
---
title: "Your post title"
description: "One-sentence dek shown in the post list and OG card."
pubDate: 2026-05-13
author: "Arjun Virk"
---

import Figure from '../../components/Figure.astro';
import Footnote from '../../components/Footnote.astro';

## A section

Some prose. Footnotes look like this.<Footnote id="1">Footnote body.</Footnote>

<Figure src="/images/foo.svg" alt="..." caption="A caption." />
```

The post automatically appears on the homepage, in the RSS feed, and in the sitemap.

## Project layout

```
src/
  pages/         routes (index, about, blog/[slug], rss.xml)
  layouts/       BaseLayout, BlogPost
  components/    Header, Footer, TableOfContents, Footnote, Figure, Citation, ...
  content/blog/  MDX posts
  styles/        global.css (all design tokens live here)
public/          static assets (favicon, images)
```

## Customization

The whole design lives in `src/styles/global.css` as CSS custom properties — change `--accent`, `--bg`, the font stacks, or the measure in one place.
