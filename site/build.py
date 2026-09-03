#!/usr/bin/env python3
"""Render docs/ from site/data, site/content, site/templates and site/static.

Deterministic: the same inputs always produce byte-identical output, so the grader can
rebuild into a temporary directory and diff. Nothing here reads the clock or git.

  python3 site/build.py                 build into docs/
  python3 site/build.py --out DIR       build into DIR (used by site/verify.py)
"""
import argparse
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DATA, CONTENT, TPL, STATIC = SITE / "data", SITE / "content", SITE / "templates", SITE / "static"
CSS_ORDER = ["tokens.css", "base.css", "layout.css", "components.css", "print.css"]
NAV = [("Experiments", "#experiments"), ("Pipeline", "#pipeline"),
       ("Reference", "#reference"), ("About", "#about")]
TAG_LABELS = [("all", "All thirteen"), ("retrieval", "Retrieval"), ("generation", "Generation & cost"),
              ("security", "Security"), ("agents", "Agents")]


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def fill(template, **slots):
    out = template
    for key, value in slots.items():
        out = out.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{(\w+)\}\}", out)
    if left:
        raise SystemExit(f"unfilled template slots: {sorted(set(left))}")
    return out


def slugify(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "section"


def heading_label(inner):
    """Plain label for a table-of-contents entry. Cut at an em dash so the label stays short
    and, more importantly, so it never duplicates one (the count is part of the evidence)."""
    label = re.sub(r"<[^>]+>", "", inner)
    label = re.split(r"&mdash;|\u2014", label)[0]
    return re.sub(r"\s+", " ", label).strip().rstrip(":,")


def add_heading_ids(body):
    """Give every h2 a stable id and return (body, [(id, label)]). Adding an attribute does
    not change the visible text, so the preserved-body check still holds."""
    entries, used = [], {}
    def repl(match):
        attrs, inner = match.group(1), match.group(2)
        base = slugify(inner)
        used[base] = used.get(base, 0) + 1
        ident = base if used[base] == 1 else f"{base}-{used[base]}"
        entries.append((ident, heading_label(inner)))
        return f'<h2 id="{ident}"{attrs}>{inner}</h2>'
    body = re.sub(r"<h2([^>]*)>(.*?)</h2>", repl, body, flags=re.S)
    return body, entries


def nav_html(base, current):
    items = []
    for label, anchor in NAV:
        href = f"{base}{anchor}"
        cur = ' aria-current="page"' if label == current else ""
        items.append(f'<li><a href="{href}"{cur}>{label}</a></li>')
    inner = "".join(items)
    return (f'<ul class="nav-list">{inner}</ul>'
            f'<details class="nav-menu"><summary>Menu</summary>'
            f'<ul class="nav-list">{inner}</ul></details>')


def page(shell, site, *, title, description, canonical, content, prefix, current=None,
         home=None, nav_base=None):
    return fill(shell,
                title=html.escape(title),
                description=html.escape(description),
                canonical=canonical,
                og_title=html.escape(title),
                og_image=site["url"] + "assets/og.png",
                prefix=prefix,
                home=home if home is not None else (f"{prefix}index.html" if prefix else "#main"),
                nav=nav_html(f"{prefix}index.html" if nav_base is None else nav_base, current),
                content=content,
                updated=site["updated"])


def build_index(site, prose, experiments, pipeline, reference):
    stats_href = {
        "experiments": "#experiments",
        "lines of Python": site["repo_tree"] + "projects/",
        "trap hallucinations": "#exp-09",
        "injection obedience": "#exp-12",
        "total API spend": site["repo"] + "#running-it",
        "disproved my own thesis": "#experiments",
    }
    stats = "".join(
        f'<a class="stat" href="{stats_href[s["l"]]}">'
        f'<span class="n">{s["n"]}</span><span class="l">{s["l"]}</span></a>'
        for s in prose["stats"])

    tiles = "".join([
        '<a class="tile" href="#exp-12"><span class="t-when">Five minutes</span>'
        '<span class="t-what">12 &middot; The attack that scores grounded</span>'
        '<span class="t-why">The threat model, the attack, the defense, the cost of the defense, '
        'and the attack that still beats it.</span></a>',
        '<a class="tile" href="#exp-09"><span class="t-when">Twenty minutes</span>'
        '<span class="t-what">Add 09 and 13</span>'
        '<span class="t-why">How the guardrail was built, then why the obvious fix for the attack '
        'catches none of it.</span></a>',
        '<a class="tile" href="#experiments"><span class="t-when">The rest</span>'
        '<span class="t-what">All thirteen, in order</span>'
        '<span class="t-why">Each one asks a question that can be answered with a number, and '
        'records the answer it got.</span></a>'])

    rail = "".join(
        f'<li class="stage"><span class="stage-name">{html.escape(s["name"])}</span>'
        f'<span class="stage-config">{s["config"]}</span>'
        + (f'<a class="stage-set" href="#exp-{s["set_in"]}">set in {s["set_in"]}</a>'
           if s.get("set_in") else "")
        + '</li>'
        for s in pipeline["stages"])

    chips = "".join(
        f'<button class="chip" type="button" data-filter="{tag}" aria-pressed="'
        f'{"true" if tag == "all" else "false"}">{html.escape(label)}</button>'
        for tag, label in TAG_LABELS)
    chips += ('<span class="chip-sep" aria-hidden="true"></span>'
              '<button class="chip" type="button" data-filter="prediction" aria-pressed="false">'
              'Prediction on record</button>')

    rows = []
    for e in experiments:
        links = "".join(f'<a href="{l["href"]}">{html.escape(l["label"])}</a>' for l in e["links"])
        prediction = "true" if any(l["label"] in ("Prediction", "Preregistration")
                                   for l in e["links"]) else "false"
        rows.append(
            f'<li class="exp" id="exp-{e["number"]}" data-tags="{" ".join(e["tags"])}" '
            f'data-prediction="{prediction}">'
            f'<div class="exp-id"><span class="exp-num">{e["number"]}</span>'
            f'<span class="exp-answer">{html.escape(e["answer"])}</span></div>'
            f'<div class="exp-body"><h3><a href="{e["href"]}">{html.escape(e["title"])}</a></h3>'
            f'<p class="exp-q">{e["question"]}</p>'
            f'<p class="exp-f">{e["finding_html"]}</p>'
            f'<p class="exp-links">{links}</p></div>'
            f'<div class="exp-metric"><span class="m-v">{e["metric_value"]}</span>'
            f'<span class="m-l">{e["metric_label"]}</span></div></li>')

    refgrid = "".join(
        f'<li><a class="refcard" href="reference/{m["slug"]}.html">'
        f'<span class="r-t">{html.escape(m["nav"])}</span>'
        f'<span class="r-b">{m["blurb"]}</span>'
        f'<span class="r-x">experiment {m["experiment"]}</span></a></li>'
        for m in reference)

    return fill((TPL / "index.html").read_text(encoding="utf-8"),
                standfirst=prose["standfirst"], what_this_is=prose["what_this_is"],
                shipping_note=prose["shipping_note"], reference_intro=prose["reference_intro"],
                scope=prose["scope"], blob=site["repo_blob"], repo=site["repo"],
                stats=stats, tiles=tiles, rail_caption=html.escape(pipeline["caption"]),
                rail=rail, chips=chips, ledger="".join(rows), refgrid=refgrid)


def build_reference(site, reference, experiments, template):
    by_slug = {m["slug"]: m for m in reference}
    by_number = {e["number"]: e for e in experiments}
    pages = {}
    for i, m in enumerate(reference):
        body = (CONTENT / "reference" / f"{m['slug']}.html").read_text(encoding="utf-8")
        body, headings = add_heading_ids(body)
        toc = "".join(f'<li><a href="#{i}">{label}</a></li>' for i, label in headings)

        exp = by_number.get(m["experiment"])
        related_exp = ""
        if exp:
            related_exp = (
                '<div><h2 id="behind-this-card">The experiment behind this card</h2>'
                f'<a class="doc-exp" href="{exp["href"]}">'
                f'<span class="d-k">Experiment {exp["number"]}</span>'
                f'<span class="d-t">{html.escape(exp["title"])}</span>'
                f'<span class="d-m">{exp["metric_value"]} {html.escape(exp["metric_label"])}</span>'
                '</a></div>')

        related_cards = ""
        if m["related"]:
            items = "".join(f'<li><a href="{s}.html">{html.escape(by_slug[s]["nav"])}</a></li>'
                            for s in m["related"] if s in by_slug)
            related_cards = f'<div><h2 id="related-cards">Related cards</h2><ul>{items}</ul></div>'

        seq = []
        if i > 0:
            p = reference[i - 1]
            seq.append(f'<a href="{p["slug"]}.html"><span class="s-k">Previous</span>'
                       f'{html.escape(p["nav"])}</a>')
        else:
            seq.append('<a href="../index.html#reference"><span class="s-k">Back</span>'
                       'All reference cards</a>')
        if i < len(reference) - 1:
            n = reference[i + 1]
            seq.append(f'<a href="{n["slug"]}.html"><span class="s-k">Next</span>'
                       f'{html.escape(n["nav"])}</a>')
        else:
            seq.append('<a href="../index.html#experiments"><span class="s-k">Next</span>'
                       'The thirteen experiments</a>')

        content = fill(template, toc=toc, toc_open=" open", eyebrow=m["eyebrow"], body=body,
                       related_exp=related_exp, related_cards=related_cards,
                       prev_next="".join(seq))
        title = re.sub(r"\s+", " ", html.unescape(m["title"])).strip()
        pages[m["slug"]] = dict(
            title=f"{title} · Muneeb Ahmad",
            description=re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m["blurb"]))).strip(),
            canonical=f'{site["url"]}reference/{m["slug"]}.html',
            content=content)
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs"))
    out = Path(ap.parse_args().out)

    site = load("site")
    experiments, pipeline, reference = load("experiments"), load("pipeline"), load("reference")
    prose = json.loads((CONTENT / "prose.json").read_text(encoding="utf-8"))
    shell = (TPL / "shell.html").read_text(encoding="utf-8")

    written = set()

    def write(rel_path, text_or_bytes):
        target = out / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text_or_bytes, bytes):
            target.write_bytes(text_or_bytes)
        else:
            target.write_text(text_or_bytes, encoding="utf-8")
        written.add(rel_path)

    write("index.html", page(shell, site,
                             title=site["title"], description=site["description"],
                             canonical=site["url"], prefix="", home="#main", nav_base="",
                             content=build_index(site, prose, experiments, pipeline, reference)))

    ref_template = (TPL / "reference.html").read_text(encoding="utf-8")
    for slug, p in build_reference(site, reference, experiments, ref_template).items():
        write(f"reference/{slug}.html",
              page(shell, site, title=p["title"], description=p["description"],
                   canonical=p["canonical"], prefix="../", current="Reference",
                   content=p["content"]))

    write("404.html", page(shell, site,
                           title="Page not found · Muneeb Ahmad",
                           description="That page is not here. Everything on this site is reachable "
                                       "from the thirteen experiments or the reference cards.",
                           canonical=site["url"] + "404.html", prefix="",
                           nav_base="index.html",
                           content=fill((TPL / "404.html").read_text(encoding="utf-8"),
                                        url=site["url"])))

    write("assets/site.css",
          "".join((STATIC / "css" / name).read_text(encoding="utf-8").rstrip() + "\n\n"
                  for name in CSS_ORDER).rstrip() + "\n")
    write("assets/site.js", (STATIC / "js" / "site.js").read_text(encoding="utf-8"))
    write("assets/favicon.svg", (STATIC / "favicon.svg").read_text(encoding="utf-8"))
    write("assets/og.png", (STATIC / "og.png").read_bytes())

    urls = [site["url"]] + [f'{site["url"]}reference/{m["slug"]}.html' for m in reference]
    write("sitemap.xml",
          '<?xml version="1.0" encoding="UTF-8"?>\n'
          '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
          + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls)
          + "</urlset>\n")
    write("robots.txt", f'User-agent: *\nAllow: /\nSitemap: {site["url"]}sitemap.xml\n')

    for existing in sorted(p for p in out.rglob("*") if p.is_file()):
        rel_path = str(existing.relative_to(out))
        if rel_path not in written and rel_path != ".nojekyll":
            existing.unlink()
            print("removed stale", rel_path)
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    print(f"built {len(written)} files into {out}")


if __name__ == "__main__":
    main()
