# Visual review rubric

You are reviewing screenshots of a personal engineering portfolio site. You have no other
context and you must not assume good faith: the author is a model that will try to stop
early. Judge only what you can see. A criterion you cannot verify from the screenshots is
a FAIL, not a pass.

Screens: index-desktop (1440 wide, light and dark), index-mobile (390 wide, light and dark),
mcp-desktop (a reference page, 1440, light and dark), chunking-mobile (a reference page,
390, light and dark).

## Hard criteria (any failure = major finding = FAIL)

1. Every screenshot renders real content. No blank page, no raw HTML text, no browser
   error page, no missing stylesheet (unstyled black-on-white Times New Roman).
2. Dark screenshots have a dark background and light text throughout. Light screenshots
   the reverse. No section keeps the wrong scheme.
3. Mobile screenshots: nothing is clipped at the right edge, no element is wider than the
   viewport, no text overlaps other text, tap targets are not crammed together.
4. Desktop index: a sticky-style header with a wordmark, navigation links, and a theme
   control is visible at the top. A hero with a large title and a panel of six statistics
   is visible without scrolling past the first 900 px.
5. Desktop index: the experiments are presented as a full-width list of thirteen rows with
   a number, a title, a question, a finding paragraph, and a large headline metric on each
   row. Not thirteen identical uniform boxes in a card grid.
6. Desktop index: the pipeline is rendered as a designed figure (nodes in a row or column
   with labels), not a monospace text block.
7. Reference desktop page: a table of contents is visible alongside the article, and the
   article typography is a comfortable reading measure (roughly 60 to 80 characters per
   line), not full width.
8. Reference mobile page: the table of contents is collapsed or compact at the top, the
   article fills the width with sane margins, tables scroll rather than overflow.
9. Text contrast is comfortably readable in both schemes, including small grey labels.
10. Consistency: header, footer, type scale, and colours match between the index and the
    reference pages in the same scheme.

## Quality criteria (each miss = minor finding; three or more minors = FAIL)

- Vertical rhythm: consistent spacing between sections, no cramped or cavernous gaps.
- Hierarchy: at a glance you can tell title from question from finding from metric.
- Numbers are set in tabular figures and aligned in the statistics panel.
- The design looks finished and deliberate, like a senior engineer's site, not a template.
- Hover/focus states are not observable in screenshots; do not penalise their absence.

## Output

Reply with a short list of what you saw per screenshot, then the JSON object on the last
line exactly as instructed. PASS only when every hard criterion passes and there are fewer
than three minor findings.
