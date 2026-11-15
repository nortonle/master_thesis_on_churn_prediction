"""Convert main.tex (this paper only, not general LaTeX) to a Word document.

Usage:  python tex_to_docx.py            -> writes main.docx next to this script

The LaTeX source is the submission format; the Word file is a single-column
reading copy with the same text, tables, figures, numbering and reference list.
"""
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
DOLLAR, NBSP = "\uE000", "\u00A0"


def slurp(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*?\n[ \t]*", "", s)


def group(s, i):
    """s[i] == '{' -> (content, index after the matching '}')."""
    assert s[i] == "{", s[i:i + 40]
    depth, j = 0, i
    while True:
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


# ---------------------------------------------------------------- source ----
src = strip_comments(slurp("main.tex"))
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", slurp("results_macros.tex") + slurp("lit_macros.tex")))
src = re.sub(r"\\input\{(tab_\w+)\}", lambda m: strip_comments(slurp(m.group(1) + ".tex")).strip(), src)
for name in sorted(macros, key=len, reverse=True):
    src = re.sub(r"\\" + name + r"(?![A-Za-z])(\{\})?", lambda m, v=macros[name]: v, src)
src = src.replace("\\$", DOLLAR)

title, _ = group(src, src.index("\\title{") + 6)
author, _ = group(src, src.index("\\IEEEauthorblockN{") + 17)
affil, _ = group(src, src.index("\\IEEEauthorblockA{") + 17)
body = src[src.index("\\maketitle") + 10:src.index("\\bibliographystyle")]

# ------------------------------------------------- numbering and labels ----
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
labels, cite_no = {}, {}
sec = sub = tab = fig = 0
env = None
for m in re.finditer(r"\\(section|subsection|begin|end|label|cite)\{([^}]*)\}", body):
    cmd, arg = m.groups()
    if cmd == "section":
        sec, sub = sec + 1, 0
    elif cmd == "subsection":
        sub += 1
    elif cmd == "begin" and arg == "table":
        tab, env = tab + 1, "table"
    elif cmd == "begin" and arg.startswith("figure"):
        fig, env = fig + 1, "figure"
    elif cmd == "end" and arg in ("table", "figure", "figure*"):
        env = None
    elif cmd == "label":
        labels[arg] = (ROMAN[tab] if env == "table" else str(fig) if env == "figure"
                       else f"{ROMAN[sec]}-{chr(64 + sub)}" if sub else ROMAN[sec])
    elif cmd == "cite":
        for k in arg.split(","):
            cite_no.setdefault(k.strip(), len(cite_no) + 1)


def cite_text(keys):
    nums = sorted(cite_no[k.strip()] for k in keys.split(","))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"[{nums[i]}]" if j == i else f"[{nums[i]}], [{nums[j]}]" if j == i + 1
                   else f"[{nums[i]}]\u2013[{nums[j]}]")
        i = j + 1
    return ", ".join(out)


# -------------------------------------------------------- inline markup ----
SYMBOLS = {"rightarrow": "\u2192", "downarrow": "\u2193", "times": "\u00D7", "cdot": "\u00B7",
           "geq": "\u2265", "rho": "\u03C1", "mu": "\u03BC", "chi": "\u03C7", "lambda": "\u03BB",
           "dots": "\u2026", "dagger": "\u2020", "quad": "\u2003", "centering": "", "footnotesize": ""}
ACCENTS = {"{\\'e}": "\u00E9", '{\\"e}': "\u00EB", "{\\v{s}}": "\u0161"}


def plain(s):
    for a, b in (("``", "\u201C"), ("''", "\u201D"), ("---", "\u2014"), ("--", "\u2013"), ("~", NBSP),
                 ("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\,", "\u2009"), (DOLLAR, "$")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s)


def math_runs(s, fmt):
    """Very small math renderer: symbols, ^ and _ with a group or a single character."""
    s = s.replace("\\%", "%").replace("\\,", "\u2009").replace(" ", "")
    s = re.sub(r"\\([A-Za-z]+)", lambda m: SYMBOLS[m.group(1)], s).replace("-", "\u2212")
    out, i, buf = [], 0, ""
    while i < len(s):
        if s[i] in "^_":
            if buf:
                out.append((buf, fmt))
                buf = ""
            kind = "sup" if s[i] == "^" else "sub"
            if s[i + 1] == "{":
                inner, i = group(s, i + 1)
            else:
                inner, i = s[i + 1], i + 2
            out.append((inner, fmt | {kind}))
        else:
            buf += s[i]
            i += 1
    if buf:
        out.append((buf, fmt))
    return out


def inline(s, fmt=frozenset(), notes=None):
    """LaTeX fragment -> list of (text, format-set) runs."""
    out, i, buf = [], 0, ""

    def flush():
        nonlocal buf
        if buf:
            out.append((plain(buf), fmt))
            buf = ""

    while i < len(s):
        c = s[i]
        if c == "$":
            flush()
            j = s.index("$", i + 1)
            out.extend(math_runs(s[i + 1:j], fmt))
            i = j + 1
        elif c == "\\" and s[i + 1] == "\\":
            flush()
            out.append(("\n", fmt))
            i += 2
        elif c == "\\" and s[i + 1].isalpha():
            m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
            cmd, i = m.group(1), i + m.end()
            if cmd in ("emph", "textit", "textbf", "texttt"):
                flush()
                inner, i = group(s, i)
                add = {"emph": "i", "textit": "i", "textbf": "b", "texttt": "tt"}[cmd]
                out.extend(inline(inner, fmt | {add}, notes))
            elif cmd == "cite":
                keys, i = group(s, i)
                buf += cite_text(keys)
            elif cmd == "ref":
                key, i = group(s, i)
                buf += labels[key]
            elif cmd == "footnote":
                flush()
                inner, i = group(s, i)
                notes.append(inner)
                out.append((str(len(notes)), fmt | {"sup"}))
            elif cmd == "label":
                _, i = group(s, i)
            elif cmd in SYMBOLS:
                buf += SYMBOLS[cmd]
            else:
                raise ValueError(f"unhandled command \\{cmd} near: {s[max(0, i - 60):i + 40]!r}")
        elif c in "{}":
            i += 1
        else:
            buf += c
            i += 1
    flush()
    return out


def add_runs(par, runs, size=None):
    for text, fmt in runs:
        if text == "\n":
            par.add_run().add_break()
            continue
        r = par.add_run(text)
        r.bold, r.italic = "b" in fmt or None, "i" in fmt or None
        r.font.superscript, r.font.subscript = "sup" in fmt or None, "sub" in fmt or None
        if "tt" in fmt:
            r.font.name = "Courier New"
        if size:
            r.font.size = Pt(size)
    return par


# ------------------------------------------------------------ document ----
doc = Document()
s0 = doc.sections[0]
s0.page_width, s0.page_height = Inches(8.5), Inches(11)
s0.left_margin = s0.right_margin = Inches(1)
s0.top_margin = s0.bottom_margin = Inches(1)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.08


def para(text, align=None, size=None, fmt=frozenset(), before=0, after=None, keep=False):
    notes = []
    p = doc.add_paragraph()
    add_runs(p, inline(text.strip(), fmt, notes), size)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = keep
    for k, note in enumerate(notes, 1):
        q = doc.add_paragraph()
        add_runs(q, [(str(k), frozenset({"sup"}))] + inline(" Note: " + note.strip()), 9)
        q.paragraph_format.left_indent = Inches(0.3)
    return p


CENTER, JUSTIFY = WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY


def parse_rows(tabular):
    rows = []
    tabular = re.sub(r"\\(toprule|midrule|bottomrule)|\\cmidrule(\([^)]*\))?\{[^}]*\}", "", tabular)
    for line in tabular.split("\\\\"):
        if not line.strip():
            continue
        cells = []
        for cell in line.split("&"):
            cell = cell.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}", cell)
            if m:
                inner, _ = group(cell, m.end())
                cells.append((inner, int(m.group(1))))
            else:
                cells.append((cell, 1))
        rows.append(cells)
    return rows


def emit_table(block):
    global n_tab
    n_tab += 1
    cap, _ = group(block, block.index("\\caption{") + 8)
    para(f"TABLE {ROMAN[n_tab]}", CENTER, 9, before=10, after=0, keep=True)
    para(cap, CENTER, 9, after=4, keep=True)
    start = block.index("\\begin{tabular}")
    _, k = group(block, start + 15)
    rows = parse_rows(block[k:block.index("\\end{tabular}")])
    ncol = max(sum(span for _, span in r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
    n_head = 2 if rows[0][0][0] == "" or rows[1][0][0] == "" and n_tab in (1,) else 1
    for ri, r in enumerate(rows):
        ci = 0
        for text, span in r:
            cell = t.cell(ri, ci)
            if span > 1:
                cell = cell.merge(t.cell(ri, ci + span - 1))
            p = cell.paragraphs[0]
            head = ri < n_head or (ri == 1 and rows[0][0][0] == "")
            add_runs(p, inline(text, frozenset({"b"}) if head else frozenset()), 9)
            p.paragraph_format.space_after = Pt(0)
            if ci > 0 or span > 1:
                p.alignment = CENTER
            ci += span
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def emit_figure(block, wide):
    global n_fig
    n_fig += 1
    cap, _ = group(block, block.index("\\caption{") + 8)
    m = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", block)
    if m:
        p = doc.add_paragraph()
        p.alignment = CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(os.path.join(HERE, "figures", m.group(1) + ".png"),
                                width=Inches(6.5 if wide else 4.6))
    else:  # the pipeline diagram: a framed box of centred lines
        i = block.index("\\parbox{")
        _, i = group(block, i + 7)
        inner, _ = group(block, i)
        t = doc.add_table(rows=1, cols=1)
        t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
        p = t.cell(0, 0).paragraphs[0]
        add_runs(p, inline(inner), 9.5)
        p.alignment = CENTER
    para(f"Fig. {n_fig}. " + cap, CENTER, 9, before=4, after=10)


def heading(text, level):
    global n_sec, n_sub
    if level == 1:
        n_sec, n_sub = n_sec + 1, 0
        para(f"{ROMAN[n_sec]}. " + text.upper(), CENTER, 11, before=12, after=6, keep=True)
    else:
        n_sub += 1
        para(f"{chr(64 + n_sub)}. " + text, None, 11, frozenset({"i"}), before=8, after=4, keep=True)


n_sec = n_sub = n_tab = n_fig = 0

para(title, CENTER, 20, after=10)
para(author, CENTER, 12, after=0)
para(affil, CENTER, 10, after=14)

TOKEN = re.compile(
    r"\\begin\{(abstract|IEEEkeywords|table|figure\*?|enumerate)\}(.*?)\\end\{\1\}"
    r"|\\(section|subsection)\{", re.S)
pos = 0


def emit_text(chunk):
    for p in re.split(r"\n\s*\n", chunk):
        p = re.sub(r"^\s*\\label\{[^}]*\}", "", p.strip())
        if p.strip():
            para(p, JUSTIFY)


while True:
    m = TOKEN.search(body, pos)
    if not m:
        emit_text(body[pos:])
        break
    emit_text(body[pos:m.start()])
    pos = m.end()
    kind, inner, sect = m.group(1), m.group(2), m.group(3)
    if sect:
        text, pos = group(body, pos - 1)
        heading(text, 1 if sect == "section" else 2)
    elif kind == "abstract":
        p = para("\\textbf{\\emph{Abstract}}\u2014" + inner.strip(), JUSTIFY, fmt=frozenset({"b"}))
    elif kind == "IEEEkeywords":
        para("\\emph{Index Terms}\u2014" + inner.strip(), JUSTIFY, fmt=frozenset({"b"}), after=10)
    elif kind == "table":
        emit_table(inner)
    elif kind.startswith("figure"):
        emit_figure(inner, kind.endswith("*"))
    elif kind == "enumerate":
        for k, item in enumerate(inner.split("\\item")[1:], 1):
            p = para(f"{k}) " + item, JUSTIFY)
            p.paragraph_format.left_indent = Inches(0.3)

# ---------------------------------------------------------- references ----
bib = slurp("references.bib")
for a, b in ACCENTS.items():
    bib = bib.replace(a, b)
entries = {}
for m in re.finditer(r"@(\w+)\{([^,]+),", bib):
    j, depth = m.end(), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(bib[j], 0)
        j += 1
    fields = {}
    chunk = bib[m.end():j - 1]
    for f in re.finditer(r"(\w+)\s*=\s*\{", chunk):
        val, _ = group(chunk, f.end() - 1)
        fields[f.group(1).lower()] = re.sub(r"\s+", " ", val.replace("{", "").replace("}", "").replace("\\&", "&"))
    entries[m.group(2).strip()] = (m.group(1).lower(), fields)


def names(a):
    out = []
    for person in a.split(" and "):
        if person.strip().lower() == "others":
            out.append("et al.")
            continue
        last, _, first = person.partition(",")
        initials = " ".join("-".join(p[0] + "." for p in w.split("-") if p) for w in first.split())
        out.append(f"{initials} {last.strip()}".strip())
    if len(out) <= 2:
        return " and ".join(out)
    return ", ".join(out[:-1]) + (", " if out[-1] == "et al." else ", and ") + out[-1]


heading_p = para("REFERENCES", CENTER, 11, before=12, after=6, keep=True)
for key, num in sorted(cite_no.items(), key=lambda kv: kv[1]):
    kind, f = entries[key]
    pages = f.get("pages", "").replace("--", "\u2013")
    pp = ("pp. " if "\u2013" in pages else "p. ") + pages if pages else ""
    runs = [(f"[{num}] {names(f['author'])}, \u201C{f['title']},\u201D ", frozenset())]
    if kind == "article":
        runs.append((f["journal"], frozenset({"i"})))
        tail = [f"vol. {f['volume']}" if "volume" in f else "", f"no. {f['number']}" if "number" in f else "",
                pp, f["year"]]
    else:
        runs += [("in ", frozenset()), (f["booktitle"], frozenset({"i"}))]
        tail = [f"vol. {f['volume']}" if "volume" in f else "", f["year"], pp]
    if "doi" in f:
        tail.append(f"doi: {f['doi']}")
    runs.append((", " + ", ".join(x for x in tail if x) + ".", frozenset()))
    p = doc.add_paragraph()
    add_runs(p, runs, 9)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.35), Inches(-0.35)
    p.paragraph_format.space_after = Pt(2)

out = os.path.join(HERE, "main.docx")
doc.save(out)
print(f"wrote {out} | sections {n_sec} | tables {n_tab} | figures {n_fig} | references {len(cite_no)}")
