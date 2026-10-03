--[[
tools/report_pdf/report.lua: the pandoc Lua filter behind tools/build_report_pdf.py.

It turns docs/report/draft.md, read as GitHub Markdown, into the body and metadata of
tools/report_pdf/template.tex. It changes layout only, never a word or a number:

* Title block. The level-1 heading is the title. Before "## Abstract", a paragraph
  starting "Author:", "Affiliation:" or "Code:" fills that field, and one starting
  "Keywords:" fills the PDF's keywords metadata only (it is not printed); the first
  other paragraph is the subtitle and any later ones are a note under the author block.
  "## Abstract" and its paragraphs become the abstract. Horizontal rules are dropped.
* Headings keep the draft's own numbers ("5.4", "Appendix D", "D.1"), which the text
  cites as "§5.4" and "App D.1", and get stable ids: sec-5, sec-5.4, app-D, app-D.1.
  The appendices start on a new page with a short list of them.
* Links. "§5.4", "App D.1" (and "D.4" in "App C.1, D.4"), "Appendix D", "Figure 3",
  "Table 1" and author-year citations such as "[Brier 1950]" link to their targets;
  "doi:" and "arXiv:" identifiers link to doi.org and arxiv.org. A reference whose
  target does not exist stays plain text and is reported.
* Figures. A paragraph holding only fig/<name>.png, followed by a paragraph that
  starts with the emphasised label "Figure N.", becomes one float: the file the
  build prepared (vector PDF or the 200-dpi PNG) at its natural width, and the
  draft's caption.
* Tables get column widths from their content (the CSS automatic layout: each
  column's longest unbreakable piece first, the rest of the line shared in
  proportion to how much more each column needs) and a font that steps down with
  the column count; a "Source:"-style paragraph right after a table is set small.
* Inline code breaks after / _ . , = : - | and inside long hashes; a hyphen before
  a digit at the start of a word (a negative number) is set as a minus sign.
* TeX math ($...$, $$...$$) passes through untouched: the link pass, the minus
  sign and the break points act on text only, and pandoc writes the math as \(...\)
  and \[...\] for the template's unicode-math. A display that is one amsmath
  environment (align, gather, equation, multline, flalign, alignat), which LaTeX
  refuses inside \[...\], is written as that environment, starred: the draft's
  displays are unnumbered, as GitHub shows them. A table sizes a math cell by its
  printed glyphs. Math in the title, the subtitle or a heading is reported, because
  the PDF's bookmarks and metadata would show its TeX source, and so is a "$" left
  as text (math that did not parse), unless it starts a price such as "$5".

Configuration: the JSON file named by the environment variable PAIEC_REPORT_CONFIG
(written by the build; read as JSON, never as Markdown): "figures" (name -> file and
natural width), "log" (where to write a JSON report of ids, links, unresolved
references and warnings) and, for --captions full, "captions" (figure label ->
caption in Markdown). Without it, images are used as the draft names them.
]]

local stringify = pandoc.utils.stringify
local List = pandoc.List

local LOG = {
  warnings = {},
  unresolved = {},
  links = {section = 0, appendix = 0, figure = 0, table = 0, citation = 0, external = 0},
  figures = {},
  tables = {},
  headings = {},
  references = {},
  front_matter = {},
}

local function warn(msg)
  table.insert(LOG.warnings, msg)
  io.stderr:write('[report.lua] ' .. msg .. '\n')
end

local function raw(s) return pandoc.RawInline('latex', s) end
local function rawblock(s) return pandoc.RawBlock('latex', s) end

local function ulen(s) return utf8.len(s) or #s end

-- The first n characters of s. string.sub counts bytes, and a cut inside a UTF-8
-- sequence leaves invalid text in the build's JSON report.
local function usub(s, n)
  local e = utf8.offset(s, n + 1)
  return e and s:sub(1, e - 1) or s
end

local function trim(s) return (s:gsub('^%s+', ''):gsub('%s+$', '')) end

local ASCII_FOLD = {
  ['á'] = 'a', ['à'] = 'a', ['â'] = 'a', ['ä'] = 'a', ['ã'] = 'a', ['é'] = 'e', ['è'] = 'e',
  ['ê'] = 'e', ['ë'] = 'e', ['í'] = 'i', ['ì'] = 'i', ['î'] = 'i', ['ï'] = 'i', ['ó'] = 'o',
  ['ò'] = 'o', ['ô'] = 'o', ['ö'] = 'o', ['õ'] = 'o', ['ú'] = 'u', ['ù'] = 'u', ['û'] = 'u',
  ['ü'] = 'u', ['ñ'] = 'n', ['ç'] = 'c', ['ø'] = 'o', ['å'] = 'a', ['ß'] = 'ss',
}

-- An ASCII id from a heading or a surname ("Martínez-Plumed" -> "martinez-plumed").
local function slug(s)
  s = s:gsub(utf8.charpattern, function(ch) return ASCII_FOLD[ch:lower()] or ASCII_FOLD[ch] or ch end)
  s = s:gsub('[\128-\255]+', ''):lower()
  s = s:gsub('[^%w]+', '-'):gsub('^%-+', ''):gsub('%-+$', '')
  return s
end

-- Ids known before the inline pass; links are only made to these.
local IDS = {}

local function unresolved(kind, text)
  table.insert(LOG.unresolved, {kind = kind, text = text})
end

------------------------------------------------------------------------------
-- Inline code and long words
------------------------------------------------------------------------------

local TT_ESC = {
  ['\\'] = '\\textbackslash{}', ['{'] = '\\{', ['}'] = '\\}', ['$'] = '\\$',
  ['&'] = '\\&', ['#'] = '\\#', ['^'] = '\\textasciicircum{}', ['_'] = '\\_',
  ['%'] = '\\%', ['~'] = '\\textasciitilde{}', [' '] = '\\ ',
}
local CODE_BREAK = {
  ['/'] = '\\allowbreak{}', ['_'] = '\\allowbreak{}', ['.'] = '\\codebreak{}',
  [','] = '\\codebreak{}', ['='] = '\\codebreak{}', [':'] = '\\codebreak{}',
  ['-'] = '\\codebreak{}', ['|'] = '\\codebreak{}',
}
local CODE_CHUNK = 10  -- a run longer than CODE_LONG (a hash) breaks every this many characters
local CODE_LONG = 16   -- ... so "experiments" or "thresholds" never breaks inside the word

local function is_code_break(ch) return ch == nil or ch == ' ' or CODE_BREAK[ch] ~= nil end

local function code_latex(text)
  local out = {'\\texttt{'}
  local cps = {}
  for _, cp in utf8.codes(text) do cps[#cps + 1] = utf8.char(cp) end
  -- break points inside long runs, never in a run's last four characters
  local chunk = {}
  local i = 1
  while i <= #cps do
    if is_code_break(cps[i]) then
      i = i + 1
    else
      local j = i
      while j < #cps and not is_code_break(cps[j + 1]) do j = j + 1 end
      if j - i + 1 > CODE_LONG then
        for p = i + CODE_CHUNK - 1, j - 4, CODE_CHUNK do chunk[p] = true end
      end
      i = j + 1
    end
  end
  for k, ch in ipairs(cps) do
    out[#out + 1] = TT_ESC[ch] or ch
    if k < #cps then
      local nxt, prev = cps[k + 1], cps[k - 1]
      if CODE_BREAK[ch] then
        -- not before another break character or a space, nor inside a leading "--"
        local option_dash = ch == '-' and (prev == nil or prev == ' ' or prev == '-')
        if not is_code_break(nxt) and not option_dash then out[#out + 1] = CODE_BREAK[ch] end
      elseif chunk[k] then
        out[#out + 1] = '\\codebreak{}'
      end
    end
  end
  out[#out + 1] = '}'
  return table.concat(out)
end

-- The longest piece of inline code no line can break, in characters (see code_latex).
local function code_min_chars(text)
  local longest = 0
  for seg in text:gmatch('[^/_%.,=:|%- ]+') do
    local l = ulen(seg)
    if l > CODE_LONG then l = CODE_CHUNK + 3 end
    longest = math.max(longest, l)
  end
  -- the break character stays on the line, and a leading "--" stays with its word
  return longest + (text:find('%-%-') and 3 or 1)
end

-- A negative number: a hyphen at the start of a word, or after ( [ /, before a digit.
local MINUS = '\u{2212}'
local function fix_minus(s)
  s = s:gsub('^%-(%.?%d)', MINUS .. '%1')
  s = s:gsub('([%(%[/])%-(%.?%d)', '%1' .. MINUS .. '%2')
  return s
end

-- Long words joined by _ or / get break points after those characters.
local function str_pieces(s)
  s = fix_minus(s)
  if ulen(s) < 12 or not s:find('[_/]') then return {pandoc.Str(s)} end
  local out = {}
  local buf = ''
  local n = #s
  for i = 1, n do
    local ch = s:sub(i, i)
    buf = buf .. ch
    if (ch == '_' or ch == '/') and i < n then
      out[#out + 1] = pandoc.Str(buf)
      out[#out + 1] = raw('\\allowbreak{}')
      buf = ''
    end
  end
  if buf ~= '' then out[#out + 1] = pandoc.Str(buf) end
  return out
end

------------------------------------------------------------------------------
-- References list: keys for author-year citations
------------------------------------------------------------------------------

local function split_plain(s, sep)
  local out = {}
  local start = 1
  while true do
    local a, b = s:find(sep, start, true)
    if not a then out[#out + 1] = s:sub(start); break end
    out[#out + 1] = s:sub(start, a - 1)
    start = b + 1
  end
  return out
end

local function words(s)
  local out = {}
  for w in s:gmatch('%S+') do out[#out + 1] = w end
  return out
end

-- "Maia Polo, F., Weber, L. and Yurochkin, M. (2024). ..." -> surnames, year
local function parse_reference(text)
  local authors, year = text:match('^(.-)%s*%((%d%d%d%d)%)')
  if not authors then return nil end
  authors = authors:gsub('%(eds?%.%)', '')
  local parts = {}
  for _, chunk in ipairs(split_plain(authors, ' and ')) do
    for _, p in ipairs(split_plain(chunk, ', ')) do
      p = trim(p):gsub(',$', '')
      if p ~= '' and p ~= 'et al.' then parts[#parts + 1] = p end
    end
  end
  local names = {}
  for _, p in ipairs(parts) do
    -- initials ("M. A.", "J.-P.") are ASCII capitals, dots, hyphens and spaces only
    if not p:match('^[%u%.%-%s]+$') then names[#names + 1] = p end
  end
  if #names == 0 then return nil end
  return {names = names, year = year}
end

local REFS = {}          -- list of {id, first = {words}, year, connectors = set}
local REF_BY_ID = {}

local function register_reference(item_text)
  local r = parse_reference(item_text)
  if not r then
    warn('reference not parsed: ' .. usub(item_text, 80))
    return nil
  end
  local id = 'ref-' .. slug(r.names[1]) .. '-' .. r.year
  local k = 2
  while REF_BY_ID[id] do id = 'ref-' .. slug(r.names[1]) .. '-' .. r.year .. '-' .. k; k = k + 1 end
  local connectors = {['and'] = true, ['et'] = true, ['al.'] = true}
  for i = 2, #r.names do
    for _, w in ipairs(words(r.names[i])) do connectors[w] = true end
  end
  local ref = {id = id, first = words(r.names[1]), year = r.year, connectors = connectors,
               names = r.names}
  REFS[#REFS + 1] = ref
  REF_BY_ID[id] = ref
  IDS[id] = true
  table.insert(LOG.references, {id = id, names = r.names, year = r.year})
  return id
end

------------------------------------------------------------------------------
-- The link pass over one list of inlines
------------------------------------------------------------------------------

local function is_space(el) return el and (el.t == 'Space' or el.t == 'SoftBreak') end
local function str_at(ils, i)
  local el = ils[i]
  if el and el.t == 'Str' then return el.text end
  return nil
end

local function link(content, id, kind)
  LOG.links[kind] = LOG.links[kind] + 1
  return pandoc.Link(content, '#' .. id)
end

-- An author-year citation starting at token i: returns prefix, link, suffix, last index.
local function match_citation(ils, i)
  local t = str_at(ils, i)
  if not t then return nil end
  local prefix, rest = t:match('^([%[%(]*)(.*)$')
  for _, ref in ipairs(REFS) do
    local first = ref.first
    local w1 = rest:gsub(',$', '')
    if w1 == first[1] then
      local j = i
      local ok = true
      for m = 2, #first do
        local tm = (is_space(ils[j + 1]) and str_at(ils, j + 2)) or nil
        if not tm or tm:gsub(',$', '') ~= first[m] then ok = false; break end
        j = j + 2
      end
      if ok then
        -- connectors, then the year
        local steps = 0
        while steps < 8 do
          local tn = is_space(ils[j + 1]) and str_at(ils, j + 2) or nil
          if not tn then break end
          local yr, suffix = tn:match('^(%d%d%d%d)(.*)$')
          if yr then
            if yr == ref.year and suffix:match('^[%];,%).:]*$') then
              local content = List()
              content:insert(pandoc.Str(rest))
              for q = i + 1, j + 1 do content:insert(ils[q]) end
              content:insert(pandoc.Str(yr))
              return prefix, link(content, ref.id, 'citation'), suffix, j + 2
            end
            break
          end
          local bare = tn:gsub(',$', '')
          if not ref.connectors[bare] then break end
          j = j + 2
          steps = steps + 1
        end
      end
    end
  end
  return nil
end

-- "App D.1" or "Appendix D" at token i (the word) and i+2 (the label).
local function parse_app_label(t)
  local L, rest = t:match('^([A-I])(.*)$')
  if not L then return nil end
  local num, rest2 = rest:match('^%.(%d+)(.*)$')
  local id = 'app-' .. L
  local label = L
  if num then id = id .. '.' .. num; label = L .. '.' .. num; rest = rest2 end
  if rest:match('^[%w]') then return nil end
  return id, label, rest
end

local function link_pass(ils)
  local out = List()
  local i = 1
  local n = #ils
  while i <= n do
    local el = ils[i]
    local done = false
    if el.t == 'Str' then
      local t = el.text
      -- author-year citation
      local prefix, lk, suffix, last = match_citation(ils, i)
      if prefix then
        if prefix ~= '' then out:insert(pandoc.Str(prefix)) end
        out:insert(lk)
        if suffix ~= '' then out:extend(str_pieces(suffix)) end
        i = last + 1
        done = true
      end
      -- App X.N / Appendix X, then ", Y.M" continuations
      if not done then
        local lead, word = t:match('^([%[%(]*)(%a+)$')
        local nxt = is_space(ils[i + 1]) and str_at(ils, i + 2) or nil
        if word and nxt and (word == 'App' or word == 'Appendix') then
          local id, label, rest = parse_app_label(nxt)
          if id then
            if IDS[id] then
              if lead ~= '' then out:insert(pandoc.Str(lead)) end
              out:insert(link({pandoc.Str(word), pandoc.Space(), pandoc.Str(label)}, id, 'appendix'))
              i = i + 3
              -- continuations: "App C.1, D.4"
              while rest == ',' and is_space(ils[i]) and str_at(ils, i + 1) do
                local id2, label2, rest2 = parse_app_label(str_at(ils, i + 1))
                if not id2 or not label2:find('%.') then break end
                out:insert(pandoc.Str(','))
                out:insert(ils[i])
                if IDS[id2] then
                  out:insert(link({pandoc.Str(label2)}, id2, 'appendix'))
                else
                  unresolved('appendix', 'App ' .. label .. ', ' .. label2)
                  out:insert(pandoc.Str(label2))
                end
                rest = rest2
                i = i + 2
              end
              if rest ~= '' then out:extend(str_pieces(rest)) end
              done = true
            else
              unresolved('appendix', word .. ' ' .. label)
            end
          end
        elseif word and nxt and (word == 'Figure' or word == 'Figures' or word == 'Table') then
          local lab, rest = nxt:match('^(%u?%d+)(.*)$')
          if lab and not rest:match('^[%w]') then
            local id = (word == 'Table' and 'tab-' or 'fig-') .. lab
            if IDS[id] then
              if lead ~= '' then out:insert(pandoc.Str(lead)) end
              out:insert(link({pandoc.Str(word), pandoc.Space(), pandoc.Str(lab)}, id,
                              word == 'Table' and 'table' or 'figure'))
              if rest ~= '' then out:extend(str_pieces(rest)) end
              i = i + 3
              done = true
            else
              unresolved(word == 'Table' and 'table' or 'figure', word .. ' ' .. lab)
            end
          end
        end
      end
      -- §N.M inside a word
      if not done and t:find('§%d') then
        local pos = 1
        while true do
          local a, b, num = t:find('§(%d+[%.%d]*)', pos)
          if not a then break end
          num = num:gsub('%.+$', '')
          b = a + #'§' + #num - 1
          if a > pos then out:extend(str_pieces(t:sub(pos, a - 1))) end
          local id = 'sec-' .. num
          local text = t:sub(a, b)
          if IDS[id] then
            out:insert(link({pandoc.Str(text)}, id, 'section'))
          else
            -- §3.2.2 of a cited paper is not ours; report only two-level numbers
            if not num:find('%d%.%d+%.%d') then unresolved('section', text) end
            out:insert(pandoc.Str(text))
          end
          pos = b + 1
        end
        if pos <= #t then out:extend(str_pieces(t:sub(pos))) end
        i = i + 1
        done = true
      end
      -- doi: and arXiv: identifiers
      if not done then
        local pre, kind, ident, post = t:match('^(.-)(doi:)(10%.%S+)$')
        if not kind then pre, kind, ident, post = t:match('^(.-)(arXiv:)(%d%d%d%d%.%d+)(.*)$') end
        if kind then
          post = post or ''
          if kind == 'doi:' then
            local strip = ident:match('[%.,;]$')
            if strip and not ident:match('CO;2$') then
              post = strip .. post
              ident = ident:sub(1, -2)
            end
          end
          local url
          if kind == 'doi:' then
            url = 'https://doi.org/' .. ident:gsub('<', '%%3C'):gsub('>', '%%3E')
          else
            url = 'https://arxiv.org/abs/' .. ident
          end
          if pre ~= '' then out:insert(pandoc.Str(pre)) end
          out:insert(pandoc.Link({pandoc.Str(kind .. ident)}, url))
          LOG.links.external = LOG.links.external + 1
          if post ~= '' then out:insert(pandoc.Str(post)) end
          i = i + 1
          done = true
        end
      end
      if not done then
        out:extend(str_pieces(t))
        i = i + 1
      end
    else
      out:insert(el)
      i = i + 1
    end
  end
  return out
end

-- amsmath's display environments, which LaTeX refuses inside the \[...\] pandoc writes
-- for display math. A display that is exactly one of them is written as the
-- environment itself, starred (unnumbered, as \[...\] and GitHub's displays are).
local DISPLAY_ENVS = {equation = true, align = true, gather = true, multline = true,
                      flalign = true, alignat = true}

local function display_env(m)
  if m.mathtype ~= 'DisplayMath' then return nil end
  local env, inner = m.text:match('^%s*\\begin{(%a+)%*?}(.*)$')
  if not (env and DISPLAY_ENVS[env]) then return nil end
  local body = inner:match('^(.*)\\end{' .. env .. '%*?}%s*$')
  if not body or body:find('\\end{' .. env .. '%*?}') then return nil end
  return raw('\\begin{' .. env .. '*}' .. body .. '\\end{' .. env .. '*}')
end

-- The inline treatment, top-down: links in every list of inlines (not inside links,
-- headings or the table-caption divs), code to breakable LaTeX, amsmath displays as
-- their environments. Math is otherwise left as it is.
local INLINE_FILTER = {
  traverse = 'topdown',
  Header = function(h) return h, false end,
  Link = function(l) return l, false end,
  Image = function(im) return im, false end,
  Div = function(d)
    if d.classes:includes('report-noref') then return d, false end
    return d
  end,
  Code = function(c) return raw(code_latex(c.text)) end,
  Math = function(m) return display_env(m) or m end,
  Inlines = function(ils) return link_pass(ils) end,
}

local function treat_inlines(ils)
  return pandoc.Inlines(ils):walk(INLINE_FILTER)
end

local function treat_blocks(blks)
  return pandoc.Blocks(blks):walk(INLINE_FILTER)
end

local function inlines_latex(ils)
  local s = pandoc.write(pandoc.Pandoc({pandoc.Plain(ils)}), 'latex')
  return (s:gsub('%s+$', ''))
end

local function blocks_latex(blks)
  local s = pandoc.write(pandoc.Pandoc(blks), 'latex')
  return (s:gsub('%s+$', ''))
end

------------------------------------------------------------------------------
-- Tables
------------------------------------------------------------------------------

local TEXTWIDTH_PT = 469.75   -- 6.5in, the template's text width
local TABCOLSEP_PT = 4
local CODE_EM = 0.50          -- advance of a code character (Menlo scaled to the text's x-height)
local BOLD = 1.06             -- bold against regular
local SLACK_EM = 0.4          -- room left beside a cell's longest unbreakable piece
local KEEP_LINES = 14         -- a table of at most this many lines is not split across pages;
                              -- a longer one may break between rows (header repeated), so a
                              -- table that misses the page by a few rows does not leave a
                              -- third of it blank
local SIZES = {
  {name = 'small', pt = 10},
  {name = 'footnotesize', pt = 9},
  {name = 'scriptsize', pt = 8},
}

-- Rough advances of a Times-like face, in em: enough to share a line between columns.
local function char_em(ch)
  if ch == ' ' then return 0.25 end
  if ch:match('^%d$') then return 0.5 end
  if ch:match('^%l$') then return ch:match('[mw]') and 0.75 or (ch:match('[ijlt f]') and 0.3 or 0.47) end
  if ch:match('^%u$') then return ch:match('[MW]') and 0.9 or 0.68 end
  if ch:match('^[%.,;:\'`!|]$') then return 0.27 end
  if ch:match('^[%(%)%[%]/%-]$') then return 0.33 end
  if ch == '_' or ch == '%' then return 0.6 end
  return 0.55   -- other ASCII and non-ASCII: ± § − Δ σ …
end

local function str_em(s, per)
  local w = 0
  for _, cp in utf8.codes(s) do
    w = w + (per or char_em(utf8.char(cp)))
  end
  return w
end

-- Inline math, roughly as printed: a command such as \mu is one glyph, the names of
-- \text, \mathrm, \operatorname and the like print nothing but their argument, sub-
-- and superscripts are set at 0.7, and a top-level relation or binary operator gets
-- its spacing. Returns the width and the longest piece no line can break: TeX breaks
-- inline math only after a top-level relation or binary operator (here = < > + and ,).
local MATH_GLYPH = '\u{3B1}'   -- a stand-in glyph; char_em gives it 0.55 em
local function math_measure(tex)
  local s = tex:gsub('\\operatorname%*?', ''):gsub('\\text%a*', ''):gsub('\\math%a+', '')
  s = s:gsub('\\left%f[^%a]', ''):gsub('\\right%f[^%a]', '')
  s = s:gsub('\\begin%b{}', ''):gsub('\\end%b{}', '')
  s = s:gsub('\\%a+', MATH_GLYPH):gsub('\\[^%a]', ' ')
  local len, best, piece = 0, 0, 0
  local stack, scripted, pending, level = {}, 0, false, 0
  for _, cp in utf8.codes(s) do
    local ch = utf8.char(cp)
    if ch == '^' or ch == '_' then
      pending = true
    elseif ch == '{' then
      stack[#stack + 1] = pending
      if pending then scripted = scripted + 1 end
      pending = false
      level = level + 1
    elseif ch == '}' then
      if table.remove(stack) then scripted = scripted - 1 end
      level = math.max(0, level - 1)
    elseif not ch:match('^%s$') then
      local w = char_em(ch) * ((scripted > 0 or pending) and 0.7 or 1)
      pending = false
      if level == 0 and ch:match('^[=<>+,%-]$') then w = w + 0.4 end
      len = len + w
      piece = piece + w
      if level == 0 and ch:match('^[=<>+,]$') then
        best = math.max(best, piece)
        piece = 0
      end
    end
  end
  return len, math.max(best, piece)
end

-- acc.len: the cell's width on one line; acc.min: its longest piece no line can break
local function measure(ils, acc, factor)
  factor = factor or 1
  for _, el in ipairs(ils) do
    if el.t == 'Str' then
      local t = el.text
      acc.len = acc.len + str_em(t) * factor
      -- pieces between break points: spaces (outside), hyphens that are not minus signs,
      -- and the _ and / that str_pieces makes breakable in long words
      local long = ulen(t) >= 12
      local pieces = {}
      local buf = ''
      for _, cp in utf8.codes(t) do
        local ch = utf8.char(cp)
        buf = buf .. ch
        if (ch == '-' and #buf > 1) or (long and (ch == '_' or ch == '/')) then
          pieces[#pieces + 1] = buf
          buf = ''
        end
      end
      if buf ~= '' then pieces[#pieces + 1] = buf end
      for _, seg in ipairs(pieces) do
        local w = str_em(seg)
        -- TeX can hyphenate a long plain word
        if ulen(seg) > 10 and seg:match('^%a+[%p]*$') then w = w * 0.6 end
        acc.min = math.max(acc.min, w * factor)
      end
    elseif el.t == 'Space' or el.t == 'SoftBreak' then
      acc.len = acc.len + 0.25 * factor
    elseif el.t == 'Code' then
      acc.len = acc.len + str_em(el.text, CODE_EM) * factor
      acc.min = math.max(acc.min, code_min_chars(el.text) * CODE_EM * factor)
    elseif el.t == 'Math' then
      local len, min = math_measure(el.text)
      acc.len = acc.len + len * factor
      acc.min = math.max(acc.min, min * factor)
    elseif el.t == 'Quoted' then
      acc.len = acc.len + 0.8 * factor
      measure(el.content, acc, factor)
    elseif el.content then
      measure(el.content, acc, el.t == 'Strong' and factor * BOLD or factor)
    end
  end
end

local function measure_cell(cell)
  local acc = {len = 0, min = 0}
  for _, b in ipairs(cell.contents) do
    if b.content and (b.t == 'Plain' or b.t == 'Para') then measure(b.content, acc) end
  end
  acc.min = acc.min + SLACK_EM
  acc.len = acc.len + SLACK_EM
  return acc
end

-- Pandoc's longtable for a captionless table, as a tabular in its own paragraph: one
-- box, which TeX moves to the next page whole instead of splitting it. Returns nil
-- (keep the longtable) if the LaTeX is not in the expected shape.
local function longtable_to_tabular(tex)
  local a, b = tex:find('\\begin{longtable}[]{', 1, true)
  if not a then return nil end
  local depth, i = 1, b + 1
  while i <= #tex and depth > 0 do
    local c = tex:sub(i, i)
    if c == '{' then depth = depth + 1 elseif c == '}' then depth = depth - 1 end
    i = i + 1
  end
  if depth ~= 0 then return nil end
  local spec = tex:sub(b + 1, i - 2)
  local rest = tex:sub(i)
  local e = rest:find('\\end{longtable}', 1, true)
  if not e then return nil end
  local inner = rest:sub(1, e - 1)
  if inner:find('\\endfirsthead', 1, true) or inner:find('\\caption', 1, true) then return nil end
  local n1, n2
  inner, n1 = inner:gsub('\n\\endhead\n', '\n', 1)
  inner, n2 = inner:gsub('\n\\bottomrule\\noalign{}\n\\endlastfoot\n', '\n', 1)
  if n1 ~= 1 or n2 ~= 1 then return nil end
  inner = inner:gsub('%s+$', '')
  return '\\par\\addvspace{4pt}{\\centering\\begin{tabular}{' .. spec .. '}' .. inner ..
    '\n\\bottomrule\\noalign{}\n\\end{tabular}\\par}\\addvspace{4pt}'
end

local function layout_table(tbl, where)
  local ncol = #tbl.colspecs
  local body_max, head_len, minw = {}, {}, {}
  for j = 1, ncol do body_max[j] = 0; head_len[j] = 0; minw[j] = 1 end
  local row_lens = {}   -- every row's cell widths on one line, header first
  local function visit(rows, is_head)
    for _, row in ipairs(rows) do
      local j = 1
      local lens = {}
      for _, cell in ipairs(row.cells) do
        local m = measure_cell(cell)
        if is_head then head_len[j] = math.max(head_len[j], m.len)
        else body_max[j] = math.max(body_max[j], m.len) end
        minw[j] = math.max(minw[j], m.min)
        lens[j] = m.len
        j = j + (cell.col_span or 1)
      end
      row_lens[#row_lens + 1] = lens
    end
  end
  visit(tbl.head.rows, true)
  for _, body in ipairs(tbl.bodies) do visit(body.body, false) end
  -- widths in em: a header may take two lines, a body cell wants one
  local maxw = {}
  local smin, smax, snat = 0, 0, 0
  for j = 1, ncol do
    maxw[j] = math.max(body_max[j], head_len[j] / 2, minw[j])
    smin = smin + minw[j]
    smax = smax + maxw[j]
    snat = snat + math.max(body_max[j], head_len[j], minw[j])
  end
  local idx = ncol <= 4 and 1 or (ncol <= 8 and 2 or 3)
  local function avail(k) return (TEXTWIDTH_PT - (ncol - 1) * 2 * TABCOLSEP_PT) / SIZES[k].pt end
  while smin > avail(idx) and idx < #SIZES do idx = idx + 1 end
  local A = avail(idx)
  local info = {where = where, columns = ncol, size = SIZES[idx].name}
  local colw = {}       -- each column's width in em, for the line estimate
  if snat <= 0.9 * A then
    -- every cell, header included, fits on one line: natural column widths
    info.layout = 'natural'
  else
    local w = {}
    if smin >= A then
      for j = 1, ncol do w[j] = minw[j] end
      info.layout = 'minimum'
      warn(string.format('table %s: its unbreakable pieces need %.0f of %.0f em', where, smin, A))
    else
      local share = (A - smin) / (smax - smin)
      for j = 1, ncol do w[j] = minw[j] + (maxw[j] - minw[j]) * share end
      info.layout = 'wrapped'
    end
    local total = 0
    for j = 1, ncol do total = total + w[j] end
    local fr = {}
    for j = 1, ncol do
      local f = math.floor(w[j] / total * 10000) / 10000   -- floor: the sum stays within the line
      tbl.colspecs[j] = {tbl.colspecs[j][1], f}
      fr[#fr + 1] = string.format('%.3f', f)
      colw[j] = f * A
    end
    info.widths = table.concat(fr, ' ')
  end
  -- lines the table takes, roughly: a short table is kept on one page
  local lines = 0
  for _, lens in ipairs(row_lens) do
    local l = 1
    for j, len in pairs(lens) do
      if colw[j] then l = math.max(l, math.ceil(len / math.max(colw[j], 1))) end
    end
    lines = lines + l
  end
  info.lines = lines
  table.insert(LOG.tables, info)
  return SIZES[idx].name, lines
end

------------------------------------------------------------------------------
-- Structure
------------------------------------------------------------------------------

local function para_text(b)
  if b and (b.t == 'Para' or b.t == 'Plain') then return stringify(b) end
  return nil
end

local function has_math(ils)
  local found = false
  pandoc.Inlines(ils):walk({Math = function(m) found = true; return m end})
  return found
end

-- A "$" the reader left as text is math that did not parse ("$ x$", "$y $", "$0.1 $."):
-- an opening "$" before a space, a closing one after a space or before a digit. Only a
-- price, a "$" that starts a word and is followed by a digit ("$5", "$20,000"), is not
-- reported. Code is not text.
local function stray_dollar(t)
  local i = t:find('$', 1, true)
  while i do
    if i > 1 or not t:sub(i + 1, i + 1):match('^%d$') then return true end
    i = t:find('$', i + 1, true)
  end
  return false
end

local function report_stray_dollars(blocks)
  local seen = {}
  blocks:walk({Str = function(s)
    local t = s.text
    if stray_dollar(t) and not seen[t] then
      seen[t] = true
      warn('"' .. usub(t, 40) .. '" holds a "$" left as text: TeX math that did not parse?')
    end
  end})
end

-- Before a table, ask for enough space that its "Table N:" caption or a short lead-in
-- paragraph, and the headings right above them, are not left at the foot of a page: room for
-- the whole table if it is kept together (kept_lines, its estimated lines), else for its
-- first rows. A lead-in longer than LEAD_MAX characters may break across pages, so the
-- space is asked for after it instead.
local PAGE_LINES = 46        -- lines of body text on a page
local LEAD_CHARS = 95        -- characters on a line of body text
local LEAD_MAX = 400

local function is_heading_block(b)
  if not b then return false end
  if b.t == 'Header' then return true end
  return b.t == 'RawBlock' and (b.text:match('^\\section{') or b.text:match('^\\subsection{')
    or b.text:match('^\\subsubsection{')) ~= nil
end

local function keep_with_table(out, kept_lines)
  local need = kept_lines and (kept_lines + 2) or 6
  local pos = #out + 1
  local prev = out[#out]
  if prev and prev.t == 'Div' and prev.classes:includes('report-noref') and #out > 1 then
    -- the caption, after its label block, whose fixed \reportneed this one replaces
    pos = #out - 1
    need = need + 2
    out[pos] = rawblock((out[pos].text:gsub('^\\reportneed{[^}]*}', '')))
  elseif prev and prev.t == 'Para' then
    local n = ulen(stringify(prev))
    if n <= LEAD_MAX then
      pos = #out
      need = need + math.ceil(n / LEAD_CHARS)
    end
  end
  -- every heading right above (a "##" heading directly over its first "###" one too),
  -- so none is left alone at the foot of a page
  while pos > 1 and is_heading_block(out[pos - 1]) do
    pos = pos - 1
    need = need + 3
  end
  need = math.min(need, PAGE_LINES - 6)
  out:insert(pos, rawblock('\\reportneed{' .. need .. '\\baselineskip}'))
end

-- Drop a leading "Label:" (and the space after it) from a paragraph's inlines.
local function after_label(ils)
  local out = List()
  local skipping = true
  for k, el in ipairs(ils) do
    if skipping then
      if k == 1 then
        local rest = el.text and el.text:match('^[%a]+:(.*)$')
        if rest and rest ~= '' then out:insert(pandoc.Str(rest)) end
      elseif is_space(el) and k == 2 then
        -- the space after the label
      else
        skipping = false
        out:insert(el)
      end
    else
      out:insert(el)
    end
  end
  return out
end

local function heading_info(h)
  local text = stringify(h.content)
  if h.level == 2 then
    local n = text:match('^(%d+)%s')
    if n then return 'sec-' .. n, n, 'section' end
    local L = text:match('^Appendix%s+(%u):')
    if L then return 'app-' .. L, nil, 'appendix' end
  elseif h.level == 3 then
    local a, b = text:match('^(%d+)%.(%d+)%s')
    if a then return 'sec-' .. a .. '.' .. b, a .. '.' .. b, 'subsection' end
    local L, m = text:match('^(%u)%.(%d+)%s')
    if L then return 'app-' .. L .. '.' .. m, L .. '.' .. m, 'appendix-subsection' end
  end
  return slug(text), nil, 'other'
end

local function figure_label(b)
  -- "*Figure 1.* ..." -> "1", the inlines after the label
  if not (b and b.t == 'Para' and b.content[1] and b.content[1].t == 'Emph') then return nil end
  local lab = stringify(b.content[1]):match('^Figure%s+(%u?%d+)%.$')
  if not lab then return nil end
  local rest = List()
  for k = 2, #b.content do
    if not (k == 2 and is_space(b.content[k])) then rest:insert(b.content[k]) end
  end
  return lab, rest
end

local function table_caption_label(b)
  if not (b and b.t == 'Para' and b.content[1] and b.content[1].t == 'Emph' and #b.content == 1) then
    return nil
  end
  return stringify(b.content[1]):match('^Table%s+(%d+):')
end

local function image_name(b)
  if not (b and b.t == 'Para' and #b.content == 1 and b.content[1].t == 'Image') then return nil end
  local src = b.content[1].src
  return src:match('^fig/([%w_%-]+)%.png$') or src, src
end

local function is_table_note(b)
  local t = para_text(b)
  if not t then return false end
  return t:match('^Sources?[%s:]') or t:match('^%a+:%s') or t:match('^¹') or t:match('^%*') ~= nil
end

local function read_config()
  local path = os.getenv('PAIEC_REPORT_CONFIG')
  if not path or path == '' then return {} end
  local f = io.open(path, 'r')
  if not f then error('report.lua: cannot read PAIEC_REPORT_CONFIG ' .. path) end
  local cfg = pandoc.json.decode(f:read('a'), false)
  f:close()
  return cfg
end

function Pandoc(doc)
  local blocks = doc.blocks
  local meta = doc.meta
  local cfg = read_config()
  local figs = cfg.figures or {}
  local captions = cfg.captions or {}
  report_stray_dollars(blocks)

  -- 1. Title block and abstract ------------------------------------------------
  local i = 1
  while blocks[i] and blocks[i].t == 'HorizontalRule' do i = i + 1 end
  local h1 = blocks[i]
  if not (h1 and h1.t == 'Header' and h1.level == 1) then
    error('report.lua: the draft must start with a level-1 heading (the title)')
  end
  local title = h1.content
  if has_math(title) then
    warn('the title holds TeX math, which the PDF title metadata would show as TeX')
  end
  i = i + 1
  local subtitle, author, affiliation, email, code, keywords
  local note = List()
  while blocks[i] and not (blocks[i].t == 'Header' and blocks[i].level <= 2) do
    local b = blocks[i]
    local t = para_text(b)
    if b.t == 'HorizontalRule' then
      -- dropped
    elseif t and t:match('^Author:') then author = after_label(b.content)
    elseif t and t:match('^Affiliation:') then affiliation = after_label(b.content)
    elseif t and t:match('^Email:') then email = after_label(b.content)
    elseif t and t:match('^Code:') then code = after_label(b.content)
    elseif t and t:match('^Keywords:') then keywords = after_label(b.content)
    elseif b.t == 'Para' and not subtitle then subtitle = b.content
    else note:insert(b) end
    i = i + 1
  end
  if not author then error('report.lua: no "Author:" paragraph before the abstract') end
  if not code then warn('no "Code:" paragraph in the title block') end
  if subtitle and has_math(subtitle) then
    warn('the subtitle holds TeX math, which the PDF subject metadata would show as TeX')
  end
  local abstract = List()
  if blocks[i] and blocks[i].t == 'Header' and stringify(blocks[i].content) == 'Abstract' then
    i = i + 1
    while blocks[i] and not (blocks[i].t == 'Header' and blocks[i].level <= 2)
          and blocks[i].t ~= 'HorizontalRule' do
      abstract:insert(blocks[i])
      i = i + 1
    end
  else
    warn('no "## Abstract" section after the title block')
  end
  local body = List()
  for k = i, #blocks do body:insert(blocks[k]) end

  LOG.front_matter = {
    title = stringify(title),
    subtitle = subtitle and stringify(subtitle) or nil,
    author = stringify(author),
    affiliation = affiliation and stringify(affiliation) or nil,
    email = email and stringify(email) or nil,
    code = code and stringify(code) or nil,
    keywords = keywords and stringify(keywords) or nil,
    note_paragraphs = #note,
    abstract_words = #words(stringify(abstract)),
  }

  -- 2. Ids: headings, figures, tables, references --------------------------------
  local in_refs = false
  for k, b in ipairs(body) do
    if b.t == 'Header' then
      local id = heading_info(b)
      IDS[id] = true
      in_refs = (b.level == 2 and stringify(b.content) == 'References') or (in_refs and b.level > 2)
    elseif b.t == 'Para' then
      local lab = figure_label(b)
      if lab and image_name(body[k - 1]) then IDS['fig-' .. lab] = true end
      local tl = table_caption_label(b)
      if tl then IDS['tab-' .. tl] = true end
    elseif b.t == 'BulletList' and in_refs then
      for _, item in ipairs(b.content) do register_reference(stringify(item)) end
    end
  end

  -- 3. Structure -----------------------------------------------------------------
  local out = List()
  local appendix_started = false
  local appendix_heads = List()
  for _, b in ipairs(body) do
    if b.t == 'Header' and b.level == 2 and stringify(b.content):match('^Appendix%s+%u:') then
      appendix_heads:insert({id = heading_info(b), text = stringify(b.content)})
    end
  end
  local last_main, last_sub = 0, 0
  local last_app, last_app_sub = nil, 0
  local k = 1
  local prev_table = false
  in_refs = false
  while k <= #body do
    local b = body[k]
    local was_table = prev_table
    prev_table = false
    if b.t == 'HorizontalRule' then
      -- dropped: headings separate the parts
    elseif b.t == 'Header' then
      local id, num, kind = heading_info(b)
      b.identifier = id
      local text = stringify(b.content)
      table.insert(LOG.headings, {id = id, level = b.level, text = text})
      if has_math(b.content) then
        warn('heading "' .. usub(text, 60) .. '" holds TeX math, which its PDF bookmark '
             .. 'would show as TeX; name the quantity in words')
      end
      in_refs = (b.level == 2 and text == 'References') or (in_refs and b.level > 2)
      if kind == 'section' then
        local n = tonumber(num)
        if n ~= last_main + 1 then warn('section ' .. num .. ' follows ' .. last_main) end
        last_main, last_sub = n, 0
      elseif kind == 'subsection' then
        local a, m = num:match('^(%d+)%.(%d+)$')
        if tonumber(a) ~= last_main or tonumber(m) ~= last_sub + 1 then
          warn('subsection ' .. num .. ' under section ' .. last_main .. ', after .' .. last_sub)
        end
        last_sub = tonumber(m)
      elseif kind == 'appendix' then
        local L = id:match('^app%-(%u)$')
        local expect = last_app and string.char(last_app:byte() + 1) or 'A'
        if L ~= expect then warn('Appendix ' .. L .. ' where ' .. expect .. ' was expected') end
        last_app, last_app_sub = L, 0
      elseif kind == 'appendix-subsection' then
        local L, m = num:match('^(%u)%.(%d+)$')
        if L ~= last_app or tonumber(m) ~= last_app_sub + 1 then
          warn('appendix subsection ' .. num .. ' after ' .. tostring(last_app) .. '.' .. last_app_sub)
        end
        last_app_sub = tonumber(m)
      end
      if kind == 'appendix' and not appendix_started then
        appendix_started = true
        out:insert(rawblock('\\reportappendixstart'))
        local lines = {'\\begin{reportappendixcontents}'}
        for _, a in ipairs(appendix_heads) do
          lines[#lines + 1] = '\\reportappendixentry{' .. a.id .. '}{' ..
            inlines_latex({pandoc.Str(a.text)}) .. '}'
        end
        lines[#lines + 1] = '\\end{reportappendixcontents}'
        out:insert(rawblock(table.concat(lines, '\n')))
      end
      -- the draft's "##" is a LaTeX \section, "###" a \subsection (its "#" is the title)
      b.level = math.max(1, b.level - 1)
      if num then
        -- "5.4 Title": the number set apart, as the draft writes it. Written as LaTeX
        -- because pandoc's own \texorpdfstring would drop the number from the bookmark.
        local rest = List()
        for q = 3, #b.content do rest:insert(b.content[q]) end
        local cmd = ({'section', 'subsection', 'subsubsection'})[math.min(b.level, 3)]
        out:insert(rawblock('\\' .. cmd .. '{\\secnum{' .. num .. '}' .. inlines_latex(rest) ..
          '}\\label{' .. id .. '}'))
      else
        out:insert(b)
      end
    elseif image_name(b) and figure_label(body[k + 1]) then
      local name, src = image_name(b)
      local lab, cap = figure_label(body[k + 1])
      if captions[lab] then
        local alt = pandoc.read(captions[lab], 'gfm+smart')
        if alt.blocks[1] and alt.blocks[1].content then cap = alt.blocks[1].content end
      end
      local spec = figs[name]
      local file, width
      if spec then
        file = spec.file
        width = spec.width
      else
        warn('figure ' .. src .. ' has no prepared file; using it as is')
        file, width = src, '\\linewidth'
      end
      local caption = inlines_latex(treat_inlines(cap))
      out:insert(rawblock(table.concat({
        '\\begin{figure}[!htbp]',
        '\\phantomsection\\label{fig-' .. lab .. '}',
        '{\\centering\\reportfig{' .. width .. '}{' .. file .. '}\\par}',
        '\\reportcaption{Figure ' .. lab .. '.}{' .. caption .. '}',
        '\\end{figure}',
      }, '\n')))
      table.insert(LOG.figures, {label = lab, name = name, file = file, width = width})
      k = k + 1   -- the caption paragraph is used
    elseif image_name(b) then
      warn('image ' .. select(2, image_name(b)) .. ' has no "*Figure N.*" caption after it')
      out:insert(b)
    elseif table_caption_label(b) then
      local tl = table_caption_label(b)
      out:insert(rawblock('\\reportneed{8\\baselineskip}\\phantomsection\\label{tab-' .. tl .. '}'))
      out:insert(pandoc.Div({b}, pandoc.Attr('', {'report-noref'})))
    elseif b.t == 'Table' then
      local size, lines = layout_table(b, 'before: ' .. usub(para_text(body[k + 1]) or '', 50))
      local kept = lines <= KEEP_LINES and longtable_to_tabular(blocks_latex(treat_blocks({b})))
      keep_with_table(out, kept and lines or nil)
      out:insert(rawblock('\\begingroup\\' .. size ..
        '\\setlength{\\tabcolsep}{' .. TABCOLSEP_PT .. 'pt}\\renewcommand{\\arraystretch}{1.15}'))
      if kept then
        -- a short table is one unbreakable box: it moves to the next page whole
        out:insert(rawblock(kept))
        LOG.tables[#LOG.tables].kept_together = true
      else
        out:insert(b)
      end
      out:insert(rawblock('\\endgroup'))
      prev_table = true
    elseif was_table and b.t == 'Para' and is_table_note(b) then
      out:insert(rawblock('\\begingroup\\small'))
      out:insert(b)
      out:insert(rawblock('\\par\\endgroup'))
    elseif b.t == 'CodeBlock' then
      if b.text:find('\\end{reportcode}', 1, true) then error('report.lua: code block closes reportcode') end
      out:insert(rawblock('\\begin{reportcode}\n' .. b.text .. '\n\\end{reportcode}'))
    elseif b.t == 'BulletList' and in_refs then
      local lines = {'\\begin{reportrefs}'}
      for _, item in ipairs(b.content) do
        local r = parse_reference(stringify(item))
        local id
        if r then
          for _, ref in ipairs(REFS) do
            if ref.first[1] == words(r.names[1])[1] and ref.year == r.year
               and table.concat(ref.names, '|') == table.concat(r.names, '|') then id = ref.id end
          end
        end
        local anchor = id and ('\\phantomsection\\label{' .. id .. '}') or ''
        lines[#lines + 1] = '\\item ' .. anchor .. blocks_latex(treat_blocks(item))
      end
      lines[#lines + 1] = '\\end{reportrefs}'
      out:insert(rawblock(table.concat(lines, '\n')))
    else
      out:insert(b)
    end
    k = k + 1
  end
  if not appendix_started then warn('no "## Appendix A: ..." heading: no appendix page break') end

  -- 4. Links and code everywhere else --------------------------------------------
  out = treat_blocks(out)

  -- 5. Metadata for the template ---------------------------------------------------
  meta.title = pandoc.MetaInlines(treat_inlines(title))
  meta['title-plain'] = pandoc.MetaString(stringify(title))
  meta.author = pandoc.MetaInlines(treat_inlines(author))
  meta['author-plain'] = pandoc.MetaString(stringify(author))
  -- the subtitle without its bracketed citations: "... NeurIPS 2026 [PAIEC organisers 2026]."
  local subject = subtitle and stringify(subtitle):gsub('%s*%[[^%]]*%]', '') or ''
  meta['subject-plain'] = pandoc.MetaString(subject)
  -- from the title block's "Keywords:" line, the one source CITATION.cff repeats
  meta['keywords-plain'] = pandoc.MetaString(keywords and stringify(keywords) or '')
  if subtitle then meta.subtitle = pandoc.MetaInlines(treat_inlines(subtitle)) end
  if affiliation then meta.affiliation = pandoc.MetaInlines(treat_inlines(affiliation)) end
  if email then meta.email = pandoc.MetaString(stringify(email)) end
  if code then meta.code = pandoc.MetaInlines(treat_inlines(code)) end
  if #note > 0 then meta.note = pandoc.MetaBlocks(treat_blocks(note)) end
  if #abstract > 0 then meta.abstract = pandoc.MetaBlocks(treat_blocks(abstract)) end
  local logpath = cfg.log
  local ids = {}
  for id in pairs(IDS) do ids[#ids + 1] = id end
  table.sort(ids)
  LOG.ids = ids
  if logpath and logpath ~= '' then
    local f = assert(io.open(logpath, 'w'))
    f:write(pandoc.json.encode(LOG))
    f:close()
  end
  return pandoc.Pandoc(out, meta)
end
