// Package translate turns ordered blocks into paragraphs and paragraphs into translations.
package translate

import (
	"fmt"
	"slices"
	"strings"

	"dplus/internal/doc"
	"dplus/internal/order"
)

const (
	// continueSnippet is how many characters on each side of a block boundary the decision model sees.
	continueSnippet = 200
	continueAt      = 0.5
	// Output budget: a translation rarely runs past four tokens per source character, and short texts still need room.
	tokensPerRune = 4
	minTokens     = 64
	maxTokens     = 4096
)

// Paragraph is one unit of translation and the blocks it came from.
type Paragraph struct {
	Kind        string   `json:"kind"`
	Score       float64  `json:"score"` // The layout model's score for the kind, from the block the paragraph took it from.
	Role        doc.Role `json:"role"`
	Pages       []int    `json:"pages"`
	Blocks      []int    `json:"blocks"`
	Source      string   `json:"source"`
	Image       string   `json:"image,omitempty"` // A picture's cropped file, relative to the folder of the output files.
	Language    string   `json:"language"`
	Spaced      string   `json:"spaced"`
	Spacing     string   `json:"spacing"`
	Translated  bool     `json:"translated"`
	Translation string   `json:"translation"`
	Stop        string   `json:"stop,omitempty"`
	Tokens      int      `json:"tokens,omitempty"`
	Seconds     float64  `json:"seconds,omitempty"`
	Check       *Check   `json:"check,omitempty"` // What the translation kept of the source.

	parts []doc.Block // The blocks in reading order, for the decisions that follow.
}

// Page is what the serialized text needs from one page.
type Page struct {
	Number   int
	Width    float64
	Height   float64
	Body     []doc.Block // In reading order.
	Inserted []doc.Block
}

// Decisions counts the questions asked to build the serialized text.
type Decisions struct {
	Continuation int `json:"continuation"`
	Caption      int `json:"caption"`
	Placement    int `json:"placement"`
	SamePlace    int `json:"same_place"`
}

// Decider answers the questions the serialized text needs.
type Decider interface {
	Choose(state any, question string, options []order.Option, images []string) (choice string, confidence float64, err error)
	order.BodyDecider
	ChooseEach(state any, questions map[string]order.Question) (map[string]string, error)
}

// Translatable reports whether the text of a kind of block is prose for the translator.
func Translatable(kind string) bool {
	switch kind {
	case "title", "section_header", "text", "list_item", "caption", "footnote", "table", "document_index", "key_value_region", "form":
		return true
	}
	return false
}

// Gather builds the serialized text: it settles the paragraphs from the body blocks, then places each inserted element
// after the paragraph the decision model chooses for it. It returns the paragraphs in output order.
func Gather(pages []Page, pageCount int, decider Decider) ([]Paragraph, Decisions, error) {
	var decisions Decisions
	body, err := joinBody(pages, pageCount, decider, &decisions)
	if err != nil {
		return nil, decisions, err
	}
	output, err := place(pages, pageCount, body, decider, &decisions)
	return output, decisions, err
}

// joinBody joins body blocks whose text continues into the next, across columns, pages and inserted elements.
func joinBody(pages []Page, pageCount int, decider Decider, decisions *Decisions) ([]Paragraph, error) {
	var body []Paragraph
	var last *doc.Block
	for _, page := range pages {
		continues, err := askBoundaries(page, pageCount, last, decider, decisions)
		if err != nil {
			return nil, err
		}
		for i := range page.Body {
			block := page.Body[i]
			if continues[i] {
				body[len(body)-1].add(block)
			} else {
				body = append(body, Paragraph{Kind: block.Kind, Score: block.Score, Role: doc.RoleBody, Pages: []int{block.Page}, Blocks: []int{block.ID}, Source: block.Text, parts: []doc.Block{block}})
			}
			last = &page.Body[i]
		}
	}
	return body, nil
}

// add continues the paragraph with a block. A paragraph that began with a formula or code takes the kind of the first
// prose that continues it, so that it is translated.
func (p *Paragraph) add(block doc.Block) {
	p.Source = joinText(p.Source, block.Text)
	p.Blocks = append(p.Blocks, block.ID)
	if p.Pages[len(p.Pages)-1] != block.Page {
		p.Pages = append(p.Pages, block.Page)
	}
	if !Translatable(p.Kind) && Translatable(block.Kind) {
		p.Kind, p.Score = block.Kind, block.Score
	}
	p.parts = append(p.parts, block)
}

// askBoundaries asks in one request whether the text continues at each boundary into the page's body blocks, the one
// from the block read before the page included. A title or section header ends a paragraph by its kind. It returns,
// for each body block, whether it continues the block before it.
func askBoundaries(page Page, pageCount int, previous *doc.Block, decider Decider, decisions *Decisions) ([]bool, error) {
	continues := make([]bool, len(page.Body))
	questions := map[string]string{}
	var boundaries []map[string]any
	position := map[string]int{}
	for i, block := range page.Body {
		before := previous
		if i > 0 {
			before = &page.Body[i-1]
		}
		if before == nil || endsParagraph(before.Kind) || endsParagraph(block.Kind) {
			continue
		}
		id := fmt.Sprintf("c%03d", len(questions)+1)
		questions[id] = fmt.Sprintf("Does the text after boundary %s continue the same sentence or paragraph as the text before it?", id)
		position[id] = i
		boundaries = append(boundaries, map[string]any{
			"id": id, "page_break_between": before.Page != block.Page,
			"before": side(*before, lastRunes(before.Text, continueSnippet)),
			"after":  side(block, firstRunes(block.Text, continueSnippet)),
		})
	}
	if len(questions) == 0 {
		return continues, nil
	}
	state := map[string]any{"page": order.PageInfo(page.Number, pageCount, page.Width, page.Height), "boundaries": boundaries}
	answers, err := decider.YesNoEach(state, questions)
	if err != nil {
		return nil, err
	}
	decisions.Continuation += len(questions)
	for id, i := range position {
		continues[i] = answers[id] >= continueAt
	}
	return continues, nil
}

func endsParagraph(kind string) bool { return kind == "title" || kind == "section_header" }

// side is one block at a boundary, with the part of its text the decision model sees.
func side(block doc.Block, text string) map[string]any {
	facts := order.Facts(block)
	facts["page"] = block.Page
	facts["text"] = text
	return facts
}

func firstRunes(text string, count int) string {
	runes := []rune(text)
	return string(runes[:min(count, len(runes))])
}

func lastRunes(text string, count int) string {
	runes := []rune(text)
	return string(runes[max(len(runes)-count, 0):])
}

func joinText(first, second string) string {
	head := []rune(strings.TrimRight(first, " "))
	tail := []rune(strings.TrimLeft(second, " "))
	if len(head) == 0 || len(tail) == 0 {
		return string(head) + string(tail)
	}
	if east(head[len(head)-1]) && east(tail[0]) {
		return string(head) + string(tail)
	}
	return string(head) + " " + string(tail)
}

func east(r rune) bool {
	return r >= 0x2E80 && r <= 0x9FFF || r >= 0xAC00 && r <= 0xD7A3 || r >= 0xF900 && r <= 0xFAFF
}

// Names of the languages the prompts mention.
var names = map[string]string{"ko": "Korean", "en": "English", "zh": "Chinese", "ja": "Japanese"}

// Name gives a language's English name, or the code itself.
func Name(code string) string {
	if name, ok := names[code]; ok {
		return name
	}
	return code
}

// kindGuides tell the translator what a paragraph of each kind is and the form its translation keeps.
var kindGuides = map[string]string{
	"title":            "the title of the document. Translate it as a concise title, not as a sentence.",
	"section_header":   "a section heading. Translate it as a concise heading and keep its number, such as 2.1, unchanged.",
	"text":             "a paragraph of the running text. Translate every sentence in order as continuous prose.",
	"list_item":        "an item of a list. Keep its bullet, number or letter at the start and translate the rest.",
	"caption":          "the caption of a figure or table. Translate its label word, such as Figure or Table, into the usual label of the target language, keep its number, and translate the description.",
	"footnote":         "a footnote. Keep its marker, a number or symbol at the start, unchanged and translate the note.",
	"table":            "the text of a table, cell by cell in reading order. Keep the line breaks and the order of the cells, translate the words of every cell, and leave numbers, units and codes unchanged.",
	"document_index":   "an entry of a table of contents or an index. Translate the entry and keep its numbers and page numbers unchanged.",
	"key_value_region": "the labels and values of a form. Translate the labels, keep each label with its value, and leave numbers, dates and codes unchanged.",
	"form":             "the labels and values of a form. Translate the labels, keep each label with its value, and leave numbers, dates and codes unchanged.",
}

// setApart describes a text block the decision model judged not part of the running text.
const setApart = "a passage set apart from the running text, such as a note, a sidebar or a label. Translate every sentence in order."

// SystemPrompt asks for a translation of a paragraph into a language. It names the paragraph's kind and the layout
// model's score for it, and the form a translation of that kind keeps. Every paragraph of a document is asked for the
// same written style, since each is translated on its own.
func SystemPrompt(p Paragraph, to string) string {
	into := Name(to)
	prompt := fmt.Sprintf("You are a professional document translator. Translate the user's text from %s into %s. "+
		"Keep numbers, names, reference marks and formulas as they are. "+
		"Use the plain written style of %s that reports and papers use. Output only the translation.", Name(p.Language), into, into)
	guide, known := kindGuides[p.Kind]
	if p.Kind == "text" && p.Role == doc.RoleInserted {
		guide, known = setApart, true
	}
	if !known {
		return prompt
	}
	return prompt + fmt.Sprintf("\n\nA layout model labeled the text %s with a score of %.2f. It is %s", p.Kind, p.Score, guide)
}

// scripts are the Unicode scripts each target language is written in. A translation may also hold Latin and Greek
// letters, for names and formulas, and the characters every script shares: digits, punctuation, symbols and marks.
var scripts = map[string][]string{
	"ko": {"Hangul"},
	"zh": {"Han"},
	"ja": {"Han", "Hiragana", "Katakana"},
	"en": {},
}

// Pattern is the regular expression a translation into a language must match: only characters of the scripts the
// language is written in. Generation takes no token outside it, so a translation cannot stay in the source language's
// script or drift into a third one. Empty for a language whose scripts are not listed.
func Pattern(to string) string {
	own, known := scripts[to]
	if !known {
		return ""
	}
	var classes strings.Builder
	for _, script := range append(slices.Clone(own), "Latin", "Greek", "Common", "Inherited") {
		classes.WriteString(`\p{` + script + `}`)
	}
	return "[" + classes.String() + "]*"
}

// TokenBudget bounds the output of one translation.
func TokenBudget(source string) int {
	return min(max(len([]rune(source))*tokensPerRune, minTokens), maxTokens)
}
