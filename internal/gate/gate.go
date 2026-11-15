// Package gate judges a page from its PDF objects and extracted characters.
package gate

import (
	"strings"
	"unicode/utf8"
)

// Verdicts of a page.
const (
	Healthy = "healthy"
	Scanned = "scanned"
	Corrupt = "corrupt"
	Blank   = "blank"
)

// alienRatio is the share of private-use characters from which the text layer counts as damaged.
const alienRatio = 0.05

// Counts holds what a page is made of.
type Counts struct {
	TextObjects  int `json:"text_objects"`
	Images       int `json:"images"`
	Paths        int `json:"paths"`
	OtherObjects int `json:"other_objects"`
	Chars        int `json:"chars"`
	Alien        int `json:"alien"`
}

// CountText fills the character counts from the text of the page.
func (c *Counts) CountText(text string) {
	trimmed := strings.TrimSpace(text)
	c.Chars = utf8.RuneCountInString(trimmed)
	c.Alien = 0
	for _, r := range text {
		if r >= 0xE000 && r <= 0xF8FF {
			c.Alien++
		}
	}
}

// Classify gives the verdict for the counts of a page.
func Classify(c Counts) string {
	if c.Chars > 0 && float64(c.Alien)/float64(c.Chars) > alienRatio {
		return Corrupt
	}
	if c.Images > c.TextObjects {
		return Scanned
	}
	if c.Chars > 0 {
		return Healthy
	}
	if c.Paths > 0 || c.OtherObjects > 0 {
		return Scanned
	}
	if c.TextObjects > 0 || c.Images > 0 {
		return Corrupt
	}
	return Blank
}
