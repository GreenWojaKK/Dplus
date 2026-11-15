// Package doc holds the types every pipeline stage shares.
package doc

import "math"

// Box is a rectangle in page points, with the origin at the top left of the displayed page.
type Box struct {
	X0 float64 `json:"x0"`
	Y0 float64 `json:"y0"`
	X1 float64 `json:"x1"`
	Y1 float64 `json:"y1"`
}

func (b Box) Width() float64   { return b.X1 - b.X0 }
func (b Box) Height() float64  { return b.Y1 - b.Y0 }
func (b Box) CenterX() float64 { return (b.X0 + b.X1) / 2 }
func (b Box) CenterY() float64 { return (b.Y0 + b.Y1) / 2 }
func (b Box) Area() float64    { return math.Max(b.Width(), 0) * math.Max(b.Height(), 0) }

// Contains reports whether the point lies inside the box.
func (b Box) Contains(x, y float64) bool { return x >= b.X0 && x <= b.X1 && y >= b.Y0 && y <= b.Y1 }

// Overlap is the area two boxes share.
func (b Box) Overlap(o Box) float64 {
	w := math.Min(b.X1, o.X1) - math.Max(b.X0, o.X0)
	h := math.Min(b.Y1, o.Y1) - math.Max(b.Y0, o.Y0)
	if w <= 0 || h <= 0 {
		return 0
	}
	return w * h
}

// Turn gives the box on a page turned by quarter clockwise turns, for a page of the given displayed size.
func (b Box) Turn(quarter int, width, height float64) Box {
	switch ((quarter % 4) + 4) % 4 {
	case 1:
		return Box{X0: height - b.Y1, Y0: b.X0, X1: height - b.Y0, Y1: b.X1}
	case 2:
		return Box{X0: width - b.X1, Y0: height - b.Y1, X1: width - b.X0, Y1: height - b.Y0}
	case 3:
		return Box{X0: b.Y0, Y0: width - b.X1, X1: b.Y1, Y1: width - b.X0}
	}
	return b
}

// Char is one character of the PDF's text layer.
type Char struct {
	Rune       rune
	Box        Box
	SpaceAfter bool // The text layer has a space right after this character.
	HyphenEnd  bool // The text layer breaks a word with a hyphen right after this character.
}

// Source says where a block's text came from.
type Source string

const (
	SourceTextLayer Source = "text_layer"
	SourceOCR       Source = "ocr"
)

// Role says how a block takes part in the serialized text.
type Role string

const (
	RoleBody     Role = "body"     // Read in order and joined into paragraphs.
	RoleInserted Role = "inserted" // Placed after the paragraph in progress at its position.
	RoleDropped  Role = "dropped"  // Left out of the text.
)

// RoleOf gives the role a kind of block starts with.
func RoleOf(kind string) Role {
	switch kind {
	case "title", "section_header", "text", "list_item", "formula", "code":
		return RoleBody
	case "page_header", "page_footer":
		return RoleDropped
	}
	return RoleInserted
}

// Block is one laid-out region of a page. Blocks from the text layer and from OCR have this same form.
type Block struct {
	ID     int     `json:"id"`
	Page   int     `json:"page"`
	Kind   string  `json:"kind"`
	Role   Role    `json:"role"`
	Box    Box     `json:"box"`
	Score  float64 `json:"score"`
	Text   string  `json:"text"`
	Source Source  `json:"source"`
	// Running is whether the decision model judged a text block part of the running text. Other kinds are not asked.
	Running *bool `json:"running_text,omitempty"`
	// Image is the file a picture was cropped into, relative to the folder of the output files.
	Image string `json:"image,omitempty"`
}
