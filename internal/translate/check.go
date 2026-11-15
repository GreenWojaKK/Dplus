package translate

import (
	"math"
	"slices"
	"unicode/utf8"

	"dplus/internal/doc"
)

// A translation is checked against the text it was given, the spaced source. Its reference marks are the source's
// numbers and footnote symbols: figure, table, equation, section and citation numbers, years and quantities, which a
// translation into any language keeps. Its length is its characters per source character against the median of the
// run's translations from the same language, so that the check does not depend on the languages.
const (
	shortAt   = 0.5 // A translation below this share of its language's usual length may have dropped part of the source.
	shortFrom = 40  // Characters a source needs before its length is judged; shorter ones vary too widely.
)

// lengthTiers group the checks by the source's length in characters.
var lengthTiers = []struct {
	label string
	below int
}{{"<100", 100}, {"100-399", 400}, {"400-1599", 1600}, {"1600+", math.MaxInt}}

// footnoteSymbols are the marks other than digits that a translation keeps.
var footnoteSymbols = []rune{'*', '†', '‡', '§', '¶'}

// Check is what a translation kept of its source.
type Check struct {
	Marks   int      `json:"marks"`             // Distinct reference marks in the source.
	Missing []string `json:"missing,omitempty"` // Marks of the source the translation lacks.
	Length  float64  `json:"length"`            // Its length against the median of its source language; 1 is usual.
	Short   bool     `json:"short,omitempty"`   // Its length is below shortAt for a source of at least shortFrom characters.
}

// Tally sums up the checks of a group of translations.
type Tally struct {
	Paragraphs   int            `json:"paragraphs"`
	Stops        map[string]int `json:"stops"`
	Marks        int            `json:"marks"`
	MissingMarks int            `json:"missing_marks"`
	WithMissing  int            `json:"with_missing"` // Translations that lack a mark of their source.
	Short        int            `json:"short"`
}

// Checks are the tallies by paragraph kind, inserted elements apart from the body, and by the source's length.
type Checks struct {
	Kinds   map[string]*Tally `json:"kinds"`
	Lengths map[string]*Tally `json:"lengths"`
}

// KindLabel names a paragraph's kind, inserted elements apart from the body.
func KindLabel(p Paragraph) string {
	if p.Role == doc.RoleInserted {
		return "inserted " + p.Kind
	}
	return p.Kind
}

// CheckAll checks every translated paragraph, stores each check in its paragraph and returns the tallies. The
// translations of one run share their target language, so a language pair is told by the source language.
func CheckAll(paragraphs []Paragraph) Checks {
	ratios := map[string][]float64{}
	for _, p := range paragraphs {
		if p.Translated && utf8.RuneCountInString(p.Spaced) >= shortFrom {
			ratios[p.Language] = append(ratios[p.Language], lengthRatio(p))
		}
	}
	medians := map[string]float64{}
	for language, list := range ratios {
		medians[language] = median(list)
	}
	checks := Checks{Kinds: map[string]*Tally{}, Lengths: map[string]*Tally{}}
	for i := range paragraphs {
		p := &paragraphs[i]
		if !p.Translated {
			continue
		}
		source := marks(p.Spaced)
		kept := marks(p.Translation)
		check := Check{Marks: len(source), Length: 1}
		for _, mark := range source {
			if !slices.Contains(kept, mark) {
				check.Missing = append(check.Missing, mark)
			}
		}
		if m := medians[p.Language]; m > 0 {
			check.Length = math.Round(lengthRatio(*p)/m*100) / 100
		}
		check.Short = check.Length < shortAt && utf8.RuneCountInString(p.Spaced) >= shortFrom
		p.Check = &check
		add(checks.Kinds, KindLabel(*p), *p)
		add(checks.Lengths, tier(utf8.RuneCountInString(p.Spaced)), *p)
	}
	return checks
}

func add(tallies map[string]*Tally, label string, p Paragraph) {
	t := tallies[label]
	if t == nil {
		t = &Tally{Stops: map[string]int{}}
		tallies[label] = t
	}
	t.Paragraphs++
	t.Stops[p.Stop]++
	t.Marks += p.Check.Marks
	t.MissingMarks += len(p.Check.Missing)
	if len(p.Check.Missing) > 0 {
		t.WithMissing++
	}
	if p.Check.Short {
		t.Short++
	}
}

func tier(length int) string {
	for _, t := range lengthTiers {
		if length < t.below {
			return t.label
		}
	}
	return lengthTiers[len(lengthTiers)-1].label
}

func lengthRatio(p Paragraph) float64 {
	source := utf8.RuneCountInString(p.Spaced)
	if source == 0 {
		return 0
	}
	return float64(utf8.RuneCountInString(p.Translation)) / float64(source)
}

func median(values []float64) float64 {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

// marks lists the distinct reference marks of a text in the order they first appear: each run of digits, full-width
// ones read as ASCII and thousands separators dropped, and each footnote symbol.
func marks(text string) []string {
	runes := []rune(text)
	var found []string
	seen := map[string]bool{}
	keep := func(mark string) {
		if !seen[mark] {
			seen[mark] = true
			found = append(found, mark)
		}
	}
	for i := 0; i < len(runes); i++ {
		if slices.Contains(footnoteSymbols, runes[i]) {
			keep(string(runes[i]))
			continue
		}
		if !isDigit(runes[i]) {
			continue
		}
		var number []rune
		for i < len(runes) && (isDigit(runes[i]) || runes[i] == ',' && thousands(runes, i+1)) {
			if isDigit(runes[i]) {
				number = append(number, asciiDigit(runes[i]))
			}
			i++
		}
		i--
		keep(string(number))
	}
	return found
}

// thousands reports whether exactly three digits start at i, as after a thousands separator.
func thousands(runes []rune, i int) bool {
	for j := i; j < i+3; j++ {
		if j >= len(runes) || !isDigit(runes[j]) {
			return false
		}
	}
	return i+3 == len(runes) || !isDigit(runes[i+3])
}

func isDigit(r rune) bool { return r >= '0' && r <= '9' || r >= '０' && r <= '９' }

func asciiDigit(r rune) rune {
	if r >= '０' && r <= '９' {
		return '0' + (r - '０')
	}
	return r
}
