package layout

import (
	"sort"
	"strings"
	"unicode"

	"dplus/internal/doc"
	"dplus/internal/pdf"
)

const (
	// coveredShare is how much of one box another must cover for the two to count as the same region.
	coveredShare = 0.8
	// sameBoxIoU is the intersection over union from which two boxes count as the same region.
	sameBoxIoU = 0.5
	// lineBand groups characters whose vertical centers differ by less than this share of the character height.
	lineBand = 0.6
	// wordGap adds a space where the text layer has none but characters on a line stand this share of the character
	// height apart, as table columns do.
	wordGap = 1.0
)

// Regions that hold other regions: their characters belong to them whatever Heron finds inside.
var containers = map[string]bool{"table": true, "picture": true}

// Result is a page's blocks and what the assignment left over.
type Result struct {
	Blocks     []doc.Block
	Unassigned int // Text-layer characters that fall in no box.
}

// Clean resolves overlapping boxes. Of two boxes on the same region the higher score stays, a larger table or picture
// takes in what it holds, and of the rest a box mostly covered by a higher-scoring one goes.
func Clean(found []Detection) []Detection {
	ranked := append([]Detection(nil), found...)
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].Score > ranked[j].Score })
	var unique []Detection
	for _, candidate := range ranked {
		if !clashes(candidate, unique, sameBox) {
			unique = append(unique, candidate)
		}
	}
	var kept []Detection
	for _, candidate := range unique {
		if !absorbed(candidate, unique) && !clashes(candidate, kept, covers) {
			kept = append(kept, candidate)
		}
	}
	return kept
}

func clashes(candidate Detection, others []Detection, conflict func(a, b doc.Box) bool) bool {
	for _, other := range others {
		if conflict(candidate.Box, other.Box) {
			return true
		}
	}
	return false
}

func absorbed(candidate Detection, all []Detection) bool {
	for _, other := range all {
		if containers[other.Kind] && other.Box.Area() > candidate.Box.Area() &&
			candidate.Box.Overlap(other.Box)/candidate.Box.Area() > coveredShare {
			return true
		}
	}
	return false
}

func sameBox(a, b doc.Box) bool {
	shared := a.Overlap(b)
	return shared > 0 && shared/(a.Area()+b.Area()-shared) > sameBoxIoU
}

func covers(a, b doc.Box) bool {
	shared := a.Overlap(b)
	return shared > 0 && (shared/a.Area() > coveredShare || shared/b.Area() > coveredShare)
}

// Build fills the detected boxes with the page's characters. Pixels are converted to page points with the given ratios.
func Build(page *pdf.Page, found []Detection, pointsPerPixelX, pointsPerPixelY float64) Result {
	regions := Clean(found)
	for i := range regions {
		box := &regions[i].Box
		box.X0, box.X1 = box.X0*pointsPerPixelX, box.X1*pointsPerPixelX
		box.Y0, box.Y1 = box.Y0*pointsPerPixelY, box.Y1*pointsPerPixelY
	}
	members := make([][]doc.Char, len(regions))
	result := Result{}
	for _, char := range page.Chars {
		best := -1
		for i, region := range regions {
			if region.Box.Contains(char.Box.CenterX(), char.Box.CenterY()) && (best < 0 || region.Score > regions[best].Score) {
				best = i
			}
		}
		if best < 0 {
			result.Unassigned++
			continue
		}
		members[best] = append(members[best], char)
	}
	for i, region := range regions {
		text := assemble(members[i], region.Kind == "table")
		if text == "" && region.Kind != "picture" {
			continue
		}
		result.Blocks = append(result.Blocks, doc.Block{
			Page: page.Number, Kind: region.Kind, Role: doc.RoleOf(region.Kind), Box: region.Box, Score: region.Score, Text: text, Source: doc.SourceTextLayer,
		})
	}
	sort.SliceStable(result.Blocks, func(i, j int) bool {
		a, b := result.Blocks[i].Box, result.Blocks[j].Box
		if a.Y0 != b.Y0 {
			return a.Y0 < b.Y0
		}
		return a.X0 < b.X0
	})
	for i := range result.Blocks {
		result.Blocks[i].ID = i + 1
	}
	return result
}

// assemble puts a block's characters into text, line by line. Table rows stay on their own lines.
func assemble(chars []doc.Char, keepLines bool) string {
	if len(chars) == 0 {
		return ""
	}
	heights := make([]float64, len(chars))
	for i, char := range chars {
		heights[i] = char.Box.Height()
	}
	sort.Float64s(heights)
	height := max(heights[len(heights)/2], 1)
	sorted := append([]doc.Char(nil), chars...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Box.CenterY() < sorted[j].Box.CenterY() })
	var lines [][]doc.Char
	for _, char := range sorted {
		if n := len(lines); n > 0 {
			last := lines[n-1]
			total := 0.0
			for _, member := range last {
				total += member.Box.CenterY()
			}
			if char.Box.CenterY()-total/float64(len(last)) < lineBand*height {
				lines[n-1] = append(last, char)
				continue
			}
		}
		lines = append(lines, []doc.Char{char})
	}
	text := ""
	for i, line := range lines {
		sort.SliceStable(line, func(a, b int) bool { return line[a].Box.X0 < line[b].Box.X0 })
		rendered := renderLine(line, height)
		if i == 0 {
			text = rendered
			continue
		}
		text = joinLines(text, rendered, keepLines)
	}
	return strings.TrimSpace(text)
}

func renderLine(line []doc.Char, height float64) string {
	var out strings.Builder
	for i, char := range line {
		out.WriteRune(char.Rune)
		if i+1 == len(line) {
			if char.HyphenEnd {
				out.WriteByte('-')
			}
			break
		}
		if char.SpaceAfter || line[i+1].Box.X0-char.Box.X1 > wordGap*height {
			out.WriteByte(' ')
		}
	}
	return out.String()
}

// joinLines adds the next line to the text. A hyphen ending a line joins a word broken across the two lines.
func joinLines(text, next string, keepLines bool) string {
	if keepLines {
		return text + "\n" + next
	}
	before := []rune(strings.TrimRight(text, " "))
	after := []rune(next)
	if len(before) == 0 || len(after) == 0 {
		return string(before) + next
	}
	last, first := before[len(before)-1], after[0]
	if last == '-' && len(before) > 1 && unicode.IsLetter(before[len(before)-2]) && unicode.IsLower(first) {
		return string(before[:len(before)-1]) + next
	}
	if eastAsian(last) && eastAsian(first) {
		return string(before) + next
	}
	return string(before) + " " + next
}

func eastAsian(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hangul, unicode.Hiragana, unicode.Katakana)
}
