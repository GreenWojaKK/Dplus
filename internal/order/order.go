// Package order settles the reading order of a page's body blocks: the decision model chooses, and page geometry limits
// and checks what it may choose.
package order

import (
	"fmt"
	"math"
	"strings"

	"dplus/internal/doc"
)

const (
	// sideTolerance is how far, in points, a block may reach past another and still lie entirely to its left.
	sideTolerance = 1.0
	// sharedWidth is the share of the narrower block two blocks must share across for them to overlap horizontally.
	sharedWidth = 0.2
	// sideBySide is the share of the shorter block two blocks must share in height to count as side by side.
	sideBySide = 0.5
	// snippetLength is how many characters of a block's start and end the model sees.
	snippetLength = 60
	// bodyAt is the probability from which a text block counts as running text.
	bodyAt = 0.5
)

// Option is one answer the decision model may give.
type Option struct {
	Key         string
	Description string
}

// Question is one choice question of a request.
type Question struct {
	Text    string
	Options []Option
}

// Decider gives the probability of each answer to a question about a page.
type Decider interface {
	// Rank asks one choice question. The state comes in two parts: fixed leads it and is the same for every question
	// about the page, moving ends it and changes from one question to the next.
	Rank(fixed, moving any, question string, options []Option, images []string) (map[string]float64, error)
}

// BodyDecider answers several yes-or-no questions about one state in one request.
type BodyDecider interface {
	YesNoEach(state any, questions map[string]string) (map[string]float64, error)
}

// Violation is a pair of blocks read against the page's geometry.
type Violation struct {
	ReadFirst  int `json:"read_first"`
	ShouldLead int `json:"should_lead"`
}

// Result is a page's reading order.
type Result struct {
	Order         []int       `json:"order"` // Body block IDs in reading order.
	Asked         int         `json:"asked"`
	MinConfidence float64     `json:"min_confidence"` // The lowest probability a block was chosen with.
	Passes        int         `json:"passes"`
	Violations    []Violation `json:"violations,omitempty"`
	Review        bool        `json:"review"`
}

// overlapsAcross reports whether two boxes share enough width to stand one above the other.
func overlapsAcross(a, b doc.Box) bool {
	shared := min(a.X1, b.X1) - max(a.X0, b.X0)
	return shared > sharedWidth*min(a.Width(), b.Width())
}

// comesAbove is the first rule: of two boxes overlapping horizontally, the upper one comes first.
func comesAbove(a, b doc.Box) bool {
	if !overlapsAcross(a, b) || a.CenterY() >= b.CenterY() {
		return false
	}
	return min(a.Y1, b.Y1)-max(a.Y0, b.Y0) < sideBySide*min(a.Height(), b.Height())
}

// comesLeft is the second rule: a box entirely to the left of another comes first, unless a box between them in height
// overlaps both horizontally.
func comesLeft(a, b doc.Box, all []doc.Box) bool {
	if a.X1 > b.X0+sideTolerance {
		return false
	}
	low, high := min(a.CenterY(), b.CenterY()), max(a.CenterY(), b.CenterY())
	for _, s := range all {
		if s.CenterY() > low && s.CenterY() < high && overlapsAcross(s, a) && overlapsAcross(s, b) {
			return false
		}
	}
	return true
}

// Check returns the pairs of blocks a reading order puts against the two rules. Separators are boxes that are not read
// but still stand between the blocks around them.
func Check(ordered []doc.Box, ids []int, separators []doc.Box) []Violation {
	all := append(append([]doc.Box(nil), ordered...), separators...)
	var found []Violation
	for i := range ordered {
		for j := i + 1; j < len(ordered); j++ {
			if comesAbove(ordered[j], ordered[i]) || comesLeft(ordered[j], ordered[i], all) {
				found = append(found, Violation{ReadFirst: ids[i], ShouldLead: ids[j]})
			}
		}
	}
	return found
}

// ClassifyBody asks in one request, for each text block of a page, whether it belongs to the running text. A block that
// does not becomes an inserted element. It returns the number of questions asked.
func ClassifyBody(page, pageCount int, width, height float64, blocks []doc.Block, decider BodyDecider) (int, error) {
	var shown []doc.Block
	questions := map[string]string{}
	for _, block := range blocks {
		if block.Role == doc.RoleDropped {
			continue
		}
		shown = append(shown, block)
		if block.Role == doc.RoleBody && block.Kind == "text" {
			questions[Key(block.ID)] = fmt.Sprintf("Is block %s part of the document's running text, the prose a reader "+
				"follows continuously from one block to the next?", Key(block.ID))
		}
	}
	if len(questions) == 0 {
		return 0, nil
	}
	answers, err := decider.YesNoEach(map[string]any{"page": PageInfo(page, pageCount, width, height), "blocks": blockEntries(shown)}, questions)
	if err != nil {
		return 0, err
	}
	for i := range blocks {
		probability, asked := answers[Key(blocks[i].ID)]
		if !asked {
			continue
		}
		running := probability >= bodyAt
		blocks[i].Running = &running
		if !running {
			blocks[i].Role = doc.RoleInserted
		}
	}
	return len(questions), nil
}

// The state of a reading-order question. Field order is the order in the prompt: what is fixed for the page comes
// first, so the model worker computes it once for all the questions of the page.
type (
	pageFacts struct {
		Number int `json:"number"`
		Of     int `json:"of"`
		Width  int `json:"width"`
		Height int `json:"height"`
	}
	entry struct {
		Kind    string  `json:"kind"`
		Score   float64 `json:"score"`
		ID      string  `json:"id"`
		Running *bool   `json:"running_text,omitempty"`
		X0      int     `json:"x0"`
		Y0      int     `json:"y0"`
		X1      int     `json:"x1"`
		Y1      int     `json:"y1"`
		Start   string  `json:"start"`
		End     string  `json:"end"`
	}
	mistake struct {
		Early string `json:"read_too_early"`
		First string `json:"should_come_first"`
	}
	// fixedState is the same for every question of one pass over a page.
	fixedState struct {
		Page     pageFacts `json:"page"`
		Blocks   []entry   `json:"blocks"`
		Mistakes []mistake `json:"earlier_attempt_mistakes,omitempty"`
	}
	// movingState changes with every block read.
	movingState struct {
		Read     []string `json:"read_so_far"`
		JustRead string   `json:"text_just_read,omitempty"`
		Next     []string `json:"may_come_next"`
	}
	// wholeState is both parts as one state, for a question that shares nothing with the others.
	wholeState struct {
		fixedState
		movingState
	}
)

// Settle orders the body blocks of one page. The model is asked which block comes next, and the answer is the most
// probable of the blocks the first rule allows. The first pass shares the fixed part of the state between its
// questions, which the model worker then computes once. A page whose order breaks a rule is decided once more with
// its violations in the state and every question computed whole.
func Settle(page, pageCount int, width, height float64, turns int, blocks []doc.Block, images []string, decider Decider) (Result, error) {
	var flow []doc.Block
	var separators []doc.Box
	for _, block := range blocks {
		switch {
		case block.Role == doc.RoleBody:
			flow = append(flow, block)
		case block.Kind == "picture" || block.Kind == "table":
			separators = append(separators, block.Box.Turn(turns, width, height))
		}
	}
	boxes := make([]doc.Box, len(flow))
	fixed := fixedState{Page: pageFacts{Number: page, Of: pageCount, Width: int(width), Height: int(height)}, Blocks: make([]entry, len(flow))}
	for i, block := range flow {
		boxes[i] = block.Box.Turn(turns, width, height)
		start, end := Snippet(block.Text, snippetLength)
		fixed.Blocks[i] = entry{
			Kind: block.Kind, Score: math.Round(block.Score*100) / 100, ID: Key(block.ID), Running: block.Running,
			X0: int(block.Box.X0), Y0: int(block.Box.Y0), X1: int(block.Box.X1), Y1: int(block.Box.Y1), Start: start, End: end,
		}
	}
	result := Result{MinConfidence: 1}
	var violations []Violation
	var order []int
	for pass := 1; pass <= 2; pass++ {
		result.Passes = pass
		fixed.Mistakes = nil
		for _, violation := range violations {
			fixed.Mistakes = append(fixed.Mistakes, mistake{Early: Key(violation.ReadFirst), First: Key(violation.ShouldLead)})
		}
		var err error
		order, err = walk(fixed, flow, boxes, pass > 1, images, decider, &result)
		if err != nil {
			return result, err
		}
		orderedBoxes := make([]doc.Box, len(order))
		ids := make([]int, len(order))
		for i, position := range order {
			orderedBoxes[i], ids[i] = boxes[position], flow[position].ID
		}
		violations = Check(orderedBoxes, ids, separators)
		if len(violations) == 0 {
			break
		}
	}
	result.Violations = violations
	result.Review = len(violations) > 0
	for _, position := range order {
		result.Order = append(result.Order, flow[position].ID)
	}
	return result, nil
}

// walk reads the page once. Every question is the same and offers every body block; what moves is the end of the
// state: the blocks read so far and the blocks the first rule allows next. With whole, a question's state is sent as
// one part.
func walk(fixed fixedState, flow []doc.Block, boxes []doc.Box, whole bool, images []string, decider Decider, result *Result) ([]int, error) {
	options := make([]Option, len(flow))
	for i, block := range flow {
		options[i] = Option{Key: Key(block.ID)}
	}
	names := func(positions []int) []string {
		list := make([]string, len(positions))
		for i, position := range positions {
			list[i] = Key(flow[position].ID)
		}
		return list
	}
	read := make([]bool, len(flow))
	order := make([]int, 0, len(flow))
	for len(order) < len(flow) {
		var ready []int
		for i := range flow {
			if !read[i] && leads(i, read, boxes) {
				ready = append(ready, i)
			}
		}
		if len(ready) == 0 { // Boxes that tie in height leave no block with every predecessor read.
			for i := range flow {
				if !read[i] {
					ready = []int{i}
					break
				}
			}
		}
		next := ready[0]
		if len(ready) > 1 {
			moving := movingState{Read: names(order), Next: names(ready)}
			if len(order) > 0 {
				start, end := Snippet(flow[order[len(order)-1]].Text, snippetLength)
				moving.JustRead = end
				if end == "" {
					moving.JustRead = start
				}
			}
			var first, rest any = fixed, moving
			if whole {
				first, rest = wholeState{fixed, moving}, nil
			}
			probabilities, err := decider.Rank(first, rest, nextQuestion(len(fixed.Mistakes) > 0), options, images)
			if err != nil {
				return nil, err
			}
			for _, position := range ready {
				if probabilities[Key(flow[position].ID)] > probabilities[Key(flow[next].ID)] {
					next = position
				}
			}
			result.Asked++
			result.MinConfidence = min(result.MinConfidence, probabilities[Key(flow[next].ID)])
		}
		read[next] = true
		order = append(order, next)
	}
	return order, nil
}

// leads reports whether every block the first rule puts before block i has been read.
func leads(i int, read []bool, boxes []doc.Box) bool {
	for j := range boxes {
		if j != i && !read[j] && comesAbove(boxes[j], boxes[i]) {
			return false
		}
	}
	return true
}

func nextQuestion(retry bool) string {
	question := "Which block comes next when a person reads this page in its natural reading order? Coordinates are page points from the top left. " +
		"The state lists the blocks read so far and the blocks that may come next."
	if retry {
		question += " An earlier attempt read a block before a block it should follow; the state lists those pairs."
	}
	return question
}

// Key names a block in decision states and options.
func Key(id int) string { return fmt.Sprintf("b%d", id) }

// Snippet gives the first and last characters of a text.
func Snippet(text string, length int) (string, string) {
	runes := []rune(strings.Join(strings.Fields(text), " "))
	if len(runes) <= length {
		return string(runes), ""
	}
	return string(runes[:length]), string(runes[max(len(runes)-length, length):])
}

// PageInfo gives the facts about a page that open a decision state.
func PageInfo(page, pageCount int, width, height float64) map[string]any {
	return map[string]any{"number": page, "of": pageCount, "width": int(width), "height": int(height)}
}

// Judged gives what earlier stages found about a block: its Heron kind and score, and whether it was judged running text.
func Judged(block doc.Block) map[string]any {
	judged := map[string]any{"kind": block.Kind, "score": math.Round(block.Score*100) / 100}
	if block.Running != nil {
		judged["running_text"] = *block.Running
	}
	return judged
}

// Facts gives what every decision receives about a block: its name and box in page points, and what was judged about it.
func Facts(block doc.Block) map[string]any {
	facts := Judged(block)
	facts["id"] = Key(block.ID)
	facts["x0"], facts["y0"], facts["x1"], facts["y1"] = int(block.Box.X0), int(block.Box.Y0), int(block.Box.X1), int(block.Box.Y1)
	return facts
}

func blockEntries(blocks []doc.Block) []map[string]any {
	entries := make([]map[string]any, len(blocks))
	for i, block := range blocks {
		start, end := Snippet(block.Text, snippetLength)
		entries[i] = Facts(block)
		entries[i]["start"], entries[i]["end"] = start, end
	}
	return entries
}
