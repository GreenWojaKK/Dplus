package order

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"dplus/internal/doc"
)

func block(id int, kind string, x0, y0, x1, y1 float64) doc.Block {
	return doc.Block{ID: id, Kind: kind, Role: doc.RoleOf(kind), Box: doc.Box{X0: x0, Y0: y0, X1: x1, Y1: y1}, Text: "text"}
}

// scripted gives all the probability to the block the callback picks among those that may come next.
type scripted struct {
	calls [][]string // The blocks that may come next, for each question.
	fixed []any
	whole []bool
	pick  func(allowed []string, call int) string
}

func (s *scripted) Rank(fixed, moving any, _ string, options []Option, _ []string) (map[string]float64, error) {
	var allowed []string
	if state, whole := fixed.(wholeState); whole {
		allowed = state.Next
	} else {
		allowed = moving.(movingState).Next
	}
	s.calls = append(s.calls, allowed)
	s.fixed = append(s.fixed, fixed)
	s.whole = append(s.whole, moving == nil)
	return map[string]float64{s.pick(allowed, len(s.calls)-1): 1}, nil
}

func first(allowed []string, _ int) string { return allowed[0] }

// inOrder picks the allowed block that comes first in a list of preferences.
func inOrder(preferences ...string) func([]string, int) string {
	return func(allowed []string, _ int) string {
		for _, want := range preferences {
			for _, name := range allowed {
				if name == want {
					return want
				}
			}
		}
		return allowed[0]
	}
}

func TestSingleColumnNeedsNoDecision(t *testing.T) {
	blocks := []doc.Block{
		block(1, "title", 50, 50, 500, 80), block(2, "text", 50, 100, 500, 200), block(3, "text", 50, 220, 500, 300),
	}
	decider := &scripted{pick: first}
	result, err := Settle(1, 1, 595, 842, 0, blocks, nil, decider)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Order, []int{1, 2, 3}) || result.Asked != 0 || result.Review {
		t.Fatalf("got %+v", result)
	}
}

func TestTwoColumnsOfferTheNextBlockOfEachColumn(t *testing.T) {
	blocks := []doc.Block{
		block(1, "text", 50, 100, 280, 300), block(2, "text", 320, 100, 550, 300),
		block(3, "text", 50, 320, 280, 500), block(4, "text", 320, 320, 550, 500),
	}
	decider := &scripted{pick: inOrder("b1", "b3", "b2", "b4")} // A reader going down the left column first.
	result, err := Settle(1, 1, 595, 842, 0, blocks, nil, decider)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Order, []int{1, 3, 2, 4}) || result.Review || result.Passes != 1 {
		t.Fatalf("got %+v", result)
	}
	if !reflect.DeepEqual(decider.calls[0], []string{"b1", "b2"}) {
		t.Fatalf("first allowed blocks %v", decider.calls[0])
	}
}

func TestZigzagAcrossColumnsIsCaughtAndAskedAgain(t *testing.T) {
	blocks := []doc.Block{
		block(1, "text", 50, 100, 280, 300), block(2, "text", 320, 100, 550, 300),
		block(3, "text", 50, 320, 280, 500), block(4, "text", 320, 320, 550, 500),
	}
	decider := &scripted{pick: inOrder("b1", "b2", "b3", "b4")}
	result, err := Settle(1, 1, 595, 842, 0, blocks, nil, decider)
	if err != nil {
		t.Fatal(err)
	}
	if result.Passes != 2 || !result.Review || len(result.Violations) == 0 {
		t.Fatalf("got %+v", result)
	}
	// The first pass shares the fixed part of its states; the second sends each state whole, with the mistakes.
	last := len(decider.whole) - 1
	if decider.whole[0] || !decider.whole[last] {
		t.Fatalf("whole states: %v", decider.whole)
	}
	if len(decider.fixed[0].(fixedState).Mistakes) != 0 || len(decider.fixed[last].(wholeState).Mistakes) == 0 {
		t.Fatalf("mistakes: first %+v, last %+v", decider.fixed[0], decider.fixed[last])
	}
}

type ranked map[string]float64

func (r *ranked) Rank(any, any, string, []Option, []string) (map[string]float64, error) {
	return *r, nil
}

func TestFullWidthBlockSeparatesColumnSections(t *testing.T) {
	boxes := []doc.Box{
		{X0: 50, Y0: 100, X1: 280, Y1: 200},  // upper left
		{X0: 320, Y0: 100, X1: 550, Y1: 200}, // upper right
		{X0: 50, Y0: 220, X1: 550, Y1: 260},  // full width
		{X0: 50, Y0: 280, X1: 280, Y1: 400},  // lower left
		{X0: 320, Y0: 280, X1: 550, Y1: 400}, // lower right
	}
	ids := []int{1, 2, 3, 4, 5}
	if found := Check(boxes, ids, nil); len(found) != 0 {
		t.Fatalf("a normal page was flagged: %+v", found)
	}
}

func TestCheckFlagsBothRules(t *testing.T) {
	lower := doc.Box{X0: 50, Y0: 300, X1: 280, Y1: 400}
	upper := doc.Box{X0: 50, Y0: 100, X1: 280, Y1: 200}
	if found := Check([]doc.Box{lower, upper}, []int{7, 8}, nil); !reflect.DeepEqual(found, []Violation{{ReadFirst: 7, ShouldLead: 8}}) {
		t.Fatalf("upper block read second: %+v", found)
	}
	left := doc.Box{X0: 50, Y0: 100, X1: 280, Y1: 200}
	right := doc.Box{X0: 320, Y0: 100, X1: 550, Y1: 200}
	if found := Check([]doc.Box{right, left}, []int{1, 2}, nil); !reflect.DeepEqual(found, []Violation{{ReadFirst: 1, ShouldLead: 2}}) {
		t.Fatalf("right block read first: %+v", found)
	}
}

func TestOnlyBodyBlocksAreOrdered(t *testing.T) {
	blocks := []doc.Block{
		block(1, "page_header", 50, 20, 500, 40), block(2, "footnote", 50, 760, 500, 800),
		block(3, "text", 50, 100, 500, 200), block(4, "page_footer", 50, 810, 500, 830),
		block(5, "caption", 50, 210, 500, 230), block(6, "table", 50, 240, 500, 400),
	}
	result, err := Settle(1, 1, 595, 842, 0, blocks, nil, &scripted{pick: first})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Order, []int{3}) {
		t.Fatalf("got %+v", result)
	}
}

func TestAPictureOrTableSeparatesColumnSections(t *testing.T) {
	boxes := []doc.Box{
		{X0: 50, Y0: 100, X1: 280, Y1: 200},  // upper left
		{X0: 320, Y0: 100, X1: 550, Y1: 200}, // upper right
		{X0: 50, Y0: 280, X1: 280, Y1: 400},  // lower left
		{X0: 320, Y0: 280, X1: 550, Y1: 400}, // lower right
	}
	ids := []int{1, 2, 3, 4}
	if found := Check(boxes, ids, nil); len(found) == 0 {
		t.Fatal("without a separator the right column must follow the whole left column")
	}
	separator := []doc.Box{{X0: 50, Y0: 220, X1: 550, Y1: 260}}
	if found := Check(boxes, ids, separator); len(found) != 0 {
		t.Fatalf("flagged across a separator: %+v", found)
	}
}

type bodyAnswers struct {
	asked map[string]string
	no    string
}

func (b *bodyAnswers) YesNoEach(_ any, questions map[string]string) (map[string]float64, error) {
	b.asked = questions
	answers := map[string]float64{}
	for key := range questions {
		answers[key] = 0.9
	}
	if _, asked := questions[b.no]; asked {
		answers[b.no] = 0.1
	}
	return answers, nil
}

func TestClassifyBodyAsksAboutTextBlocksOnly(t *testing.T) {
	blocks := []doc.Block{
		block(1, "title", 50, 50, 500, 80), block(2, "text", 50, 90, 500, 110),
		block(3, "text", 50, 120, 500, 300), block(4, "footnote", 50, 760, 500, 800),
	}
	decider := &bodyAnswers{no: "b2"}
	asked, err := ClassifyBody(1, 1, 595, 842, blocks, decider)
	if err != nil {
		t.Fatal(err)
	}
	if asked != 2 || len(decider.asked) != 2 {
		t.Fatalf("asked %d: %v", asked, decider.asked)
	}
	if blocks[1].Role != doc.RoleInserted || blocks[2].Role != doc.RoleBody || blocks[0].Role != doc.RoleBody {
		t.Fatalf("got %+v", blocks)
	}
	if blocks[0].Running != nil || *blocks[1].Running || !*blocks[2].Running {
		t.Fatalf("verdicts %v, %v, %v", blocks[0].Running, blocks[1].Running, blocks[2].Running)
	}
}

func TestReadingOrderStateCarriesWhatWasJudged(t *testing.T) {
	running := true
	blocks := []doc.Block{block(1, "text", 50, 100, 280, 300), block(2, "list_item", 320, 100, 550, 300)}
	blocks[0].Score, blocks[0].Running = 0.876, &running
	decider := &scripted{pick: first}
	if _, err := Settle(3, 9, 595, 842, 0, blocks, nil, decider); err != nil {
		t.Fatal(err)
	}
	fixed := decider.fixed[0].(fixedState)
	if fixed.Page != (pageFacts{Number: 3, Of: 9, Width: 595, Height: 842}) {
		t.Fatalf("page %+v", fixed.Page)
	}
	if entry := fixed.Blocks[0]; entry.Kind != "text" || entry.Score != 0.88 || entry.ID != "b1" || entry.Running == nil || !*entry.Running {
		t.Fatalf("got %+v", entry)
	}
	if entry := fixed.Blocks[1]; entry.Running != nil || entry.Kind != "list_item" {
		t.Fatalf("got %+v", entry)
	}
}

func TestTheStateLeadsWithItsFixedPart(t *testing.T) {
	blocks := []doc.Block{
		block(1, "text", 50, 100, 280, 300), block(2, "text", 320, 100, 550, 300), block(3, "text", 50, 320, 280, 500),
	}
	decider := &scripted{pick: inOrder("b1", "b3", "b2")}
	if _, err := Settle(1, 1, 595, 842, 0, blocks, nil, decider); err != nil {
		t.Fatal(err)
	}
	whole, err := json.Marshal(wholeState{decider.fixed[0].(fixedState), movingState{Read: []string{"b1"}, JustRead: "text", Next: []string{"b2", "b3"}}})
	if err != nil {
		t.Fatal(err)
	}
	part, err := json.Marshal(decider.fixed[0])
	if err != nil {
		t.Fatal(err)
	}
	// The model worker finds what to compute once by where the fixed part alone parts from the whole state.
	if !strings.HasPrefix(string(whole), strings.TrimSuffix(string(part), "}")) {
		t.Fatalf("the whole state does not start with its fixed part:\n%s\n%s", whole, part)
	}
	if !strings.HasSuffix(string(whole), `"read_so_far":["b1"],"text_just_read":"text","may_come_next":["b2","b3"]}`) {
		t.Fatalf("got %s", whole)
	}
	if !reflect.DeepEqual(decider.calls, [][]string{{"b1", "b2"}, {"b2", "b3"}}) {
		t.Fatalf("questions %v", decider.calls)
	}
}

func TestTurnedPageIsCheckedUpright(t *testing.T) {
	// Text running up the page: its first line is the leftmost, which the turn puts on top.
	blocks := []doc.Block{block(1, "text", 100, 100, 200, 500), block(2, "text", 300, 100, 400, 500)}
	result, err := Settle(1, 1, 595, 842, 1, blocks, nil, &scripted{pick: first})
	if err != nil {
		t.Fatal(err)
	}
	if result.Asked != 0 || result.Review || !reflect.DeepEqual(result.Order, []int{1, 2}) {
		t.Fatalf("got %+v", result)
	}
}

func TestPicturesAreNeitherOfferedNorJudged(t *testing.T) {
	blocks := []doc.Block{
		block(1, "picture", 50, 60, 90, 100), block(2, "section_header", 110, 60, 400, 100), block(3, "text", 50, 120, 500, 300),
	}
	decider := &scripted{pick: first}
	result, err := Settle(1, 1, 595, 842, 0, blocks, nil, decider)
	if err != nil {
		t.Fatal(err)
	}
	if result.Asked != 0 || result.Review || len(decider.calls) != 0 || !reflect.DeepEqual(result.Order, []int{2, 3}) {
		t.Fatalf("got %+v", result)
	}
}
