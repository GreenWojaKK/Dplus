package translate

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"dplus/internal/doc"
	"dplus/internal/order"
)

// script answers decisions from fixed answers and records what was asked.
type script struct {
	yes     []float64                   // Probabilities in question order, across requests. No otherwise.
	choices map[string]string           // By question key. The first option otherwise.
	next    func([]order.Option) string // Picks among elements at one place. The first option otherwise.

	asked          int
	yesStates      []any
	yesRequests    []map[string]string
	chooseStates   []any
	chooseRequests []map[string]order.Question
	nextCalls      [][]order.Option
}

func (s *script) YesNoEach(state any, questions map[string]string) (map[string]float64, error) {
	s.yesStates, s.yesRequests = append(s.yesStates, state), append(s.yesRequests, questions)
	keys := make([]string, 0, len(questions))
	for key := range questions {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	answers := make(map[string]float64, len(keys))
	for _, key := range keys {
		if s.asked < len(s.yes) {
			answers[key] = s.yes[s.asked]
		}
		s.asked++
	}
	return answers, nil
}

func (s *script) ChooseEach(state any, questions map[string]order.Question) (map[string]string, error) {
	s.chooseStates, s.chooseRequests = append(s.chooseStates, state), append(s.chooseRequests, questions)
	answers := make(map[string]string, len(questions))
	for key, question := range questions {
		answers[key] = question.Options[0].Key
		if choice, given := s.choices[key]; given {
			answers[key] = choice
		}
	}
	return answers, nil
}

func (s *script) Choose(_ any, _ string, options []order.Option, _ []string) (string, float64, error) {
	s.nextCalls = append(s.nextCalls, options)
	if s.next != nil {
		return s.next(options), 1, nil
	}
	return options[0].Key, 1, nil
}

func text(id, page int, kind, value string) doc.Block {
	return doc.Block{ID: id, Page: page, Kind: kind, Role: doc.RoleOf(kind), Text: value}
}

func inserted(id, page int, kind, value string) doc.Block {
	block := text(id, page, kind, value)
	block.Role = doc.RoleInserted
	return block
}

func onPage(number int, body []doc.Block, elements ...doc.Block) Page {
	return Page{Number: number, Width: 595, Height: 842, Body: body, Inserted: elements}
}

func heading(id, page int, value string) []doc.Block {
	return []doc.Block{text(id, page, "section_header", value)}
}

func sources(paragraphs []Paragraph) []string {
	out := make([]string, len(paragraphs))
	for i, paragraph := range paragraphs {
		out[i] = paragraph.Source
	}
	return out
}

func optionKeys(options []order.Option) []string {
	keys := make([]string, len(options))
	for i, option := range options {
		keys[i] = option.Key
	}
	return keys
}

func wantSources(t *testing.T, paragraphs []Paragraph, want ...string) {
	t.Helper()
	if got := sources(paragraphs); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestGatherJoinsContinuingTextAcrossPages(t *testing.T) {
	pages := []Page{
		onPage(1, []doc.Block{text(1, 1, "section_header", "Heading"), text(2, 1, "text", "The first half")}),
		onPage(2, []doc.Block{text(1, 2, "text", "of a sentence."), text(2, 2, "text", "A new paragraph.")}),
	}
	paragraphs, decisions, err := Gather(pages, 2, &script{yes: []float64{0.9, 0.1}})
	if err != nil {
		t.Fatal(err)
	}
	if decisions.Continuation != 2 || len(paragraphs) != 3 {
		t.Fatalf("decisions %+v, got %+v", decisions, paragraphs)
	}
	joined := paragraphs[1]
	if joined.Source != "The first half of a sentence." || !reflect.DeepEqual(joined.Pages, []int{1, 2}) || !reflect.DeepEqual(joined.Blocks, []int{2, 1}) {
		t.Fatalf("got %+v", joined)
	}
}

func TestGatherJoinsEastAsianTextDirectly(t *testing.T) {
	pages := []Page{onPage(1, []doc.Block{text(1, 1, "text", "문장의 앞"), text(2, 1, "text", "부분")})}
	paragraphs, _, err := Gather(pages, 1, &script{yes: []float64{0.8}})
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "문장의 앞부분")
}

func TestEveryBoundaryIsAskedExceptAroundHeadings(t *testing.T) {
	pages := []Page{onPage(1, []doc.Block{
		text(1, 1, "title", "Title"), text(2, 1, "text", "A"), text(3, 1, "list_item", "B"), text(4, 1, "formula", "C"),
		text(5, 1, "section_header", "D"), text(6, 1, "code", "E"), text(7, 1, "text", "F"),
	})}
	_, decisions, err := Gather(pages, 1, &script{})
	if err != nil {
		t.Fatal(err)
	}
	if decisions.Continuation != 3 { // A to B, B to C and E to F.
		t.Fatalf("got %+v", decisions)
	}
}

func TestOnePageIsOneRequestAndTheBoundaryIntoAPageBelongsToIt(t *testing.T) {
	pages := []Page{
		onPage(1, []doc.Block{text(1, 1, "text", "A"), text(2, 1, "text", "B")}),
		onPage(2, []doc.Block{text(1, 2, "text", "C"), text(2, 2, "text", "D")}),
	}
	s := &script{}
	if _, _, err := Gather(pages, 2, s); err != nil {
		t.Fatal(err)
	}
	if len(s.yesRequests) != 2 || len(s.yesRequests[0]) != 1 || len(s.yesRequests[1]) != 2 {
		t.Fatalf("got %v", s.yesRequests)
	}
	boundaries := s.yesStates[1].(map[string]any)["boundaries"].([]map[string]any)
	if boundaries[0]["page_break_between"] != true || boundaries[1]["page_break_between"] != false {
		t.Fatalf("got %v", boundaries)
	}
	if boundaries[0]["before"].(map[string]any)["page"] != 1 || boundaries[0]["before"].(map[string]any)["text"] != "B" {
		t.Fatalf("got %v", boundaries[0])
	}
}

func TestBoundariesCarryKindsAndScores(t *testing.T) {
	first, second := text(1, 1, "list_item", "One"), text(2, 1, "text", "Two")
	first.Score, second.Score = 0.876, 0.5
	running := true
	second.Running = &running
	s := &script{}
	if _, _, err := Gather([]Page{onPage(1, []doc.Block{first, second})}, 1, s); err != nil {
		t.Fatal(err)
	}
	boundary := s.yesStates[0].(map[string]any)["boundaries"].([]map[string]any)[0]
	before, after := boundary["before"].(map[string]any), boundary["after"].(map[string]any)
	if before["kind"] != "list_item" || before["score"] != 0.88 || before["text"] != "One" {
		t.Fatalf("before %v", before)
	}
	if after["kind"] != "text" || after["score"] != 0.5 || after["running_text"] != true || after["text"] != "Two" {
		t.Fatalf("after %v", after)
	}
}

func TestAFormulaContinuedByProseIsTranslated(t *testing.T) {
	pages := []Page{onPage(1, []doc.Block{text(1, 1, "formula", "x = 1"), text(2, 1, "text", "where x is one.")})}
	paragraphs, _, err := Gather(pages, 1, &script{yes: []float64{0.9}})
	if err != nil {
		t.Fatal(err)
	}
	if len(paragraphs) != 1 || !Translatable(paragraphs[0].Kind) {
		t.Fatalf("got %+v", paragraphs)
	}
}

func TestAnElementGoesAfterTheParagraphTheModelChose(t *testing.T) {
	table := inserted(3, 1, "table", "A table")
	table.Box = doc.Box{X0: 50, Y0: 20, X1: 500, Y1: 60} // Above both paragraphs on the page.
	body := []doc.Block{text(1, 1, "section_header", "One"), text(2, 1, "section_header", "Two")}
	body[0].Box = doc.Box{X0: 50, Y0: 100, X1: 500, Y1: 140}
	body[1].Box = doc.Box{X0: 50, Y0: 300, X1: 500, Y1: 340}
	paragraphs, decisions, err := Gather([]Page{onPage(1, body, table)}, 1, &script{choices: map[string]string{"e1_1": "p002"}})
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "One", "Two", "A table")
	if decisions.Placement != 1 || paragraphs[0].Role != doc.RoleBody || paragraphs[2].Role != doc.RoleInserted {
		t.Fatalf("decisions %+v, got %+v", decisions, paragraphs)
	}
}

func TestAFootnoteDoesNotBreakASentenceAcrossPages(t *testing.T) {
	pages := []Page{
		onPage(1, []doc.Block{text(1, 1, "text", "The first half")}, inserted(2, 1, "footnote", "A note")),
		onPage(2, []doc.Block{text(1, 2, "text", "of a sentence.")}),
	}
	paragraphs, _, err := Gather(pages, 2, &script{yes: []float64{0.9}, choices: map[string]string{"e1_1": "p001"}})
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "The first half of a sentence.", "A note")
}

func TestPlacesAreTheParagraphsOnThePage(t *testing.T) {
	continuing := []doc.Block{text(1, 2, "text", "The first half"), text(2, 2, "text", "Second")}
	pages := []Page{
		onPage(1, heading(1, 1, "H1")),
		onPage(2, continuing, inserted(3, 2, "table", "A table")),
		onPage(3, []doc.Block{text(1, 3, "text", "of a sentence."), text(2, 3, "section_header", "H3")}),
	}
	s := &script{yes: []float64{0.1, 0.9}} // Second continues onto page 3.
	if _, _, err := Gather(pages, 3, s); err != nil {
		t.Fatal(err)
	}
	if got := optionKeys(s.chooseRequests[0]["e2_1"].Options); !reflect.DeepEqual(got, []string{"p001", "p002"}) {
		t.Fatalf("options %v", got)
	}
	listed := s.chooseStates[0].(map[string]any)["paragraphs"].([]map[string]any)
	if listed[0]["text"] != "The first half" || listed[1]["text"] != "Second of a sentence." {
		t.Fatalf("paragraphs %v", listed)
	}
}

func TestOnlyTheFirstPageOffersThePlaceBeforeTheDocument(t *testing.T) {
	pages := []Page{
		onPage(1, heading(1, 1, "H1"), inserted(2, 1, "text", "An author")),
		onPage(2, []doc.Block{text(1, 2, "section_header", "H2"), text(2, 2, "section_header", "H3")}, inserted(3, 2, "table", "A table")),
	}
	s := &script{choices: map[string]string{"e2_1": "p002"}}
	paragraphs, _, err := Gather(pages, 2, s)
	if err != nil {
		t.Fatal(err)
	}
	if got := optionKeys(s.chooseRequests[0]["e1_1"].Options); !reflect.DeepEqual(got, []string{atStart, "p001"}) {
		t.Fatalf("first page options %v", got)
	}
	if got := optionKeys(s.chooseRequests[1]["e2_1"].Options); !reflect.DeepEqual(got, []string{"p001", "p002"}) {
		t.Fatalf("second page options %v", got)
	}
	wantSources(t, paragraphs, "An author", "H1", "H2", "H3", "A table")
}

func TestAnElementWithOnePlaceOnItsPageIsNotAsked(t *testing.T) {
	pages := []Page{onPage(1, heading(1, 1, "H1")), onPage(2, heading(1, 2, "H2"), inserted(2, 2, "table", "A table"))}
	s := &script{}
	paragraphs, decisions, err := Gather(pages, 2, s)
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "H1", "H2", "A table")
	if decisions.Placement != 0 {
		t.Fatalf("got %+v", decisions)
	}
}

func TestPlacementStateHoldsFullTextAndWhatWasJudged(t *testing.T) {
	long := strings.Repeat("a long paragraph ", 40)
	body := text(1, 1, "section_header", long)
	body.Score = 0.9
	table := inserted(2, 1, "table", strings.Repeat("cell ", 80))
	table.Score = 0.7
	notRunning := false
	author := inserted(3, 1, "text", "An author")
	author.Running = &notRunning
	s := &script{}
	if _, _, err := Gather([]Page{onPage(1, []doc.Block{body}, table, author)}, 1, s); err != nil {
		t.Fatal(err)
	}
	if len(s.chooseRequests) != 1 || len(s.chooseRequests[0]) != 2 {
		t.Fatalf("requests %v", s.chooseRequests)
	}
	state := s.chooseStates[0].(map[string]any)
	paragraph := state["paragraphs"].([]map[string]any)[0]
	if paragraph["text"] != long {
		t.Fatalf("paragraph %v", paragraph)
	}
	judged := paragraph["blocks"].([]map[string]any)[0]
	if judged["kind"] != "section_header" || judged["score"] != 0.9 {
		t.Fatalf("judged %v", judged)
	}
	elements := state["elements"].([]map[string]any)
	first := elements[0]["members"].([]map[string]any)[0]
	if first["text"] != table.Text || first["kind"] != "table" || first["score"] != 0.7 {
		t.Fatalf("first element %v", first)
	}
	if second := elements[1]["members"].([]map[string]any)[0]; second["running_text"] != false {
		t.Fatalf("second element %v", second)
	}
}

func TestACaptionMovesWithWhatItDescribes(t *testing.T) {
	page := onPage(1, heading(1, 1, "Heading"), inserted(2, 1, "table", "A table"), inserted(3, 1, "caption", "Table 1"))
	s := &script{choices: map[string]string{"b3": "b2", "e1_1": "p001"}}
	paragraphs, decisions, err := Gather([]Page{page}, 1, s)
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "Heading", "Table 1", "A table")
	if decisions.Caption != 1 || decisions.Placement != 1 {
		t.Fatalf("got %+v", decisions)
	}
	if got := optionKeys(s.chooseRequests[0]["b3"].Options); !reflect.DeepEqual(got, []string{"b2", noTarget}) {
		t.Fatalf("options %v", got)
	}
}

func TestACaptionOfNothingOnItsPageIsPlacedAlone(t *testing.T) {
	page := onPage(1, heading(1, 1, "Heading"), inserted(2, 1, "picture", ""), inserted(3, 1, "caption", "Figure 1"))
	s := &script{choices: map[string]string{"b3": noTarget}}
	_, decisions, err := Gather([]Page{page}, 1, s)
	if err != nil {
		t.Fatal(err)
	}
	if decisions.Caption != 1 || decisions.Placement != 2 {
		t.Fatalf("got %+v", decisions)
	}
}

func TestACaptionWithNoTableOrPictureOnItsPageIsNotMatched(t *testing.T) {
	page := onPage(1, heading(1, 1, "Heading"), inserted(2, 1, "caption", "Figure 1"))
	_, decisions, err := Gather([]Page{page}, 1, &script{})
	if err != nil {
		t.Fatal(err)
	}
	if decisions.Caption != 0 || decisions.Placement != 1 {
		t.Fatalf("got %+v", decisions)
	}
}

func TestElementsAtOnePlaceAreOrderedByTheModel(t *testing.T) {
	page := onPage(1, heading(1, 1, "Heading"), inserted(2, 1, "footnote", "A"), inserted(3, 1, "footnote", "B"), inserted(4, 1, "footnote", "C"))
	s := &script{
		choices: map[string]string{"e1_1": "p001", "e1_2": "p001", "e1_3": "p001"},
		next:    func(options []order.Option) string { return options[len(options)-1].Key },
	}
	paragraphs, decisions, err := Gather([]Page{page}, 1, s)
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "Heading", "C", "B", "A")
	if decisions.SamePlace != 2 || len(s.nextCalls[0]) != 3 || len(s.nextCalls[1]) != 2 {
		t.Fatalf("decisions %+v, calls %v", decisions, s.nextCalls)
	}
}

func TestAPictureIsPlacedWithItsImage(t *testing.T) {
	picture := inserted(2, 1, "picture", "")
	picture.Image = "paper.pictures/page-1-block-2.png"
	page := onPage(1, heading(1, 1, "Heading"), picture)
	paragraphs, _, err := Gather([]Page{page}, 1, &script{choices: map[string]string{"e1_1": "p001"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(paragraphs) != 2 || paragraphs[1].Kind != "picture" || paragraphs[1].Role != doc.RoleInserted || paragraphs[1].Image != picture.Image {
		t.Fatalf("got %+v", paragraphs)
	}
}

func TestAnElementWithNoParagraphNearFollowsTheTextReadBeforeIt(t *testing.T) {
	pages := []Page{onPage(1, heading(1, 1, "H1")), onPage(4, nil, inserted(1, 4, "table", "A table")), onPage(7, heading(1, 7, "H7"))}
	s := &script{}
	paragraphs, decisions, err := Gather(pages, 7, s)
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "H1", "A table", "H7")
	if decisions.Placement != 0 || len(s.chooseRequests) != 0 {
		t.Fatalf("decisions %+v", decisions)
	}
}

func TestADocumentWithNoParagraphsKeepsItsElements(t *testing.T) {
	paragraphs, _, err := Gather([]Page{onPage(1, nil, inserted(1, 1, "table", "A table"))}, 1, &script{})
	if err != nil {
		t.Fatal(err)
	}
	wantSources(t, paragraphs, "A table")
}

func TestAPlaceNotOfferedIsAnError(t *testing.T) {
	page := onPage(1, heading(1, 1, "Heading"), inserted(2, 1, "table", "A table"))
	if _, _, err := Gather([]Page{page}, 1, &script{choices: map[string]string{"e1_1": "p999"}}); err == nil {
		t.Fatal("expected an error")
	}
}

func TestACaptionTargetNotOfferedIsAnError(t *testing.T) {
	page := onPage(1, heading(1, 1, "Heading"), inserted(2, 1, "table", "A table"), inserted(3, 1, "caption", "Table 1"))
	if _, _, err := Gather([]Page{page}, 1, &script{choices: map[string]string{"b3": "b9"}}); err == nil {
		t.Fatal("expected an error")
	}
}

func TestTranslatable(t *testing.T) {
	for kind, want := range map[string]bool{"text": true, "table": true, "code": false, "formula": false, "picture": false} {
		if Translatable(kind) != want {
			t.Errorf("%s", kind)
		}
	}
}

func TestTokenBudget(t *testing.T) {
	if TokenBudget("짧음") != minTokens || TokenBudget(string(make([]rune, 2000))) != maxTokens {
		t.Fatal("budget out of range")
	}
	if got := TokenBudget(string(make([]rune, 100))); got != 400 {
		t.Fatalf("got %d", got)
	}
}
