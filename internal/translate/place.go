package translate

import (
	"fmt"

	"dplus/internal/doc"
	"dplus/internal/order"
)

const (
	// atStart is the option for the place before the document's first paragraph.
	atStart = "start"
	// noTarget is the option for a caption that describes none of its page's tables and pictures.
	noTarget = "none"
	// optionSnippet is how many characters of an element its option shows.
	optionSnippet = 60
)

// element is what is placed as one: a table or picture with the captions that describe it, or a lone inserted element.
type element struct {
	label   string
	members []doc.Block // Captions come before what they describe.
}

func (e element) paragraphs() []Paragraph {
	out := make([]Paragraph, len(e.members))
	for i, block := range e.members {
		out[i] = Paragraph{
			Kind: block.Kind, Score: block.Score, Role: doc.RoleInserted, Pages: []int{block.Page}, Blocks: []int{block.ID},
			Source: block.Text, Image: block.Image,
		}
	}
	return out
}

// place decides where each inserted element goes, now that the paragraphs are settled, and returns the serialized text.
func place(pages []Page, pageCount int, body []Paragraph, decider Decider, decisions *Decisions) ([]Paragraph, error) {
	after := make([][]element, len(body)+1) // Slot 0 comes before the first paragraph, slot i+1 after paragraph i.
	for _, page := range pages {
		elements, err := group(page, pageCount, decider, decisions)
		if err != nil {
			return nil, err
		}
		if len(elements) == 0 {
			continue
		}
		slots, err := chooseSlots(page, pageCount, elements, body, decider, decisions)
		if err != nil {
			return nil, err
		}
		for i, slot := range slots {
			after[slot] = append(after[slot], elements[i])
		}
	}
	output := make([]Paragraph, 0, len(body))
	for slot, items := range after {
		ordered, err := orderSlot(items, slot, body, decider, decisions)
		if err != nil {
			return nil, err
		}
		for _, item := range ordered {
			output = append(output, item.paragraphs()...)
		}
		if slot < len(body) {
			output = append(output, body[slot])
		}
	}
	return output, nil
}

// group turns a page's inserted blocks into what is placed, each caption together with what it describes.
func group(page Page, pageCount int, decider Decider, decisions *Decisions) ([]element, error) {
	targetOf, err := matchCaptions(page, pageCount, decider, decisions)
	if err != nil {
		return nil, err
	}
	captions := map[int][]doc.Block{}
	for _, block := range page.Inserted {
		if target, matched := targetOf[block.ID]; matched {
			captions[target] = append(captions[target], block)
		}
	}
	var elements []element
	for _, block := range page.Inserted {
		if _, matched := targetOf[block.ID]; matched {
			continue
		}
		elements = append(elements, element{
			label:   fmt.Sprintf("e%d_%d", page.Number, len(elements)+1),
			members: append(append([]doc.Block(nil), captions[block.ID]...), block),
		})
	}
	return elements, nil
}

// matchCaptions asks in one request which of the page's tables and pictures each caption describes. It returns the
// block ID of the target of each matched caption.
func matchCaptions(page Page, pageCount int, decider Decider, decisions *Decisions) (map[int]int, error) {
	var captions, targets []doc.Block
	for _, block := range page.Inserted {
		switch block.Kind {
		case "caption":
			captions = append(captions, block)
		case "table", "picture":
			targets = append(targets, block)
		}
	}
	if len(captions) == 0 || len(targets) == 0 {
		return nil, nil
	}
	options := make([]order.Option, 0, len(targets)+1)
	targetIDs := make(map[string]int, len(targets))
	for _, target := range targets {
		key := order.Key(target.ID)
		options = append(options, order.Option{Key: key, Description: summary(target)})
		targetIDs[key] = target.ID
	}
	options = append(options, order.Option{Key: noTarget, Description: "None of these"})
	questions := make(map[string]order.Question, len(captions))
	for _, caption := range captions {
		key := order.Key(caption.ID)
		questions[key] = order.Question{Text: fmt.Sprintf("Which table or picture does caption %s describe?", key), Options: options}
	}
	state := map[string]any{
		"page":                order.PageInfo(page.Number, pageCount, page.Width, page.Height),
		"captions":            blockEntries(captions),
		"tables_and_pictures": blockEntries(targets),
	}
	answers, err := decider.ChooseEach(state, questions)
	if err != nil {
		return nil, err
	}
	decisions.Caption += len(questions)
	targetOf := make(map[int]int, len(captions))
	for _, caption := range captions {
		choice := answers[order.Key(caption.ID)]
		if choice == noTarget {
			continue
		}
		target, offered := targetIDs[choice]
		if !offered {
			return nil, fmt.Errorf("the decision model chose %q for a caption on page %d, which was not offered", choice, page.Number)
		}
		targetOf[caption.ID] = target
	}
	return targetOf, nil
}

// chooseSlots asks in one request which paragraph each of the page's elements follows. The options are the paragraphs
// on the page, and the state holds their full text and the elements'.
func chooseSlots(page Page, pageCount int, elements []element, body []Paragraph, decider Decider, decisions *Decisions) ([]int, error) {
	var options []order.Option
	slotOf := map[string]int{}
	if len(body) == 0 || page.Number <= body[0].Pages[0] {
		options = append(options, order.Option{Key: atStart, Description: "Before the document's first paragraph"})
		slotOf[atStart] = 0
	}
	near := onThePage(page.Number, body)
	paragraphs := make([]map[string]any, len(near))
	for rank, index := range near {
		label := fmt.Sprintf("p%03d", rank+1)
		options = append(options, order.Option{Key: label, Description: "After paragraph " + label})
		slotOf[label] = index + 1
		paragraphs[rank] = paragraphEntry(label, body[index])
	}
	slots := make([]int, len(elements))
	if len(options) <= 1 {
		slot := followingSlot(page.Number, body)
		if len(options) == 1 {
			slot = slotOf[options[0].Key]
		}
		for i := range slots {
			slots[i] = slot
		}
		return slots, nil
	}
	questions := make(map[string]order.Question, len(elements))
	entries := make([]map[string]any, len(elements))
	for i, item := range elements {
		questions[item.label] = order.Question{Text: fmt.Sprintf("Which place in the text does element %s belong to?", item.label), Options: options}
		entries[i] = elementEntry(item)
	}
	state := map[string]any{
		"page": order.PageInfo(page.Number, pageCount, page.Width, page.Height), "elements": entries, "paragraphs": paragraphs,
	}
	answers, err := decider.ChooseEach(state, questions)
	if err != nil {
		return nil, err
	}
	decisions.Placement += len(questions)
	for i, item := range elements {
		slot, offered := slotOf[answers[item.label]]
		if !offered {
			return nil, fmt.Errorf("the decision model chose %q for an element on page %d, which was not offered", answers[item.label], page.Number)
		}
		slots[i] = slot
	}
	return slots, nil
}

// onThePage gives the indexes of the paragraphs with text on the page.
func onThePage(number int, body []Paragraph) []int {
	var found []int
	for i, paragraph := range body {
		for _, page := range paragraph.Pages {
			if page == number {
				found = append(found, i)
				break
			}
		}
	}
	return found
}

// followingSlot is the place after the text read before the page, for an element on a page with no paragraph.
func followingSlot(number int, body []Paragraph) int {
	slot := 0
	for i, paragraph := range body {
		if paragraph.Pages[0] < number {
			slot = i + 1
		}
	}
	return slot
}

// orderSlot asks which element comes next until the elements that share a place are in order.
func orderSlot(items []element, slot int, body []Paragraph, decider Decider, decisions *Decisions) ([]element, error) {
	if len(items) < 2 {
		return items, nil
	}
	remaining := append([]element(nil), items...)
	entries := make([]map[string]any, len(items))
	for i, item := range items {
		entries[i] = elementEntry(item)
	}
	state := map[string]any{"elements": entries}
	if slot > 0 {
		state["paragraph_before"] = body[slot-1].Source
	}
	var ordered []element
	placed := []string{}
	for len(remaining) > 1 {
		options := make([]order.Option, len(remaining))
		position := make(map[string]int, len(remaining))
		for i, item := range remaining {
			options[i] = order.Option{Key: item.label, Description: summary(item.members[0])}
			position[item.label] = i
		}
		state["placed_so_far"] = placed
		choice, _, err := decider.Choose(state, "Which element comes next at this place in the text?", options, nil)
		if err != nil {
			return nil, err
		}
		decisions.SamePlace++
		next, offered := position[choice]
		if !offered {
			return nil, fmt.Errorf("the decision model chose %q as the next element, which was not offered", choice)
		}
		ordered = append(ordered, remaining[next])
		placed = append(placed, choice)
		remaining = append(remaining[:next], remaining[next+1:]...)
	}
	return append(ordered, remaining[0]), nil
}

func summary(block doc.Block) string {
	start, _ := order.Snippet(block.Text, optionSnippet)
	if start == "" {
		return fmt.Sprintf("%s on page %d", block.Kind, block.Page)
	}
	return fmt.Sprintf("%s on page %d: %s", block.Kind, block.Page, start)
}

// blockEntries lists blocks with what was judged about them and their full text.
func blockEntries(blocks []doc.Block) []map[string]any {
	entries := make([]map[string]any, len(blocks))
	for i, block := range blocks {
		entries[i] = side(block, block.Text)
	}
	return entries
}

func elementEntry(item element) map[string]any {
	return map[string]any{"id": item.label, "members": blockEntries(item.members)}
}

func paragraphEntry(label string, paragraph Paragraph) map[string]any {
	blocks := make([]map[string]any, len(paragraph.parts))
	for i, part := range paragraph.parts {
		blocks[i] = order.Judged(part)
	}
	return map[string]any{"id": label, "pages": paragraph.Pages, "blocks": blocks, "text": paragraph.Source}
}
