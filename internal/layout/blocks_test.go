package layout

import (
	"image"
	"image/color"
	"strings"
	"testing"

	"dplus/internal/doc"
	"dplus/internal/pdf"
)

func char(r rune, x, y float64) doc.Char {
	return doc.Char{Rune: r, Box: doc.Box{X0: x, Y0: y, X1: x + 5, Y1: y + 10}}
}

func TestCleanKeepsContainerOverWhatItHolds(t *testing.T) {
	kept := Clean([]Detection{
		{Kind: "text", Score: 0.99, Box: doc.Box{X0: 10, Y0: 10, X1: 50, Y1: 20}},
		{Kind: "table", Score: 0.6, Box: doc.Box{X0: 0, Y0: 0, X1: 100, Y1: 100}},
	})
	if len(kept) != 1 || kept[0].Kind != "table" {
		t.Fatalf("got %+v", kept)
	}
}

func TestCleanKeepsTheBetterKindOfTheSameBox(t *testing.T) {
	kept := Clean([]Detection{
		{Kind: "table", Score: 0.51, Box: doc.Box{X0: 423, Y0: 20, X1: 518, Y1: 29}},
		{Kind: "page_header", Score: 0.88, Box: doc.Box{X0: 422, Y0: 20, X1: 519, Y1: 29}},
	})
	if len(kept) != 1 || kept[0].Kind != "page_header" {
		t.Fatalf("got %+v", kept)
	}
}

func TestCleanDropsARegionItsParts(t *testing.T) {
	kept := Clean([]Detection{
		{Kind: "key_value_region", Score: 0.7, Box: doc.Box{X0: 0, Y0: 0, X1: 100, Y1: 100}},
		{Kind: "text", Score: 0.9, Box: doc.Box{X0: 5, Y0: 5, X1: 95, Y1: 40}},
		{Kind: "text", Score: 0.85, Box: doc.Box{X0: 5, Y0: 50, X1: 95, Y1: 90}},
	})
	if len(kept) != 2 || kept[0].Kind != "text" || kept[1].Kind != "text" {
		t.Fatalf("got %+v", kept)
	}
}

func TestCleanKeepsHigherScoreOfSameRegion(t *testing.T) {
	kept := Clean([]Detection{
		{Kind: "text", Score: 0.6, Box: doc.Box{X0: 0, Y0: 0, X1: 100, Y1: 50}},
		{Kind: "section_header", Score: 0.9, Box: doc.Box{X0: 2, Y0: 0, X1: 100, Y1: 50}},
		{Kind: "text", Score: 0.8, Box: doc.Box{X0: 0, Y0: 60, X1: 100, Y1: 90}},
	})
	if len(kept) != 2 || kept[0].Kind != "section_header" {
		t.Fatalf("got %+v", kept)
	}
}

func TestBuildAssignsEveryCharacterOnceAndCountsTheRest(t *testing.T) {
	page := &pdf.Page{Number: 3, Width: 200, Height: 200, Chars: []doc.Char{
		char('b', 16, 10), char('a', 10, 10), char('c', 10, 22), char('x', 180, 180),
	}}
	result := Build(page, []Detection{{Kind: "text", Score: 0.9, Box: doc.Box{X0: 5, Y0: 5, X1: 50, Y1: 40}}}, 1, 1)
	if result.Unassigned != 1 || len(result.Blocks) != 1 {
		t.Fatalf("got %+v", result)
	}
	block := result.Blocks[0]
	if block.Text != "ab c" || block.Page != 3 || block.ID != 1 || block.Source != doc.SourceTextLayer {
		t.Fatalf("got %+v", block)
	}
}

func TestBuildScalesPixelsToPoints(t *testing.T) {
	page := &pdf.Page{Number: 1, Chars: []doc.Char{char('a', 100, 100)}}
	result := Build(page, []Detection{{Kind: "text", Score: 0.9, Box: doc.Box{X0: 40, Y0: 40, X1: 60, Y1: 60}}}, 2, 2)
	if len(result.Blocks) != 1 || result.Unassigned != 0 {
		t.Fatalf("got %+v", result)
	}
}

func TestAssembleJoinsLines(t *testing.T) {
	chars := func(text string, y float64) []doc.Char {
		var out []doc.Char
		for i, r := range []rune(text) {
			out = append(out, char(r, float64(i)*5, y))
		}
		return out
	}
	cases := []struct {
		name  string
		lines []string
		table bool
		want  string
	}{
		{"latin lines take a space", []string{"one", "two"}, false, "one two"},
		{"a hyphen at the line end joins the word", []string{"exam-", "ple"}, false, "example"},
		{"a capital after a hyphen stays apart", []string{"pre-", "War"}, false, "pre- War"},
		{"korean lines join directly", []string{"가나", "다라"}, false, "가나다라"},
		{"chinese lines join directly", []string{"发明", "专利"}, false, "发明专利"},
		{"table rows keep their lines", []string{"a", "b"}, true, "a\nb"},
	}
	for _, c := range cases {
		var all []doc.Char
		for i, line := range c.lines {
			all = append(all, chars(line, float64(i)*14)...)
		}
		if got := assemble(all, c.table); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func TestAssembleJoinsAWordTheTextLayerBrokeWithAHyphen(t *testing.T) {
	broken := char('s', 0, 0)
	broken.HyphenEnd = true
	if got := assemble([]doc.Char{broken, char('u', 0, 14)}, false); got != "su" {
		t.Fatalf("got %q", got)
	}
}

func TestAssembleUsesTheTextLayerSpaces(t *testing.T) {
	first := char('a', 0, 0)
	first.SpaceAfter = true
	if got := assemble([]doc.Char{first, char('b', 5, 0)}, false); got != "a b" {
		t.Fatalf("got %q", got)
	}
}

func TestResizeKeepsAFlatImage(t *testing.T) {
	source := image.NewRGBA(image.Rect(0, 0, 595, 842))
	for y := range 842 {
		for x := range 595 {
			source.Set(x, y, color.RGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}
	out := resizeRGB(source, 640, 640)
	if len(out) != 640*640*3 {
		t.Fatalf("length %d", len(out))
	}
	if !strings.EqualFold(string(out[:3]), string([]byte{200, 100, 50})) || out[len(out)-1] != 50 {
		t.Fatalf("got %v and %d", out[:3], out[len(out)-1])
	}
}
