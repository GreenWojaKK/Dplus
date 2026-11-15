package translate

import (
	"reflect"
	"strings"
	"testing"

	"dplus/internal/doc"
)

func TestThePromptNamesTheKindAndItsScore(t *testing.T) {
	caption := SystemPrompt(Paragraph{Kind: "caption", Score: 0.934, Role: doc.RoleInserted, Language: "en"}, "ko")
	if !strings.Contains(caption, "from English into Korean") || !strings.Contains(caption, "plain written style of Korean") ||
		!strings.Contains(caption, "labeled the text caption with a score of 0.93") || !strings.Contains(caption, "label word") {
		t.Fatalf("caption prompt: %s", caption)
	}
	body := SystemPrompt(Paragraph{Kind: "text", Score: 0.9, Role: doc.RoleBody, Language: "zh"}, "ko")
	apart := SystemPrompt(Paragraph{Kind: "text", Score: 0.9, Role: doc.RoleInserted, Language: "zh"}, "ko")
	if !strings.Contains(body, "running text. Translate every sentence") || !strings.Contains(apart, "set apart from the running text") {
		t.Fatalf("text prompts: %s | %s", body, apart)
	}
	if other := SystemPrompt(Paragraph{Kind: "picture", Language: "en"}, "ko"); strings.Contains(other, "layout model") {
		t.Fatalf("a kind without a guide gets the plain prompt: %s", other)
	}
}

func TestThePatternHoldsTheScriptsOfTheTargetLanguage(t *testing.T) {
	if got := Pattern("ko"); got != `[\p{Hangul}\p{Latin}\p{Greek}\p{Common}\p{Inherited}]*` {
		t.Fatalf("got %s", got)
	}
	if got := Pattern("ja"); !strings.Contains(got, `\p{Han}\p{Hiragana}\p{Katakana}\p{Latin}`) {
		t.Fatalf("got %s", got)
	}
	if got := Pattern("en"); got != `[\p{Latin}\p{Greek}\p{Common}\p{Inherited}]*` {
		t.Fatalf("got %s", got)
	}
	if got := Pattern("xx"); got != "" {
		t.Fatalf("a language without listed scripts has no pattern: %s", got)
	}
}

func TestMarksAreNumbersAndFootnoteSymbols(t *testing.T) {
	got := marks("Figure 2 shows 1,000 samples [3, 4] from 2021†, see 图１２ and 3.5%*")
	want := []string{"2", "1000", "3", "4", "2021", "†", "12", "5", "*"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	if got := marks("no marks here"); got != nil {
		t.Fatalf("got %q", got)
	}
}

func TestChecksFindMissingMarksAndShortTranslations(t *testing.T) {
	long := strings.Repeat("words of the source ", 5) // 100 characters
	paragraphs := []Paragraph{
		{Kind: "text", Role: doc.RoleBody, Language: "en", Spaced: long + "as Table 3 shows [12].", Translated: true, Stop: "eos",
			Translation: strings.Repeat("번역", 30) + " 표 3 [12]."},
		{Kind: "text", Role: doc.RoleBody, Language: "en", Spaced: long + "in 2021.", Translated: true, Stop: "eos",
			Translation: strings.Repeat("번역", 27) + " 2021년."},
		{Kind: "text", Role: doc.RoleBody, Language: "en", Spaced: long + "see Figure 12 and [7].", Translated: true, Stop: "length",
			Translation: "그림 1"},
		{Kind: "caption", Role: doc.RoleInserted, Language: "en", Spaced: "Figure 4: results", Translated: true, Stop: "eos",
			Translation: "그림 4: 결과"},
		{Kind: "text", Role: doc.RoleBody, Language: "ko", Spaced: "번역하지 않는 문단 1", Translated: false},
	}
	checks := CheckAll(paragraphs)
	if c := paragraphs[0].Check; c.Marks != 2 || len(c.Missing) != 0 || c.Short {
		t.Fatalf("kept everything: %+v", c)
	}
	if c := paragraphs[2].Check; !reflect.DeepEqual(c.Missing, []string{"12", "7"}) || !c.Short {
		t.Fatalf("dropped marks and text: %+v", c)
	}
	if c := paragraphs[3].Check; c.Short || len(c.Missing) != 0 {
		t.Fatalf("a source too short to judge its length: %+v", c)
	}
	if paragraphs[4].Check != nil {
		t.Fatal("an untranslated paragraph is not checked")
	}
	text := checks.Kinds["text"]
	if text.Paragraphs != 3 || text.WithMissing != 1 || text.MissingMarks != 2 || text.Short != 1 || text.Stops["length"] != 1 {
		t.Fatalf("text tally %+v", text)
	}
	if checks.Kinds["inserted caption"].Paragraphs != 1 || checks.Lengths["<100"].Paragraphs != 1 || checks.Lengths["100-399"].Paragraphs != 3 {
		t.Fatalf("tallies %+v %+v", checks.Kinds, checks.Lengths)
	}
}
