package spacing

import "testing"

func TestLanguage(t *testing.T) {
	cases := map[string]string{
		"본 발명은 스마트 폰에 관한 것이다": "ko",
		"本发明涉及一种装置":           "zh",
		"The layout model":    "en",
		"これはテストです":            "ja",
		"1234 ...":            "",
	}
	for text, want := range cases {
		if got := Language(text); got != want {
			t.Errorf("%q: got %q, want %q", text, got, want)
		}
	}
}

func TestJoinEastAsian(t *testing.T) {
	cases := map[string]string{
		"发 明 专 利":    "发明专利",
		"专利 ABC 发明":  "专利 ABC 发明",
		"第一 ，第二":     "第一，第二",
		"abc def":    "abc def",
		"  发 明":      "  发明",
		"发明  \n  专利": "发明专利",
	}
	for text, want := range cases {
		if got := JoinEastAsian(text); got != want {
			t.Errorf("%q: got %q, want %q", text, got, want)
		}
	}
}

// The expectations come from the same inputs run through dokey.spacing.apply_lexicon.
func TestApplyLexicon(t *testing.T) {
	lexicon := map[string]string{"안전보건": "안전 보건", "산업재해": "산업재해"}
	cases := map[string]string{
		"안전보건 관리":    "안전 보건 관리",
		"안 전 보 건 관리": "안전 보건 관리",
		"산업 재해 예방":   "산업재해 예방",
		"관리 산업재해":    "관리 산업재해",
		"무관한 문장":     "무관한 문장",
	}
	for text, want := range cases {
		if got := ApplyLexicon(text, lexicon); got != want {
			t.Errorf("%q: got %q, want %q", text, got, want)
		}
	}
	if got := ApplyLexicon("그대로 둔다", nil); got != "그대로 둔다" {
		t.Errorf("no lexicon changed the text: %q", got)
	}
}

func TestApplyLexiconKeepsTermsFromOverlapping(t *testing.T) {
	lexicon := map[string]string{"가나다": "가 나다", "나다라": "나다 라"}
	if got := ApplyLexicon("가나다라", lexicon); got != "가 나다라" {
		t.Errorf("got %q", got)
	}
}

func TestApplyLexiconSkipsATermWhoseSpacingDoesNotFit(t *testing.T) {
	if got := ApplyLexicon("가나", map[string]string{"가나": "가나다"}); got != "가나" {
		t.Errorf("got %q", got)
	}
}
