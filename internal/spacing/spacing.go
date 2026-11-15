// Package spacing restores word spacing in ordered text, per language.
package spacing

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

const (
	paddingIndex   = 0
	unknownIndex   = 1
	spaceThreshold = 0.5
	// strongShare is the share of letters a script needs to name the language of a text.
	strongShare = 0.3
)

// Language names the script family of a text.
func Language(text string) string {
	var hangul, han, kana, latin, letters int
	for _, r := range text {
		if !unicode.IsLetter(r) {
			continue
		}
		letters++
		switch {
		case unicode.Is(unicode.Hangul, r):
			hangul++
		case unicode.Is(unicode.Han, r):
			han++
		case unicode.In(r, unicode.Hiragana, unicode.Katakana):
			kana++
		case unicode.Is(unicode.Latin, r):
			latin++
		}
	}
	if letters == 0 {
		return ""
	}
	share := func(count int) float64 { return float64(count) / float64(letters) }
	switch {
	case share(hangul) >= strongShare:
		return "ko"
	case share(kana) >= strongShare:
		return "ja"
	case share(han) >= strongShare:
		return "zh"
	case share(latin) >= strongShare:
		return "en"
	}
	return ""
}

// Restorer holds the spacing models.
type Restorer struct {
	korean  *korean
	lexicon map[string]string
}

// New loads the Korean spacing model from its folder and the terms that fix spacing.
func New(koreanDir string, lexicon map[string]string) (*Restorer, error) {
	model, err := loadKorean(koreanDir)
	if err != nil {
		return nil, err
	}
	return &Restorer{korean: model, lexicon: lexicon}, nil
}

// Close releases the models.
func (r *Restorer) Close() { r.korean.session.Close() }

// Restore returns the text with its spacing restored and the method used.
func (r *Restorer) Restore(ctx context.Context, text, language string) (string, string, error) {
	switch language {
	case "ko":
		spaced, err := r.korean.space(ctx, text)
		if err != nil {
			return "", "", err
		}
		return ApplyLexicon(spaced, r.lexicon), "korean-model", nil
	case "zh", "ja":
		return JoinEastAsian(text), "east-asian-rule", nil
	}
	return text, "unchanged", nil
}

// JoinEastAsian removes the spaces between two East Asian characters.
func JoinEastAsian(text string) string {
	runes := []rune(text)
	out := make([]rune, 0, len(runes))
	for i := 0; i < len(runes); i++ {
		if unicode.IsSpace(runes[i]) && len(out) > 0 {
			j := i
			for j < len(runes) && unicode.IsSpace(runes[j]) {
				j++
			}
			if j < len(runes) && eastAsian(out[len(out)-1]) && eastAsian(runes[j]) {
				i = j - 1
				continue
			}
		}
		out = append(out, runes[i])
	}
	return string(out)
}

func eastAsian(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana) ||
		(r >= 0x3000 && r <= 0x303F) || (r >= 0xFF00 && r <= 0xFFEF)
}

type korean struct {
	session *ort.Session
	vocab   map[rune]int64
}

func loadKorean(dir string) (*korean, error) {
	raw, err := os.ReadFile(dir + "/vocab.txt")
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(raw), "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	vocab := make(map[rune]int64, len(lines))
	for i, line := range lines {
		if runes := []rune(line); len(runes) == 1 { // Multi-character entries such as the padding token are not text.
			vocab[runes[0]] = int64(i)
		}
	}
	session, err := ort.NewSession(dir+"/model.onnx", nil)
	if err != nil {
		return nil, fmt.Errorf("loading the Korean spacing model: %w", err)
	}
	return &korean{session: session, vocab: vocab}, nil
}

// space drops the text's own spaces and asks the model for each position whether a space follows.
func (k *korean) space(ctx context.Context, text string) (string, error) {
	var chars []rune
	for _, r := range text {
		if !unicode.IsSpace(r) {
			chars = append(chars, r)
		}
	}
	if len(chars) == 0 {
		return "", nil
	}
	indices := make([]int64, len(chars))
	for i, r := range chars {
		if value, ok := k.vocab[r]; ok {
			indices[i] = value
		} else {
			indices[i] = unknownIndex
		}
	}
	input, err := ort.CreateTensor([]int64{1, int64(len(indices))}, indices)
	if err != nil {
		return "", err
	}
	defer input.Close()
	outputs, err := k.session.Run(ctx, map[string]*ort.Tensor{"indices": input}, nil)
	if err != nil {
		return "", err
	}
	defer func() {
		for _, tensor := range outputs {
			tensor.Close()
		}
	}()
	probabilities, err := ort.TensorData[float32](outputs["probabilities"])
	if err != nil {
		return "", err
	}
	var out strings.Builder
	for i, r := range chars {
		out.WriteRune(r)
		if i < len(chars)-1 && probabilities[i] >= spaceThreshold {
			out.WriteByte(' ')
		}
	}
	return out.String(), nil
}

// ApplyLexicon imposes each term's own spacing wherever the term occurs, comparing without spaces and without case.
func ApplyLexicon(text string, lexicon map[string]string) string {
	if len(lexicon) == 0 {
		return text
	}
	var compact []rune
	spaceAfter := map[int]bool{}
	for _, r := range text {
		if unicode.IsSpace(r) {
			if len(compact) > 0 {
				spaceAfter[len(compact)-1] = true
			}
			continue
		}
		compact = append(compact, r)
	}
	haystack := make([]rune, len(compact))
	for i, r := range compact {
		haystack[i] = unicode.ToLower(r)
	}
	keys := make([]string, 0, len(lexicon))
	for key := range lexicon {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if len([]rune(keys[i])) != len([]rune(keys[j])) {
			return len([]rune(keys[i])) > len([]rune(keys[j]))
		}
		return keys[i] < keys[j]
	})
	claimed := make([]bool, len(compact))
	for _, key := range keys {
		needle := []rune(key)
		canonical := map[int]bool{}
		position := 0
		for _, r := range lexicon[key] {
			if unicode.IsSpace(r) {
				canonical[position-1] = true
			} else {
				position++
			}
		}
		if position != len(needle) || len(needle) == 0 {
			continue
		}
	search:
		for start := 0; start+len(needle) <= len(haystack); start++ {
			for offset, r := range needle {
				if haystack[start+offset] != r || claimed[start+offset] {
					continue search
				}
			}
			for offset := range needle {
				claimed[start+offset] = true
				if offset == len(needle)-1 {
					continue // The edge belongs to the neighbors.
				}
				if canonical[offset] {
					spaceAfter[start+offset] = true
				} else {
					delete(spaceAfter, start+offset)
				}
			}
		}
	}
	var out strings.Builder
	for i, r := range compact {
		out.WriteRune(r)
		if spaceAfter[i] && i < len(compact)-1 {
			out.WriteByte(' ')
		}
	}
	return out.String()
}
