package gate

import "testing"

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		c    Counts
		want string
	}{
		{"text page", Counts{TextObjects: 40, Images: 1, Chars: 900}, Healthy},
		{"image page", Counts{TextObjects: 0, Images: 1}, Scanned},
		{"more images than text objects", Counts{TextObjects: 2, Images: 3, Chars: 20}, Scanned},
		{"glyphs drawn as paths", Counts{Paths: 500}, Scanned},
		{"other objects only", Counts{OtherObjects: 2}, Scanned},
		{"text objects without characters", Counts{TextObjects: 3}, Corrupt},
		{"private use characters", Counts{TextObjects: 30, Chars: 100, Alien: 6}, Corrupt},
		{"just under the private use limit", Counts{TextObjects: 30, Chars: 100, Alien: 5}, Healthy},
		{"empty page", Counts{}, Blank},
	}
	for _, c := range cases {
		if got := Classify(c.c); got != c.want {
			t.Errorf("%s: got %s, want %s", c.name, got, c.want)
		}
	}
}

func TestCountText(t *testing.T) {
	var c Counts
	c.CountText("  ab \n")
	if c.Chars != 3 || c.Alien != 1 {
		t.Fatalf("got %d chars, %d alien", c.Chars, c.Alien)
	}
}
