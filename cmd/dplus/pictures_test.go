package main

import (
	"encoding/base64"
	"image"
	"os"
	"path/filepath"
	"testing"

	"dplus/internal/doc"
)

func TestAPictureCoversWholePixels(t *testing.T) {
	bounds := image.Rect(0, 0, 1700, 2200)
	got := pictureRect(doc.Box{X0: 10.2, Y0: 20.7, X1: 100.1, Y1: 200.5}, 2, 3, bounds)
	if want := image.Rect(20, 62, 201, 602); got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestAPictureStaysOnThePage(t *testing.T) {
	bounds := image.Rect(0, 0, 1700, 2200)
	if got := pictureRect(doc.Box{X0: 10, Y0: 1150, X1: 900, Y1: 1200}, 2, 2, bounds); !got.Empty() {
		t.Fatalf("got %v for a box below the page, want an empty rectangle", got)
	}
	got := pictureRect(doc.Box{X0: -5, Y0: 1000, X1: 900, Y1: 1200}, 2, 2, bounds)
	if want := image.Rect(0, 2000, 1700, 2200); got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestAnEmbeddedPictureIsItsFileAsBase64(t *testing.T) {
	folder := t.TempDir()
	base := filepath.Join(folder, "paper.dplus")
	if err := os.MkdirAll(base+".pictures", 0o755); err != nil {
		t.Fatal(err)
	}
	raw := []byte("\x89PNG picture bytes")
	if err := os.WriteFile(filepath.Join(base+".pictures", "page-1-block-2.png"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	image := "paper.dplus.pictures/page-1-block-2.png"
	if got, want := pictureLink(base, image, true), "data:image/png;base64,"+base64.StdEncoding.EncodeToString(raw); got != want {
		t.Fatalf("embedded: got %q, want %q", got, want)
	}
	if got, want := pictureLink(base, image, false), "<"+image+">"; got != want {
		t.Fatalf("linked: got %q, want %q", got, want)
	}
	if got, want := pictureLink(base, "paper.dplus.pictures/missing.png", true), "<paper.dplus.pictures/missing.png>"; got != want {
		t.Fatalf("missing file: got %q, want %q", got, want)
	}
}
