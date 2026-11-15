// Package pdf reads pages through pdfium: the objects the gate counts, characters with boxes, and page images.
package pdf

import (
	"fmt"
	"image"
	"image/draw"
	"math"
	"os"
	"strings"
	"time"
	"unicode"

	"github.com/klippa-app/go-pdfium"
	"github.com/klippa-app/go-pdfium/enums"
	"github.com/klippa-app/go-pdfium/references"
	"github.com/klippa-app/go-pdfium/requests"
	"github.com/klippa-app/go-pdfium/webassembly"

	"dplus/internal/doc"
	"dplus/internal/gate"
)

const (
	// maxFormDepth is how deep form objects are followed when counting a page's objects.
	maxFormDepth = 3
	// angleSamples is how many characters decide the page's text direction.
	angleSamples = 400
	// deviceScale keeps the rounding of FPDF_PageToDevice far below a point.
	deviceScale = 100
)

// Document is an open PDF.
type Document struct {
	pool     pdfium.Pool
	instance pdfium.Pdfium
	ref      references.FPDF_DOCUMENT
	Pages    int
}

// Open reads the file and opens it in pdfium.
func Open(path string) (*Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pool, err := webassembly.Init(webassembly.Config{MinIdle: 1, MaxIdle: 1, MaxTotal: 1, ReuseWorkers: true})
	if err != nil {
		return nil, fmt.Errorf("starting pdfium: %w", err)
	}
	instance, err := pool.GetInstance(time.Minute)
	if err != nil {
		pool.Close()
		return nil, fmt.Errorf("starting pdfium: %w", err)
	}
	opened, err := instance.OpenDocument(&requests.OpenDocument{File: &data})
	if err != nil {
		instance.Close()
		pool.Close()
		return nil, fmt.Errorf("opening the PDF: %w", err)
	}
	count, err := instance.FPDF_GetPageCount(&requests.FPDF_GetPageCount{Document: opened.Document})
	if err != nil {
		instance.Close()
		pool.Close()
		return nil, err
	}
	return &Document{pool: pool, instance: instance, ref: opened.Document, Pages: count.PageCount}, nil
}

// Close releases pdfium.
func (d *Document) Close() {
	d.instance.FPDF_CloseDocument(&requests.FPDF_CloseDocument{Document: d.ref})
	d.instance.Close()
	d.pool.Close()
}

// Page is what the pipeline needs from one page.
type Page struct {
	Number int // From one.
	Width  float64
	Height float64
	Counts gate.Counts
	Chars  []doc.Char
	Turns  int // Quarter clockwise turns that bring the page's text upright.
}

func (d *Document) page(index int) requests.Page {
	return requests.Page{ByIndex: &requests.PageByIndex{Document: d.ref, Index: index}}
}

// Read gets the objects and the characters of a page. Index counts from zero.
func (d *Document) Read(index int) (*Page, error) {
	ref := d.page(index)
	width, err := d.instance.FPDF_GetPageWidthF(&requests.FPDF_GetPageWidthF{Page: ref})
	if err != nil {
		return nil, err
	}
	height, err := d.instance.FPDF_GetPageHeightF(&requests.FPDF_GetPageHeightF{Page: ref})
	if err != nil {
		return nil, err
	}
	page := &Page{Number: index + 1, Width: float64(width.PageWidth), Height: float64(height.PageHeight)}
	if err := d.countObjects(ref, &page.Counts); err != nil {
		return nil, err
	}
	toDisplay, err := d.displayMap(ref, page.Width, page.Height)
	if err != nil {
		return nil, err
	}
	loaded, err := d.instance.FPDFText_LoadPage(&requests.FPDFText_LoadPage{Page: ref})
	if err != nil {
		return nil, err
	}
	defer d.instance.FPDFText_ClosePage(&requests.FPDFText_ClosePage{TextPage: loaded.TextPage})
	count, err := d.instance.FPDFText_CountChars(&requests.FPDFText_CountChars{TextPage: loaded.TextPage})
	if err != nil {
		return nil, err
	}
	var text strings.Builder
	votes := [4]int{}
	step := max(count.Count/angleSamples, 1)
	for i := range count.Count {
		unicodeValue, err := d.instance.FPDFText_GetUnicode(&requests.FPDFText_GetUnicode{TextPage: loaded.TextPage, Index: i})
		if err != nil {
			return nil, err
		}
		r := rune(unicodeValue.Unicode)
		text.WriteRune(r)
		if isLineBreakHyphen(r) {
			if n := len(page.Chars); n > 0 {
				page.Chars[n-1].HyphenEnd = true
			}
			continue
		}
		if unicode.IsSpace(r) || unicode.IsControl(r) || r == 0 {
			if n := len(page.Chars); n > 0 {
				page.Chars[n-1].SpaceAfter = true
			}
			continue
		}
		box, err := d.instance.FPDFText_GetCharBox(&requests.FPDFText_GetCharBox{TextPage: loaded.TextPage, Index: i})
		if err != nil {
			return nil, err
		}
		page.Chars = append(page.Chars, doc.Char{Rune: r, Box: toDisplay.box(box.Left, box.Right, box.Bottom, box.Top)})
		if i%step == 0 {
			if angle, err := d.instance.FPDFText_GetCharAngle(&requests.FPDFText_GetCharAngle{TextPage: loaded.TextPage, Index: i}); err == nil && angle.CharAngle >= 0 {
				votes[int(math.Round(float64(angle.CharAngle)/(math.Pi/2)))%4]++
			}
		}
	}
	page.Counts.CountText(text.String())
	page.Turns = turnsFromVotes(votes)
	return page, nil
}

// isLineBreakHyphen recognizes the marks pdfium and PDF producers use for a hyphen that breaks a word at a line end.
func isLineBreakHyphen(r rune) bool { return r == 0x02 || r == 0xFFFE || r == 0xAD }

// turnsFromVotes gives the quarter clockwise turns that undo the most common text angle.
func turnsFromVotes(votes [4]int) int {
	best := 0
	for i, v := range votes {
		if v > votes[best] {
			best = i
		}
	}
	return best
}

// affine maps PDF page coordinates to displayed page points, whatever the page's rotation and origin.
type affine struct{ a, b, c, d, e, f float64 }

func (m affine) point(x, y float64) (float64, float64) {
	return m.a*x + m.c*y + m.e, m.b*x + m.d*y + m.f
}

func (m affine) box(left, right, bottom, top float64) doc.Box {
	x0, y0 := m.point(left, bottom)
	x1, y1 := m.point(right, top)
	return doc.Box{X0: math.Min(x0, x1), Y0: math.Min(y0, y1), X1: math.Max(x0, x1), Y1: math.Max(y0, y1)}
}

func (d *Document) displayMap(page requests.Page, width, height float64) (affine, error) {
	const span = 100.0
	device := func(x, y float64) (float64, float64, error) {
		result, err := d.instance.FPDF_PageToDevice(&requests.FPDF_PageToDevice{
			Page: page, SizeX: int(width * deviceScale), SizeY: int(height * deviceScale),
			Rotate: enums.FPDF_PAGE_ROTATION_NONE, PageX: x, PageY: y,
		})
		if err != nil {
			return 0, 0, err
		}
		return float64(result.DeviceX) / deviceScale, float64(result.DeviceY) / deviceScale, nil
	}
	x0, y0, err := device(0, 0)
	if err != nil {
		return affine{}, err
	}
	xa, ya, err := device(span, 0)
	if err != nil {
		return affine{}, err
	}
	xc, yc, err := device(0, span)
	if err != nil {
		return affine{}, err
	}
	return affine{a: (xa - x0) / span, b: (ya - y0) / span, c: (xc - x0) / span, d: (yc - y0) / span, e: x0, f: y0}, nil
}

func (d *Document) countObjects(page requests.Page, counts *gate.Counts) error {
	total, err := d.instance.FPDFPage_CountObjects(&requests.FPDFPage_CountObjects{Page: page})
	if err != nil {
		return err
	}
	for i := range total.Count {
		object, err := d.instance.FPDFPage_GetObject(&requests.FPDFPage_GetObject{Page: page, Index: i})
		if err != nil {
			return err
		}
		if err := d.countObject(object.PageObject, 0, counts); err != nil {
			return err
		}
	}
	return nil
}

func (d *Document) countObject(object references.FPDF_PAGEOBJECT, depth int, counts *gate.Counts) error {
	kind, err := d.instance.FPDFPageObj_GetType(&requests.FPDFPageObj_GetType{PageObject: object})
	if err != nil {
		return err
	}
	switch kind.Type {
	case enums.FPDF_PAGEOBJ_TEXT:
		counts.TextObjects++
	case enums.FPDF_PAGEOBJ_PATH:
		counts.Paths++
	case enums.FPDF_PAGEOBJ_IMAGE:
		counts.Images++
	default:
		counts.OtherObjects++
	}
	if kind.Type != enums.FPDF_PAGEOBJ_FORM || depth+1 >= maxFormDepth {
		return nil
	}
	inner, err := d.instance.FPDFFormObj_CountObjects(&requests.FPDFFormObj_CountObjects{PageObject: object})
	if err != nil {
		return err
	}
	for i := range inner.Count {
		child, err := d.instance.FPDFFormObj_GetObject(&requests.FPDFFormObj_GetObject{PageObject: object, Index: uint64(i)})
		if err != nil {
			return err
		}
		if err := d.countObject(child.PageObject, depth+1, counts); err != nil {
			return err
		}
	}
	return nil
}

// Render draws a page at the given resolution on white.
func (d *Document) Render(index, dpi int) (*image.RGBA, error) {
	rendered, err := d.instance.RenderPageInDPI(&requests.RenderPageInDPI{Page: d.page(index), DPI: dpi})
	if err != nil {
		return nil, err
	}
	defer rendered.Cleanup()
	source := rendered.Result.RenderedImage
	bounds := source.Bounds()
	target := image.NewRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(target, target.Bounds(), image.White, image.Point{}, draw.Src)
	draw.Draw(target, target.Bounds(), source, bounds.Min, draw.Over)
	return target, nil
}
