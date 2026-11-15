// Command dplus translates a PDF.
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"

	"dplus/internal/doc"
	"dplus/internal/gate"
	"dplus/internal/layout"
	"dplus/internal/order"
	"dplus/internal/pdf"
	"dplus/internal/spacing"
	"dplus/internal/translate"
	"dplus/internal/worker"
)

const (
	layoutDPI = 72
	// defaultBatch is how many paragraphs are translated together unless -batch says otherwise.
	defaultBatch = 16
)

type options struct {
	input       string
	output      string
	to          string
	from        string
	pages       string
	models      string
	python      string
	device      string
	lexicon     string
	batch       int
	noTranslate bool
	pageImage   bool
	embed       bool
	dev         bool
	devEvery    int
	verbose     bool
}

// PageReport is what the pipeline found on one page.
type PageReport struct {
	Number        int          `json:"number"`
	Verdict       string       `json:"verdict"`
	Counts        gate.Counts  `json:"counts"`
	Turns         int          `json:"turns"`
	Blocks        []doc.Block  `json:"blocks,omitempty"`
	Unassigned    int          `json:"unassigned_chars"`
	BodyQuestions int          `json:"body_questions"`
	Order         order.Result `json:"order"`
	Seconds       float64      `json:"seconds"`
}

// Report is the JSON output.
type Report struct {
	Input      string                `json:"input"`
	To         string                `json:"to"`
	Backbone   string                `json:"backbone"`
	Worker     worker.Ready          `json:"worker"`
	PeakMiB    int64                 `json:"peak_mib"`
	Pages      []PageReport          `json:"pages"`
	Paragraphs []translate.Paragraph `json:"paragraphs"`
	Summary    map[string]any        `json:"summary"`
}

func main() {
	command := ""
	if len(os.Args) > 1 {
		command = os.Args[1]
	}
	var err error
	switch command {
	case "translate":
		err = run(parse(os.Args[2:]))
	case "bench":
		err = bench(os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, "usage: dplus translate [flags] file.pdf\n       dplus bench [flags] report.json")
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "dplus:", err)
		os.Exit(1)
	}
}

func environment(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func parse(arguments []string) options {
	var o options
	set := flag.NewFlagSet("translate", flag.ExitOnError)
	home, _ := os.UserHomeDir()
	set.StringVar(&o.output, "output", "", "output path without extension (default: next to the input)")
	set.StringVar(&o.to, "to", "ko", "language to translate into")
	set.StringVar(&o.from, "from", "auto", "source language, or auto for each paragraph")
	set.StringVar(&o.pages, "pages", "", "pages to read, such as 1-3,7 (default: all)")
	set.StringVar(&o.models, "models", environment("DPLUS_MODELS", filepath.Join(home, "models")), "folder of the models")
	set.StringVar(&o.python, "python", environment("DPLUS_PYTHON", "python3"), "Python that runs the model worker")
	set.StringVar(&o.device, "device", "cuda:0", "device of the model worker")
	set.StringVar(&o.lexicon, "lexicon", "", "JSON file of terms and their spacing")
	set.IntVar(&o.batch, "batch", defaultBatch, "paragraphs translated together")
	set.BoolVar(&o.noTranslate, "no-translate", false, "stop after the paragraphs are ready")
	set.BoolVar(&o.pageImage, "page-image", false, "show the decision model the page image")
	set.BoolVar(&o.embed, "embed-pictures", false, "write the pictures into the .md file as base64 data instead of linking their files")
	set.BoolVar(&o.verbose, "verbose", false, "print translations as they are produced")
	set.BoolVar(&o.dev, "dev", false, "developer mode: record where the device's time goes, next to the output as .dev.jsonl")
	set.IntVar(&o.devEvery, "dev-every", defaultDevEvery, "with -dev, how many spans of a phase share one whose int4 products are timed; 0 times none")
	set.Parse(arguments)
	if set.NArg() != 1 || o.batch < 1 || o.devEvery < 0 {
		set.Usage()
		os.Exit(2)
	}
	o.input = set.Arg(0)
	if o.output == "" {
		o.output = strings.TrimSuffix(o.input, filepath.Ext(o.input)) + ".dplus"
	}
	return o
}

func pageRange(text string, total int) ([]int, error) {
	if text == "" {
		all := make([]int, total)
		for i := range all {
			all[i] = i
		}
		return all, nil
	}
	seen := map[int]bool{}
	for _, part := range strings.Split(text, ",") {
		first, last, ranged := strings.Cut(part, "-")
		low, err := strconv.Atoi(strings.TrimSpace(first))
		if err != nil {
			return nil, fmt.Errorf("bad page range %q", part)
		}
		high := low
		if ranged {
			if high, err = strconv.Atoi(strings.TrimSpace(last)); err != nil {
				return nil, fmt.Errorf("bad page range %q", part)
			}
		}
		if low < 1 || high > total || low > high {
			return nil, fmt.Errorf("page range %q is outside 1-%d", part, total)
		}
		for page := low; page <= high; page++ {
			seen[page-1] = true
		}
	}
	list := make([]int, 0, len(seen))
	for page := range seen {
		list = append(list, page)
	}
	sort.Ints(list)
	return list, nil
}

func libraryPath() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	name := "libonnxruntime.so"
	if runtime.GOOS == "windows" {
		name = "onnxruntime.dll"
	}
	return filepath.Join(filepath.Dir(exe), name), nil
}

// newClient gives the client of the model worker. The device kernel library is read from the app's folder.
func newClient(models, python, device string) *worker.Client {
	client := &worker.Client{
		Python: python, Model: filepath.Join(models, "d3-ultra-INT4"),
		Head: filepath.Join(models, "Qwen3.5-27B-lm_head", "lm_head.safetensors"), Device: device,
	}
	if exe, err := os.Executable(); err == nil {
		client.Kernels = filepath.Join(filepath.Dir(exe), "librdna4_w4a16.so")
	}
	return client
}

func run(o options) error {
	began := time.Now()
	ctx := context.Background()
	library, err := libraryPath()
	if err != nil {
		return err
	}
	ort.SetSharedLibraryPath(library)
	if err := ort.Init(); err != nil {
		return fmt.Errorf("loading ONNX Runtime from %s: %w", library, err)
	}
	defer ort.Shutdown()

	lexicon := map[string]string{}
	if o.lexicon != "" {
		raw, err := os.ReadFile(o.lexicon)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(raw, &lexicon); err != nil {
			return fmt.Errorf("reading %s: %w", o.lexicon, err)
		}
	}
	document, err := pdf.Open(o.input)
	if err != nil {
		return err
	}
	defer document.Close()
	selected, err := pageRange(o.pages, document.Pages)
	if err != nil {
		return err
	}
	heron, err := layout.NewHeron(filepath.Join(o.models, "docling-layout-heron-onnx", "model.onnx"), max(runtime.NumCPU()/2, 1))
	if err != nil {
		return err
	}
	defer heron.Close()
	restorer, err := spacing.New(filepath.Join(o.models, "dplus-spacing-ko"), lexicon)
	if err != nil {
		return err
	}
	defer restorer.Close()
	client := newClient(o.models, o.python, o.device)
	defer client.Close()
	if o.dev {
		setDev(client, o.output+".dev.jsonl", o.devEvery)
	}

	report := Report{Input: o.input, To: o.to, Backbone: client.Model}
	var pages []translate.Page
	totals := map[string]int{}
	for _, index := range selected {
		started := time.Now()
		entry, page, err := readPage(ctx, document, index, heron, client, o.pageImage, o.output+".pictures")
		if err != nil {
			return fmt.Errorf("page %d: %w", index+1, err)
		}
		entry.Seconds = time.Since(started).Seconds()
		report.Pages = append(report.Pages, entry)
		pages = append(pages, page)
		totals[entry.Verdict]++
		fmt.Fprintf(os.Stderr, "page %d: %s, %d blocks, %d body questions, %d order decisions, %.1fs%s\n", entry.Number, entry.Verdict, len(entry.Blocks), entry.BodyQuestions, entry.Order.Asked, entry.Seconds, marks(entry))
	}
	paragraphs, decisions, err := translate.Gather(pages, document.Pages, client)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d paragraphs (decisions: %d continuation, %d caption, %d placement, %d same place)\n",
		len(paragraphs), decisions.Continuation, decisions.Caption, decisions.Placement, decisions.SamePlace)
	generation := finish(ctx, paragraphs, restorer, client, o)
	dev, err := devReport(client)
	if err != nil {
		return err
	}
	report.Paragraphs = paragraphs
	report.Worker = client.Info
	report.PeakMiB = generation.PeakMiB
	report.Summary = map[string]any{
		"pages": totals, "decisions": decisions, "generation": generation, "seconds": time.Since(began).Seconds(),
	}
	if !o.noTranslate {
		checks := translate.CheckAll(paragraphs)
		for _, line := range checkLines(checks) {
			fmt.Fprintln(os.Stderr, line)
		}
		report.Summary["check"] = checks
	}
	if dev != nil {
		report.Summary["dev"] = dev
	}
	if err := write(report, o.output, o.embed); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "done in %.1fs\n", time.Since(began).Seconds())
	return nil
}

func marks(entry PageReport) string {
	var parts []string
	if entry.Order.Review {
		parts = append(parts, fmt.Sprintf("order needs review (%d violations)", len(entry.Order.Violations)))
	}
	if entry.Unassigned > 0 {
		parts = append(parts, fmt.Sprintf("%d characters in no box", entry.Unassigned))
	}
	if len(parts) == 0 {
		return ""
	}
	return ", " + strings.Join(parts, ", ")
}

// readPage finds the blocks of a page, decides their roles and reading order, and saves its pictures into the folder
// pictures.
func readPage(ctx context.Context, document *pdf.Document, index int, heron *layout.Heron, client *worker.Client, pageImage bool, pictures string) (PageReport, translate.Page, error) {
	var result translate.Page
	page, err := document.Read(index)
	if err != nil {
		return PageReport{}, result, err
	}
	entry := PageReport{Number: page.Number, Counts: page.Counts, Turns: page.Turns, Verdict: gate.Classify(page.Counts)}
	if entry.Verdict != gate.Healthy {
		return entry, result, nil // OCR is not part of the pipeline yet, so these pages stay marked in the report.
	}
	rendered, err := document.Render(index, layoutDPI)
	if err != nil {
		return entry, result, err
	}
	found, err := heron.Detect(ctx, rendered)
	if err != nil {
		return entry, result, err
	}
	bounds := rendered.Bounds()
	built := layout.Build(page, found, page.Width/float64(bounds.Dx()), page.Height/float64(bounds.Dy()))
	if err := savePictures(document, index, page.Width, page.Height, built.Blocks, pictures); err != nil {
		return entry, result, fmt.Errorf("saving pictures: %w", err)
	}
	entry.BodyQuestions, err = order.ClassifyBody(page.Number, document.Pages, page.Width, page.Height, built.Blocks, client)
	if err != nil {
		return entry, result, err
	}
	entry.Blocks, entry.Unassigned = built.Blocks, built.Unassigned
	var images []string
	if pageImage {
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, rendered); err != nil {
			return entry, result, err
		}
		images = []string{"data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())}
	}
	entry.Order, err = order.Settle(page.Number, document.Pages, page.Width, page.Height, page.Turns, built.Blocks, images, client)
	if err != nil {
		return entry, result, err
	}
	result = translate.Page{Number: page.Number, Width: page.Width, Height: page.Height}
	byID := make(map[int]doc.Block, len(built.Blocks))
	for _, block := range built.Blocks {
		byID[block.ID] = block
		if block.Role == doc.RoleInserted {
			result.Inserted = append(result.Inserted, block)
		}
	}
	for _, id := range entry.Order.Order {
		result.Body = append(result.Body, byID[id])
	}
	return entry, result, nil
}

// finish restores the spacing of each paragraph and translates the paragraphs that need it, several at a time.
func finish(ctx context.Context, paragraphs []translate.Paragraph, restorer *spacing.Restorer, client *worker.Client, o options) Generation {
	for i := range paragraphs {
		p := &paragraphs[i]
		p.Language = o.from
		if o.from == "auto" {
			p.Language = spacing.Language(p.Source)
		}
		spaced, method, err := restorer.Restore(ctx, p.Source, p.Language)
		if err != nil {
			fmt.Fprintf(os.Stderr, "spacing failed on paragraph %d: %v\n", i+1, err)
			spaced, method = p.Source, "failed"
		}
		p.Spaced, p.Spacing = spaced, method
	}
	if o.noTranslate {
		return Generation{BatchSize: o.batch, Stops: map[string]int{}}
	}
	requests := requestsFor(paragraphs, o.to)
	shown := -1 // The paragraph whose source is on the screen.
	show := func(index int) {
		if shown != index {
			fmt.Fprintf(os.Stderr, "\n[%d/%d] %s\n> ", index+1, len(paragraphs), paragraphs[index].Spaced)
			shown = index
		}
	}
	var onText func(key, text string)
	if o.verbose && o.batch == 1 { // One paragraph runs at a time, so its text can be shown as it forms.
		onText = func(key, text string) {
			index, _ := strconv.Atoi(key)
			show(index)
			fmt.Fprint(os.Stderr, text)
		}
	}
	finished := 0
	generation, err := generate(client, requests, worker.Options{BatchSize: o.batch}, func(outcome worker.Generated) {
		index, _ := strconv.Atoi(outcome.Key)
		finished++
		if outcome.Error != "" {
			fmt.Fprintf(os.Stderr, "translation failed on paragraph %d: %s\n", index+1, outcome.Error)
			return
		}
		p := &paragraphs[index]
		p.Translated, p.Translation, p.Stop, p.Tokens = true, strings.TrimSpace(outcome.Text), outcome.Stop, outcome.Tokens
		p.Seconds = outcome.Seconds
		switch {
		case !o.verbose:
			fmt.Fprintf(os.Stderr, "translated paragraph %d (%d of %d, %s, %d tokens)\n", index+1, finished, len(requests), outcome.Stop, outcome.Tokens)
		case onText != nil:
			show(index)
			fmt.Fprintf(os.Stderr, "\n(%s, %d tokens)\n", outcome.Stop, outcome.Tokens)
		default:
			show(index)
			fmt.Fprintf(os.Stderr, "%s\n(%s, %d tokens)\n", p.Translation, outcome.Stop, outcome.Tokens)
		}
	}, onText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "translation stopped: %v\n", err)
	}
	if len(requests) > 0 {
		fmt.Fprintf(os.Stderr, "translated %d of %d paragraphs in %.1fs: %d tokens, %.1f tokens/s, %d steps of up to %d rows, idle %.0f%%, padding %.0f%%, peak %d MiB\n",
			len(requests)-generation.Stops["failed"], len(requests), generation.Seconds, generation.OutputTokens, generation.TokensPerSecond,
			generation.Steps, o.batch, generation.IdleShare*100, generation.PaddingShare*100, generation.PeakMiB)
	}
	return generation
}

// kindMark is the line that tells a paragraph's kind and sets inserted elements apart from the body.
func kindMark(paragraph translate.Paragraph) string {
	if paragraph.Role == doc.RoleInserted {
		return "[inserted " + paragraph.Kind + "]"
	}
	return "[" + paragraph.Kind + "]"
}

// pictureLink is where the .md file finds a picture: its file beside the output, or with embed the picture itself as
// base64 data, so the .md file shows it on its own. A picture whose file cannot be read stays linked.
func pictureLink(base, image string, embed bool) string {
	if embed {
		raw, err := os.ReadFile(filepath.Join(filepath.Dir(base), filepath.FromSlash(image)))
		if err == nil {
			return "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)
		}
		fmt.Fprintf(os.Stderr, "linking %s instead of embedding it: %v\n", image, err)
	}
	// Angle brackets keep a link whose path holds spaces whole.
	return "<" + image + ">"
}

func write(report Report, base string, embed bool) error {
	raw, err := json.MarshalIndent(report, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(base+".json", raw, 0o644); err != nil {
		return err
	}
	var text strings.Builder
	fmt.Fprintf(&text, "# %s\n\n", filepath.Base(report.Input))
	for _, paragraph := range report.Paragraphs {
		text.WriteString(kindMark(paragraph) + "\n")
		if paragraph.Image != "" {
			fmt.Fprintf(&text, "![%s](%s)\n", paragraph.Kind, pictureLink(base, paragraph.Image, embed))
		}
		if paragraph.Spaced != "" {
			for _, line := range strings.Split(paragraph.Spaced, "\n") {
				fmt.Fprintf(&text, "> %s\n", line)
			}
		}
		text.WriteString("\n")
		if paragraph.Translated {
			text.WriteString(paragraph.Translation + "\n\n")
		}
	}
	for _, page := range report.Pages {
		if page.Verdict != gate.Healthy {
			fmt.Fprintf(&text, "<!-- page %d: %s, not read -->\n", page.Number, page.Verdict)
		}
	}
	if err := os.WriteFile(base+".md", []byte(text.String()), 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s.md and %s.json\n", base, base)
	return nil
}
