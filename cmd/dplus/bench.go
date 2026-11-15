package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"dplus/internal/translate"
	"dplus/internal/worker"
)

// BenchRun is one line of the benchmark log: the same requests at one batch size.
type BenchRun struct {
	Time   string `json:"time"`
	Report string `json:"report"`
	Device string `json:"device"`
	// Whether the int4 layers multiplied straight from their packed weights.
	PackedLinear bool `json:"packed_linear"`
	Generation
	// Against the first batch size of the same invocation.
	Speedup         float64 `json:"speedup"`
	SameOutcomes    int     `json:"same_outcomes"`     // Requests whose tokens are the same.
	Divergences     []int   `json:"divergences"`       // For the others, the first token that differs.
	FirstLogitShift float64 `json:"first_logit_shift"` // Largest change in the logit of a most probable first token.

	Outcomes []worker.Generated `json:"outcomes"`
}

// bench measures generation on the paragraphs of an earlier report: the same requests once at each batch size, each
// run a line appended to the log.
func bench(arguments []string) error {
	set := flag.NewFlagSet("bench", flag.ExitOnError)
	home, _ := os.UserHomeDir()
	sizes := set.String("batch", "1,2,4,8,16", "batch sizes to run, the first being what the others are compared to")
	state := set.Float64("state", 0, "device memory in GiB the states of the rows may take (default: the worker's bound)")
	logPath := set.String("log", "", "file the runs are appended to (default: next to the report)")
	to := set.String("to", "", "language to translate into (default: the report's)")
	models := set.String("models", environment("DPLUS_MODELS", filepath.Join(home, "models")), "folder of the models")
	python := set.String("python", environment("DPLUS_PYTHON", "python3"), "Python that runs the model worker")
	device := set.String("device", "cuda:0", "device of the model worker")
	dev := set.Bool("dev", false, "developer mode: record where the device's time goes, next to the log as .dev.jsonl")
	devEvery := set.Int("dev-every", defaultDevEvery, "with -dev, how many spans of a phase share one whose int4 products are timed; 0 times none")
	set.Parse(arguments)
	if set.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: dplus bench [flags] report.json")
		set.PrintDefaults()
		os.Exit(2)
	}
	path := set.Arg(0)
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var report struct {
		To         string                `json:"to"`
		Paragraphs []translate.Paragraph `json:"paragraphs"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	if *to == "" {
		*to = report.To
	}
	requests := requestsFor(report.Paragraphs, *to)
	if len(requests) == 0 {
		return fmt.Errorf("%s has no paragraph to translate into %s", path, *to)
	}
	var batchSizes []int
	for _, part := range strings.Split(*sizes, ",") {
		size, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || size < 1 {
			return fmt.Errorf("bad batch size %q", part)
		}
		batchSizes = append(batchSizes, size)
	}
	if *logPath == "" {
		*logPath = strings.TrimSuffix(path, filepath.Ext(path)) + ".bench.jsonl"
	}
	log, err := os.OpenFile(*logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer log.Close()
	client := newClient(*models, *python, *device)
	defer client.Close()
	if *dev {
		setDev(client, strings.TrimSuffix(*logPath, filepath.Ext(*logPath))+".dev.jsonl", max(*devEvery, 0))
	}

	// The runs measure a loaded model whose kernels are compiled: the shortest request runs once before them.
	if err := client.Start(); err != nil {
		return err
	}
	shortest := requests[0]
	for _, request := range requests {
		if len(request.User) < len(shortest.User) {
			shortest = request
		}
	}
	if _, err := client.GenerateEach([]worker.Request{shortest}, worker.Options{BatchSize: 1}, func(worker.Generated) {}, nil); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d requests from %s, runs appended to %s\n", len(requests), path, *logPath)
	var first *BenchRun
	for _, size := range batchSizes {
		run := BenchRun{Time: time.Now().Format(time.RFC3339), Report: path}
		outcomes := map[string]worker.Generated{}
		options := worker.Options{BatchSize: size, StateMiB: int(*state * 1024), Probe: true}
		run.Generation, err = generate(client, requests, options, func(outcome worker.Generated) { outcomes[outcome.Key] = outcome }, nil)
		if err != nil {
			return err
		}
		run.Device, run.PackedLinear = client.Info.Device, client.Info.PackedLinear
		for _, request := range requests {
			run.Outcomes = append(run.Outcomes, outcomes[request.Key])
		}
		if first == nil {
			first = &run
		}
		compare(&run, first)
		line, err := json.Marshal(run)
		if err != nil {
			return err
		}
		if _, err := log.Write(append(line, '\n')); err != nil {
			return err
		}
		fmt.Fprintln(os.Stderr, describeRun(&run, first))
	}
	_, err = devReport(client)
	return err
}

// compare sets how a run differs from the reference run of the same requests.
func compare(run, reference *BenchRun) {
	if run.Seconds > 0 {
		run.Speedup = reference.Seconds / run.Seconds
	}
	run.Divergences = []int{}
	for i, outcome := range run.Outcomes {
		other := reference.Outcomes[i]
		at := divergence(outcome.TokenIDs, other.TokenIDs)
		if at < 0 && outcome.Error == other.Error {
			run.SameOutcomes++
		} else {
			run.Divergences = append(run.Divergences, at)
		}
		logits := make(map[int]float64, len(other.FirstTop))
		for _, candidate := range other.FirstTop {
			logits[candidate.Token] = candidate.Logit
		}
		for _, candidate := range outcome.FirstTop {
			if logit, shared := logits[candidate.Token]; shared {
				run.FirstLogitShift = max(run.FirstLogitShift, math.Abs(candidate.Logit-logit))
			}
		}
	}
	sort.Ints(run.Divergences)
}

// divergence is the index of the first token two generations do not share, or -1 when they are the same.
func divergence(a, b []int) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

func describeRun(run, first *BenchRun) string {
	text := fmt.Sprintf("batch %d: %.1fs (prefill %.1fs, decode %.1fs), %d tokens, %.1f tokens/s, %d steps, %d joins, at most %d rows, idle %.0f%%, padding %.0f%%, peak %d MiB",
		run.BatchSize, run.Seconds, run.PrefillSeconds, run.DecodeSeconds, run.OutputTokens, run.TokensPerSecond, run.Steps, run.Joins, run.Rows,
		run.IdleShare*100, run.PaddingShare*100, run.PeakMiB)
	if failed := run.Stops["failed"]; failed > 0 {
		text += fmt.Sprintf(", %d failed", failed)
	}
	if run == first {
		return text
	}
	text += fmt.Sprintf("; against batch %d: %.2fx, %d of %d outcomes the same", first.BatchSize, run.Speedup, run.SameOutcomes, len(run.Outcomes))
	if len(run.Divergences) > 0 {
		text += fmt.Sprintf(" (the others differ from token %v)", run.Divergences)
	}
	return text + fmt.Sprintf(", first logits shift by at most %.3f", run.FirstLogitShift)
}
