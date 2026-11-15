package main

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"time"

	"dplus/internal/translate"
	"dplus/internal/worker"
)

// Generation sums up what the generations of one run took.
type Generation struct {
	BatchSize       int            `json:"batch_size"`
	StateMiB        int            `json:"state_mib,omitempty"` // The bound on the rows' states, when the run set one.
	Requests        int            `json:"requests"`
	Seconds         float64        `json:"seconds"` // From sending the requests to the last outcome.
	OutputTokens    int            `json:"output_tokens"`
	TokensPerSecond float64        `json:"tokens_per_second"`
	IdleShare       float64        `json:"idle_share"`    // Share of the places, batch size times decode steps, that stood empty.
	PaddingShare    float64        `json:"padding_share"` // Share of prefilled tokens that were padding.
	Stops           map[string]int `json:"stops"`
	worker.Pool
}

// requestsFor lists the translations the paragraphs need, each keyed by its paragraph's index.
func requestsFor(paragraphs []translate.Paragraph, to string) []worker.Request {
	var list []worker.Request
	for i, p := range paragraphs {
		if !translate.Translatable(p.Kind) || p.Language == "" || p.Language == to {
			continue
		}
		list = append(list, worker.Request{
			Key: strconv.Itoa(i), System: translate.SystemPrompt(p, to), User: p.Spaced, MaxTokens: translate.TokenBudget(p.Spaced),
			Pattern: translate.Pattern(to),
		})
	}
	return list
}

// checkLines says what the translations kept of their sources, a line for each kind of paragraph and for each length
// of source.
func checkLines(checks translate.Checks) []string {
	if len(checks.Kinds) == 0 {
		return nil
	}
	lines := []string{"translations by kind:"}
	lines = append(lines, tallyLines(checks.Kinds)...)
	lines = append(lines, "translations by source length in characters:")
	return append(lines, tallyLines(checks.Lengths)...)
}

func tallyLines(tallies map[string]*translate.Tally) []string {
	var lines []string
	for _, label := range slices.Sorted(maps.Keys(tallies)) {
		t := tallies[label]
		var stops []string
		for _, stop := range slices.Sorted(maps.Keys(t.Stops)) {
			stops = append(stops, fmt.Sprintf("%s %d", stop, t.Stops[stop]))
		}
		lines = append(lines, fmt.Sprintf("  %-22s %4d, %d lacking a mark (%d of %d marks), %d short, stops: %s",
			label, t.Paragraphs, t.WithMissing, t.MissingMarks, t.Marks, t.Short, strings.Join(stops, ", ")))
	}
	return lines
}

// generate runs the requests and sums up what they took. A request that failed counts under the stop "failed".
func generate(client *worker.Client, requests []worker.Request, options worker.Options, onDone func(worker.Generated), onText func(key, text string)) (Generation, error) {
	result := Generation{BatchSize: options.BatchSize, StateMiB: options.StateMiB, Requests: len(requests), Stops: map[string]int{}}
	started := time.Now()
	tokens := 0
	pool, err := client.GenerateEach(requests, options, func(outcome worker.Generated) {
		if outcome.Error != "" {
			result.Stops["failed"]++
		} else {
			result.Stops[outcome.Stop]++
			tokens += outcome.Tokens
		}
		onDone(outcome)
	}, onText)
	result.Seconds = time.Since(started).Seconds()
	result.Pool = pool
	result.OutputTokens = tokens
	if places := options.BatchSize * pool.Steps; places > 0 {
		result.IdleShare = 1 - float64(pool.RowSteps)/float64(places)
	}
	if pool.PaddedTokens > 0 {
		result.PaddingShare = 1 - float64(pool.PromptTokens)/float64(pool.PaddedTokens)
	}
	if result.Seconds > 0 {
		result.TokensPerSecond = float64(tokens) / result.Seconds
	}
	return result, err
}
