package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"dplus/internal/worker"
)

// defaultDevEvery is how many spans of a phase share one whose int4 products are timed, unless -dev-every says
// otherwise. A timed span runs about two and a half times as long, so the whole run takes about a tenth longer.
const defaultDevEvery = 16

// devPhase is what the CLI shows of a phase of the developer mode's summary. The record's format is defined in
// rdna4-w4a16.
type devPhase struct {
	Phase          string  `json:"phase"`
	Spans          int64   `json:"spans"`
	Tokens         int64   `json:"tokens"`
	GPUMs          float64 `json:"gpu_ms"`
	HeadMs         float64 `json:"head_ms"`
	TimedGPUMs     float64 `json:"timed_gpu_ms"`
	TimedInt4Ms    float64 `json:"timed_int4_ms"`
	TimedHeadMs    float64 `json:"timed_head_ms"`
	TimedInt4Calls int64   `json:"timed_int4_calls"`
}

// devLines says where the device's time went, a line for each phase that ran. A phase's time is that of its spans,
// the timed ones without what their events cost: pairMs for each timed int4 product. The share of the int4 products
// is what they took in the timed spans against the time of those spans.
func devLines(phases []devPhase, pairMs float64) []string {
	var lines []string
	for _, phase := range phases {
		if phase.Spans == 0 {
			continue
		}
		timed := max(phase.TimedGPUMs-pairMs*float64(phase.TimedInt4Calls), 0)
		line := fmt.Sprintf("  %-16s %5d spans %8d tokens %8.1fs", phase.Phase, phase.Spans, phase.Tokens, (phase.GPUMs+timed)/1000)
		if head := phase.HeadMs + phase.TimedHeadMs; head > 0 {
			line += fmt.Sprintf(", head %.1fs", head/1000)
		}
		if phase.TimedInt4Calls > 0 && timed > 0 {
			line += fmt.Sprintf(", int4 %.0f%%", min(phase.TimedInt4Ms/timed, 1)*100)
		}
		lines = append(lines, line)
	}
	return lines
}

// setDev turns the developer mode of a client on: the worker records into the file.
func setDev(client *worker.Client, record string, every int) {
	client.Record, client.Every = record, every
}

// devReport ends the developer mode, prints its phases and returns what the report keeps of its summary: the record
// file's name, the costs of its events, the phases and the counters. Nil when the mode is off.
func devReport(client *worker.Client) (map[string]any, error) {
	if client.Record == "" {
		return nil, nil
	}
	if client.Started() && !client.Info.Dev {
		fmt.Fprintln(os.Stderr, "the developer mode is off: build the kernel library and its Python binding with tools/build.sh")
		return nil, nil
	}
	summary, err := client.Recorded()
	if err != nil || summary == nil {
		return nil, err
	}
	var parts map[string]json.RawMessage
	if err := json.Unmarshal(summary, &parts); err != nil {
		return nil, fmt.Errorf("reading the developer mode's summary: %w", err)
	}
	var phases []devPhase
	if err := json.Unmarshal(parts["phases"], &phases); err != nil {
		return nil, fmt.Errorf("reading the developer mode's phases: %w", err)
	}
	var pairMs float64
	if err := json.Unmarshal(parts["pair_ms"], &pairMs); err != nil {
		return nil, fmt.Errorf("reading the developer mode's costs: %w", err)
	}
	fmt.Fprintf(os.Stderr, "developer mode: %s\n", client.Record)
	for _, line := range devLines(phases, pairMs) {
		fmt.Fprintln(os.Stderr, line)
	}
	return map[string]any{
		"record": filepath.Base(client.Record), "every": parts["every"], "event_ms": parts["event_ms"], "pair_ms": parts["pair_ms"],
		"phases": parts["phases"], "counters": parts["counters"],
	}, nil
}
