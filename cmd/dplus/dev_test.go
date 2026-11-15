package main

import (
	"reflect"
	"testing"
)

func TestAPhaseShowsItsTimeWithoutWhatItsEventsCost(t *testing.T) {
	got := devLines([]devPhase{
		// Seven spans of 100 ms and a timed one of 250 ms whose 400 timed products cost 0.375 ms each: 100 ms are
		// left of it, 45 of them the int4 products'.
		{Phase: "decode", Spans: 8, Tokens: 128, GPUMs: 700, HeadMs: 28, TimedGPUMs: 250, TimedInt4Ms: 45, TimedHeadMs: 4, TimedInt4Calls: 400},
		{Phase: "prefill"},
		{Phase: "decision_state", Spans: 1, Tokens: 1500, TimedGPUMs: 1350, TimedInt4Ms: 900, TimedInt4Calls: 400},
		{Phase: "decision_suffix", Spans: 2, Tokens: 500, GPUMs: 600},
	}, 0.375)
	want := []string{
		"  decode               8 spans      128 tokens      0.8s, head 0.0s, int4 45%",
		"  decision_state       1 spans     1500 tokens      1.2s, int4 75%",
		"  decision_suffix      2 spans      500 tokens      0.6s",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got\n%q\nwant\n%q", got, want)
	}
}
