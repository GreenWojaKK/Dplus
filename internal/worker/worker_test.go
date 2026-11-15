package worker

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"dplus/internal/order"
)

// fake answers the protocol the way the real worker does, without a model.
const fake = `
import json, sys
record, every = sys.argv[5], sys.argv[6]
print(json.dumps({"ready": True, "device": "fake", "parameters": 5, "head_parameters": 0, "allocated_mib": 7, "dev": bool(record)}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    identifier = request["id"]
    if request["op"] == "decide":
        questions = request["questions"]
        if any(q.get("instructions") == "fail" for q in questions.values()):
            print(json.dumps({"id": identifier, "error": "ValueError: bad"}), flush=True)
        else:
            answers = {}
            for key, q in questions.items():
                if q.get("instructions") == "refuse":
                    answers[key] = {"type": q["type"], "error": "max_length_exceeded", "reason": "question_tokens", "message": "too long"}
                elif q["type"] == "choice":
                    # The state names the choice when it comes in two parts; an option with no description comes as null.
                    choice = request["moving"]["pick"] if "moving" in request else sorted(q["criteria"])[-1]
                    shown = sum(description is not None for description in q["criteria"].values())
                    answers[key] = {"choice": choice, "confidence": 0.25, "probabilities": {k: 0.75 if k == choice else 0.01 * shown for k in q["criteria"]}}
                else:
                    answers[key] = {"noul": 0.75}
            print(json.dumps({"id": identifier, "answers": answers}), flush=True)
    elif request["op"] == "dev":
        print(json.dumps({"id": identifier, "summary": {"record": "summary", "file": record, "every": int(every)}}), flush=True)
    elif request["requests"][0]["user"] == "crash":
        print(json.dumps({"id": identifier, "error": "RuntimeError: broken"}), flush=True)
    else:
        size = request["batch_size"]
        for start in range(0, len(request["requests"]), size):
            rows = request["requests"][start:start + size]
            for row in rows:
                if row["user"] == "fail":
                    print(json.dumps({"id": identifier, "key": row["key"], "done": True, "error": "ValueError: bad"}), flush=True)
                    continue
                if request["deltas"]:
                    for piece in ("가", "나"):
                        print(json.dumps({"id": identifier, "key": row["key"], "delta": piece}), flush=True)
                outcome = {"id": identifier, "key": row["key"], "done": True, "text": "가나", "stop": "eos", "tokens": 2, "seconds": 0.5}
                if request["probe"]:
                    outcome["token_ids"], outcome["first_top"] = [7, 8], [{"token": 7, "logit": 1.5}]
                print(json.dumps(outcome), flush=True)
            print(json.dumps({"id": identifier, "pool": {"rows": len(rows), "steps": 2, "row_steps": 3, "joins": 1, "peak_mib": 11}}), flush=True)
        print(json.dumps({"id": identifier, "finished": True}), flush=True)
`

func startFake(t *testing.T) *Client {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is not available")
	}
	original := source
	source = fake
	t.Cleanup(func() { source = original })
	client := &Client{Python: python, Model: "unused", Device: "cpu"}
	t.Cleanup(client.Close)
	return client
}

func TestTheWorkerImportsItsModulesAndTheirFolderGoesWithIt(t *testing.T) {
	client := startFake(t)
	source = `
import json, sys
import dry_penalty
penalties = dry_penalty.dry_penalties([1, 2, 3] * 4, frozenset(), 0.8, 1.75, 2, dry_penalty.max_exponent(1.75))
print(json.dumps({"ready": True, "device": "fake", "parameters": len(penalties), "head_parameters": 0, "allocated_mib": 0}), flush=True)
sys.stdin.read()
`
	if err := client.Start(); err != nil {
		t.Fatal(err)
	}
	if client.Info.Parameters != 1 {
		t.Fatalf("the module charged %d tokens", client.Info.Parameters)
	}
	folder := client.modules
	if _, err := os.Stat(filepath.Join(folder, "dry_penalty.py")); err != nil {
		t.Fatal(err)
	}
	client.Close()
	if _, err := os.Stat(folder); !os.IsNotExist(err) {
		t.Fatalf("the modules' folder stays after the worker ends: %v", err)
	}
}

func TestAChoiceIsAnswered(t *testing.T) {
	client := startFake(t)
	choice, confidence, err := client.Choose(map[string]any{"page": 1}, "next?", []order.Option{{Key: "b1", Description: "one"}, {Key: "b2", Description: "two"}}, nil)
	if err != nil || choice != "b2" || confidence != 0.25 {
		t.Fatalf("choice %q, confidence %v, error %v", choice, confidence, err)
	}
	if client.Info.Device != "fake" || client.Info.AllocatedMiB != 7 {
		t.Fatalf("ready reply %+v", client.Info)
	}
}

func TestRankSendsTheStateInTwoPartsAndReturnsEveryProbability(t *testing.T) {
	client := startFake(t)
	options := []order.Option{{Key: "b1"}, {Key: "b2"}, {Key: "b3", Description: "three"}}
	probabilities, err := client.Rank(map[string]any{"blocks": 3}, map[string]any{"pick": "b1"}, "next?", options, nil)
	if err != nil || len(probabilities) != 3 || probabilities["b1"] != 0.75 || probabilities["b2"] != 0.01 {
		t.Fatalf("probabilities %v, error %v", probabilities, err)
	}
}

func requests(users ...string) []Request {
	list := make([]Request, len(users))
	for i, user := range users {
		list[i] = Request{Key: string(rune('a' + i)), System: "system", User: user, MaxTokens: 8}
	}
	return list
}

func TestGenerateEachReturnsEveryOutcomeAndWhatThePoolTook(t *testing.T) {
	client := startFake(t)
	var outcomes []Generated
	pieces := map[string]string{}
	pool, err := client.GenerateEach(requests("one", "two", "three"), Options{BatchSize: 2},
		func(outcome Generated) { outcomes = append(outcomes, outcome) },
		func(key, text string) { pieces[key] += text })
	if err != nil || len(outcomes) != 3 {
		t.Fatalf("outcomes %+v, error %v", outcomes, err)
	}
	if outcomes[2].Key != "c" || outcomes[2].Text != "가나" || outcomes[2].Stop != "eos" || outcomes[2].Seconds != 0.5 || pieces["c"] != "가나" {
		t.Fatalf("outcome %+v, pieces %v", outcomes[2], pieces)
	}
	// The worker reported twice: the reports add up, and the most rows and the peak are kept.
	if pool.Rows != 2 || pool.Steps != 4 || pool.RowSteps != 6 || pool.Joins != 2 || pool.PeakMiB != 11 || outcomes[0].TokenIDs != nil {
		t.Fatalf("pool %+v, outcome %+v", pool, outcomes[0])
	}
	if client.next != 1 {
		t.Fatalf("sent %d requests", client.next)
	}
}

func TestGenerateEachSendsNoTextUnlessAsked(t *testing.T) {
	client := startFake(t)
	var outcomes []Generated
	if _, err := client.GenerateEach(requests("one"), Options{BatchSize: 1, Probe: true}, func(outcome Generated) { outcomes = append(outcomes, outcome) }, nil); err != nil {
		t.Fatal(err)
	}
	if len(outcomes) != 1 || len(outcomes[0].TokenIDs) != 2 || outcomes[0].FirstTop[0].Logit != 1.5 {
		t.Fatalf("outcomes %+v", outcomes)
	}
}

func TestAFailedGenerationDoesNotEndTheOthers(t *testing.T) {
	client := startFake(t)
	var outcomes []Generated
	_, err := client.GenerateEach(requests("one", "fail", "three"), Options{BatchSize: 4}, func(outcome Generated) { outcomes = append(outcomes, outcome) }, nil)
	if err != nil || len(outcomes) != 3 || outcomes[0].Error != "" || !strings.Contains(outcomes[1].Error, "bad") || outcomes[2].Text != "가나" {
		t.Fatalf("outcomes %+v, error %v", outcomes, err)
	}
}

func TestAGenerateRequestThatFailsIsReported(t *testing.T) {
	client := startFake(t)
	if _, err := client.GenerateEach(requests("crash"), Options{BatchSize: 1}, func(Generated) {}, nil); err == nil || !strings.Contains(err.Error(), "broken") {
		t.Fatalf("got %v", err)
	}
	if pool, err := client.GenerateEach(nil, Options{BatchSize: 1}, func(Generated) {}, nil); err != nil || pool.Steps != 0 || client.next != 1 {
		t.Fatalf("an empty list reached the worker: %+v, %v, %d", pool, err, client.next)
	}
}

func TestYesNoEachAnswersEveryQuestionInOneRequest(t *testing.T) {
	client := startFake(t)
	answers, err := client.YesNoEach("state", map[string]string{"b1": "one?", "b2": "two?"})
	if err != nil || len(answers) != 2 || answers["b1"] != 0.75 || answers["b2"] != 0.75 {
		t.Fatalf("answers %v, error %v", answers, err)
	}
	if client.next != 1 {
		t.Fatalf("sent %d requests", client.next)
	}
}

func TestChooseEachAnswersEveryQuestionInOneRequest(t *testing.T) {
	client := startFake(t)
	options := []order.Option{{Key: "p001", Description: "After p001"}, {Key: "p002", Description: "After p002"}}
	answers, err := client.ChooseEach("state", map[string]order.Question{"e1": {Text: "where?", Options: options}, "e2": {Text: "where?", Options: options}})
	if err != nil || len(answers) != 2 || answers["e1"] != "p002" || answers["e2"] != "p002" {
		t.Fatalf("answers %v, error %v", answers, err)
	}
	if client.next != 1 {
		t.Fatalf("sent %d requests", client.next)
	}
}

func TestAQuestionTheModelRefusesIsReported(t *testing.T) {
	client := startFake(t)
	_, err := client.YesNoEach("state", map[string]string{"b1": "one?", "b2": "refuse"})
	if err == nil || !strings.Contains(err.Error(), "b2") || !strings.Contains(err.Error(), "too long") {
		t.Fatalf("got %v", err)
	}
}

func TestAFailedRequestIsReportedAndTheWorkerGoesOn(t *testing.T) {
	client := startFake(t)
	if _, err := client.YesNoEach("state", map[string]string{"q": "fail"}); err == nil || !strings.Contains(err.Error(), "bad") {
		t.Fatalf("got %v", err)
	}
	if answers, err := client.YesNoEach("state", map[string]string{"q": "continue?"}); err != nil || answers["q"] != 0.75 {
		t.Fatalf("the worker did not go on: %v, %v", answers, err)
	}
}

func TestTheDeveloperModeGivesItsSummary(t *testing.T) {
	client := startFake(t)
	client.Record, client.Every = "run.dev.jsonl", 8
	if summary, err := client.Recorded(); err != nil || summary != nil {
		t.Fatalf("before the worker started: summary %s, error %v", summary, err)
	}
	if err := client.Start(); err != nil {
		t.Fatal(err)
	}
	summary, err := client.Recorded()
	if err != nil || !client.Info.Dev {
		t.Fatalf("error %v, ready reply %+v", err, client.Info)
	}
	if got := strings.ReplaceAll(string(summary), " ", ""); got != `{"record":"summary","file":"run.dev.jsonl","every":8}` {
		t.Fatalf("summary %s", got)
	}
}

func TestWithoutARecordFileTheDeveloperModeIsOff(t *testing.T) {
	client := startFake(t)
	if err := client.Start(); err != nil {
		t.Fatal(err)
	}
	if summary, err := client.Recorded(); err != nil || summary != nil || client.Info.Dev {
		t.Fatalf("summary %s, error %v, ready reply %+v", summary, err, client.Info)
	}
}

func TestAWorkerThatEndsIsReported(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is not available")
	}
	original := source
	source = "import sys; print('model file missing', file=sys.stderr); sys.exit(3)"
	t.Cleanup(func() { source = original })
	client := &Client{Python: python, Model: "unused", Device: "cpu"}
	t.Cleanup(client.Close)
	if _, err := client.YesNoEach("state", map[string]string{"q": "continue?"}); err == nil || !strings.Contains(err.Error(), "did not start") {
		t.Fatalf("got %v", err)
	}
}
