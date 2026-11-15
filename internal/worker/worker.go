// Package worker starts and talks to the Python process that owns the model.
package worker

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"dplus/internal/order"
)

//go:embed backbone.py
var source string

const (
	maxReply  = 64 << 20
	logLines  = 30
	endingFor = 5 * time.Second
)

// Ready is what the worker reports once the model is loaded.
type Ready struct {
	Device         string `json:"device"`
	Parameters     int64  `json:"parameters"`
	HeadParameters int64  `json:"head_parameters"`
	PackedLinear   bool   `json:"packed_linear"` // The int4 layers multiply up to 512 rows straight from their packed weights.
	Dev            bool   `json:"dev"`           // The developer mode records where the device's time goes.
	AllocatedMiB   int64  `json:"allocated_mib"`
}

// Request is one generation. The worker runs requests of similar length together.
type Request struct {
	Key       string `json:"key"`
	System    string `json:"system"`
	User      string `json:"user"`
	MaxTokens int    `json:"max_tokens"`
	// Pattern is a regular expression the whole output must match. Generation takes no token that leaves it.
	Pattern string `json:"pattern,omitempty"`
}

// Candidate is a token and its logit.
type Candidate struct {
	Token int     `json:"token"`
	Logit float64 `json:"logit"`
}

// Generated is the outcome of one generation.
type Generated struct {
	Key          string  `json:"key"`
	Text         string  `json:"text"`
	Stop         string  `json:"stop"`
	PromptTokens int     `json:"prompt_tokens"`
	Tokens       int     `json:"tokens"`
	Seconds      float64 `json:"seconds"` // From the start of its prefill to its last token.
	Error        string  `json:"error,omitempty"`
	// With a probe: the generated tokens, and the most probable first tokens with their logits.
	TokenIDs []int       `json:"token_ids,omitempty"`
	FirstTop []Candidate `json:"first_top,omitempty"`
}

// Pool is what the pool of generations took.
type Pool struct {
	Rows           int     `json:"rows"`          // The most rows that ran together.
	Steps          int     `json:"steps"`         // Decode steps.
	RowSteps       int     `json:"row_steps"`     // Rows that ran, summed over the decode steps.
	Joins          int     `json:"joins"`         // Times waiting requests were prefilled and added.
	PromptTokens   int     `json:"prompt_tokens"` // Over all requests, before padding.
	PaddedTokens   int     `json:"padded_tokens"` // Prompt tokens prefilled, padding included.
	OutputTokens   int     `json:"output_tokens"`
	PrefillSeconds float64 `json:"prefill_seconds"`
	DecodeSeconds  float64 `json:"decode_seconds"`
	PeakMiB        int64   `json:"peak_mib"`
}

// add sums up two reports of the worker: it starts a new pool after one that failed.
func (p *Pool) add(other Pool) {
	p.Rows, p.PeakMiB = max(p.Rows, other.Rows), max(p.PeakMiB, other.PeakMiB)
	p.Steps += other.Steps
	p.RowSteps += other.RowSteps
	p.Joins += other.Joins
	p.PromptTokens += other.PromptTokens
	p.PaddedTokens += other.PaddedTokens
	p.OutputTokens += other.OutputTokens
	p.PrefillSeconds += other.PrefillSeconds
	p.DecodeSeconds += other.DecodeSeconds
}

// Client runs the worker on first use.
type Client struct {
	Python string
	Model  string // The d3-ultra-INT4 folder.
	Head   string // The safetensors file holding the vocabulary head.
	Device string
	// The library of device kernels. Without the file the worker runs on the checkpoint's own layers.
	Kernels string
	// The developer mode: the file the kernel library records the worker's spans and products into, and how many
	// spans of a phase share one whose int4 products are timed. Off without a file.
	Record string
	Every  int

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	logs   *tail
	next   int
	Info   Ready
}

type tail struct {
	mu    sync.Mutex
	lines []string
}

func (t *tail) add(line string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lines = append(t.lines, line)
	if len(t.lines) > logLines {
		t.lines = t.lines[len(t.lines)-logLines:]
	}
}

func (t *tail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return strings.Join(t.lines, "\n")
}

func (c *Client) start() error {
	if c.cmd != nil {
		return nil
	}
	cmd := exec.Command(c.Python, "-c", source, c.Model, c.Head, c.Device, c.Kernels, c.Record, strconv.Itoa(c.Every))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting the model worker: %w", err)
	}
	c.cmd, c.stdin = cmd, stdin
	c.logs = &tail{}
	go func() {
		lines := bufio.NewScanner(stderr)
		lines.Buffer(make([]byte, 64<<10), 4<<20)
		for lines.Scan() {
			c.logs.add(lines.Text())
		}
	}()
	c.stdout = bufio.NewScanner(stdout)
	c.stdout.Buffer(make([]byte, 64<<10), maxReply)
	var ready struct {
		Ready
		IsReady bool `json:"ready"`
	}
	if err := c.read(&ready); err != nil {
		return fmt.Errorf("the model worker did not start: %w", err)
	}
	c.Info = ready.Ready
	return nil
}

// Start loads the model now instead of at the first request.
func (c *Client) Start() error { return c.start() }

// Started reports whether the model has been loaded.
func (c *Client) Started() bool { return c.cmd != nil }

func (c *Client) read(target any) error {
	if !c.stdout.Scan() {
		if err := c.stdout.Err(); err != nil {
			return err
		}
		return fmt.Errorf("the model worker ended:\n%s", c.logs)
	}
	return json.Unmarshal(c.stdout.Bytes(), target)
}

func (c *Client) send(request map[string]any) (int, error) {
	if err := c.start(); err != nil {
		return 0, err
	}
	c.next++
	request["id"] = c.next
	line, err := json.Marshal(request)
	if err != nil {
		return 0, err
	}
	if _, err := c.stdin.Write(append(line, '\n')); err != nil {
		return 0, fmt.Errorf("writing to the model worker: %w\n%s", err, c.logs)
	}
	return c.next, nil
}

// Recorded ends the developer mode and returns its summary, which is also the last line of the record file. It is
// nil when the mode is off or the worker never started.
func (c *Client) Recorded() (json.RawMessage, error) {
	if c.cmd == nil || !c.Info.Dev {
		return nil, nil
	}
	identifier, err := c.send(map[string]any{"op": "dev"})
	if err != nil {
		return nil, err
	}
	var reply struct {
		ID      int             `json:"id"`
		Error   string          `json:"error"`
		Summary json.RawMessage `json:"summary"`
	}
	if err := c.read(&reply); err != nil {
		return nil, err
	}
	if reply.Error != "" {
		return nil, errors.New(reply.Error)
	}
	if reply.ID != identifier {
		return nil, fmt.Errorf("the model worker answered request %d, not %d", reply.ID, identifier)
	}
	if string(reply.Summary) == "null" {
		return nil, nil
	}
	return reply.Summary, nil
}

// Close ends the worker: it stops when its input closes.
func (c *Client) Close() {
	if c.cmd == nil {
		return
	}
	c.stdin.Close()
	done := make(chan struct{})
	go func() {
		c.cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(endingFor):
		c.cmd.Process.Kill()
		<-done
	}
	c.cmd = nil
}

type answer struct {
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	Noul          float64            `json:"noul"`
	Error         string             `json:"error"`
	Message       string             `json:"message"`
}

func (c *Client) decide(state any, questions map[string]any, images []string) (map[string]answer, error) {
	return c.ask(map[string]any{"op": "decide", "state": state, "questions": questions}, images)
}

func (c *Client) ask(request map[string]any, images []string) (map[string]answer, error) {
	if len(images) > 0 {
		request["images"] = images
	}
	identifier, err := c.send(request)
	if err != nil {
		return nil, err
	}
	var reply struct {
		ID      int               `json:"id"`
		Error   string            `json:"error"`
		Answers map[string]answer `json:"answers"`
	}
	if err := c.read(&reply); err != nil {
		return nil, err
	}
	if reply.Error != "" {
		return nil, errors.New(reply.Error)
	}
	if reply.ID != identifier {
		return nil, fmt.Errorf("the model worker answered request %d, not %d", reply.ID, identifier)
	}
	for key, item := range reply.Answers { // The model refuses single questions and answers the rest of the request.
		if item.Error != "" {
			return nil, fmt.Errorf("the decision model did not answer question %s: %s: %s", key, item.Error, item.Message)
		}
	}
	return reply.Answers, nil
}

func choiceQuestion(question string, options []order.Option) map[string]any {
	criteria := make(map[string]any, len(options))
	for _, option := range options {
		criteria[option.Key] = nil // An option with no description is shown by its key alone.
		if option.Description != "" {
			criteria[option.Key] = option.Description
		}
	}
	return map[string]any{"type": "choice", "instructions": question, "criteria": criteria}
}

// Rank asks a choice question about a state in two parts and returns the probability of each option. The worker
// computes the prompt up to the end of the fixed part once for the questions that share it.
func (c *Client) Rank(fixed, moving any, question string, options []order.Option, images []string) (map[string]float64, error) {
	request := map[string]any{"op": "decide", "state": fixed, "moving": moving, "questions": map[string]any{"q": choiceQuestion(question, options)}}
	answers, err := c.ask(request, images)
	if err != nil {
		return nil, err
	}
	return answers["q"].Probabilities, nil
}

// Choose asks a choice question.
func (c *Client) Choose(state any, question string, options []order.Option, images []string) (string, float64, error) {
	answers, err := c.decide(state, map[string]any{"q": choiceQuestion(question, options)}, images)
	return answers["q"].Choice, answers["q"].Confidence, err
}

// ChooseEach asks several choice questions about one state in one request and returns the option chosen for each.
func (c *Client) ChooseEach(state any, questions map[string]order.Question) (map[string]string, error) {
	asked := make(map[string]any, len(questions))
	for key, question := range questions {
		asked[key] = choiceQuestion(question.Text, question.Options)
	}
	answers, err := c.decide(state, asked, nil)
	if err != nil {
		return nil, err
	}
	choices := make(map[string]string, len(answers))
	for key, item := range answers {
		choices[key] = item.Choice
	}
	return choices, nil
}

// YesNoEach asks several yes-or-no questions about one state in one request.
func (c *Client) YesNoEach(state any, questions map[string]string) (map[string]float64, error) {
	asked := make(map[string]any, len(questions))
	for key, question := range questions {
		asked[key] = map[string]any{"type": "noul", "instructions": question}
	}
	answers, err := c.decide(state, asked, nil)
	if err != nil {
		return nil, err
	}
	probabilities := make(map[string]float64, len(answers))
	for key, item := range answers {
		probabilities[key] = item.Noul
	}
	return probabilities, nil
}

// Options say how a list of generations runs.
type Options struct {
	BatchSize int  // Rows that may run together.
	StateMiB  int  // Device memory the states of the rows may take, or 0 for the worker's own bound.
	Probe     bool // Each outcome carries its tokens and first logits.
}

// GenerateEach runs the requests, up to BatchSize together, and calls onDone as each finishes, in the order they
// finish. A request that fails carries its error. With onText the worker also sends each request's text as it forms.
// It returns what the pool of generations took.
func (c *Client) GenerateEach(requests []Request, options Options, onDone func(Generated), onText func(key, text string)) (Pool, error) {
	var pool Pool
	if len(requests) == 0 {
		return pool, nil
	}
	request := map[string]any{
		"op": "generate", "requests": requests, "batch_size": options.BatchSize, "deltas": onText != nil, "probe": options.Probe,
	}
	if options.StateMiB > 0 {
		request["state_mib"] = options.StateMiB
	}
	identifier, err := c.send(request)
	if err != nil {
		return pool, err
	}
	for {
		var reply struct {
			Generated
			ID       int    `json:"id"`
			Delta    string `json:"delta"`
			Done     bool   `json:"done"`
			Pool     *Pool  `json:"pool"`
			Finished bool   `json:"finished"`
		}
		if err := c.read(&reply); err != nil {
			return pool, err
		}
		if reply.ID != identifier {
			return pool, fmt.Errorf("the model worker answered request %d, not %d", reply.ID, identifier)
		}
		switch {
		case reply.Done:
			onDone(reply.Generated)
		case reply.Error != "":
			return pool, errors.New(reply.Error)
		case reply.Pool != nil:
			pool.add(*reply.Pool)
		case reply.Finished:
			return pool, nil
		case onText != nil:
			onText(reply.Key, reply.Delta)
		}
	}
}
