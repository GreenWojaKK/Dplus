// Package layout finds a page's blocks: Heron gives the boxes and their kinds, and the text layer fills them.
package layout

import (
	"context"
	"fmt"
	"image"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"

	"dplus/internal/doc"
)

const inputSize = 640

// labels are the kinds Heron's export numbers, in order.
var labels = []string{
	"caption", "footnote", "formula", "list_item", "page_footer", "page_header", "picture", "section_header", "table",
	"text", "title", "document_index", "code", "checkbox_selected", "checkbox_unselected", "form", "key_value_region",
}

// minimumScores are the confidence a box of each kind needs. Docling uses the same values for Heron.
var minimumScores = map[string]float64{
	"caption": 0.5, "footnote": 0.5, "formula": 0.5, "list_item": 0.5, "page_footer": 0.5, "page_header": 0.5,
	"picture": 0.5, "section_header": 0.45, "table": 0.5, "text": 0.5, "title": 0.45, "document_index": 0.45,
	"code": 0.45, "checkbox_selected": 0.45, "checkbox_unselected": 0.45, "form": 0.45, "key_value_region": 0.45,
}

// Detection is one box Heron found, in image pixels.
type Detection struct {
	Kind  string
	Score float64
	Box   doc.Box
}

// Heron runs the layout model.
type Heron struct{ session *ort.Session }

// NewHeron loads the ONNX export.
func NewHeron(modelPath string, threads int) (*Heron, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Close()
	if err := options.SetIntraOpNumThreads(threads); err != nil {
		return nil, err
	}
	session, err := ort.NewSession(modelPath, options)
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", modelPath, err)
	}
	return &Heron{session: session}, nil
}

// Close releases the model.
func (h *Heron) Close() { h.session.Close() }

// Detect runs the model on a page image and keeps the boxes that reach their kind's score.
func (h *Heron) Detect(ctx context.Context, page *image.RGBA) ([]Detection, error) {
	rgb := resizeRGB(page, inputSize, inputSize)
	pixels := make([]uint8, inputSize*inputSize*3)
	for i := range inputSize * inputSize {
		pixels[i] = rgb[i*3]
		pixels[inputSize*inputSize+i] = rgb[i*3+1]
		pixels[2*inputSize*inputSize+i] = rgb[i*3+2]
	}
	images, err := ort.CreateTensor([]int64{1, 3, inputSize, inputSize}, pixels)
	if err != nil {
		return nil, err
	}
	defer images.Close()
	bounds := page.Bounds()
	sizes, err := ort.CreateTensor([]int64{1, 2}, []int64{int64(bounds.Dx()), int64(bounds.Dy())})
	if err != nil {
		return nil, err
	}
	defer sizes.Close()
	outputs, err := h.session.Run(ctx, map[string]*ort.Tensor{"images": images, "orig_target_sizes": sizes}, nil)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, tensor := range outputs {
			tensor.Close()
		}
	}()
	kinds, err := ort.TensorData[int64](outputs["labels"])
	if err != nil {
		return nil, err
	}
	boxes, err := ort.TensorData[float32](outputs["boxes"])
	if err != nil {
		return nil, err
	}
	scores, err := ort.TensorData[float32](outputs["scores"])
	if err != nil {
		return nil, err
	}
	var found []Detection
	for i, score := range scores {
		if kinds[i] < 0 || int(kinds[i]) >= len(labels) {
			continue
		}
		kind := labels[kinds[i]]
		if float64(score) < minimumScores[kind] {
			continue
		}
		found = append(found, Detection{Kind: kind, Score: float64(score), Box: doc.Box{
			X0: float64(boxes[i*4]), Y0: float64(boxes[i*4+1]), X1: float64(boxes[i*4+2]), Y1: float64(boxes[i*4+3]),
		}})
	}
	return found, nil
}
