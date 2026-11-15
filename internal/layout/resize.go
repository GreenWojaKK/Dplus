package layout

import (
	"image"
	"math"
)

// The resizing Heron was trained with: PIL's bilinear filter, which widens its support when shrinking. Coefficients are
// fixed point with 22 fractional bits, horizontal pass first, as PIL does it.
const coefficientBits = 22

type coefficients struct {
	start  []int
	values [][]int32
}

func precompute(in, out int) coefficients {
	scale := float64(in) / float64(out)
	filterScale := math.Max(scale, 1)
	support := filterScale
	result := coefficients{start: make([]int, out), values: make([][]int32, out)}
	for o := range out {
		center := (float64(o) + 0.5) * scale
		low := max(int(center-support+0.5), 0)
		high := min(int(center+support+0.5), in)
		weights := make([]float64, high-low)
		total := 0.0
		for i := range weights {
			distance := math.Abs((float64(i+low) - center + 0.5) / filterScale)
			if distance < 1 {
				weights[i] = 1 - distance
			}
			total += weights[i]
		}
		fixed := make([]int32, len(weights))
		for i, w := range weights {
			value := w / total * (1 << coefficientBits)
			if value < 0 {
				fixed[i] = int32(value - 0.5)
			} else {
				fixed[i] = int32(value + 0.5)
			}
		}
		result.start[o], result.values[o] = low, fixed
	}
	return result
}

func clip(value int64) uint8 {
	value >>= coefficientBits
	if value < 0 {
		return 0
	}
	if value > 255 {
		return 255
	}
	return uint8(value)
}

// resizeRGB returns width x height x 3 bytes in row order, red first.
func resizeRGB(source *image.RGBA, width, height int) []byte {
	bounds := source.Bounds()
	sourceWidth, sourceHeight := bounds.Dx(), bounds.Dy()
	across := precompute(sourceWidth, width)
	down := precompute(sourceHeight, height)
	horizontal := make([]byte, width*sourceHeight*3)
	rounding := int64(1) << (coefficientBits - 1)
	for y := range sourceHeight {
		row := source.Pix[y*source.Stride:]
		for x := range width {
			for channel := range 3 {
				sum := rounding
				for i, k := range across.values[x] {
					sum += int64(row[(across.start[x]+i)*4+channel]) * int64(k)
				}
				horizontal[(y*width+x)*3+channel] = clip(sum)
			}
		}
	}
	out := make([]byte, width*height*3)
	for y := range height {
		for x := range width * 3 {
			sum := rounding
			for i, k := range down.values[y] {
				sum += int64(horizontal[(down.start[y]+i)*width*3+x]) * int64(k)
			}
			out[y*width*3+x] = clip(sum)
		}
	}
	return out
}
