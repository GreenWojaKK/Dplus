package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"

	"dplus/internal/doc"
	"dplus/internal/pdf"
)

// pictureDPI is the resolution a page is drawn at to crop its pictures.
const pictureDPI = 200

// pictureRect gives the pixels a box in page points covers on a page drawn scale pixels to the point, widened to whole
// pixels and kept on the image.
func pictureRect(box doc.Box, scaleX, scaleY float64, bounds image.Rectangle) image.Rectangle {
	return image.Rect(
		int(math.Floor(box.X0*scaleX)), int(math.Floor(box.Y0*scaleY)),
		int(math.Ceil(box.X1*scaleX)), int(math.Ceil(box.Y1*scaleY)),
	).Intersect(bounds)
}

// savePictures crops each picture of a page from the page drawn at pictureDPI and saves it as a PNG file in folder.
// Each picture block records its file as the folder's name and the file's name, so the output files beside the folder
// can link it.
func savePictures(document *pdf.Document, index int, width, height float64, blocks []doc.Block, folder string) error {
	var pictures []int
	for i, block := range blocks {
		if block.Kind == "picture" {
			pictures = append(pictures, i)
		}
	}
	if len(pictures) == 0 {
		return nil
	}
	page, err := document.Render(index, pictureDPI)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(folder, 0o755); err != nil {
		return err
	}
	bounds := page.Bounds()
	scaleX, scaleY := float64(bounds.Dx())/width, float64(bounds.Dy())/height
	for _, i := range pictures {
		block := &blocks[i]
		rect := pictureRect(block.Box, scaleX, scaleY, bounds)
		if rect.Empty() {
			continue
		}
		name := fmt.Sprintf("page-%d-block-%d.png", block.Page, block.ID)
		if err := writePNG(filepath.Join(folder, name), page.SubImage(rect)); err != nil {
			return err
		}
		block.Image = filepath.Base(folder) + "/" + name
	}
	return nil
}

func writePNG(path string, picture image.Image) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(file, picture); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}
