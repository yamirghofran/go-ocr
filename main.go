package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"image/jpeg"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gen2brain/go-fitz"
	"github.com/joho/godotenv"
	"google.golang.org/genai"
)

// Constants for Gemini API
const (
	DEFAULT_SYSTEM_PROMPT = `
Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ` + "```" + `markdown or ` + "```" + `html.

RULES:
  - You must include all information on the page. Do not exclude headers, footers, or subtext.
  - Return tables in an HTML format.
  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
  - Logos should be wrapped in brackets. Ex: <logo>Coca-Cola<logo>
  - Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY<watermark>
  - Page numbers should be wrapped in brackets. Ex: <page_number>14<page_number> or <page_number>9/22<page_number>
  - Prefer using ☐ and ☑ for check boxes.
`
	MATCH_MARKDOWN_BLOCKS = `^` + "```" + `[a-z]*\n([\s\S]*?)\n` + "```" + `$`
	MATCH_CODE_BLOCKS     = `^` + "```" + `\n([\s\S]*?)\n` + "```" + `$`
	DEFAULT_BATCH_SIZE    = 50
	DEFAULT_CONCURRENCY   = 50
	MAX_RETRIES           = 3
	RETRY_DELAY_BASE_MS   = 1000
	RETRY_DELAY_MAX_MS    = 10000
)

// ConvertPDFToImages converts each page of a PDF file to a JPEG image using go-fitz
// pdfPath: path to the input PDF file
// outputDir: directory where the images will be saved
// Returns the number of pages converted and any error encountered
func ConvertPDFToImages(pdfPath string, outputDir string) (int, error) {
	// Make sure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return 0, fmt.Errorf("failed to create output directory: %v", err)
	}

	// Open the PDF document
	doc, err := fitz.New(pdfPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open PDF file: %v", err)
	}
	defer doc.Close()

	// Get the number of pages in the document
	numPages := doc.NumPage()

	// Process each page
	for i := 0; i < numPages; i++ {
		// Extract the page as an image
		img, err := doc.Image(i)
		if err != nil {
			return i, fmt.Errorf("failed to convert page %d to image: %v", i+1, err)
		}

		// Create output file
		outputPath := filepath.Join(outputDir, fmt.Sprintf("page_%d.jpg", i+1))
		outFile, err := os.Create(outputPath)
		if err != nil {
			return i, fmt.Errorf("failed to create output file for page %d: %v", i+1, err)
		}

		// Write the image as JPEG
		err = jpeg.Encode(outFile, img, &jpeg.Options{Quality: 85})
		outFile.Close()
		if err != nil {
			return i, fmt.Errorf("failed to encode image for page %d: %v", i+1, err)
		}

		fmt.Printf("Page %d converted to image: %s\n", i+1, outputPath)
	}

	return numPages, nil
}

// ProcessImageWithGemini sends an image to Gemini API for OCR processing
// imgPath: path to the image file
// apiKey: Gemini API key
// Returns the markdown text extracted from the image and any error encountered
func ProcessImageWithGemini(ctx context.Context, imgPath string, apiKey string) (string, error) {
	// Read the image file
	imgData, err := os.ReadFile(imgPath)
	if err != nil {
		return "", fmt.Errorf("error reading image: %v", err)
	}

	// Determine the MIME type
	mimeType := getMimeType(imgPath)
	if mimeType == "" {
		return "", fmt.Errorf("unsupported image format: %s", imgPath)
	}

	// Create Gemini client
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %w", err)
	}

	// Create content with the image and a prompt
	parts := []*genai.Part{
		{Text: DEFAULT_SYSTEM_PROMPT},
		{InlineData: &genai.Blob{Data: imgData, MIMEType: mimeType}},
	}

	// Generate content
	resp, err := client.Models.GenerateContent(ctx, "gemini-1.5-pro", []*genai.Content{{Parts: parts}}, nil)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	// Extract text from response
	text, err := resp.Text()
	if err != nil {
		return "", fmt.Errorf("failed to extract text from response: %w", err)
	}

	if text == "" {
		return "", fmt.Errorf("no text found in response")
	}

	// Extract markdown from response
	markdown := extractMarkdown(text)
	return markdown, nil
}

// ProcessImageWithRetry processes an image with retry logic
// ctx: context for cancellation
// imgPath: path to the image file
// apiKey: Gemini API key
// Returns the markdown text and any error encountered
func ProcessImageWithRetry(ctx context.Context, imgPath string, apiKey string) (string, error) {
	var lastErr error

	for attempt := 0; attempt <= MAX_RETRIES; attempt++ {
		// Check if context is cancelled
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		default:
			// Continue processing
		}

		// Process the image
		markdown, err := ProcessImageWithGemini(ctx, imgPath, apiKey)
		if err == nil {
			return markdown, nil
		}

		lastErr = err

		// If this wasn't our last attempt, wait before retrying
		if attempt < MAX_RETRIES {
			// Check for rate limit errors
			if strings.Contains(err.Error(), "rate limit") ||
				strings.Contains(err.Error(), "quota exceeded") ||
				strings.Contains(err.Error(), "resource exhausted") {

				// Exponential backoff with jitter
				backoffTime := time.Duration(math.Pow(2, float64(attempt+1))) * time.Second
				jitter := time.Duration(rand.Intn(1000)) * time.Millisecond

				fmt.Printf("Rate limit hit for %s, retrying in %.1f seconds (attempt %d/%d)...\n",
					filepath.Base(imgPath),
					float64(backoffTime+jitter)/float64(time.Second),
					attempt+1, MAX_RETRIES)

				select {
				case <-time.After(backoffTime + jitter):
					// Continue to next retry
				case <-ctx.Done():
					return "", ctx.Err()
				}
			} else {
				// For other errors, use a shorter backoff
				backoffTime := time.Duration(attempt+1) * 500 * time.Millisecond
				fmt.Printf("Error processing %s: %v, retrying in %.1f seconds (attempt %d/%d)...\n",
					filepath.Base(imgPath),
					err,
					float64(backoffTime)/float64(time.Second),
					attempt+1, MAX_RETRIES)
				time.Sleep(backoffTime)
			}
		}
	}

	return "", fmt.Errorf("failed after %d attempts: %w", MAX_RETRIES, lastErr)
}

// Helper function to get MIME type based on file extension
func getMimeType(filePath string) string {
	switch {
	case strings.HasSuffix(strings.ToLower(filePath), ".jpg"),
		strings.HasSuffix(strings.ToLower(filePath), ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(strings.ToLower(filePath), ".png"):
		return "image/png"
	case strings.HasSuffix(strings.ToLower(filePath), ".gif"):
		return "image/gif"
	case strings.HasSuffix(strings.ToLower(filePath), ".webp"):
		return "image/webp"
	// Add other image formats here
	default:
		return "" // Unknown or unsupported format
	}
}

// ProcessBase64ImageWithGemini processes a base64 encoded image with Gemini
func ProcessBase64ImageWithGemini(ctx context.Context, base64Img string, apiKey string, prompt string) (string, error) {
	// Decode the base64 string
	data, err := base64.StdEncoding.DecodeString(base64Img)
	if err != nil {
		return "", fmt.Errorf("error decoding base64 image: %w", err)
	}

	// Detect Mime Type from the base64 string prefix
	mimeType := detectMimeType(base64Img)
	if mimeType == "" {
		return "", fmt.Errorf("unsupported or invalid image format")
	}

	// Create Gemini client
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %w", err)
	}

	// Create content with the image and a prompt
	parts := []*genai.Part{
		{Text: prompt},
		{InlineData: &genai.Blob{Data: data, MIMEType: mimeType}},
	}

	// Generate content
	resp, err := client.Models.GenerateContent(ctx, "gemini-1.5-pro", []*genai.Content{{Parts: parts}}, nil)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	// Extract text from response
	text, err := resp.Text()
	if err != nil {
		return "", fmt.Errorf("failed to extract text from response: %w", err)
	}

	if text == "" {
		return "", fmt.Errorf("no text found in response")
	}

	return text, nil
}

// Detect Mime Type from the base64 data
func detectMimeType(base64String string) string {
	parts := strings.SplitN(base64String, ",", 2)
	if len(parts) < 2 {
		return ""
	}
	header := parts[0]

	switch {
	case strings.Contains(header, "image/jpeg"):
		return "image/jpeg"
	case strings.Contains(header, "image/png"):
		return "image/png"
	case strings.Contains(header, "image/gif"):
		return "image/gif"
	case strings.Contains(header, "image/webp"):
		return "image/webp"
	//add other types if you want to support
	default:
		return ""
	}
}

// ExtractMarkdown extracts markdown content from Gemini response
// text: the text response from Gemini
// Returns the extracted markdown
func extractMarkdown(text string) string {
	// Try to match markdown blocks first
	mdRegex := regexp.MustCompile(MATCH_MARKDOWN_BLOCKS)
	matches := mdRegex.FindStringSubmatch(text)
	if len(matches) > 1 {
		return matches[1]
	}

	// Try to match code blocks
	codeRegex := regexp.MustCompile(MATCH_CODE_BLOCKS)
	matches = codeRegex.FindStringSubmatch(text)
	if len(matches) > 1 {
		return matches[1]
	}

	// If no blocks found, return the original text
	return text
}

// BatchResult represents the result of processing a batch of images
type BatchResult struct {
	BatchNumber int
	Results     []string
	Error       error
}

// ProcessImagesInBatches processes images in batches and saves the combined markdown
// imgDir: directory containing the images
// outputPath: path to save the markdown file
// apiKey: Gemini API key
// batchSize: number of images to process in each batch (0 for default)
// maxConcurrent: maximum number of concurrent requests (0 for default)
// cleanupImages: whether to delete the image files after processing
// Returns any error encountered
func ProcessImagesInBatches(ctx context.Context, imgDir string, outputPath string, apiKey string,
	batchSize int, maxConcurrent int, cleanupImages bool) error {
	// Create temp directory for batch results
	tempDir := filepath.Join(imgDir, "batch_results")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return fmt.Errorf("failed to create temp directory: %v", err)
	}

	// Get all jpg files in the directory
	files, err := filepath.Glob(filepath.Join(imgDir, "*.jpg"))
	if err != nil {
		return fmt.Errorf("failed to list image files: %v", err)
	}

	if len(files) == 0 {
		return fmt.Errorf("no image files found in directory: %s", imgDir)
	}

	// Sort files by page number
	sort.Slice(files, func(i, j int) bool {
		numI := extractPageNumber(files[i])
		numJ := extractPageNumber(files[j])
		return numI < numJ
	})

	// Set default batch size if not specified or invalid
	if batchSize <= 0 {
		batchSize = DEFAULT_BATCH_SIZE
	}

	// Set default concurrency if not specified or invalid
	if maxConcurrent <= 0 {
		maxConcurrent = DEFAULT_CONCURRENCY
	}

	// Calculate number of batches
	totalFiles := len(files)
	totalBatches := (totalFiles + batchSize - 1) / batchSize // Ceiling division

	fmt.Printf("Processing %d images in %d batches (batch size: %d, concurrency: %d)\n",
		totalFiles, totalBatches, batchSize, maxConcurrent)

	// Check for existing batch results
	existingBatches := make(map[int]bool)
	existingBatchFiles, _ := filepath.Glob(filepath.Join(tempDir, "batch_*.md"))
	for _, batchFile := range existingBatchFiles {
		batchNumStr := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(batchFile), "batch_"), ".md")
		batchNum, err := strconv.Atoi(batchNumStr)
		if err == nil {
			existingBatches[batchNum] = true
			fmt.Printf("Found existing batch result: %s\n", filepath.Base(batchFile))
		}
	}

	// Process in batches
	var allResults []string
	for batchNum := 0; batchNum < totalBatches; batchNum++ {
		batchFilePath := filepath.Join(tempDir, fmt.Sprintf("batch_%d.md", batchNum))

		// Skip already processed batches
		if existingBatches[batchNum] {
			// Load existing batch result
			batchContent, err := os.ReadFile(batchFilePath)
			if err != nil {
				return fmt.Errorf("error reading existing batch file %s: %v", batchFilePath, err)
			}

			// Split content into lines (results)
			batchResults := strings.Split(string(batchContent), "\n")
			allResults = append(allResults, batchResults...)

			fmt.Printf("Loaded existing batch %d/%d\n", batchNum+1, totalBatches)
			continue
		}

		// Calculate batch range
		startIdx := batchNum * batchSize
		endIdx := startIdx + batchSize
		if endIdx > totalFiles {
			endIdx = totalFiles
		}

		batchFiles := files[startIdx:endIdx]

		fmt.Printf("Processing batch %d/%d (images %d-%d)...\n",
			batchNum+1, totalBatches, startIdx+1, endIdx)

		// Process this batch with retry logic
		var batchResults []string
		var batchErr error

		for retryCount := 0; retryCount <= 2; retryCount++ {
			// Process the batch
			batchResults, batchErr = processBatch(ctx, batchFiles, apiKey, maxConcurrent)

			if batchErr == nil {
				break // Success, no need to retry
			}

			// If this wasn't our last retry, wait before retrying
			if retryCount < 2 {
				backoffTime := time.Duration(math.Pow(2, float64(retryCount+1))) * time.Second
				fmt.Printf("Batch %d failed: %v, retrying in %v seconds (attempt %d/3)...\n",
					batchNum+1, batchErr, backoffTime/time.Second, retryCount+1)
				time.Sleep(backoffTime)
			}
		}

		if batchErr != nil {
			return fmt.Errorf("error processing batch %d after multiple retries: %w", batchNum+1, batchErr)
		}

		// Save batch results to temporary file
		batchContent := strings.Join(batchResults, "\n")
		if err := os.WriteFile(batchFilePath, []byte(batchContent), 0644); err != nil {
			return fmt.Errorf("error saving batch results: %v", err)
		}

		// Add batch results to overall results
		allResults = append(allResults, batchResults...)

		fmt.Printf("Completed batch %d/%d\n", batchNum+1, totalBatches)
	}

	// Combine all markdown content
	combinedMarkdown := strings.Join(allResults, "\n\n---\n\n")

	// Write to file
	err = os.WriteFile(outputPath, []byte(combinedMarkdown), 0644)
	if err != nil {
		return fmt.Errorf("failed to write markdown file: %v", err)
	}

	fmt.Printf("Successfully saved markdown to %s\n", outputPath)

	// Clean up images if requested
	if cleanupImages {
		// Clean up batch results directory
		if err := os.RemoveAll(tempDir); err != nil {
			fmt.Printf("Warning: failed to clean up batch results directory: %v\n", err)
		}

		// Clean up image files
		if err := CleanupImagesDirectory(imgDir); err != nil {
			return fmt.Errorf("error cleaning up images directory: %v", err)
		}
		fmt.Printf("Successfully cleaned up images directory: %s\n", imgDir)
	}

	return nil
}

// processBatch processes a batch of image files concurrently
// files: slice of image file paths to process
// apiKey: Gemini API key
// maxConcurrent: maximum number of concurrent requests
// Returns the markdown results and any error encountered
func processBatch(ctx context.Context, files []string, apiKey string, maxConcurrent int) ([]string, error) {
	// Create a semaphore to limit concurrency
	sem := make(chan struct{}, maxConcurrent)

	// Process images concurrently with limited concurrency
	var wg sync.WaitGroup
	resultsChan := make(chan struct {
		index    int
		markdown string
		err      error
	}, len(files))

	// Track progress
	var processedCount int32
	totalCount := int32(len(files))

	fmt.Printf("Starting batch processing with concurrency limit of %d for %d files\n", maxConcurrent, len(files))

	for i, file := range files {
		wg.Add(1)
		go func(idx int, imgPath string) {
			defer wg.Done()

			// Log when starting to process an image
			imgName := filepath.Base(imgPath)
			fmt.Printf("Starting to process image: %s\n", imgName)

			// Acquire semaphore (blocks if maxConcurrent goroutines are already running)
			select {
			case sem <- struct{}{}:
				// We acquired the semaphore
				fmt.Printf("Acquired semaphore for image: %s\n", imgName)
				defer func() {
					<-sem // Release semaphore when done
					fmt.Printf("Released semaphore for image: %s\n", imgName)
				}()
			case <-ctx.Done():
				// Context was cancelled
				fmt.Printf("Context cancelled while waiting for semaphore: %s\n", imgName)
				resultsChan <- struct {
					index    int
					markdown string
					err      error
				}{idx, "", ctx.Err()}
				return
			}

			startTime := time.Now()

			// Process the image with retry logic
			markdown, err := ProcessImageWithRetry(ctx, imgPath, apiKey)

			// Log when finished processing an image
			elapsed := time.Since(startTime)
			if err != nil {
				fmt.Printf("Failed processing image: %s (took %.2f seconds): %v\n",
					imgName, elapsed.Seconds(), err)
			} else {
				fmt.Printf("Finished processing image: %s (took %.2f seconds)\n",
					imgName, elapsed.Seconds())
			}

			// Update progress
			count := atomic.AddInt32(&processedCount, 1)
			fmt.Printf("Progress: %d/%d (%.1f%%) complete\n",
				count, totalCount, float64(count)/float64(totalCount)*100)

			resultsChan <- struct {
				index    int
				markdown string
				err      error
			}{idx, markdown, err}
		}(i, file)
	}

	// Close channel when all goroutines are done
	go func() {
		wg.Wait()
		close(resultsChan)
		fmt.Println("All goroutines completed for this batch")
	}()

	// Collect results
	results := make([]string, len(files))
	for result := range resultsChan {
		if result.err != nil {
			return nil, fmt.Errorf("error processing image %s: %w",
				filepath.Base(files[result.index]), result.err)
		}
		results[result.index] = result.markdown
	}

	fmt.Printf("Batch processing complete for %d files\n", len(files))
	return results, nil
}

// CleanupImagesDirectory removes all image files from the specified directory
// imgDir: directory containing the images to delete
// Returns any error encountered
func CleanupImagesDirectory(imgDir string) error {
	// Get all image files in the directory
	patterns := []string{"*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp"}

	for _, pattern := range patterns {
		files, err := filepath.Glob(filepath.Join(imgDir, pattern))
		if err != nil {
			return fmt.Errorf("failed to list image files: %v", err)
		}

		// Delete each file
		for _, file := range files {
			if err := os.Remove(file); err != nil {
				return fmt.Errorf("failed to delete file %s: %v", file, err)
			}
		}
	}

	return nil
}

// extractPageNumber extracts the page number from a filename like "page_1.jpg"
func extractPageNumber(filename string) int {
	// Extract the base filename without directory and extension
	base := filepath.Base(filename)
	// Remove extension
	base = strings.TrimSuffix(base, filepath.Ext(base))
	// Split by underscore and get the last part
	parts := strings.Split(base, "_")
	if len(parts) < 2 {
		return 0 // Default to 0 if format doesn't match
	}
	// Convert to integer
	num, err := strconv.Atoi(parts[len(parts)-1])
	if err != nil {
		return 0 // Default to 0 if conversion fails
	}
	return num
}

// DownloadPDF downloads a PDF from a URL if the file doesn't exist locally.
// It returns the local file path where the PDF is stored.
func DownloadPDF(pdfPath string) (string, error) {
	// Check if pdfPath is a URL
	_, err := url.ParseRequestURI(pdfPath)
	if err != nil || (!strings.HasPrefix(pdfPath, "http://") && !strings.HasPrefix(pdfPath, "https://")) {
		// Not a URL, assume it's a local file
		if _, err := os.Stat(pdfPath); os.IsNotExist(err) {
			return "", fmt.Errorf("local file not found: %s", pdfPath)
		}
		return pdfPath, nil
	}

	// It's a URL, download it
	// Create a temporary file name based on the URL
	fileName := filepath.Base(pdfPath)
	if fileName == "" || fileName == "." || fileName == "/" {
		fileName = fmt.Sprintf("downloaded_%d.pdf", time.Now().Unix())
	}

	// Check if the file already exists
	if _, err := os.Stat(fileName); err == nil {
		return fileName, nil
	}

	// Create the file
	out, err := os.Create(fileName)
	if err != nil {
		return "", fmt.Errorf("error creating file: %w", err)
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(pdfPath)
	if err != nil {
		return "", fmt.Errorf("error downloading file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("bad status: %s", resp.Status)
	}

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return "", fmt.Errorf("error writing file: %w", err)
	}

	return fileName, nil
}

// OcrPdfToMarkdown converts a PDF to markdown using Gemini API.
// It takes the PDF path (local file path or URL) and API key as input
// and returns the markdown content or an error.
func OcrPdfToMarkdown(pdfPath string, apiKey string) (string, error) {
	// Download the PDF if it's a URL
	localPdfPath, err := DownloadPDF(pdfPath)
	if err != nil {
		return "", fmt.Errorf("error getting PDF: %w", err)
	}

	outputDir := "output_images" // You can make this configurable if needed
	markdownPath := "output.md"  // You can make this configurable if needed
	ctx := context.Background()

	// Convert PDF to images
	numConverted, err := ConvertPDFToImages(localPdfPath, outputDir)
	if err != nil {
		return "", fmt.Errorf("error converting PDF to images: %w", err)
	}
	fmt.Printf("Successfully converted %d pages to images\n", numConverted)

	// Process images in batches with concurrency limit and save markdown
	err = ProcessImagesInBatches(ctx, outputDir, markdownPath, apiKey,
		DEFAULT_BATCH_SIZE, DEFAULT_CONCURRENCY, true)
	if err != nil {
		return "", fmt.Errorf("error processing images in batches: %w", err)
	}

	// Read the generated markdown file
	markdownContent, err := os.ReadFile(markdownPath)
	if err != nil {
		return "", fmt.Errorf("error reading markdown file: %w", err)
	}

	return string(markdownContent), nil
}

func main() {
	// Load environment variables from .env file
	if err := godotenv.Load(); err != nil {
		fmt.Println("Error loading .env file:", err)
		os.Exit(1)
	}
	// Example usage
	pdfPath := "input.pdf"
	outputDir := "output_images"
	markdownPath := "output.md"
	apiKey := os.Getenv("GEMINI_API_KEY")

	if apiKey == "" {
		fmt.Println("Error: GEMINI_API_KEY environment variable not set")
		os.Exit(1)
	}

	// Convert PDF to images
	numConverted, err := ConvertPDFToImages(pdfPath, outputDir)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Successfully converted %d pages to images\n", numConverted)

	// Process images in batches with concurrency limit and save markdown
	ctx := context.Background()

	// Use batch size of 50, concurrency of 25, and clean up after processing
	err = ProcessImagesInBatches(ctx, outputDir, markdownPath, apiKey,
		DEFAULT_BATCH_SIZE, DEFAULT_CONCURRENCY, true)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}
