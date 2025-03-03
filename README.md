# Go OCR with Gemini

This application converts PDF documents to markdown text using Google's Gemini AI. It works by:

1. Converting each page of a PDF to an image
2. Sending each image to Gemini for OCR processing
3. Extracting the markdown text from Gemini's response
4. Combining all pages into a single markdown file

## Requirements

- Go 1.21 or later
- Google Gemini API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yamirghofran0/go-ocr.git
cd go-ocr

# Install dependencies
go mod download
```

## Usage

1. Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

2. Place your PDF file in the project directory and name it `input.pdf`

3. Run the application:

```bash
go run main.go
```

4. The application will:
   - Convert the PDF to images in the `output_images` directory
   - Process each image with Gemini
   - Save the combined markdown to `output.md`

## Customization

You can modify the following variables in `main.go` to customize the behavior:

```go
pdfPath := "input.pdf"        // Path to your input PDF
outputDir := "output_images"  // Directory for storing images
markdownPath := "output.md"   // Output markdown file
```

## How It Works

1. **PDF to Images**: Uses the `go-fitz` library to convert each page of the PDF to a JPEG image.
2. **OCR Processing**: Sends each image to Gemini with a system prompt that instructs it to convert the document to markdown.
3. **Concurrent Processing**: Processes multiple images concurrently for faster results.
4. **Markdown Extraction**: Extracts and formats the markdown text from Gemini's response.
5. **Result Aggregation**: Combines all pages into a single markdown file with page separators.

## License

MIT 