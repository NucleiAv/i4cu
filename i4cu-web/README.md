# I fore see you - AI Deepfake Detection

A web application that uses AI to detect if images are deepfakes or AI-generated. Built with Cloudflare Workers AI and Durable Objects.

## What It Does

Upload images and the system will analyze them using multiple AI models to determine if they're authentic or AI-generated. It checks for:
- Visual inconsistencies and artifacts
- Watermarks and text markers
- EXIF metadata patterns
- Signs of manipulation

## Prerequisites

- Node.js 18+ installed
- A Cloudflare account
- Cloudflare Workers AI enabled (free tier available)

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd cloudflare-proj
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Login to Cloudflare**
   ```bash
   npx wrangler login
   ```

4. **Enable Workers AI** (if not already enabled)
   - Go to your Cloudflare dashboard
   - Navigate to Workers & Pages
   - Enable Workers AI in your account

## Running Locally

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:8787`

## Deploying

Deploy to Cloudflare:
```bash
npm run deploy
```

After deployment, you'll get a URL where your app is live.

## How to Use

1. Open the app in your browser
2. Upload an image (drag and drop or click to select)
3. Click "Analyze for Deepfake"
4. Wait for the analysis (takes 10-30 seconds)
5. View the results showing if it's authentic or AI-generated

## Features

- **Multiple AI Models**: Uses ensemble detection with 3+ AI models for accuracy
- **Watermark Detection**: Finds AI platform watermarks (Gemini, DALL-E, Midjourney, etc.)
- **OCR**: Extracts text that might indicate AI generation
- **Metadata Analysis**: Checks EXIF data for suspicious patterns
- **Detection History**: Saves all analyses in a database
- **Batch Processing**: Upload and analyze multiple images at once

## Project Structure

```
cloudflare-proj/
├── src/
│   ├── index.ts      # Main worker entry point
│   └── agent.ts      # Deepfake detection logic
├── package.json      # Dependencies
├── wrangler.toml     # Cloudflare configuration
└── tsconfig.json     # TypeScript configuration
```

## Technologies Used

- **Cloudflare Workers**: Serverless runtime
- **Cloudflare Workers AI**: AI models (LLaVA, Llama 3.3)
- **Durable Objects**: Database for storing detection history
- **TypeScript**: Programming language

## Notes

- Free tier of Cloudflare Workers AI has rate limits
- Large images may take longer to process
- Detection accuracy depends on image quality and AI model capabilities

## License

MIT

