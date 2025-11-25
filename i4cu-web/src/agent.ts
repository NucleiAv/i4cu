import { Agent } from "agents";

interface DetectionResult {
  isDeepfake: boolean;
  confidence: number;
  analysis: string;
  indicators: string[];
  timestamp: number;
}

export class DeepfakeDetectorAgent extends Agent {
  constructor(state: any, env: any) {
    super(state, env);
    (this as any).env = env;
  }

  async ensureTables() {
    await this.sql`
      CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        is_deepfake INTEGER NOT NULL,
        confidence REAL NOT NULL,
        analysis TEXT NOT NULL,
        indicators TEXT NOT NULL,
        created_at INTEGER NOT NULL
      )
    `;
    
    await this.sql`
      CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        total_detections INTEGER DEFAULT 0,
        deepfake_count INTEGER DEFAULT 0,
        real_count INTEGER DEFAULT 0,
        avg_confidence REAL DEFAULT 0
      )
    `;
  }

  async onConnect(connection: any, ctx: any) {
    await this.ensureTables();
    const recent = await this.sql`
      SELECT * FROM detections 
      ORDER BY created_at DESC 
      LIMIT 20
    `;
    const stats = await this.getStats();
    await this.broadcast(new TextEncoder().encode(JSON.stringify({ 
      type: "init", 
      detections: recent,
      stats 
    })));
  }

  async detectImage(imageData: string, fileName: string, fileSize: number) {
    await this.ensureTables();
    
    try {
      await this.broadcast(new TextEncoder().encode(JSON.stringify({ type: "status", message: "Running ensemble detection..." })));
    } catch {}
    
    const env = (this as any).env || {};
    const ai = env.AI;
    
    if (!ai) {
      console.error("AI binding not available in agent");
      const fallbackResult = this.fallbackDetection("AI binding not configured");
      return { 
        success: true, 
        result: {
          ...fallbackResult,
          fileName,
          indicators: [fallbackResult.indicators],
          timestamp: Date.now()
        },
        error: "AI binding not available"
      };
    }
    
    const base64Data = imageData.includes(',') ? imageData.split(',')[1] : imageData;
    const imageBytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
    const imageArray = Array.from(imageBytes);
    
    try {
      await this.broadcast(new TextEncoder().encode(JSON.stringify({ type: "status", message: "Analyzing metadata and watermarks..." })));
    } catch {}
    
    try {
      const [exifData, watermarkText, ocrText] = await Promise.all([
        this.extractEXIFData(imageBytes),
        this.detectWatermarks(ai, imageArray),
        this.performOCR(ai, imageArray)
      ]);
      
      console.log("EXIF Data:", exifData);
      console.log("Watermark Detection:", watermarkText);
      console.log("OCR Result:", ocrText);
      
      const ensembleResults = await this.runEnsembleDetection(ai, imageArray, base64Data);
      const combinedResults = [...ensembleResults, {
        model: "metadata",
        weight: 0.15,
        response: `EXIF: ${JSON.stringify(exifData)}, Watermarks: ${watermarkText}, OCR: ${ocrText}`,
        raw: { exif: exifData, watermarks: watermarkText, ocr: ocrText }
      }];
      
      return this.combineEnsembleResults(combinedResults, fileName);
    } catch (error: any) {
      console.error("Ensemble detection error:", error);
      const fallbackResult = this.fallbackDetection(error.message || "Unknown error");
      return { 
        success: true, 
        result: {
          ...fallbackResult,
          fileName,
          indicators: [fallbackResult.indicators],
          timestamp: Date.now()
        },
        error: error.message 
      };
    }
  }

  async extractEXIFData(imageBytes: Uint8Array): Promise<any> {
    try {
      // Check more bytes for better detection (first 2000 bytes for comprehensive analysis)
      const hex = Array.from(imageBytes.slice(0, 2000))
        .map(b => b.toString(16).padStart(2, '0'))
        .join('');

      let format = "UNKNOWN";
      if (hex.startsWith("ffd8ff")) format = "JPEG";
      else if (hex.startsWith("89504e470d0a1a0a")) format = "PNG";
      else if (hex.startsWith("47494638")) format = "GIF";
      else if (hex.startsWith("52494646") && hex.includes("57454250")) format = "WebP";
      else if (hex.startsWith("52494646") && hex.includes("41564920")) format = "AVIF";
      else if (hex.startsWith("6674797068656963")) format = "HEIC";

      let hasEXIF = false;
      let hasXMP = false;
      let hasIPTC = false;
      let hasICC = false;
      
      // Check for EXIF markers in JPEG
      if (format === "JPEG") {
        // EXIF markers: FF E1 (APP1), FF E2 (APP2), etc.
        // Also check for "Exif" string (45786966 in hex)
        if (hex.includes("ffe1") || hex.includes("ffe2") || hex.includes("ffe3") || 
            hex.includes("ff01") || hex.includes("ff00") || hex.includes("45786966")) {
          hasEXIF = true;
        }
      }
      
      // Check for XMP metadata (XML-based, commonly used by AI tools like Adobe Firefly, Midjourney)
      // XMP signature: "http://ns.adobe.com/xmp/" or "XMP" marker
      if (hex.includes("584d50") || hex.includes("687474703a2f2f6e732e61646f62652e636f6d2f786d702f")) {
        hasXMP = true;
      }
      
      // Check for IPTC metadata (International Press Telecommunications Council)
      // IPTC signature: "1C01" or "IPTC" string
      if (hex.includes("49505443") || hex.includes("1c01")) {
        hasIPTC = true;
      }
      
      // Check for ICC color profile (International Color Consortium)
      // ICC signature: "ICC_PROFILE" or "icc_profile"
      if (hex.includes("4943435f50524f46494c45") || hex.includes("6963635f70726f66696c65")) {
        hasICC = true;
      }

      const hasMetadata = hasEXIF || hasXMP || hasIPTC || hasICC;

      let suspicious = false;
      let suspiciousReasons: string[] = [];
      
      // AI-generated images often have distinctive metadata patterns:
      // 1. No EXIF data (stripped during generation or never added)
      // 2. Only XMP metadata (added by AI tools like Adobe Firefly, Midjourney)
      // 3. Missing camera/device info (no camera make/model)
      // 4. Unusual metadata patterns (XMP without EXIF is suspicious)
      // 5. WebP/AVIF formats are commonly used by AI tools
      
      if (format === "JPEG") {
        if (!hasEXIF && !hasMetadata) {
          suspicious = true;
          suspiciousReasons.push("JPEG with no metadata (common in AI-generated images)");
        } else if (!hasEXIF && hasXMP && !hasIPTC) {
          suspicious = true;
          suspiciousReasons.push("JPEG with only XMP metadata (typical of AI tools like Adobe Firefly, Midjourney)");
        } else if (hasXMP && !hasEXIF) {
          suspicious = true;
          suspiciousReasons.push("XMP metadata without EXIF (suspicious pattern - AI tools often add XMP but strip EXIF)");
        }
      } else if (format === "PNG") {
        // PNGs typically have less metadata, but AI-generated ones often have none or only XMP
        if (!hasXMP && !hasIPTC && !hasICC) {
          suspicious = true;
          suspiciousReasons.push("PNG with no metadata (common in AI-generated images)");
        } else if (hasXMP && !hasIPTC && !hasICC) {
          suspicious = true;
          suspiciousReasons.push("PNG with only XMP metadata (typical of AI tools)");
        }
      } else if (format === "WebP") {
        // WebP is commonly used by AI tools (e.g., DALL-E, Midjourney exports)
        if (!hasXMP && !hasEXIF) {
          suspicious = true;
          suspiciousReasons.push("WebP with minimal metadata (suspicious - commonly used by AI tools)");
        }
      } else if (format === "AVIF" || format === "HEIC") {
        // These modern formats are sometimes used by AI tools
        if (!hasMetadata) {
          suspicious = true;
          suspiciousReasons.push(`${format} with no metadata (suspicious)`);
        }
      }

      return { 
        format, 
        hasEXIF, 
        hasXMP,
        hasIPTC,
        hasICC,
        hasMetadata, 
        suspicious,
        suspiciousReasons: suspiciousReasons.length > 0 ? suspiciousReasons : undefined
      };
    } catch (e: any) {
      console.error("EXIF extraction error:", e);
      return { 
        format: "ERROR", 
        hasEXIF: false, 
        hasXMP: false,
        hasIPTC: false,
        hasICC: false,
        hasMetadata: false, 
        suspicious: true, 
        error: e.message 
      };
    }
  }

  async detectWatermarks(ai: any, imageArray: number[]): Promise<string> {
    try {
      const watermarkPrompt = `You are a forensic watermark detection expert. Examine this image SYSTEMATICALLY for ANY watermarks, logos, text, or markers that indicate its origin or AI generation.

LOOK SPECIFICALLY FOR ALL AI GENERATION PLATFORMS:
1. GOOGLE AI:
   - "Gemini" or "Google Gemini"
   - "Imagen" or "Google Imagen"
   - "Veo" or "Google Veo"

2. OPENAI:
   - "DALL-E" or "DALL-E 2" or "DALL-E 3"
   - "ChatGPT" or "ChatGPT Image"
   - "Sora" (video generation, but may appear in stills)
   - "OpenAI" logo or text

3. ANTHROPIC:
   - "Claude" or "Anthropic Claude"
   - "Anthropic" branding

4. MICROSOFT:
   - "Bing Image Creator" or "Bing Creator"
   - "Microsoft Designer" or "Designer"
   - "Copilot" or "Microsoft Copilot"

5. ADOBE:
   - "Adobe Firefly" or "Firefly"
   - "Adobe" logo or text
   - "Photoshop" (AI features)

6. MIDJOURNEY & STABLE DIFFUSION:
   - "Midjourney" logo or text
   - "Stable Diffusion" or "SDXL"
   - "Stability AI" or "Stable AI"

7. OTHER AI PLATFORMS:
   - "Runway" or "Runway ML"
   - "Leonardo AI" or "Leonardo"
   - "ElevenLabs" (for images)
   - "Jasper" or "Jasper Art"
   - "Craiyon" (formerly DALL-E mini)
   - "NightCafe" or "NightCafe Creator"
   - "Artbreeder" or "Art Breeder"
   - "DeepAI" or "Deep Dream"
   - "This Person Does Not Exist" markers
   - "Generated Photos" or "Generated.photos"
   - "Synthesia" (for video stills)
   - "Pika" or "Pika Labs"
   - "Kling" or "Kling AI"
   - "Luma" or "Luma AI"
   - "Ideogram" or "Ideogram AI"
   - "Flux" or "Black Forest Labs"
   - "Civitai" or "Civit AI"
   - "Hugging Face" or "HF" markers

8. GENERIC AI MARKERS:
   - "AI Generated" or "Generated by AI"
   - "AI Art" or "Artificial Intelligence"
   - "Synthetic" or "Synthetic Media"
   - "Deepfake" (if self-labeled)
   - "GAN" or "Neural Network"
   - "Machine Learning" or "ML Generated"

9. PLATFORM WATERMARKS:
   - Social media logos (Instagram, Facebook, Twitter/X, TikTok, Reddit)
   - Stock photo site watermarks (Getty, Shutterstock, Adobe Stock, Unsplash)
   - Any platform branding or logos

10. VISIBLE WATERMARKS:
   - Semi-transparent text overlays (especially in corners)
   - Copyright symbols (©) or "© 2024" text
   - Brand names or company logos
   - Signature marks or creator names
   - QR codes or barcodes

11. SUBTLE MARKERS:
   - Small text in corners or edges
   - Embedded logos or symbols
   - Any text that might indicate the image source
   - Metadata watermarks (invisible but may leave traces)

EXAMINE THE ENTIRE IMAGE SYSTEMATICALLY:
- All four corners (top-left, top-right, bottom-left, bottom-right)
- Bottom center, top center, left center, right center
- All edges and borders
- Any text overlays anywhere in the image
- Background areas that might contain watermarks
- Foreground elements that might be watermarks

DETECTION INSTRUCTIONS:
1. Scan the image in a grid pattern (top to bottom, left to right)
2. Pay special attention to corners and edges (most common watermark locations)
3. Look for text in different sizes (large, medium, small, tiny)
4. Check for text in different styles (bold, italic, transparent, colored)
5. Look for logos, symbols, or icons
6. Check for patterns that might be watermarks

If you find ANY watermark, logo, or text (even if small, subtle, or partially obscured), describe it in DETAIL including:
- EXACT text (spell it out character by character if possible)
- Location (e.g., "bottom-right corner", "top-left edge", "center-bottom")
- Size and visibility (large, medium, small, tiny, very subtle)
- Color and style (white, black, transparent, colored, bold, italic)
- Any logos or symbols you see

If you find NO watermarks, logos, or text markers at all after thorough examination, respond with exactly: "No watermarks detected."

Be EXTREMELY thorough - even tiny text in corners or subtle logos are critical for detection!`;
      
      const response = await ai.run("@cf/llava-hf/llava-1.5-7b-hf", {
        image: imageArray,
        prompt: watermarkPrompt,
      });
      
      return response?.description || response?.response || response?.text || "Watermark analysis completed";
    } catch (e) {
      return "Watermark detection failed";
    }
  }

  async performOCR(ai: any, imageArray: number[]): Promise<string> {
    try {
      const ocrPrompt = `You are an expert OCR (Optical Character Recognition) analyst. Extract ALL visible text from this image, no matter how small, subtle, or partially obscured.

EXAMINE THE ENTIRE IMAGE SYSTEMATICALLY IN A GRID PATTERN:
1. Check ALL FOUR CORNERS (top-left, top-right, bottom-left, bottom-right)
2. Check CENTER areas (top-center, bottom-center, left-center, right-center, middle)
3. Check ALL EDGES and borders (top edge, bottom edge, left edge, right edge)
4. Scan the main image content for any embedded text
5. Check background areas for watermarks or text
6. Look for text in different orientations (horizontal, vertical, diagonal)

LOOK SPECIFICALLY FOR ALL AI GENERATION PLATFORMS:
GOOGLE AI:
- "Gemini", "Google Gemini", "Imagen", "Veo", "Google Veo"

OPENAI:
- "DALL-E", "DALL-E 2", "DALL-E 3", "ChatGPT", "Sora", "OpenAI"

ANTHROPIC:
- "Claude", "Anthropic", "Anthropic Claude"

MICROSOFT:
- "Bing Image Creator", "Bing Creator", "Microsoft Designer", "Designer", "Copilot", "Microsoft Copilot"

ADOBE:
- "Adobe Firefly", "Firefly", "Adobe", "Photoshop"

OTHER AI PLATFORMS:
- "Midjourney", "Stable Diffusion", "SDXL", "Stability AI"
- "Runway", "Runway ML", "Leonardo AI", "Leonardo"
- "ElevenLabs", "Jasper", "Jasper Art", "Craiyon"
- "NightCafe", "Artbreeder", "DeepAI", "Deep Dream"
- "Pika", "Pika Labs", "Kling", "Kling AI", "Luma", "Luma AI"
- "Ideogram", "Ideogram AI", "Flux", "Black Forest Labs"
- "Civitai", "Civit AI", "Hugging Face", "HF"

GENERIC AI MARKERS:
- "AI Generated", "Generated by AI", "AI Art", "Artificial Intelligence"
- "Synthetic", "Synthetic Media", "Deepfake", "GAN", "Neural Network"
- "Machine Learning", "ML Generated", "Created by AI", "AI Created"
- "This Person Does Not Exist", "Generated Photos", "Synthesia"

PLATFORM MARKERS:
- Social media: "Instagram", "Facebook", "Twitter", "X", "TikTok", "Reddit"
- Stock photos: "Getty", "Shutterstock", "Adobe Stock", "Unsplash"
- Copyright: "©", "© 2024", "Copyright", "All Rights Reserved"
- Any brand names, company names, or platform identifiers

EXTRACTION INSTRUCTIONS:
- Read ALL text you can see, even if it's:
  * Extremely small or tiny (even 1-2 pixels high)
  * In corners, edges, or margins
  * Partially obscured or cut off
  * Low contrast (light on light, dark on dark)
  * Stylized, decorative, or unusual fonts
  * Transparent or semi-transparent
  * Colored or multi-colored
  * Rotated or at an angle
  * Blurred or out of focus
- Include EXACT spelling and capitalization (case-sensitive)
- Note the precise location of each text element
- If text is unclear, describe what you think it says and why
- Include any numbers, symbols, or special characters

FORMAT YOUR RESPONSE:
- List each piece of text you find with its location
- Format: "[LOCATION]: '[EXACT TEXT]'"
- Example: "bottom-right corner: 'Gemini'", "top-left edge: 'AI Generated'"
- If multiple text elements, list them all
- If text is unclear, write: "[LOCATION]: '[UNCLEAR TEXT - appears to say: ...]'"

If you find ABSOLUTELY NO text anywhere in the image after thorough systematic examination, respond with exactly: "No text detected."

Be EXTREMELY thorough - even 1-2 pixel high text in corners is critical for detection! Scan every pixel if necessary!`;
      
      const response = await ai.run("@cf/llava-hf/llava-1.5-7b-hf", {
        image: imageArray,
        prompt: ocrPrompt,
      });
      
      return response?.description || response?.response || response?.text || "OCR completed";
    } catch (e) {
      return "OCR failed";
    }
  }

  async runEnsembleDetection(ai: any, imageArray: number[], base64Data: string) {
    console.log("Running ensemble detection with multiple models...");
    
    const visionPrompt = `You are a balanced forensic image analyst. Analyze this image OBJECTIVELY for deepfake/manipulation indicators.

IMPORTANT: Real photos can have natural variations, lighting differences, and minor imperfections. Only flag as deepfake if you see CLEAR, OBVIOUS signs of manipulation.

EXAMINE FOR (but be conservative):
1. Facial inconsistencies: CLEAR misalignment (eyes pointing in completely different directions, not just slight asymmetry)
2. Lighting mismatches: OBVIOUS lighting conflicts (face lit from opposite side than background, not just different angles)
3. Blur artifacts: UNNATURAL blurring (artificial-looking edges, not just depth-of-field blur)
4. Skin texture anomalies: OBVIOUSLY fake texture (plastic-like, clearly AI-generated, not just smooth skin)
5. GAN/neural network artifacts: VISIBLE patterns (checkerboard, grid lines, not just compression artifacts)
6. Resolution inconsistencies: CLEAR mismatch (face at 4K while background is 240p, not just focus differences)
7. Unnatural proportions: OBVIOUS distortion (clearly wrong geometry, not just perspective)
8. Color inconsistencies: OBVIOUS mismatches (completely different color temperature, not just variations)
9. Edge artifacts: VISIBLE seams (clear cut-and-paste lines, not just shadows)
10. Too perfect: OBVIOUSLY AI-generated (unnaturally flawless, clearly synthetic, not just good lighting)

BALANCE: Real photos can have:
- Slight lighting variations
- Natural depth-of-field blur
- Minor facial asymmetry (normal)
- Smooth skin (good photography/lighting)
- Focus differences (normal photography)

Only rate high (70+) if you see MULTIPLE CLEAR indicators. Rate 40-60 if uncertain. Rate 0-40 if image appears natural.

Rate likelihood of being a deepfake 0-100. Be conservative - when in doubt, rate lower.`;
    
    try {
      await this.broadcast(new TextEncoder().encode(JSON.stringify({ type: "status", message: "Model 1/3: Analyzing image..." })));
    } catch {}
    
    const visionResponse = await ai.run("@cf/llava-hf/llava-1.5-7b-hf", {
      image: imageArray,
      prompt: visionPrompt,
    });
    
    const visionText = visionResponse?.description || visionResponse?.response || visionResponse?.text || "";
    console.log("Vision model response:", visionText);
    
    const detectionPrompt1 = `You are a BALANCED forensic deepfake detection expert. Be OBJECTIVE, not overly suspicious.

Image Analysis: "${visionText}"

CRITICAL RULES:
- Only flag as DEEPFAKE if analysis mentions MULTIPLE CLEAR indicators (not just one minor issue)
- If analysis says "slight", "minor", "subtle", "may be", "could be" → Be CONSERVATIVE, likely authentic
- Real photos can have: lighting variations, depth-of-field blur, minor asymmetry, smooth skin (good photography)
- Only flag if analysis mentions OBVIOUS/CLEAR/STRONG indicators of manipulation
- When analysis is uncertain or mentions "normal photography" → Mark as AUTHENTIC
- If only 1-2 minor issues mentioned → Mark as AUTHENTIC with confidence 30-45

CONFIDENCE SCALE:
- 0-30: Clearly authentic, natural photo
- 31-50: Likely authentic, minor concerns
- 51-70: Suspicious, multiple indicators (flag as deepfake)
- 71-100: Definitely deepfake, strong evidence

Be BALANCED: Real photos are common. Only flag when evidence is strong.

Respond with JSON only:
{"isDeepfake": true/false, "confidence": 0-100, "indicators": ["specific issue 1", "specific issue 2"], "analysis": "brief explanation"}`;

    const detectionPrompt2 = `You are a BALANCED deepfake detector. Be OBJECTIVE, not overly aggressive.

Image Analysis: "${visionText}"

DETECTION RULES:
- Only flag as DEEPFAKE if analysis mentions STRONG/CLEAR/OBVIOUS indicators
- If analysis uses words like "slight", "minor", "subtle", "may", "could" → Mark as AUTHENTIC
- Real photos can have: good lighting (smooth skin), depth-of-field (blur), natural variations
- Perfect symmetry or flawless features CAN be real (good photography, professional lighting)
- Only flag if MULTIPLE clear indicators are present
- If unsure → Mark as AUTHENTIC (real photos are more common than deepfakes)

BALANCE: Be conservative. Only flag when evidence is strong.

Respond with JSON:
{"isDeepfake": true/false, "confidence": 0-100, "indicators": ["issue"], "analysis": "explanation"}`;

    try {
      await this.broadcast(new TextEncoder().encode(JSON.stringify({ type: "status", message: "Model 2/3: Deepfake analysis..." })));
    } catch {}
    
    const llamaResponse1 = await ai.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
      messages: [{ role: "user", content: detectionPrompt1 }],
    });
    
    try {
      await this.broadcast(new TextEncoder().encode(JSON.stringify({ type: "status", message: "Model 3/3: Cross-validation..." })));
    } catch {}
    
    const llamaResponse2 = await ai.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
      messages: [{ role: "user", content: detectionPrompt2 }],
    });
    
    return [
      { model: "llava", weight: 0.3, response: visionText, raw: visionResponse },
      { model: "llama1", weight: 0.35, response: llamaResponse1?.response || llamaResponse1?.text || "", raw: llamaResponse1 },
      { model: "llama2", weight: 0.35, response: llamaResponse2?.response || llamaResponse2?.text || "", raw: llamaResponse2 }
    ];
  }

  combineEnsembleResults(results: any[], fileName: string): any {
    let totalDeepfakeScore = 0;
    let totalWeight = 0;
    const allIndicators: string[] = [];
    const deepfakeAnalyses: string[] = [];
    const authenticAnalyses: string[] = [];
    let deepfakeCount = 0;
    let authenticCount = 0;
    
    for (const result of results) {
      if (result.error || !result.response) continue;
      
      const parsed = this.parseModelResponse(result.model, result.response, result.raw);
      if (parsed) {
        const weight = result.weight || 0.33;
        totalWeight += weight;
        
        if (parsed.isDeepfake) {
          totalDeepfakeScore += parsed.confidence * weight;
          deepfakeCount++;
          if (parsed.analysis && !parsed.analysis.toLowerCase().includes("not a deepfake") && 
              !parsed.analysis.toLowerCase().includes("no indicators")) {
            deepfakeAnalyses.push(parsed.analysis);
          }
        } else {
          totalDeepfakeScore += (100 - parsed.confidence) * weight;
          authenticCount++;
          if (parsed.analysis) authenticAnalyses.push(parsed.analysis);
        }
        
        allIndicators.push(...(parsed.indicators || []));
      }
    }
    
    if (totalWeight === 0) {
      return {
        success: true,
        result: this.fallbackDetection("All models failed"),
        error: "All detection models failed"
      };
    }
    
    const avgDeepfakeScore = totalDeepfakeScore / totalWeight;
    const isDeepfake = avgDeepfakeScore > 52;
    let finalConfidence = isDeepfake ? avgDeepfakeScore : (100 - avgDeepfakeScore);
    
    if (isDeepfake && finalConfidence < 60) {
      finalConfidence = 60;
    }
    if (!isDeepfake && finalConfidence > 50) {
      finalConfidence = Math.min(finalConfidence, 45);
    }
    
    const uniqueIndicators = [...new Set(allIndicators)].filter(ind => 
      ind && !ind.toLowerCase().includes("no indicators") && 
      !ind.toLowerCase().includes("standard analysis")
    );
    
    let combinedAnalysis = "";
    if (isDeepfake) {
      if (deepfakeAnalyses.length > 0) {
        const mainAnalysis = deepfakeAnalyses[0];
        const cleanAnalysis = mainAnalysis
          .replace(/no indicators of deepfake/gi, "")
          .replace(/not a deepfake/gi, "")
          .replace(/appears to be natural/gi, "may show signs of manipulation")
          .substring(0, 400);
        combinedAnalysis = `Ensemble detection (${deepfakeCount}/${results.length} models flagged as deepfake): ${cleanAnalysis}`;
      } else {
        combinedAnalysis = `Ensemble analysis: ${deepfakeCount} of ${results.length} models detected deepfake indicators. Confidence: ${Math.round(finalConfidence)}%.`;
      }
    } else {
      if (authenticAnalyses.length > 0) {
        combinedAnalysis = `Ensemble analysis: ${authenticAnalyses[0].substring(0, 400)}`;
      } else {
        combinedAnalysis = `Ensemble analysis: ${authenticCount} of ${results.length} models indicate authentic content.`;
      }
    }
    
    if (combinedAnalysis.length === 0) {
      combinedAnalysis = `Ensemble analysis: ${results.length} models analyzed. ${isDeepfake ? 'Deepfake detected' : 'Likely authentic'}.`;
    }
    
    const detection: DetectionResult = {
      isDeepfake,
      confidence: Math.round(Math.min(100, Math.max(0, finalConfidence))),
      analysis: combinedAnalysis,
      indicators: uniqueIndicators.length > 0 ? uniqueIndicators : (isDeepfake ? ["Suspicious content detected"] : ["No major issues"]),
      timestamp: Date.now()
    };
    
    return this.saveDetectionResult(detection, fileName, "image", 0);
  }

  parseModelResponse(modelName: string, response: string, rawResponse: any): DetectionResult | null {
    try {
      if (modelName === "metadata" && rawResponse) {
        const exif = rawResponse.exif || {};
        const watermarks = (rawResponse.watermarks || "").toLowerCase();
        const ocr = (rawResponse.ocr || "").toLowerCase();
        
        const aiWatermarkKeywords = [
          // Google AI
          "gemini", "google gemini", "imagen", "veo", "google veo",
          // OpenAI
          "dall-e", "dall-e 2", "dall-e 3", "dall e", "chatgpt", "openai", "sora",
          // Anthropic
          "claude", "anthropic", "anthropic claude",
          // Microsoft
          "bing image creator", "bing creator", "microsoft designer", "designer", "copilot", "microsoft copilot",
          // Adobe
          "adobe firefly", "firefly", "adobe", "photoshop",
          // Midjourney & Stable Diffusion
          "midjourney", "stable diffusion", "sdxl", "stability ai", "stable ai",
          // Other AI Platforms
          "runway", "runway ml", "leonardo ai", "leonardo", "elevenlabs", "jasper", "jasper art", "craiyon",
          "nightcafe", "artbreeder", "deepai", "deep dream", "pika", "pika labs", "kling", "kling ai",
          "luma", "luma ai", "ideogram", "ideogram ai", "flux", "black forest labs", "civitai", "civit ai",
          "hugging face", "hf",
          // Generic AI Markers
          "ai generated", "generated by ai", "ai art", "artificial intelligence", "synthetic", "synthetic media",
          "deepfake", "gan", "neural network", "machine learning", "ml generated", "created by ai", "ai created",
          "this person does not exist", "generated photos", "synthesia"
        ];
        const watermarkLower = (watermarks || "").toLowerCase();
        const ocrLower = (ocr || "").toLowerCase();
        const hasAIWatermark = aiWatermarkKeywords.some(kw => watermarkLower.includes(kw) || ocrLower.includes(kw));
        const hasGemini = watermarkLower.includes("gemini") || ocrLower.includes("gemini");
        const hasNoEXIF = exif.suspicious || (!exif.hasEXIF && exif.format === 'JPEG');
        const hasWatermark = !watermarkLower.includes("no watermark") && !watermarkLower.includes("no text") && !watermarkLower.includes("no watermarks detected");
        
        let isDeepfake = false;
        let confidence = 50;
        const indicators: string[] = [];
        
        // Check for Gemini specifically (highest priority - 90% confidence)
        if (hasGemini) {
          isDeepfake = true;
          confidence = 90;
          indicators.push("Gemini AI generation detected");
        }
        
        if (hasAIWatermark && !hasGemini) {
          isDeepfake = true;
          confidence = Math.max(confidence, 85);
          indicators.push("AI generation watermark detected");
        }
        
        if (exif.suspicious) {
          isDeepfake = true;
          confidence = Math.max(confidence, 70);
          if (exif.suspiciousReasons && exif.suspiciousReasons.length > 0) {
            indicators.push(...exif.suspiciousReasons);
          } else {
            indicators.push("Suspicious metadata pattern detected");
          }
        }
        
        // Additional checks for specific suspicious patterns
        if (exif.format === 'WebP' || exif.format === 'AVIF' || exif.format === 'HEIC') {
          if (!exif.hasMetadata) {
            isDeepfake = true;
            confidence = Math.max(confidence, 65);
            indicators.push(`${exif.format} format with no metadata (commonly used by AI tools)`);
          }
        }
        
        if (exif.hasXMP && !exif.hasEXIF && exif.format === 'JPEG') {
          isDeepfake = true;
          confidence = Math.max(confidence, 75);
          indicators.push("XMP metadata without EXIF (typical of AI generation tools)");
        }
        
        if (hasWatermark && !hasAIWatermark) {
          indicators.push("Watermark detected");
        }
        
        if (ocr && !ocrLower.includes("no text") && !ocrLower.includes("no text detected")) {
          if (ocrLower.includes("gemini")) {
            isDeepfake = true;
            confidence = Math.max(confidence, 90);
            indicators.push("Gemini text detected via OCR");
          } else if (aiWatermarkKeywords.some(kw => ocrLower.includes(kw))) {
            isDeepfake = true;
            confidence = Math.max(confidence, 80);
            indicators.push("AI generation text detected via OCR");
          } else {
            // Any text found is worth noting
            indicators.push("Text detected via OCR");
          }
        }
        
        return {
          isDeepfake,
          confidence,
          analysis: `Metadata Analysis: Format: ${exif.format || 'Unknown'}, EXIF: ${exif.hasEXIF ? 'Present' : 'Missing'}, Watermarks: ${watermarks.substring(0, 100)}, OCR: ${ocr.substring(0, 100)}`,
          indicators: indicators.length > 0 ? indicators : ["Standard metadata check"],
          timestamp: Date.now()
        };
      }
      
      if ((modelName === "llama1" || modelName === "llama2") && rawResponse) {
        if (typeof rawResponse === 'object' && rawResponse.response) {
          const data = rawResponse.response;
          if (typeof data === 'object' && data.isDeepfake !== undefined) {
            return {
              isDeepfake: data.isDeepfake,
              confidence: data.confidence || 50,
              analysis: data.analysis || response,
              indicators: Array.isArray(data.indicators) ? data.indicators : [],
              timestamp: Date.now()
            };
          }
        }
        
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            const data = JSON.parse(jsonMatch[0]);
            return {
              isDeepfake: data.isDeepfake || false,
              confidence: data.confidence || 50,
              analysis: data.analysis || response,
              indicators: Array.isArray(data.indicators) ? data.indicators : [],
              timestamp: Date.now()
            };
          } catch {}
        }
      }
      
      const lower = response.toLowerCase();
      
      const deepfakeKeywords = ["deepfake", "fake", "synthetic", "artificial", "generated", "manipulated", "altered", "forged"];
      const strongIssueKeywords = ["clear", "obvious", "visible", "strong", "definite", "checkerboard", "grid pattern", "seam", "halo effect", "cut-and-paste"];
      const weakIssueKeywords = ["slight", "minor", "subtle", "may be", "could be", "possible", "might", "suggest"];
      const authenticKeywords = ["authentic", "real", "genuine", "natural", "not a deepfake", "no indicators", "no signs", "clearly authentic", "normal photography", "good photography", "professional", "depth-of-field"];
      
      const hasDeepfakeKeyword = deepfakeKeywords.some(kw => lower.includes(kw));
      const hasStrongIssues = strongIssueKeywords.some(kw => lower.includes(kw)) && (lower.includes("inconsist") || lower.includes("artifact") || lower.includes("mismatch"));
      const hasWeakIssues = weakIssueKeywords.some(kw => lower.includes(kw));
      const hasAuthenticKeyword = authenticKeywords.some(kw => lower.includes(kw));
      
      const isTooPerfect = (lower.includes("too perfect") || lower.includes("unnaturally flawless")) && 
                           (lower.includes("clear") || lower.includes("obvious") || lower.includes("definite"));
      
      const hasStrongAuthentic = lower.includes("clearly authentic") || lower.includes("definitely real") || lower.includes("genuine photo") || lower.includes("normal photography");
      
      const isDeepfake = hasDeepfakeKeyword || (hasStrongIssues && !hasWeakIssues && !hasStrongAuthentic) || isTooPerfect;
      
      const confidenceMatch = response.match(/confidence[:\s]+(\d+)|(\d+)%/i) || response.match(/(\d+)/);
      let confidence = confidenceMatch ? parseInt(confidenceMatch[1] || confidenceMatch[2] || confidenceMatch[0]) : 50;
      
      if (isDeepfake) {
        if (confidence < 60 && hasStrongIssues) confidence = 60;
        if (hasWeakIssues) confidence = Math.min(confidence, 45);
        if (isTooPerfect && hasStrongIssues) confidence = Math.max(confidence, 70);
        if (hasDeepfakeKeyword) confidence = Math.max(confidence, 70);
      } else {
        if (hasWeakIssues) confidence = Math.min(confidence, 40);
        if (hasStrongAuthentic) confidence = Math.min(confidence, 30);
        if (confidence > 50 && !hasStrongIssues) confidence = 40;
      }
      
      const indicators: string[] = [];
      if (lower.includes("inconsist")) indicators.push("Facial inconsistencies");
      if (lower.includes("artifact")) indicators.push("Artifacts detected");
      if (lower.includes("blur")) indicators.push("Blur patterns");
      if (lower.includes("lighting")) indicators.push("Lighting mismatches");
      if (lower.includes("texture")) indicators.push("Texture anomalies");
      if (lower.includes("resolution")) indicators.push("Resolution issues");
      if (lower.includes("gan") || lower.includes("generative")) indicators.push("GAN patterns");
      if (lower.includes("chatgpt") || lower.includes("ai-generated") || lower.includes("dall-e") || lower.includes("midjourney")) indicators.push("AI-generated content");
      if (indicators.length === 0 && isDeepfake) indicators.push("Suspicious content");
      if (indicators.length === 0) indicators.push("Model analysis");
      
      return {
        isDeepfake,
        confidence: Math.min(100, Math.max(0, confidence)),
        analysis: response.substring(0, 300),
        indicators,
        timestamp: Date.now()
      };
    } catch (e) {
      console.error(`Error parsing ${modelName} response:`, e);
      return null;
    }
  }

  async saveDetectionResult(detection: DetectionResult, fileName: string, fileType: string, fileSize: number) {
    const indicators = Array.isArray(detection.indicators) 
      ? detection.indicators 
      : [detection.indicators || "Analysis completed"];

    await this.sql`
      INSERT INTO detections (
        file_name, file_type, file_size, is_deepfake, 
        confidence, analysis, indicators, created_at
      )
      VALUES (
        ${fileName}, ${fileType}, ${fileSize}, 
        ${detection.isDeepfake ? 1 : 0}, 
        ${detection.confidence}, 
        ${detection.analysis || "Analysis completed"}, 
        ${JSON.stringify(indicators)}, 
        ${Date.now()}
      )
    `;

    const stats = await this.getStats();
    
    try {
      await this.broadcast(new TextEncoder().encode(JSON.stringify({ 
        type: "detection", 
        result: {
          ...detection,
          fileName,
          indicators,
          timestamp: Date.now()
        },
        stats
      })));
    } catch {}

    return {
      success: true,
      result: {
        ...detection,
        fileName,
        indicators,
        timestamp: Date.now()
      },
      stats
    };
  }

  private fallbackDetection(prompt: string): DetectionResult {
    return {
      isDeepfake: false,
      confidence: 50,
      analysis: "Analysis completed using fallback method. For best results, ensure Workers AI models are properly configured.",
      indicators: ["Standard analysis"],
      timestamp: Date.now()
    };
  }

  async getHistory(limit: number = 50) {
    await this.ensureTables();
    const detections = await this.sql`
      SELECT * FROM detections 
      ORDER BY created_at DESC 
      LIMIT ${limit}
    `;
    return { type: "history", data: detections };
  }

  async getStats() {
    await this.ensureTables();
    const total = await this.sql`SELECT COUNT(*) as count FROM detections`;
    const deepfakes = await this.sql`SELECT COUNT(*) as count FROM detections WHERE is_deepfake = 1`;
    const real = await this.sql`SELECT COUNT(*) as count FROM detections WHERE is_deepfake = 0`;
    const avgConf = await this.sql`SELECT AVG(confidence) as avg FROM detections`;

    return {
      total: total[0]?.count || 0,
      deepfakes: deepfakes[0]?.count || 0,
      real: real[0]?.count || 0,
      avgConfidence: Math.round((Number(avgConf[0]?.avg) || 0) * 10) / 10
    };
  }

  async chat(message: string) {
    await this.ensureTables();
    const stats = await this.getStats();
    
    const context = `You are an expert deepfake detection assistant. 
Recent detection stats: ${stats.total} total, ${stats.deepfakes} deepfakes detected, ${stats.real} real media.
Average confidence: ${stats.avgConfidence}%

User question: ${message}

Provide helpful, technical insights about deepfake detection.`;

    const env = (this as any).env || {};
    const ai = env.AI;
    
    if (!ai) {
      return { type: "chat", message: "AI service not available." };
    }

    const response = await ai.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
      messages: [{ role: "user", content: context }],
    });

    const reply = response?.response || response?.text || "I'm here to help with deepfake detection questions.";
    return { type: "chat", message: reply };
  }
}
