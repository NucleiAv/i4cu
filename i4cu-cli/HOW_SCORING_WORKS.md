# How Scoring Works - Simple Explanation

## 🎯 The Big Picture

The tool gives each image/video a **score from 0 to 1** (0% to 100%):
- **0.0 (0%)** = Definitely Real
- **0.5 (50%)** = Unsure / Neutral
- **1.0 (100%)** = Definitely Deepfake

If the score is **above 0.55 (55%)** AND confidence is **above 0.6 (60%)**, it's flagged as a deepfake.

---

## 📊 How the Score is Calculated

The final score combines **3 different checks**:

### 1. **EXIF Data Check** (10% weight)
- Looks at image metadata (camera info, timestamps, etc.)
- Finds suspicious patterns like:
  - Missing timestamps
  - Suspicious software names ("deepfake", "faceswap", etc.)
  - Unusual camera models
- Gives a "suspicious score" from 0 to 1

### 2. **OCR Text Check** (10% weight)
- Extracts any text from the image
- Looks for suspicious keywords like:
  - "deepfake", "generated", "fake", "manipulated"
  - Watermarks or copyright text
- Gives a "suspicious score" from 0 to 1

### 3. **ML Model Check** (80% weight) ⭐ **Most Important**
- Uses CLIP-ViT (or other ML model) to analyze the image
- The model has been trained on thousands of real vs fake images
- Gives a score from 0 to 1:
  - **0.0-0.5** = Looks Real
  - **0.5** = Unsure
  - **0.5-1.0** = Looks Fake

**For CLIP-ViT specifically:**
- CLIP compares the image to text descriptions:
  - "a real authentic photograph" 
  - "a fake manipulated deepfake image"
- Sees which description matches better
- Converts this to a 0-1 score

---

## 🧮 The Math (Simplified)

```
Final Score = (EXIF Score × 10%) + (OCR Score × 10%) + (ML Score × 80%)
```

**Example:**
- EXIF finds something suspicious: 0.3 (30%)
- OCR finds nothing: 0.0 (0%)
- ML model thinks it's fake: 0.7 (70%)

```
Final Score = (0.3 × 0.10) + (0.0 × 0.10) + (0.7 × 0.80)
            = 0.03 + 0.00 + 0.56
            = 0.59 (59%)
```

Since 0.59 > 0.55 (threshold), it would be flagged as deepfake (if confidence is also high enough).

---

## 🎚️ Special Adjustments

### CLIP-ViT Calibration
CLIP scores are adjusted to reduce false positives:
- If CLIP says 0.6 (60% fake), it might be scaled to 0.52 (52%)
- This makes the model more conservative

### Confidence Boost
If multiple indicators agree (EXIF + OCR + ML all suspicious), the score gets a small boost (+10%).

### Confidence Score
Separate from the main score, confidence measures how sure the tool is:
- High confidence = Strong signals, consistent results
- Low confidence = Weak signals, conflicting results

---

## ✅ Final Decision

An image is flagged as **DEEPFAKE** if:
1. Final Score > **0.55** (55%)
2. AND Confidence > **0.6** (60%)

Both conditions must be true! This prevents false positives.

---

## 📈 What Each Score Range Means

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.0 - 0.4 | Very likely real | ✅ Marked as Real |
| 0.4 - 0.55 | Possibly real, uncertain | ✅ Marked as Real (below threshold) |
| 0.55 - 0.7 | Suspicious, possibly fake | 🔴 Marked as Deepfake (if confidence high) |
| 0.7 - 1.0 | Very likely fake | 🔴 Marked as Deepfake |

---

## 🎯 Why This Approach?

1. **ML Model (80%)** - Most accurate, trained on real data
2. **EXIF/OCR (20%)** - Help catch obvious cases, but less reliable
3. **Threshold (0.55)** - Balanced to catch fakes without too many false alarms
4. **Confidence Check** - Extra safety to prevent mistakes

---

## 💡 In Simple Terms

Think of it like a **jury of 3 judges**:
- **Judge 1 (EXIF)**: Checks the paperwork - 10% say
- **Judge 2 (OCR)**: Checks for obvious signs - 10% say  
- **Judge 3 (ML Model)**: The expert who's seen thousands of cases - 80% say

They all vote, but the expert's opinion matters most. If the final vote is >55% "fake" AND they're confident (>60%), it's declared a deepfake!

