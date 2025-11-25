import { AgentNamespace } from "agents";
import { DeepfakeDetectorAgent } from "./agent";

export { DeepfakeDetectorAgent };

export interface Env {
  DeepfakeDetectorAgent: AgentNamespace<DeepfakeDetectorAgent>;
  AI: any;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/agent")) {
      const id = env.DeepfakeDetectorAgent.idFromName("main");
      const agent = env.DeepfakeDetectorAgent.get(id);
      return agent.fetch(request);
    }
    
    if (url.pathname.startsWith("/api/")) {
      const id = env.DeepfakeDetectorAgent.idFromName("main");
      const agent = env.DeepfakeDetectorAgent.get(id);
      
      if (request.method === "POST") {
        try {
          const body = await request.json();
          const method = body.method;
          const params = body.params || [];
          
          if (method === "detectImage") {
            const agentInstance = agent as any;
            if (agentInstance.env) {
              agentInstance.env.AI = env.AI;
            } else {
              agentInstance.env = { AI: env.AI };
            }
            const result = await agentInstance.detectImage(params[0], params[1], params[2]);
            return Response.json(result);
          } else if (method === "getHistory") {
            const result = await (agent as any).getHistory(params[0] || 20);
            return Response.json(result);
          } else if (method === "chat") {
            const agentInstance = agent as any;
            if (agentInstance.env) {
              agentInstance.env.AI = env.AI;
            } else {
              agentInstance.env = { AI: env.AI };
            }
            const result = await agentInstance.chat(params[0]);
            return Response.json(result);
          } else if (method === "getStats") {
            const result = await (agent as any).getStats();
            return Response.json({ stats: result });
          }
        } catch (error: any) {
          console.error("API error:", error);
          return Response.json({ 
            error: error.message || "Internal server error",
            stack: error.stack 
          }, { status: 500 });
        }
      }
      
      return Response.json({ error: "Method not found" }, { status: 404 });
    }

    if (url.pathname === "/logo.png") {
      return new Response("Logo file not found. Please add logo.png to the project root directory (same folder as package.json).\n\nAlternatively, you can:\n1. Convert logo.png to base64 and I can embed it in the HTML\n2. Use Cloudflare Pages to serve static files\n3. Keep using the embedded SVG logo (already working)", {
        status: 404,
        headers: { "Content-Type": "text/plain" },
      });
    }

    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(getHTML(), {
        headers: { "Content-Type": "text/html" },
      });
    }

    return new Response("Not found", { status: 404 });
  },
};

function getHTML() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>I fore see you - AI-Powered Deepfake Detection</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    body {
      font-family: 'Rajdhani', 'Segoe UI', sans-serif;
      background: #0a0e27;
      background-image: 
        radial-gradient(circle at 20% 50%, rgba(0, 102, 255, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(255, 0, 102, 0.1) 0%, transparent 50%),
        linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
      min-height: 100vh;
      padding: 20px;
      color: #e0e7ff;
      position: relative;
      overflow-x: hidden;
    }
    
    body::before {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-image: 
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 102, 255, 0.03) 2px, rgba(0, 102, 255, 0.03) 4px),
        repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(0, 102, 255, 0.03) 2px, rgba(0, 102, 255, 0.03) 4px);
      pointer-events: none;
      z-index: 0;
    }
    .container {
      max-width: 1400px;
      margin: 0 auto;
      position: relative;
      z-index: 1;
    }
    .header {
      background: rgba(10, 14, 39, 0.8);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 12px;
      padding: 40px 30px;
      margin-bottom: 30px;
      box-shadow: 
        0 0 30px rgba(0, 102, 255, 0.2),
        inset 0 0 30px rgba(0, 102, 255, 0.05);
      text-align: center;
      backdrop-filter: blur(10px);
      position: relative;
      overflow: hidden;
    }
    .header::before {
      content: '';
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: radial-gradient(circle, rgba(0, 102, 255, 0.1) 0%, transparent 70%);
      animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 0.3; }
      50% { opacity: 0.6; }
    }
    .logo-container {
      margin: 0 auto 20px;
      display: block;
      position: relative;
      z-index: 1;
    }
    .logo {
      max-width: 500px;
      width: 100%;
      height: auto;
      display: block;
      margin: 0 auto;
      filter: drop-shadow(0 0 30px rgba(0, 102, 255, 0.6));
    }
    .logo .eye {
      animation: pulse-eye 3s ease-in-out infinite;
    }
    @keyframes pulse-eye {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.8; transform: scale(1.05); }
    }
    .logo .circuit-lines {
      animation: circuit-pulse 4s ease-in-out infinite;
    }
    @keyframes circuit-pulse {
      0%, 100% { opacity: 0.4; }
      50% { opacity: 0.8; }
    }
    .main-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 2.2em;
      margin: 15px 0;
      background: linear-gradient(135deg, #00ccff 0%, #0066ff 50%, #ff0066 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      font-weight: 700;
      letter-spacing: 3px;
      text-transform: uppercase;
      position: relative;
      z-index: 1;
      text-shadow: 0 0 30px rgba(0, 102, 255, 0.5);
    }
    .header p { 
      color: rgba(224, 231, 255, 0.7); 
      font-size: 16px; 
      letter-spacing: 1px;
      position: relative;
      z-index: 1;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 30px;
    }
    .stat-card {
      background: rgba(10, 14, 39, 0.6);
      border: 1px solid rgba(0, 102, 255, 0.2);
      border-radius: 8px;
      padding: 25px 20px;
      box-shadow: 
        0 0 20px rgba(0, 102, 255, 0.1),
        inset 0 0 20px rgba(0, 102, 255, 0.05);
      text-align: center;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }
    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(0, 102, 255, 0.2), transparent);
      transition: left 0.5s;
    }
    .stat-card:hover {
      border-color: rgba(0, 102, 255, 0.5);
      box-shadow: 
        0 0 30px rgba(0, 102, 255, 0.3),
        inset 0 0 20px rgba(0, 102, 255, 0.1);
      transform: translateY(-2px);
    }
    .stat-card:hover::before {
      left: 100%;
    }
    .stat-card h3 {
      font-family: 'Orbitron', sans-serif;
      font-size: 12px;
      color: rgba(224, 231, 255, 0.6);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 2px;
      font-weight: 600;
    }
    .stat-card .value {
      font-family: 'Orbitron', sans-serif;
      font-size: 36px;
      font-weight: 700;
      background: linear-gradient(135deg, #00ccff 0%, #0066ff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      text-shadow: 0 0 20px rgba(0, 102, 255, 0.5);
    }
    .main-content {
      display: grid;
      grid-template-columns: 1fr 400px;
      gap: 20px;
    }
    .upload-section {
      background: rgba(10, 14, 39, 0.6);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 12px;
      padding: 30px;
      box-shadow: 
        0 0 30px rgba(0, 102, 255, 0.2),
        inset 0 0 30px rgba(0, 102, 255, 0.05);
      backdrop-filter: blur(10px);
    }
    .upload-area {
      border: 2px dashed rgba(0, 102, 255, 0.4);
      border-radius: 8px;
      padding: 60px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
      background: rgba(0, 102, 255, 0.05);
      position: relative;
      overflow: hidden;
    }
    .upload-area::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(0, 102, 255, 0.1), transparent);
      transition: left 0.5s;
    }
    .upload-area:hover {
      background: rgba(0, 102, 255, 0.1);
      border-color: rgba(0, 204, 255, 0.6);
      box-shadow: 0 0 30px rgba(0, 102, 255, 0.3);
    }
    .upload-area:hover::before {
      left: 100%;
    }
    .upload-area.dragover {
      background: rgba(0, 102, 255, 0.15);
      border-color: #00ccff;
      box-shadow: 0 0 40px rgba(0, 204, 255, 0.5);
      transform: scale(1.01);
    }
    .upload-icon {
      font-size: 48px;
      margin-bottom: 20px;
      filter: drop-shadow(0 0 10px rgba(0, 102, 255, 0.5));
    }
    .upload-text {
      font-family: 'Orbitron', sans-serif;
      font-size: 18px;
      color: #00ccff;
      font-weight: 600;
      margin-bottom: 10px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    .upload-hint {
      color: rgba(224, 231, 255, 0.5);
      font-size: 13px;
      letter-spacing: 0.5px;
    }
    #fileInput { display: none; }
    .preview-section {
      margin-top: 20px;
      display: none;
    }
    .preview-section.active {
      display: block;
    }
    .image-preview {
      max-width: 100%;
      max-height: 400px;
      border-radius: 8px;
      margin-bottom: 20px;
      border: 1px solid rgba(0, 102, 255, 0.3);
      box-shadow: 
        0 0 30px rgba(0, 102, 255, 0.3),
        inset 0 0 20px rgba(0, 102, 255, 0.1);
    }
    .detect-btn {
      width: 100%;
      padding: 18px;
      background: linear-gradient(135deg, #0066ff 0%, #00ccff 100%);
      color: #0a0e27;
      border: 1px solid rgba(0, 204, 255, 0.5);
      border-radius: 8px;
      font-family: 'Orbitron', sans-serif;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 
        0 0 20px rgba(0, 102, 255, 0.4),
        inset 0 0 20px rgba(0, 204, 255, 0.2);
      position: relative;
      overflow: hidden;
    }
    .detect-btn::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 0;
      height: 0;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.3);
      transform: translate(-50%, -50%);
      transition: width 0.6s, height 0.6s;
    }
    .detect-btn:hover {
      transform: translateY(-2px);
      box-shadow: 
        0 0 30px rgba(0, 204, 255, 0.6),
        inset 0 0 30px rgba(0, 204, 255, 0.3);
      border-color: #00ccff;
    }
    .detect-btn:hover::before {
      width: 300px;
      height: 300px;
    }
    .detect-btn:active {
      transform: translateY(0);
    }
    .detect-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }
    .status {
      margin-top: 15px;
      padding: 15px;
      background: rgba(0, 102, 255, 0.1);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 8px;
      color: #00ccff;
      font-size: 14px;
      display: none;
      font-family: 'Orbitron', sans-serif;
      letter-spacing: 1px;
      box-shadow: 0 0 20px rgba(0, 102, 255, 0.2);
    }
    .status.active { display: block; }
    .status.error {
      background: rgba(255, 0, 102, 0.1);
      border-color: rgba(255, 0, 102, 0.3);
      color: #ff0066;
      box-shadow: 0 0 20px rgba(255, 0, 102, 0.2);
    }
    .result {
      margin-top: 20px;
      padding: 25px;
      background: rgba(10, 14, 39, 0.8);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 12px;
      border-left: 4px solid #0066ff;
      display: none;
      backdrop-filter: blur(10px);
      box-shadow: 
        0 0 30px rgba(0, 102, 255, 0.2),
        inset 0 0 30px rgba(0, 102, 255, 0.05);
    }
    .result.active { display: block; }
    .result.deepfake {
      border-left-color: #ff0066;
      box-shadow: 
        0 0 30px rgba(255, 0, 102, 0.3),
        inset 0 0 30px rgba(255, 0, 102, 0.1);
    }
    .result.authentic {
      border-left-color: #00ff88;
      box-shadow: 
        0 0 30px rgba(0, 255, 136, 0.3),
        inset 0 0 30px rgba(0, 255, 136, 0.1);
    }
    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    .result-label {
      font-family: 'Orbitron', sans-serif;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
    }
    .result-label.deepfake { 
      color: #ff0066;
      text-shadow: 0 0 20px rgba(255, 0, 102, 0.8);
    }
    .result-label.real { 
      color: #00ff88;
      text-shadow: 0 0 20px rgba(0, 255, 136, 0.8);
    }
    .confidence-bar {
      width: 100%;
      height: 35px;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 8px;
      overflow: hidden;
      margin: 20px 0;
      position: relative;
      box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.5);
    }
    .confidence-fill {
      height: 100%;
      background: linear-gradient(90deg, #00ff88 0%, #00ccff 50%, #ff0066 100%);
      transition: width 0.5s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #0a0e27;
      font-weight: 700;
      font-size: 14px;
      font-family: 'Orbitron', sans-serif;
      letter-spacing: 1px;
      box-shadow: 0 0 20px rgba(0, 102, 255, 0.5);
      position: relative;
    }
    .confidence-fill::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
      animation: shimmer 2s infinite;
    }
    @keyframes shimmer {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(100%); }
    }
    .indicators {
      margin-top: 15px;
    }
    .indicator {
      display: inline-block;
      padding: 8px 14px;
      background: rgba(0, 102, 255, 0.1);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 6px;
      margin: 5px 5px 5px 0;
      font-size: 12px;
      color: #00ccff;
      font-weight: 500;
      letter-spacing: 0.5px;
      transition: all 0.3s ease;
    }
    .indicator:hover {
      background: rgba(0, 102, 255, 0.2);
      border-color: rgba(0, 204, 255, 0.5);
      box-shadow: 0 0 10px rgba(0, 102, 255, 0.3);
    }
    .analysis-text {
      margin-top: 15px;
      padding: 20px;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(0, 102, 255, 0.2);
      border-radius: 8px;
      color: rgba(224, 231, 255, 0.9);
      font-size: 14px;
      line-height: 1.8;
      white-space: pre-wrap;
      max-height: 300px;
      overflow-y: auto;
      font-family: 'Rajdhani', sans-serif;
      letter-spacing: 0.5px;
    }
    .analysis-text::-webkit-scrollbar {
      width: 8px;
    }
    .analysis-text::-webkit-scrollbar-track {
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
    }
    .analysis-text::-webkit-scrollbar-thumb {
      background: rgba(0, 102, 255, 0.5);
      border-radius: 4px;
    }
    .analysis-text::-webkit-scrollbar-thumb:hover {
      background: rgba(0, 204, 255, 0.7);
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .history-card, .chat-card {
      background: rgba(10, 14, 39, 0.6);
      border: 1px solid rgba(0, 102, 255, 0.3);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 
        0 0 30px rgba(0, 102, 255, 0.2),
        inset 0 0 30px rgba(0, 102, 255, 0.05);
      max-height: 500px;
      display: flex;
      flex-direction: column;
      backdrop-filter: blur(10px);
    }
    .card-header {
      font-family: 'Orbitron', sans-serif;
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 15px;
      color: #00ccff;
      display: flex;
      align-items: center;
      gap: 10px;
      letter-spacing: 2px;
      text-transform: uppercase;
    }
    .history-list {
      overflow-y: auto;
      flex: 1;
    }
    .history-list::-webkit-scrollbar {
      width: 6px;
    }
    .history-list::-webkit-scrollbar-track {
      background: rgba(0, 0, 0, 0.3);
      border-radius: 3px;
    }
    .history-list::-webkit-scrollbar-thumb {
      background: rgba(0, 102, 255, 0.5);
      border-radius: 3px;
    }
    .history-item {
      padding: 12px;
      background: rgba(0, 102, 255, 0.05);
      border: 1px solid rgba(0, 102, 255, 0.2);
      border-radius: 6px;
      margin-bottom: 10px;
      border-left: 3px solid #0066ff;
      cursor: pointer;
      transition: all 0.3s ease;
    }
    .history-item:hover {
      background: rgba(0, 102, 255, 0.15);
      border-color: rgba(0, 204, 255, 0.4);
      transform: translateX(5px);
      box-shadow: 0 0 15px rgba(0, 102, 255, 0.3);
    }
    .history-item.deepfake {
      border-left-color: #ff0066;
    }
    .history-item.real {
      border-left-color: #00ff88;
    }
    .history-item .name {
      font-weight: 600;
      color: rgba(224, 231, 255, 0.9);
      margin-bottom: 8px;
      font-family: 'Rajdhani', sans-serif;
      letter-spacing: 0.5px;
    }
    .history-item .meta {
      font-size: 12px;
      color: rgba(224, 231, 255, 0.6);
      display: flex;
      justify-content: space-between;
      font-family: 'Orbitron', sans-serif;
      letter-spacing: 1px;
    }
    .chat-messages {
      flex: 1;
      overflow-y: auto;
      margin-bottom: 15px;
      padding: 10px;
      background: #f8f9ff;
      border-radius: 8px;
      max-height: 300px;
    }
    .chat-message {
      padding: 10px;
      margin-bottom: 10px;
      border-radius: 8px;
      font-size: 14px;
    }
    .chat-message.user {
      background: #667eea;
      color: white;
      margin-left: 20%;
    }
    .chat-message.assistant {
      background: white;
      color: #333;
      margin-right: 20%;
    }
    .chat-input-area {
      display: flex;
      gap: 10px;
    }
    .chat-input {
      flex: 1;
      padding: 10px;
      border: 1px solid #e0e7ff;
      border-radius: 8px;
      font-size: 14px;
    }
    .chat-send {
      padding: 10px 20px;
      background: #667eea;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    @media (max-width: 1024px) {
      .main-content {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo-container">
        <svg class="logo" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#00ccff;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#0066ff;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="redGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#ff0066;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#cc0044;stop-opacity:1" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <rect width="800" height="200" fill="#0a0e27"/>
          <g opacity="0.3">
            <path d="M0,0 L800,0 L800,200 L0,200 Z" fill="url(#blueGrad)" opacity="0.1"/>
            <circle cx="100" cy="100" r="80" fill="none" stroke="#0066ff" stroke-width="1" opacity="0.2"/>
            <circle cx="700" cy="100" r="60" fill="none" stroke="#ff0066" stroke-width="1" opacity="0.2"/>
          </g>
          <g class="eye">
            <circle cx="120" cy="100" r="45" fill="none" stroke="url(#blueGrad)" stroke-width="3" filter="url(#glow)"/>
            <circle cx="120" cy="100" r="30" fill="none" stroke="url(#redGrad)" stroke-width="2" stroke-dasharray="5,5" filter="url(#glow)"/>
            <circle cx="120" cy="100" r="15" fill="url(#redGrad)" filter="url(#glow)"/>
            <circle cx="120" cy="100" r="8" fill="#ff0066"/>
          </g>
          <text x="200" y="120" font-family="Orbitron, monospace" font-size="80" font-weight="900" fill="url(#blueGrad)" filter="url(#glow)">4CU</text>
          <text x="50" y="170" font-family="Rajdhani, sans-serif" font-size="24" font-weight="600" fill="#00ccff" letter-spacing="3">I fore see you</text>
          <g class="circuit-lines" opacity="0.6">
            <line x1="0" y1="50" x2="800" y2="50" stroke="#0066ff" stroke-width="1" filter="url(#glow)"/>
            <line x1="0" y1="150" x2="800" y2="150" stroke="#0066ff" stroke-width="1" filter="url(#glow)"/>
            <line x1="100" y1="0" x2="100" y2="200" stroke="#ff0066" stroke-width="1" filter="url(#glow)"/>
            <line x1="700" y1="0" x2="700" y2="200" stroke="#ff0066" stroke-width="1" filter="url(#glow)"/>
          </g>
        </svg>
      </div>
      <h1 class="main-title">I fore see you</h1>
      <p>AI-Powered Media Authenticity Analysis using Cloudflare Workers AI</p>
    </div>
    
    <div class="stats" id="stats">
      <div class="stat-card">
        <h3>Total Scans</h3>
        <div class="value" id="statTotal">0</div>
      </div>
      <div class="stat-card">
        <h3>Deepfakes Detected</h3>
        <div class="value" id="statDeepfakes">0</div>
      </div>
      <div class="stat-card">
        <h3>Authentic Media</h3>
        <div class="value" id="statReal">0</div>
      </div>
      <div class="stat-card">
        <h3>Avg Confidence</h3>
        <div class="value" id="statConfidence">0%</div>
      </div>
    </div>

    <div class="main-content">
      <div class="upload-section">
        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
          <div class="upload-icon">📸</div>
          <div class="upload-text">Drop images here or click to upload</div>
          <div class="upload-hint">Supports JPG, PNG, WebP (Max 10MB each). Multiple files supported.</div>
        </div>
        <input type="file" id="fileInput" accept="image/*" multiple />
        
        <div class="preview-section" id="previewSection">
          <img id="imagePreview" class="image-preview" />
          <button class="detect-btn" id="detectBtn" onclick="detectImage()">🔍 Analyze for Deepfake</button>
        </div>
        
        <div class="status" id="status"></div>
        
        <div class="result" id="result">
          <div class="result-header">
            <div class="result-label" id="resultLabel">Result</div>
          </div>
          <div class="confidence-bar">
            <div class="confidence-fill" id="confidenceFill" style="width: 0%">0%</div>
          </div>
          <div class="indicators" id="indicators"></div>
          <div class="analysis-text" id="analysisText"></div>
        </div>
      </div>

      <div class="sidebar">
        <div class="history-card">
          <div class="card-header">📋 Detection History</div>
          <div class="history-list" id="historyList"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentImageData = null;
    let currentFileName = null;

    async function init() {
      setupUpload();
      
      try {
        const response = await fetch('/api/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ method: 'getHistory', params: [20] })
        });
        const data = await response.json();
        if (data.type === 'history') {
          updateHistory(data.data);
        }
        
        const statsResponse = await fetch('/api/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ method: 'getStats', params: [] })
        });
        const statsData = await statsResponse.json();
        if (statsData.stats) {
          updateStats(statsData.stats);
        }
      } catch (e) {
        console.error('Error initializing:', e);
      }
    }

    function setupUpload() {
      const uploadArea = document.getElementById('uploadArea');
      const fileInput = document.getElementById('fileInput');
      
      uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
      });
      
      uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
      });
      
      uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) {
          if (files.length === 1) {
            handleFile(files[0]);
          } else {
            handleMultipleFiles(files);
          }
        }
      });
      
      fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length > 0) {
          if (files.length === 1) {
            handleFile(files[0]);
          } else {
            handleMultipleFiles(files);
          }
        }
      });
    }

    function handleFile(file) {
      if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Max 10MB.');
        return;
      }
      
      currentFileName = file.name;
      const reader = new FileReader();
      reader.onload = (e) => {
        currentImageData = e.target.result;
        document.getElementById('imagePreview').src = currentImageData;
        document.getElementById('previewSection').classList.add('active');
        document.getElementById('result').classList.remove('active');
      };
      reader.readAsDataURL(file);
    }

    async function handleMultipleFiles(files) {
      const validFiles = files.filter(function(f) {
        if (f.size > 10 * 1024 * 1024) {
          console.warn('File ' + f.name + ' is too large. Skipping.');
          return false;
        }
        return true;
      });

      if (validFiles.length === 0) {
        alert('No valid files to process.');
        return;
      }

      if (validFiles.length > 10) {
        if (!confirm('You selected ' + validFiles.length + ' files. Processing more than 10 files may take a while. Continue?')) {
          return;
        }
      }

      showStatus('Processing ' + validFiles.length + ' files...', 'info');
      document.getElementById('previewSection').classList.remove('active');
      document.getElementById('result').classList.remove('active');

      const results = [];
      for (let i = 0; i < validFiles.length; i++) {
        const file = validFiles[i];
        showStatus('Processing ' + (i + 1) + '/' + validFiles.length + ': ' + file.name + '...', 'info');
        
        try {
          const imageData = await readFileAsDataURL(file);
          const response = await fetch('/api/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              method: 'detectImage',
              params: [imageData, file.name, file.size]
            })
          });
          
          const data = await response.json();
          if (data.success && data.result) {
            results.push({ fileName: file.name, result: data.result });
          }
        } catch (e) {
          console.error('Error processing ' + file.name + ':', e);
          results.push({ fileName: file.name, error: e.message });
        }
      }

      showBatchResults(results);
      loadHistory();
    }

    function readFileAsDataURL(file) {
      return new Promise(function(resolve, reject) {
        const reader = new FileReader();
        reader.onload = function(e) { resolve(e.target.result); };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }

    function showBatchResults(results) {
      const resultDiv = document.getElementById('result');
      const label = document.getElementById('resultLabel');
      const fill = document.getElementById('confidenceFill');
      const indicators = document.getElementById('indicators');
      const analysis = document.getElementById('analysisText');

      const deepfakes = results.filter(function(r) { return r.result && r.result.isDeepfake; });
      const authentic = results.filter(function(r) { return r.result && !r.result.isDeepfake; });
      const avgConfidence = results
        .filter(function(r) { return r.result; })
        .reduce(function(sum, r) { return sum + (r.result.confidence || 0); }, 0) / results.length || 0;

      label.textContent = 'Batch: ' + deepfakes.length + ' Deepfake' + (deepfakes.length !== 1 ? 's' : '') + ', ' + authentic.length + ' Authentic';
      label.className = 'result-label ' + (deepfakes.length > 0 ? 'deepfake' : 'real');
      
      fill.style.width = avgConfidence + '%';
      fill.textContent = Math.round(avgConfidence) + '% Avg Confidence';
      
      indicators.innerHTML = '';
      const allIndicators = new Set();
      results.forEach(function(r) {
        if (r.result && r.result.indicators) {
          r.result.indicators.forEach(function(ind) { allIndicators.add(ind); });
        }
      });
      allIndicators.forEach(function(ind) {
        const span = document.createElement('span');
        span.className = 'indicator';
        span.textContent = ind;
        indicators.appendChild(span);
      });
      
      let analysisText = 'Batch Analysis Results:\\n\\n';
      results.forEach(function(r) {
        if (r.error) {
          analysisText += '❌ ' + r.fileName + ': Error - ' + r.error + '\\n';
        } else if (r.result) {
          const status = r.result.isDeepfake ? '⚠️ Deepfake' : '✅ Authentic';
          analysisText += status + ' ' + r.fileName + ': ' + r.result.confidence + '% confidence\\n';
        }
      });
      
      analysis.textContent = analysisText;
      resultDiv.classList.add('active');
      document.getElementById('status').classList.remove('active');
    }

    async function detectImage() {
      console.log('detectImage called', { hasImage: !!currentImageData });
      
      if (!currentImageData) {
        alert('Please upload an image first');
        return;
      }
      
      const btn = document.getElementById('detectBtn');
      btn.disabled = true;
      btn.textContent = 'Analyzing...';
      
      document.getElementById('result').classList.remove('active');
      showStatus('Processing image with AI models...', 'info');
      
      try {
        const response = await fetch('/api/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            method: 'detectImage',
            params: [currentImageData, currentFileName, 0]
          })
        });
        
        const data = await response.json();
        console.log('Detection result:', data);
        
        if (data.error) {
          console.error('Detection error:', data.error);
          showStatus('Error: ' + data.error + ' (Check console for details)', 'error');
        }
        
        if (data.success && data.result) {
          showResult(data.result);
          if (data.stats) {
            updateStats(data.stats);
          }
          loadHistory();
        } else if (data.result) {
          showResult(data.result);
          if (data.stats) {
            updateStats(data.stats);
          }
          loadHistory();
        } else {
          showStatus('Detection failed. Check browser console (F12) for details.', 'error');
        }
      } catch (e) {
        console.error('Error detecting image:', e);
        showStatus('Error: ' + e.message + ' (Check console F12 for details)', 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = '🔍 Analyze for Deepfake';
      }
    }

    function handleMessage(data) {
      if (data.type === 'init') {
        updateStats(data.stats);
        updateHistory(data.detections || []);
      } else if (data.type === 'status') {
        showStatus(data.message, 'info');
      } else if (data.type === 'detection') {
        showResult(data.result);
        updateStats(data.stats);
        loadHistory();
        document.getElementById('detectBtn').disabled = false;
        document.getElementById('detectBtn').textContent = '🔍 Analyze for Deepfake';
      } else if (data.type === 'history') {
        updateHistory(data.data);
      } else if (data.type === 'chat') {
        addChatMessage('assistant', data.message);
      }
    }

    function showResult(result) {
      const resultDiv = document.getElementById('result');
      const label = document.getElementById('resultLabel');
      const fill = document.getElementById('confidenceFill');
      const indicators = document.getElementById('indicators');
      const analysis = document.getElementById('analysisText');
      
      label.textContent = result.isDeepfake ? '⚠️ Deepfake Detected' : '✅ Authentic Media';
      label.className = 'result-label ' + (result.isDeepfake ? 'deepfake' : 'real');
      
      fill.style.width = result.confidence + '%';
      fill.textContent = result.confidence + '% Confidence';
      
      indicators.innerHTML = '';
      if (result.indicators && result.indicators.length > 0) {
        result.indicators.forEach(ind => {
          const span = document.createElement('span');
          span.className = 'indicator';
          span.textContent = ind;
          indicators.appendChild(span);
        });
      }
      
      analysis.textContent = result.analysis || 'Analysis completed.';
      
      resultDiv.classList.add('active');
      document.getElementById('status').classList.remove('active');
    }

    function showStatus(message, type) {
      const status = document.getElementById('status');
      status.textContent = message;
      status.className = 'status active';
    }

    function updateStats(stats) {
      if (!stats) return;
      document.getElementById('statTotal').textContent = stats.total || 0;
      document.getElementById('statDeepfakes').textContent = stats.deepfakes || 0;
      document.getElementById('statReal').textContent = stats.real || 0;
      document.getElementById('statConfidence').textContent = (stats.avgConfidence || 0) + '%';
    }

    async function loadHistory() {
      try {
        const response = await fetch('/api/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ method: 'getHistory', params: [20] })
        });
        const data = await response.json();
        if (data.type === 'history') {
          updateHistory(data.data);
        }
      } catch (e) {
        console.error('Error loading history:', e);
      }
    }

    function updateHistory(detections) {
      const list = document.getElementById('historyList');
      list.innerHTML = '';
      
      if (detections.length === 0) {
        list.innerHTML = '<div style="text-align:center;color:#999;padding:20px;">No detections yet</div>';
        return;
      }
      
      detections.forEach(d => {
        const item = document.createElement('div');
        item.className = 'history-item ' + (d.is_deepfake ? 'deepfake' : 'real');
        item.innerHTML = \`
          <div class="name">\${d.file_name}</div>
          <div class="meta">
            <span>\${d.is_deepfake ? '⚠️ Deepfake' : '✅ Real'}</span>
            <span>\${d.confidence}%</span>
          </div>
        \`;
        list.appendChild(item);
      });
    }

    async function sendChat() {
      const input = document.getElementById('chatInput');
      const message = input.value.trim();
      if (!message) return;
      
      addChatMessage('user', message);
      input.value = '';
      
      try {
        const response = await fetch('/api/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ method: 'chat', params: [message] })
        });
        const data = await response.json();
        if (data.type === 'chat') {
          addChatMessage('assistant', data.message);
        }
      } catch (e) {
        console.error('Error sending chat:', e);
        addChatMessage('assistant', 'Error: Could not get response');
      }
    }

    function addChatMessage(type, text) {
      const messages = document.getElementById('chatMessages');
      const msg = document.createElement('div');
      msg.className = 'chat-message ' + type;
      msg.textContent = text;
      messages.appendChild(msg);
      messages.scrollTop = messages.scrollHeight;
    }

    init();
  </script>
</body>
</html>`;
}
