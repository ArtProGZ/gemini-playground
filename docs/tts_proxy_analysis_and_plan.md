# Analysis and Plan for TTS Proxy Modification

This document outlines the analysis of the current proxy mechanism in the `gemini-playground` fork, the reasons for Text-to-Speech (TTS) request failures, and the plan to enable TTS functionality.

## Request Flow Diagram

```mermaid
graph TD
    Client[Client Application e.g., Web UI, Python Script] -->|HTTP Request /v1/chat/completions| DenoServer(Deno Server - src/deno_index.ts);
    DenoServer -->|Forwards API calls| ProxyWorker(API Proxy Worker - src/api_proxy/worker.mjs);

    subgraph ProxyWorker [src/api_proxy/worker.mjs]
        direction LR
        Router{Request Router} --> HandleCompletions(handleCompletions);
        HandleCompletions -- TTS Request? --> HandleTTS(handleTTSGeneration - NEW);
        HandleCompletions -- Chat Request --> ProcessChat(Existing Chat Logic);
        HandleTTS --> GeminiTTSReq(Construct Gemini TTS Payload);
        ProcessChat --> GeminiChatReq(Construct Gemini Chat Payload);
        GeminiTTSReq --> |POST /v1beta/models/tts-model:generateContent| GeminiAPI[Google Gemini API];
        GeminiChatReq --> |POST /v1beta/models/chat-model:{stream}generateContent| GeminiAPI;
    end

    GeminiAPI -->|Audio Data (Base64)| HandleTTS;
    GeminiAPI -->|Text Data (JSON/Stream)| ProcessChat;
    HandleTTS -->|Raw Audio Bytes| Client;
    ProcessChat -->|Formatted Text Response| Client;

    style DenoServer fill:#f9f,stroke:#333,stroke-width:2px;
    style ProxyWorker fill:#ccf,stroke:#333,stroke-width:2px;
    style Client fill:#bbf,stroke:#333,stroke-width:2px;
    style GeminiAPI fill:#ff9,stroke:#333,stroke-width:2px;
    style HandleTTS fill:#cfc,stroke:#333,stroke-width:2px;
```

## Detailed Explanation

### 1. Current End-to-End Request Flow (General API Requests):

*   **Client to Deno Server:** A client application sends an HTTP request (e.g., to `/v1/chat/completions`) to the Deno server, which is managed by `src/deno_index.ts`.
*   **Deno Server Routing (`src/deno_index.ts`):**
    *   The main `handleRequest` function (around line 87) inspects the request URL.
    *   If the URL path matches known API endpoints like `/chat/completions`, `/embeddings`, or `/models` (lines 96-98), the request is passed to `handleAPIRequest` (line 99).
    *   `handleAPIRequest` (around line 70) dynamically imports the `worker.mjs` module (line 72) and calls its `fetch` method (line 73), effectively delegating the request to the API proxy worker.
*   **API Proxy Worker (`src/api_proxy/worker.mjs`):**
    *   The `fetch` method (around line 8) acts as the entry point for the worker.
    *   It performs basic routing based on the request path. For `/chat/completions` (line 26), it calls the `handleCompletions` function (line 28).
    *   The `handleCompletions` function (around line 145) is currently designed *only* for text-based chat completions. It transforms the incoming OpenAI-compatible request into a Gemini chat request using `transformRequest` (line 163), sends it to the Gemini API, and processes the text-based response.

### 2. Why Text-to-Speech (TTS) Requests Are Failing:

Based on the current implementation of `src/api_proxy/worker.mjs` and the information in `docs/modify_proxy_for_tts.md`, TTS requests fail for the following reasons:

*   **No TTS-Specific Handling:** The `handleCompletions` function (`src/api_proxy/worker.mjs` line 145) lacks any logic to detect or differentiate a TTS request from a standard chat completion request. It assumes all requests to `/v1/chat/completions` are for text generation.
*   **Incorrect Request Transformation:** The `transformRequest` function (`src/api_proxy/worker.mjs` line 331) is tailored to convert OpenAI-style chat messages into Gemini's `contents` format for text. A Gemini TTS request, as detailed in `docs/modify_proxy_for_tts.md` (lines 161-172), requires a different payload structure, notably including `responseModalities: ['Audio']` and `speechConfig`. The current transformation logic does not create this structure.
*   **Mismatched Response Processing:** The existing `handleCompletions` function expects a JSON response containing text candidates from Gemini. It processes this either as a stream for `streamGenerateContent` or as a complete JSON object for `generateContent` (`src/api_proxy/worker.mjs` lines 166-188). A successful Gemini TTS API call, however, returns a JSON object containing base64 encoded audio data and its MIME type (`docs/modify_proxy_for_tts.md` lines 201-213). The current code would not be able to parse this audio data correctly or return it to the client as raw audio bytes with the appropriate `Content-Type` header.
*   **Model and Endpoint Mismatch:** Even if a TTS model name were passed, the request would still be sent to a chat-oriented Gemini endpoint (like `generateContent` or `streamGenerateContent` without the specific TTS payload structure), leading to errors from the Gemini API.

### 3. Necessary Modifications for TTS Functionality (as per `docs/modify_proxy_for_tts.md`):

The document `docs/modify_proxy_for_tts.md` provides a clear roadmap to fix this:

*   **Modify `handleCompletions` in `src/api_proxy/worker.mjs` (around line 145):**
    *   **Detect TTS Intent:** The function will be updated to first inspect the incoming JSON request body for specific fields that signify a TTS request, namely `tts_settings` and `input_text` (`docs/modify_proxy_for_tts.md` lines 43-44).
    *   **Delegate to a New Handler:** If these fields are present, the request will be delegated to a new, dedicated function: `handleTTSGeneration(reqBody, apiKey)` (`docs/modify_proxy_for_tts.md` line 46).
    *   **Fallback to Chat:** If the TTS-specific fields are not found, the function will proceed with its existing logic for handling chat completions. An added check will prevent TTS models from being erroneously used for chat requests (`docs/modify_proxy_for_tts.md` lines 62-66).

*   **Add New `handleTTSGeneration` Function to `src/api_proxy/worker.mjs` (as per `docs/modify_proxy_for_tts.md` lines 135-234):**
    *   **Input Validation:** This new function will validate the presence of `input_text` and `tts_settings.voice` in the request.
    *   **Model Selection:** It will use the TTS model specified in the request (e.g., `gemini-2.5-flash-preview-tts`) or a default TTS model.
    *   **Construct Gemini TTS Request:** It will build the correct JSON payload required by the Gemini TTS API. This payload includes the text prompt, `responseModalities: ['Audio']`, and `speechConfig` with the specified voice (`docs/modify_proxy_for_tts.md` lines 161-172).
    *   **Call Gemini API:** It will make a non-streaming `generateContent` call to the appropriate Gemini model endpoint for TTS (e.g., `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent`).
    *   **Process Gemini Audio Response:**
        *   It will parse the JSON response from Gemini.
        *   It will extract the base64 encoded audio data (from `candidates[0].content.parts[0].inlineData.data`) and its MIME type (`inlineData.mimeType`).
    *   **Return Raw Audio to Client:**
        *   The base64 audio data will be decoded into raw audio bytes (using `Buffer.from(..., 'base64')`, requiring `Buffer` to be imported at the top of the file - `docs/modify_proxy_for_tts.md` lines 239-243).
        *   A new `Response` object will be created with these audio bytes as the body.
        *   Crucially, the `Content-Type` header of this response will be set to the MIME type received from Gemini (e.g., `audio/L16;codec=pcm;rate=24000`), and `Content-Length` will also be set.
        *   CORS headers will be applied using the existing `fixCors` utility.

### 4. End-to-End Flow for a TTS Request (with the fix):

1.  Client sends a POST request to `/v1/chat/completions` with a JSON body like:
    ```json
    {
      "model": "gemini-2.5-flash-preview-tts",
      "input_text": "Hello world",
      "tts_settings": { "voice": "VoiceName" }
    }
    ```
2.  Request flows through `src/deno_index.ts` to `src/api_proxy/worker.mjs`.
3.  The modified `handleCompletions` in `src/api_proxy/worker.mjs` detects `input_text` and `tts_settings`, then calls `handleTTSGeneration`.
4.  `handleTTSGeneration` constructs the Gemini TTS API payload and calls the Gemini API.
5.  Gemini API responds with JSON containing base64 audio.
6.  `handleTTSGeneration` decodes the audio, creates a new `Response` with raw audio bytes and the correct `Content-Type` (e.g., `audio/L16;codec=pcm;rate=24000`).
7.  This audio response is sent back through the Deno server to the client.