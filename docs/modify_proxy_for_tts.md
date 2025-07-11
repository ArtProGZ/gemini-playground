# Modifying `src/api_proxy/worker.mjs` for Text-to-Speech (TTS) Support

This document outlines the necessary changes to the Deno proxy worker script (`src/api_proxy/worker.mjs`) to enable Text-to-Speech (TTS) functionality using Google's Gemini API.

## Goal

The goal is to allow clients (like a Python script) to send a request to the `/v1/chat/completions` endpoint of this proxy, indicating a TTS operation, and receive raw audio data in response.

## Strategy

1.  **Detect TTS Intent:** The `handleCompletions` function will be modified to check for specific fields in the incoming JSON request (e.g., `tts_settings` and `input_text`) to identify a TTS request.
2.  **Delegate to New Handler:** TTS requests will be delegated to a new, dedicated function: `handleTTSGeneration`.
3.  **Construct Gemini TTS Request:** `handleTTSGeneration` will create the appropriate JSON payload for the Gemini TTS API (e.g., `gemini-2.5-flash-preview-tts`), including the text prompt, `responseModalities: ['Audio']`, and `speechConfig`.
4.  **Call Gemini API:** This handler will make a non-streaming `generateContent` call to the Gemini API.
5.  **Process Gemini Response:** It will extract the base64 encoded audio data and its MIME type from the Gemini JSON response.
6.  **Return Audio to Client:** The proxy will decode the base64 audio and send the raw audio bytes back to the client with the appropriate `Content-Type` header (e.g., `audio/L16;codec=pcm;rate=24000`).

## Detailed Modifications

### 1. Modify `handleCompletions` function

Locate the `async function handleCompletions (req, apiKey)` (around line 145).

**Current Structure (Simplified):**
```javascript
async function handleCompletions (req, apiKey) {
  let model = DEFAULT_MODEL;
  // ... model selection logic ...
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  // ... URL construction ...
  const geminiRequestBody = await transformRequest(req);
  const response = await fetch(url, { /* ... */ });
  // ... response processing for chat ...
  return new Response(body, fixCors(response));
}
```

**Proposed Modifications to `handleCompletions`:**

```javascript
async function handleCompletions (reqBody, apiKey) { // Renamed req to reqBody for clarity
  // Check for TTS specific settings in the request body
  // Client will send: {"model": "gemini-tts-model", "input_text": "text to speak", "tts_settings": {"voice": "VoiceName"}}
  if (reqBody.tts_settings && reqBody.input_text) {
    // It's a TTS request, delegate to a new handler
    return handleTTSGeneration(reqBody, apiKey);
  }

  // Existing logic for chat completions / text generation
  let model = DEFAULT_MODEL;
  switch(true) {
    case typeof reqBody.model !== "string":
      break;
    case reqBody.model.startsWith("models/"):
      model = reqBody.model.substring(7);
      break;
    case reqBody.model.startsWith("gemini-"):
    case reqBody.model.startsWith("learnlm-"):
      model = reqBody.model;
  }

  // Ensure 'model' here refers to a chat model, not a TTS model if not a TTS request
  if (model.includes("-tts")) {
      console.error("TTS models should be called via tts_settings. For chat, use a chat model. Received:", model);
      throw new HttpError("TTS models should be called via tts_settings. For chat, use a chat model.", 400);
  }

  const TASK = reqBody.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (reqBody.stream) { url += "?alt=sse"; }

  const geminiRequestBody = await transformRequest(reqBody); // Existing transformation

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(geminiRequestBody),
  });

  let body = response.body;
  if (response.ok) {
    let id = generateChatcmplId();
    if (reqBody.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: reqBody.stream_options?.include_usage,
          model, id, last: [],
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        const parsedBody = JSON.parse(body);
        // Check if the response has the expected structure for chat
        if (parsedBody.candidates && parsedBody.candidates[0]?.content?.parts) {
             body = processCompletionsResponse(parsedBody, model, id);
        } else if (parsedBody.error) {
            // If Gemini returned an error object
            console.error("Gemini API returned an error for chat:", parsedBody.error);
            throw new HttpError(parsedBody.error.message || "Gemini API error", parsedBody.error.code || response.status || 500);
        } else {
            // If it's not a valid chat response (e.g. Gemini error for a misrouted TTS attempt)
            console.error("Received non-standard chat response from Gemini:", body);
            throw new HttpError(parsedBody.error?.message || "Failed to process Gemini chat response", response.status || 500);
        }
      } catch (e) {
        console.error("Error parsing or processing chat response:", e, "Original body:", body);
        throw new HttpError(e.message || "Invalid JSON response from Gemini for chat", 500);
      }
    }
  } else { // Handle non-ok responses for chat
      const errorBodyText = await response.text();
      let errorMessage = `Gemini chat API request failed with status ${response.status}`;
      try {
          const parsedError = JSON.parse(errorBodyText);
          if (parsedError.error && parsedError.error.message) {
              errorMessage = parsedError.error.message;
          }
      } catch(e) { /* ignore parsing error, use generic message */ }
      console.error(`Gemini chat API error (${response.status}):`, errorBodyText);
      throw new HttpError(errorMessage, response.status);
  }
  return new Response(body, fixCors(response));
}
```

### 2. Add New `handleTTSGeneration` Function

Add this new function definition, for example, after the `handleCompletions` function or alongside other handlers like `handleModels` and `handleEmbeddings`.

```javascript
async function handleTTSGeneration(requestBody, apiKey) {
  const { input_text, model: requestedModel, tts_settings } = requestBody;

  if (!input_text) {
    throw new HttpError("input_text is required for TTS", 400);
  }
  if (!tts_settings || !tts_settings.voice) {
    throw new HttpError("tts_settings with a voice (e.g., tts_settings.voice = 'VoiceName') is required", 400);
  }

  // Use the model specified in the request if it's a TTS model, otherwise default.
  // Ensure the model name used here is the actual Gemini model ID, not an alias.
  const geminiTtsModelId = (requestedModel && requestedModel.includes("-tts"))
                           ? requestedModel
                           : "gemini-2.5-flash-preview-tts"; // Default TTS model

  const voiceName = tts_settings.voice;

  // Construct the prompt for Gemini TTS as per Gemini documentation
  const ttsPrompt = `Say: ${input_text}`; // Or just input_text if "Say:" is not needed by the model

  const geminiPayload = {
    contents: [{ parts: [{ text: ttsPrompt }] }],
    generationConfig: { // Matches structure from Gemini docs for TTS
      responseModalities: ['Audio'],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: voiceName }
        }
      }
    }
    // Note: TTS models generally don't use safetySettings or other chat-specific generationConfig fields.
  };

  // The Gemini endpoint for TTS is typically the non-streaming generateContent
  const url = `${BASE_URL}/${API_VERSION}/models/${geminiTtsModelId}:generateContent`;

  console.log(`Sending TTS request to Gemini: ${url} with voice: ${voiceName}`);

  try {
    const geminiResponse = await fetch(url, {
      method: "POST",
      headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify(geminiPayload),
    });

    if (!geminiResponse.ok) {
      const errorBodyText = await geminiResponse.text();
      let errorMessage = `Gemini TTS API request failed with status ${geminiResponse.status}`;
      try {
        const parsedError = JSON.parse(errorBodyText);
        if (parsedError.error && parsedError.error.message) {
          errorMessage = parsedError.error.message;
        }
      } catch (e) { /* Ignore parsing error, use generic message */ }
      console.error(`Gemini TTS API error (${geminiResponse.status}):`, errorBodyText);
      throw new HttpError(errorMessage, geminiResponse.status);
    }

    const responseJson = await geminiResponse.json();

    if (!responseJson.candidates || !responseJson.candidates[0] ||
        !responseJson.candidates[0].content || !responseJson.candidates[0].content.parts ||
        !responseJson.candidates[0].content.parts[0] ||
        !responseJson.candidates[0].content.parts[0].inlineData ||
        !responseJson.candidates[0].content.parts[0].inlineData.data) {
      console.error("Invalid response structure from Gemini TTS:", JSON.stringify(responseJson, null, 2));
      throw new HttpError("Received invalid or incomplete audio data from Gemini TTS", 500);
    }

    const inlineData = responseJson.candidates[0].content.parts[0].inlineData;
    const base64AudioData = inlineData.data;
    // Gemini TTS typically returns 'audio/L16;codec=pcm;rate=24000' for raw PCM.
    const mimeType = inlineData.mimeType || 'audio/L16;codec=pcm;rate=24000';

    // The 'Buffer' class needs to be available (e.g. via `import { Buffer } from "node:buffer";` at the top of the file if not already present)
    const audioBytes = Buffer.from(base64AudioData, 'base64');

    const responseHeaders = new Headers({
      "Content-Type": mimeType,
      "Content-Length": audioBytes.length.toString(),
    });

    // Apply CORS headers using the existing fixCors utility
    const corsFixedResponseOptions = fixCors({ headers: responseHeaders, status: 200 });

    return new Response(audioBytes, corsFixedResponseOptions);

  } catch (err) {
    console.error("Error in handleTTSGeneration:", err);
    // Re-throw HttpError instances directly, wrap others
    if (err instanceof HttpError) throw err;
    throw new HttpError(err.message || "Failed to generate TTS audio due to an unexpected error", 500);
  }
}
```

### 3. Ensure `Buffer` is Imported

At the top of `src/api_proxy/worker.mjs`, make sure `Buffer` is imported if it's not already:

```javascript
import { Buffer } from "node:buffer"; // Add this if not present
```

## Testing the Modified Proxy

After applying these changes, the proxy should be tested with a `curl` command similar to this:

```bash
curl -X POST \
  https://YOUR_DENO_PROXY_URL/v1/chat/completions \
  -H "Authorization: Bearer YOUR_OPENAI_COMPATIBLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash-preview-tts",
    "input_text": "Hello, this is a test from the updated proxy.",
    "tts_settings": {
        "voice": "Kore"
    }
  }' \
  --output test_audio_from_proxy.raw \
  --verbose
```

**Expected Outcome:**

*   HTTP Status: `200 OK`.
*   `Content-Type` Response Header: `audio/L16;codec=pcm;rate=24000` (or similar, based on Gemini's output).
*   `test_audio_from_proxy.raw`: Should contain the raw audio bytes. This file can then be imported into an audio editor (specifying PCM, 16-bit, 24000Hz, mono) or processed with `pydub` in Python.

## Notes for the Coder

*   **Error Handling:** The suggested code includes basic error handling. Enhance as needed.
*   **Model Naming:** The client can specify a TTS model like `"gemini-2.5-flash-preview-tts"` in the `"model"` field of the JSON payload. The `handleTTSGeneration` function uses this or defaults to `"gemini-2.5-flash-preview-tts"`.
*   **Clarity of `reqBody`:** The incoming `req` parameter in `handleCompletions` has been renamed to `reqBody` in the example modifications for clarity, as it represents the parsed JSON body of the request.
*   **Idempotency & Retries:** This guide does not cover idempotency or retry logic for calls to the Gemini API.
*   **Proxy's Own API Key:** Ensure the proxy environment correctly provides the actual `apiKey` (Google Gemini API Key) to the `makeHeaders` function when calling the Gemini backend.

This concludes the instructions for modifying the proxy.