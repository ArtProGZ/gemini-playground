//Author: PublicAffairs
//Project: https://github.com/PublicAffairs/openai-gemini
//MIT License : https://github.com/PublicAffairs/openai-gemini/blob/main/LICENSE

import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error("Error caught by errHandler:", err); // General log for Deno Deploy

      let responseMessage = "An unknown error occurred during API proxying.";
      let responseStatus = 500;
      let errorDetails = err.stack || "No stack available.";

      if (err instanceof HttpError) {
        // HttpError's message should contain the text from Gemini's error response
        responseMessage = err.message;
        responseStatus = err.status;
      } else if (err instanceof Error) {
        responseMessage = err.message;
      } else if (typeof err === 'string') {
        responseMessage = err;
      }

      // Log what will be sent back for debugging on Deno Deploy
      console.error(`Responding to client with status: ${responseStatus}, message: ${responseMessage}`);

      // Construct a JSON response containing the error details
      const errorResponsePayload = {
        error: {
          message: responseMessage,
          status: responseStatus,
          details_from_worker: errorDetails // Add stack or original error string
        }
      };

      return new Response(JSON.stringify(errorResponsePayload), fixCors({
        status: responseStatus,
        headers: { ...new Headers(fixCors({}).headers), 'Content-Type': 'application/json' } // Ensure JSON content type
      }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const { pathname } = new URL(request.url);

      // Corrected flexible routing logic
      switch (true) {
        case /^\/v1\/chat/.test(pathname) || pathname.endsWith("/chat/completions"):
          if (request.method !== "POST") {
            throw new HttpError("The specified HTTP method is not allowed. Please use POST.", 405);
          }
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);

        case /^\/v1\/embeddings/.test(pathname) || pathname.endsWith("/embeddings"):
           if (request.method !== "POST") {
            throw new HttpError("The specified HTTP method is not allowed. Please use POST.", 405);
          }
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);

        case /^\/v1\/models/.test(pathname) || pathname.endsWith("/models"):
          if (request.method !== "GET") {
            throw new HttpError("The specified HTTP method is not allowed. Please use GET.", 405);
          }
          return handleModels(apiKey)
            .catch(errHandler);

        default:
          console.error(`404 Not Found for pathname: ${pathname}`);
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels (apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  if (!Array.isArray(req.input)) {
    req.input = [ req.input ];
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-1.5-pro-latest";
async function handleCompletions (req, apiKey) {
  // Check for TTS-specific fields
  if (req.input_text && req.tts_settings) {
    return handleTTSGeneration(req, apiKey);
  }

  // The 'transformRequest' function will now handle both OpenAI and Anthropic formats.
  const geminiPayload = await transformRequest(req);
  const model = geminiPayload.model || DEFAULT_MODEL; // Get model from payload or use default.
  delete geminiPayload.model; // Clean up model property before sending.

  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(geminiPayload),
  });

  let body = response.body;
  if (response.ok) {
    let id = generateChatcmplId();
    if (req.stream) {
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
          streamIncludeUsage: req.stream_options?.include_usage,
          model, id, last: [],
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      body = processCompletionsResponse(JSON.parse(body), model, id);
    }
  }
  return new Response(body, fixCors(response));
}


async function handleTTSGeneration(reqBody, apiKey) {
  console.log("TTS function with NEW payload structure was triggered!");
  const model = reqBody.model || "gemini-2.5-flash-preview-tts";
  const url = `${BASE_URL}/${API_VERSION}/models/${model}:generateContent`;

  const ttsPayload = {
    "contents": [{
      "parts": [{ "text": reqBody.input_text }]
    }],
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "voiceConfig": {
          "prebuiltVoiceConfig": {
            "voiceName": reqBody.tts_settings.voice
          }
        }
      }
    }
  };

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(ttsPayload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Gemini TTS API Error:", errorText);
    throw new HttpError(`Gemini TTS API Error: ${response.status} ${response.statusText} - ${errorText}`, response.status);
  }

  const responseData = await response.json();

  if (!responseData.candidates || !responseData.candidates[0] || !responseData.candidates[0].content || !responseData.candidates[0].content.parts || !responseData.candidates[0].content.parts[0] || !responseData.candidates[0].content.parts[0].inlineData) {
    console.error("Invalid TTS response structure:", responseData);
    throw new HttpError("Invalid TTS response structure from Gemini API", 500);
  }

  const audioData = responseData.candidates[0].content.parts[0].inlineData.data;
  const mimeType = responseData.candidates[0].content.parts[0].inlineData.mimeType || "audio/L16;codec=pcm;rate=24000";

  const audioBytes = Buffer.from(audioData, 'base64');

  const headers = new Headers(fixCors({}).headers);
  headers.set("Content-Type", mimeType);
  headers.set("Content-Length", audioBytes.length.toString());

  return new Response(audioBytes, {
    status: 200,
    headers: headers,
  });
}

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));
const fieldsMap = {
  stop: "stopSequences",
  n: "candidateCount",
  max_tokens: "maxOutputTokens",
  max_completion_tokens: "maxOutputTokens",
  temperature: "temperature",
  top_p: "topP",
  top_k: "topK",
  frequency_penalty: "frequencyPenalty",
  presence_penalty: "presencePenalty",
};
const transformConfig = (req) => {
  let cfg = {};
  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }
  if (req.response_format) {
    switch(req.response_format.type) {
      case "json_schema":
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
      // eslint-disable-next-line no-fallthrough
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new Error("Invalid image data: " + url);
    }
    ({ mimeType, data } = match.groups);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

const transformMsg = async ({ role, content }) => {
  const parts = [];
  if (!Array.isArray(content)) {
    parts.push({ text: content });
    return { role, parts };
  }
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          }
        });
        break;
      default:
        throw new TypeError(`Unknown "content" item type: "${item.type}"`);
    }
  }
  if (content.every(item => item.type === "image_url")) {
    parts.push({ text: "" });
  }
  return { role, parts };
};

const transformRequest = async (req) => {
  // Check if the request uses the Anthropic format (has a top-level 'system' property)
  const isAnthropicFormat = req.system && Array.isArray(req.system) && req.system.length > 0;

  // Default 'req.messages' to an empty array if it's missing to prevent 'undefined.find' error.
  let messages = req.messages || [];
  let system_instruction;

  if (isAnthropicFormat) {
    // For Anthropic, the system prompt is in req.system[0].text
    console.log("Anthropic format detected.");
    system_instruction = {
      role: "system",
      parts: [{ text: req.system[0].text }],
    };
  } else {
    // For OpenAI, this code is now safe because 'messages' is guaranteed to be an array.
    const systemMessage = messages.find(msg => msg.role === "system");
    if (systemMessage) {
      system_instruction = await transformMsg(systemMessage);
      messages = messages.filter(msg => msg.role !== "system");
    }
  }

  // Transform the remaining messages (user, assistant/model)
  const contents = [];
  for (const item of messages) {
    const messageCopy = { ...item };
    messageCopy.role = messageCopy.role === "assistant" ? "model" : "user";
    contents.push(await transformMsg(messageCopy));
  }


  if (system_instruction && contents.length === 0) {
    contents.push({ role: "model", parts: { text: " " } });
  }

  let model = req.model || DEFAULT_MODEL;
  if (model.startsWith("models/")) {
      model = model.substring(7);
  }

  return {
    ...(system_instruction && { system_instruction: { parts: system_instruction.parts } }),
    contents,
    safetySettings,
    generationConfig: transformConfig(req),
    model, // Pass the model up to handleCompletions
  };
};

const generateChatcmplId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = {
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
};
const SEP = "\n\n|>";
const transformCandidates = (key, cand) => ({
  index: cand.index || 0,
  [key]: {
    role: "assistant",
    content: cand.content?.parts.map(p => p.text).join(SEP) },
  logprobs: null,
  finish_reason: reasonsMap[cand.finishReason] || cand.finishReason,
});
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data) => ({
  completion_tokens: data.candidatesTokenCount,
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount
});

const processCompletionsResponse = (data, model, id) => {
  return JSON.stringify({
    id,
    choices: data.candidates.map(transformCandidatesMessage),
    created: Math.floor(Date.now()/1000),
    model,
    object: "chat.completion",
    usage: transformUsage(data.usageMetadata),
  });
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream (chunk, controller) {
  chunk = await chunk;
  if (!chunk) { return; }
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true);
}
async function parseStreamFlush (controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
  }
}

function transformResponseStream (data, stop, first) {
  const item = transformCandidatesDelta(data.candidates[0]);
  if (stop) { item.delta = {}; } else { item.finish_reason = null; }
  if (first) { item.delta.content = ""; } else { delete item.delta.role; }
  const output = {
    id: this.id,
    choices: [item],
    created: Math.floor(Date.now()/1000),
    model: this.model,
    object: "chat.completion.chunk",
  };
  if (data.usageMetadata && this.streamIncludeUsage) {
    output.usage = stop ? transformUsage(data.usageMetadata) : null;
  }
  return "data: " + JSON.stringify(output) + delimiter;
}
const delimiter = "\n\n";
async function toOpenAiStream (chunk, controller) {
  const transform = transformResponseStream.bind(this);
  const line = await chunk;
  if (!line) { return; }
  let data;
  try {
    data = JSON.parse(line);
  } catch (err) {
    console.error(line);
    console.error(err);
    const length = this.last.length || 1;
    const candidates = Array.from({ length }, (_, index) => ({
      finishReason: "error",
      content: { parts: [{ text: err }] },
      index,
    }));
    data = { candidates };
  }
  const cand = data.candidates[0];
  console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
  cand.index = cand.index || 0;
  if (!this.last[cand.index]) {
    controller.enqueue(transform(data, false, "first"));
  }
  this.last[cand.index] = data;
  if (cand.content) {
    controller.enqueue(transform(data));
  }
}
async function toOpenAiStreamFlush (controller) {
  const transform = transformResponseStream.bind(this);
  if (this.last.length > 0) {
    for (const data of this.last) {
      controller.enqueue(transform(data, "stop"));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}