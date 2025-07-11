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
        responseMessage = err.message;
        responseStatus = err.status;
      } else if (err instanceof Error) {
        responseMessage = err.message;
      } else if (typeof err === 'string') {
        responseMessage = err;
      }

      console.error(`Responding to client with status: ${responseStatus}, message: ${responseMessage}`);

      const errorResponsePayload = {
        error: {
          message: responseMessage,
          status: responseStatus,
          details_from_worker: errorDetails
        }
      };

      return new Response(JSON.stringify(errorResponsePayload), fixCors({
        status: responseStatus,
        headers: { ...new Headers(fixCors({}).headers), 'Content-Type': 'application/json' }
      }));
    };
    try {
      // --- START of REFACTORED AUTHENTICATION LOGIC ---
      // This new logic checks for the API key in multiple headers to support
      // different client implementations.

      let apiKey;
      const authHeader = request.headers.get("Authorization");

      // Priority 1: Check for a standard "Bearer <token>" in the Authorization header.
      if (authHeader?.startsWith("Bearer ")) {
        apiKey = authHeader.substring(7);
        console.log("Found API key in 'Authorization: Bearer' header.");
      }
      // Priority 2: If no Bearer token, check for the key directly in 'x-goog-api-key'.
      // This handles the user's specific client behavior for Gemini requests.
      else {
        apiKey = request.headers.get("x-goog-api-key");
        if(apiKey) {
            console.log("Found API key in 'x-goog-api-key' header.");
        }
      }
      // --- END of REFACTORED AUTHENTICATION LOGIC ---

      const { pathname } = new URL(request.url);

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
const API_CLIENT = "genai-js/0.21.0";
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  // This will now correctly use the apiKey found from either header
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
  if (req.input_text && req.tts_settings) {
    return handleTTSGeneration(req, apiKey);
  }

  const geminiPayload = await transformRequest(req);
  const model = geminiPayload.model || DEFAULT_MODEL;
  delete geminiPayload.model;

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
  "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));
const fieldsMap = {
  stop: "stopSequences", n: "candidateCount", max_tokens: "maxOutputTokens", max_completion_tokens: "maxOutputTokens", temperature: "temperature", top_p: "topP", top_k: "topK", frequency_penalty: "frequencyPenalty", presence_penalty: "presencePenalty",
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
  let systemPromptText = null;
  let conversationMessages = [];

  const isAnthropicFormat = req.system && Array.isArray(req.system) && req.system.length > 0;
  const incomingMessages = req.messages || [];

  if (isAnthropicFormat) {
    console.log("Anthropic format detected.");
    systemPromptText = req.system[0].text;
    conversationMessages = incomingMessages;
  } else {
    console.log("OpenAI format detected.");
    const systemMessage = incomingMessages.find(msg => msg.role === "system");
    if (systemMessage) {
      systemPromptText = systemMessage.content;
      conversationMessages = incomingMessages.filter(msg => msg.role !== "system");
    } else {
      conversationMessages = incomingMessages;
    }
  }

  let geminiSystemInstruction = null;
  if (systemPromptText) {
    geminiSystemInstruction = { parts: [{ text: systemPromptText }] };
  }

  const contents = [];
  for (const item of conversationMessages) {
    const messageCopy = { ...item };
    messageCopy.role = messageCopy.role === "assistant" ? "model" : "user";
    contents.push(await transformMsg(messageCopy));
  }
  
  if (geminiSystemInstruction && contents.length === 0) {
    contents.push({ role: "model", parts: { text: " " } });
  }

  let model = req.model || DEFAULT_MODEL;
  if (model.startsWith("models/")) {
      model = model.substring(7);
  }

  return {
    ...(geminiSystemInstruction && { system_instruction: geminiSystemInstruction }),
    contents,
    safetySettings,
    generationConfig: transformConfig(req),
    model,
  };
};

const generateChatcmplId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = {
  "STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter",
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