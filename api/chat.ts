import type { IncomingMessage, ServerResponse } from "node:http";
import {
  MAX_CONTEXT_MESSAGES,
  MODEL_ATTEMPTS,
  PRIMARY_MODEL,
  getGeminiClient,
  parseGeminiError,
  sanitizeMessages,
} from "./_gemini.ts";

type RequestWithBody = IncomingMessage & { body?: unknown };
type ResponseWithFlush = ServerResponse & { flushHeaders?: () => void };

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req: RequestWithBody): Promise<unknown> {
  if (req.body !== undefined) {
    if (typeof req.body === "string") {
      try {
        return JSON.parse(req.body);
      } catch {
        return null;
      }
    }
    return req.body;
  }

  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    const maxBytes = 1024 * 1024;

    req.on("data", (chunk: Buffer) => {
      totalBytes += chunk.length;
      if (totalBytes > maxBytes) {
        reject(new Error("Request body too large."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });

    req.on("end", () => {
      if (chunks.length === 0) {
        resolve(null);
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch {
        resolve(null);
      }
    });

    req.on("error", reject);
  });
}

export default async function handler(
  req: RequestWithBody,
  res: ResponseWithFlush
): Promise<void> {
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.setHeader("Allow", "POST, OPTIONS");
    res.end();
    return;
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST, OPTIONS");
    sendJson(res, 405, { error: "Method not allowed. Use POST." });
    return;
  }

  const body = (await readJsonBody(req)) as { messages?: unknown } | null;
  if (!body || !Array.isArray(body.messages) || body.messages.length === 0) {
    sendJson(res, 400, { error: "A non-empty messages array is required." });
    return;
  }

  const sanitizedMessages = sanitizeMessages(body.messages);
  if (
    sanitizedMessages.length === 0 ||
    sanitizedMessages[sanitizedMessages.length - 1].role !== "user"
  ) {
    sendJson(res, 400, {
      error: "Conversation must end with a non-empty user message.",
    });
    return;
  }

  let clientAborted = false;
  let activeAbortController: AbortController | null = null;

  res.on("close", () => {
    if (!res.writableEnded) {
      clientAborted = true;
      activeAbortController?.abort();
    }
  });

  req.socket?.setNoDelay?.(true);

  try {
    const ai = getGeminiClient();

    res.statusCode = 200;
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();
    res.write(": open\n\n");

    const recentMessages = sanitizedMessages.slice(-MAX_CONTEXT_MESSAGES);
    const contents = recentMessages.map((msg) => ({
      role: msg.role,
      parts: [{ text: msg.text }],
    }));

    const systemInstruction =
      "You are a fast, helpful, and concise AI assistant. Give direct, clear answers without unnecessary filler.";

    let lastError: unknown = null;
    let completed = false;

    for (let i = 0; i < MODEL_ATTEMPTS.length; i++) {
      if (clientAborted) return;
      const attempt = MODEL_ATTEMPTS[i];
      const controller = new AbortController();
      activeAbortController = controller;

      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, attempt.firstTokenTimeoutMs);

      try {
        const stream = await ai.models.generateContentStream({
          model: attempt.model,
          contents,
          config: {
            abortSignal: controller.signal,
            thinkingConfig: { thinkingLevel: attempt.thinkingLevel },
            systemInstruction,
          },
        });

        const iterator = stream[Symbol.asyncIterator]();
        let firstText = "";

        while (true) {
          const step = await iterator.next();
          if (step.done) break;
          const text = step.value.text ?? "";
          if (text) {
            firstText = text;
            break;
          }
        }

        clearTimeout(timer);

        if (!firstText) {
          throw new Error("Model returned an empty response.");
        }

        if (clientAborted) return;

        res.write(`data: ${JSON.stringify({ text: firstText })}\n\n`);

        while (true) {
          if (clientAborted) break;
          const step = await iterator.next();
          if (step.done) break;
          const text = step.value.text ?? "";
          if (text) {
            res.write(`data: ${JSON.stringify({ text })}\n\n`);
          }
        }

        if (!clientAborted) {
          res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
          res.end();
        }

        completed = true;
        break;
      } catch (err) {
        clearTimeout(timer);
        if (clientAborted) return;

        if (timedOut) {
          lastError = new Error(
            "Gemini took too long to respond. Please try again."
          );
          if (i < MODEL_ATTEMPTS.length - 1) continue;
          throw lastError;
        }

        lastError = err;
        const parsed = parseGeminiError(err);
        if (!parsed.shouldFallback || i === MODEL_ATTEMPTS.length - 1) {
          throw err;
        }
        if (
          i === 0 &&
          MODEL_ATTEMPTS[1]?.model === PRIMARY_MODEL &&
          (parsed.status === 503 || parsed.status === 404)
        ) {
          i = 1;
        }
      } finally {
        if (activeAbortController === controller) {
          activeAbortController = null;
        }
      }
    }

    if (!completed && !clientAborted) {
      throw lastError ?? new Error("Failed to generate a response from Gemini.");
    }
  } catch (err) {
    const { status, message } = parseGeminiError(err);
    if (!res.headersSent) {
      sendJson(res, status, { error: message });
    } else if (!clientAborted) {
      res.write(`data: ${JSON.stringify({ error: message })}\n\n`);
      res.end();
    }
  }
}
