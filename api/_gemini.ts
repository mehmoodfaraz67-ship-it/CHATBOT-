import { GoogleGenAI, ThinkingLevel } from "@google/genai";

export const PRIMARY_MODEL = "gemini-3.1-flash-lite";
export const FALLBACK_MODEL = "gemini-4-hydrogen";
export const MAX_CONTEXT_MESSAGES = 16;

export interface ChatMessageInput {
  role: "user" | "model";
  text: string;
}

let cachedClient: { apiKey: string; client: GoogleGenAI } | null = null;

export function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error(
      "Gemini API key is not configured. Please add GEMINI_API_KEY in Settings > Secrets or Vercel Environment Variables."
    );
  }
  if (cachedClient && cachedClient.apiKey === apiKey) {
    return cachedClient.client;
  }
  const client = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
  cachedClient = { apiKey, client };
  return client;
}

function extractReadableErrorMessage(raw: string): string {
  let current = raw.trim();
  for (let i = 0; i < 3; i++) {
    if (!current.startsWith("{")) break;
    try {
      const parsed = JSON.parse(current) as Record<string, unknown>;
      if (parsed.error && typeof parsed.error === "object") {
        const inner = parsed.error as Record<string, unknown>;
        if (typeof inner.message === "string") {
          current = inner.message.trim();
          continue;
        }
      }
      if (typeof parsed.message === "string") {
        current = parsed.message.trim();
        continue;
      }
    } catch {
      break;
    }
    break;
  }
  return current;
}

export function parseGeminiError(err: unknown): {
  status: number;
  message: string;
  shouldFallback: boolean;
} {
  const rawMessage =
    err instanceof Error ? err.message : "An unexpected error occurred.";
  const readable = extractReadableErrorMessage(rawMessage);

  if (
    rawMessage.includes("API_KEY_INVALID") ||
    rawMessage.includes("PERMISSION_DENIED") ||
    rawMessage.toLowerCase().includes("gemini api key is not configured")
  ) {
    return {
      status: 403,
      message:
        "Invalid or missing Gemini API key. Please check GEMINI_API_KEY in Settings > Secrets or Vercel Environment Variables.",
      shouldFallback: false,
    };
  }

  if (rawMessage.includes("RESOURCE_EXHAUSTED") || rawMessage.includes("429")) {
    return {
      status: 429,
      message:
        "Rate limit or quota exceeded. Please wait a moment and try again.",
      shouldFallback: false,
    };
  }

  if (
    rawMessage.includes("UNAVAILABLE") ||
    rawMessage.includes("503") ||
    rawMessage.includes("high demand") ||
    rawMessage.includes("fetch failed") ||
    rawMessage.includes("ECONNRESET") ||
    rawMessage.includes("ETIMEDOUT") ||
    rawMessage.includes("socket")
  ) {
    return {
      status: 503,
      message:
        "Gemini is temporarily busy. Please try again in a moment.",
      shouldFallback: true,
    };
  }

  if (rawMessage.includes("NOT_FOUND") || rawMessage.includes("404")) {
    return {
      status: 404,
      message: "Requested Gemini model is unavailable.",
      shouldFallback: true,
    };
  }

  return {
    status: 500,
    message: readable || "Failed to generate a response from Gemini.",
    shouldFallback:
      rawMessage.includes("500") || rawMessage.includes("INTERNAL"),
  };
}

export function sanitizeMessages(rawMessages: unknown): ChatMessageInput[] {
  if (!Array.isArray(rawMessages)) return [];
  return rawMessages
    .filter((item): item is ChatMessageInput => {
      if (!item || typeof item !== "object") return false;
      const candidate = item as Record<string, unknown>;
      return (
        (candidate.role === "user" || candidate.role === "model") &&
        typeof candidate.text === "string" &&
        candidate.text.trim().length > 0
      );
    })
    .map((item) => ({
      role: item.role,
      text: item.text.trim(),
    }));
}

export const MODEL_ATTEMPTS = [
  {
    model: PRIMARY_MODEL,
    thinkingLevel: ThinkingLevel.MINIMAL,
    firstTokenTimeoutMs: 7500,
  },
  {
    model: PRIMARY_MODEL,
    thinkingLevel: ThinkingLevel.MINIMAL,
    firstTokenTimeoutMs: 9000,
  },
  {
    model: FALLBACK_MODEL,
    thinkingLevel: ThinkingLevel.LOW,
    firstTokenTimeoutMs: 11000,
  },
] as const;
