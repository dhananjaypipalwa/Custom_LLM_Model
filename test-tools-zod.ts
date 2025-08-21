// test-tools-zod.ts
// Run: EMERGE_BASE_URL=http://165.73.253.125:8000 EMERGE_API_TOKEN=emergegpt-secure-token OPENWEATHER_API_KEY=YOUR_KEY npx tsx test-tools-zod.ts

import "dotenv/config";
import fetch from "node-fetch";

/** ====== CONFIG ====== */
const BASE_URL = process.env.EMERGE_BASE_URL ?? "http://165.73.253.125:8000";
const API_TOKEN = process.env.EMERGE_API_TOKEN ?? "emergegpt-secure-token";

/** ====== Types (subset of OpenAI style) ====== */
type Role = "system" | "user" | "assistant" | "tool";
type Message = { role: Role; content: string };
type ToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};
type ChatChoice =
  | { index: number; finish_reason: "tool_calls"; message: { role: "assistant"; tool_calls: ToolCall[] } }
  | { index: number; finish_reason: "stop"; message: { role: "assistant"; content: string } };

type ChatResp = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatChoice[];
  usage: Record<string, unknown>;
};

/** ====== Weather tool (real fetch, no hardcoded data) ====== */
type WeatherResult = {
  city: string;
  units: "F" | "C";
  temp: number;
  conditions: string;
};

function extractCityFromText(t: string): string | undefined {
  // very simple heuristic: "... in <City> ...", grabs the word(s) after "in "
  const m = t.match(/\bin\s+([A-Za-z][\w\s\-']{1,40})\b/i);
  if (m) return m[1].trim();
  // fallback: if text looks like "Weather: <City>"
  const m2 = t.match(/\bweather\s*[:\-]\s*([A-Za-z][\w\s\-']{1,40})\b/i);
  if (m2) return m2[1].trim();
  return undefined;
}

async function getWeather(args: { city?: string; units?: "F" | "C"; userText?: string } = {}): Promise<WeatherResult> {
  const apiKey = process.env.OPENWEATHER_API_KEY;
  if (!apiKey) throw new Error("Missing OPENWEATHER_API_KEY");

  //let city = (args.city || extractCityFromText(args.userText || "") || "").trim();
  let city = (args.city || "").trim(); // <-- only trust model
  //if (!city) throw new Error("City not provided and not detected from user text");
  if (!city) throw new Error("City not provided by model");

  //const units: "F" | "C" = args.units ?? "F";
  const units = args.units as "F" | "C"; // <-- only trust model
  if (units !== "F" && units !== "C") {
    throw new Error('Units not provided by model (expected "F" or "C")');
  }
  const owUnits = units === "C" ? "metric" : "imperial";

  const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&units=${owUnits}&appid=${apiKey}`;
  const r = await fetch(url);
  if (!r.ok) {
    const txt = await r.text().catch(() => "");
    throw new Error(`weather fetch failed: ${r.status} ${txt}`);
  }
  const j: any = await r.json();
  return {
    city,
    units,
    temp: Math.round(j.main?.temp ?? 0),
    conditions: j.weather?.[0]?.main ?? "Unknown",
  };
}

/** ====== Helpers ====== */
async function postChat(body: any): Promise<ChatResp> {
  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as ChatResp;
}

/** ====== Main (non-streaming roundtrip) ====== */
async function main() {
  const question = process.argv.slice(2).join(" ").trim() || "What's the weather in Delhi?";
  const userMsg: Message = { role: "user", content: question };

  // Round 1: model asks for tool
  const first = await postChat({
    model: "emergegpt",
    messages: [userMsg],
    tools: [
      {
        type: "function",
        function: {
          name: "getWeather",
          parameters: {
            type: "object",
            properties: {
              city: { type: "string" },
              units: { type: "string", enum: ["F", "C"] },
            },
            required: ["city", "units"], // <-- enforce model to provide both
          },
        },
      },
    ],
    tool_choice: "required",
  });

  const choice1 = first.choices[0];
  if (choice1.finish_reason !== "tool_calls") {
    throw new Error(`Expected tool_calls, got finish_reason=${(choice1 as any).finish_reason}`);
  }

  const toolCalls = (choice1 as Extract<ChatChoice, { finish_reason: "tool_calls" }>).message.tool_calls;
  if (!toolCalls?.length) throw new Error("No tool_calls returned");

  // Execute tools locally (real fetch)
  const outputs: any[] = [];
  for (const tc of toolCalls) {
    const name = tc.function.name;
    const raw = tc.function.arguments ?? "{}";

    let args: any = {};
    try {
      args = JSON.parse(raw);
    } catch {
      console.warn("Could not parse tool args JSON:", raw);
    }

    // DEBUG: show exactly what the model sent
    console.log("Tool call from model:", { name, raw, parsed: args });

    if (name === "getWeather") {
      //const result = await getWeather({ ...args, userText: userMsg.content });
      const result = await getWeather(args);
      outputs.push({ id: tc.id, name, input: args, output: result });
    } else {
      outputs.push({ id: tc.id, name, input: args, output: { error: `Unknown tool ${name}` } });
    }
  }

  // Print raw tool results (no prefixes)
  console.log(JSON.stringify(outputs, null, 2));

  // Round 2: send tool result back to model
  const second = await postChat({
    model: "emergegpt",
    messages: [
      userMsg,
      {
        role: "tool",
        content: JSON.stringify(outputs[0].output),
      } as Message,
    ],
  });

  const choice2 = second.choices[0];
  if (choice2.finish_reason !== "stop") {
    throw new Error(`Expected final text, got finish_reason=${(choice2 as any).finish_reason}`);
  }

  const finalText = (choice2 as Extract<ChatChoice, { finish_reason: "stop" }>).message.content;
  // Print final answer only (no prefixes)
  console.log(finalText);
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
