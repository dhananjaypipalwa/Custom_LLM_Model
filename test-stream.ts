import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

const emerge = createOpenAI({
  apiKey: process.env.EMERGE_API_KEY || "emergegpt-secure-token",
  baseURL: "http://165.73.253.125:8000/v1",
  compatibility: "strict",
});

async function run() {
  const stream = await streamText({
    model: emerge.chat("emergegpt"),
    messages: [{ role: "user", content: "Stream a short welcome line." }],
    maxTokens: 60,
  });

  for await (const chunk of stream.textStream) process.stdout.write(chunk);
  console.log("\n[DONE]");
}
run().catch(console.error);
