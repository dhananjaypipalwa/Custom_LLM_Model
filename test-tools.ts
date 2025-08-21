import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

const emerge = createOpenAI({
  apiKey: process.env.EMERGE_API_KEY || "emergegpt-secure-token",
  baseURL: "http://165.73.253.125:8000/v1",
  compatibility: "strict",
});

async function run() {
  const result = await generateText({
    model: emerge.chat("emergegpt"),
    messages: [{ role: "user", content: "What’s the weather in Delhi?" }],
    tools: {
      getWeather: {
        description: "Get current weather for a city",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" }
          },
          required: ["city"]
        }
      }
    }
  });

  console.log(JSON.stringify(result, null, 2));
}
run().catch(console.error);
