import json
import os
import re

# === CONFIG ===
input_file = "formatted_dataset.txt"            # your original .txt dataset
output_file = "sectionwise_dataset.jsonl"       # output section-level jsonl

section_map = {
    "education": "Extract education history from this section:",
    "experiences": "Extract work experience from this section:",
    "projects": "Extract project details from this section:",
    "skills": "Extract skills from this section:",
    "achievements_and_awards": "Extract achievements from this section:"
}

def extract_blocks(text):
    """Extract all <|startoftext|> ... <|endoftext|> blocks"""
    pattern = r"<\|startoftext\|>(.*?)<\|endoftext\|>"
    return re.findall(pattern, text, re.DOTALL)

def parse_output_block(block):
    """Extract the JSON part from Output: ..."""
    try:
        output_part = block.split("Output:", 1)[1].strip()
        return json.loads(output_part)
    except:
        return None

def generate_sectionwise_examples(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        content = infile.read()

    blocks = extract_blocks(content)
    print(f"📦 Found {len(blocks)} examples in formatted_dataset.txt")

    count = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for block in blocks:
            parsed = parse_output_block(block)
            if not parsed:
                continue
            full_output = parsed.get("full_output", {})
            for section, instruction in section_map.items():
                section_data = full_output.get(section)
                if section_data:
                    prompt = f"{instruction}\n{json.dumps(section_data, ensure_ascii=False)}"
                    output_obj = {section: section_data}
                    json.dump({"input": prompt, "output": output_obj}, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    count += 1

    print(f"✅ Done. Wrote {count} section-level samples to {output_path}")

if __name__ == "__main__":
    generate_sectionwise_examples(input_file, output_file)
