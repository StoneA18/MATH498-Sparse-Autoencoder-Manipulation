from openai import OpenAI
import random
import os

client = OpenAI(api_key=os.environ["AGENT_KEY"])

tones = ["positive and optimistic", "negative and pessimistic"]

with open("./samples/prompts.txt") as f:
    prompts = f.readlines()

with open("plus_minus_dataset.txt", 'a') as f:
    for sub_prompt in prompts:
        sub_prompt = sub_prompt.strip()
        if not sub_prompt:
            continue
        tone = random.choice(tones)
        master_prompt = f"""You are tasked with generating responses in a given tone. Please write a plain text response, approximately 1000 words, to the following prompt in a tone that is {tone}:
        {sub_prompt}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": master_prompt}],
        )
        result = response.choices[0].message.content
        f.write(f"PROMPT: {sub_prompt}\nTONE: {tone}\nRESPONSE:\n{result}\n\n---\n\n")