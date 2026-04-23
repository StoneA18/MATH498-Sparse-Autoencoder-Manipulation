

tone = "optimistic"

with open("../samples") as f:
    prompts = f.read_lines()

for sub_promptin prompts:
master_prompt = f"""
You are tasked with generating responses in a given tone. Please write a plain text response to the following prompt in a tone that is {tone}.

{sub_prompt}
"""