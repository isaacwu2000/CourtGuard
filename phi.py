import openai
from dotenv import load_dotenv
import asyncio
from prompts import get_prompt
import weave

def phi(prompt):
    load_dotenv()
    client = openai.OpenAI(
        base_url='https://api.inference.wandb.ai/v1',
        project="isaacwu3/evals-nt" # Replace with your actual project name
    )
    response = client.chat.completions.create(
        model="microsoft/Phi-4-mini-instruct",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

class Court_Detector():
    async def predict(self, prompt):
        loop = asyncio.get_running_loop()
        benign_setup = loop.run_in_executor(None, phi, get_prompt(prompt_name = "defense_prompt", user_input = prompt))
        adversarial_setup = loop.run_in_executor(None, phi, get_prompt(prompt_name = "prosecution_prompt", user_input = prompt))

        # Runs the formulation of the benign and adversarial prompts in parrallel to reduce latency
        benign, adversarial = await asyncio.gather(benign_setup, adversarial_setup)
        
        judgement = await loop.run_in_executor(None, phi, get_prompt(prompt_name = "judge_prompt", user_input = prompt, benign = benign, adversarial = adversarial))
        verdict = await loop.run_in_executor(None, phi, get_prompt(prompt_name = "verdict_prompt", verdict_info = judgement))
        
        return {"benign":benign, "adversarial":adversarial, "judgement":judgement, "verdict":verdict.lower().replace(".","")}

# Serves as comparison to the Court Detector
class Direct_Detector():
    def predict(self, prompt):
        direct_thought = phi(get_prompt(prompt_name = "direct_prompt", user_input = prompt))
        verdict = phi(get_prompt(prompt_name = "verdict_prompt", verdict_info = direct_thought))
        return {"direct_throught":direct_thought, "verdict":verdict.lower().replace(".","")}