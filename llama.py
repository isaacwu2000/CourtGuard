from llama_api_client import LlamaAPIClient
from dotenv import load_dotenv
import asyncio
from prompts import get_prompt

def llama(prompt):
    load_dotenv()
    client = LlamaAPIClient()
    completion = client.chat.completions.create(
        model="Llama-3.3-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0
    )
    return completion.completion_message.content.text

class Court_Detector():
    async def predict(self, prompt):
        loop = asyncio.get_running_loop()
        benign_setup = loop.run_in_executor(None, llama, get_prompt(prompt_name = "benign_prompt", user_input = prompt))
        adversarial_setup = loop.run_in_executor(None, llama, get_prompt(prompt_name = "adversarial_prompt", user_input = prompt))

        # Runs the formulation of the benign and adversarial prompts in parrallel to reduce latency
        benign, adversarial = await asyncio.gather(benign_setup, adversarial_setup)
        
        judgement = await loop.run_in_executor(None, llama, get_prompt(prompt_name = "judge_prompt", user_input = prompt, benign = benign, adversarial = adversarial))
        verdict = await loop.run_in_executor(None, llama, get_prompt(prompt_name = "verdict_prompt", verdict_info = judgement))
        
        return {"benign":benign, "adversarial":adversarial, "judgement":judgement, "verdict":verdict.lower().replace(".","")}

# Serves as comparison to the Court Detector
class Direct_Detector():
    def predict(self, prompt):
        direct_thought = llama(get_prompt(prompt_name = "direct_prompt", user_input = prompt))
        verdict = llama(get_prompt(prompt_name = "verdict_prompt", verdict_info = direct_thought))
        return {"direct_throught":direct_thought, "verdict":verdict.lower().replace(".","")}