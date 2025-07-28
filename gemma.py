from google import genai
from google.genai import types
from dotenv import load_dotenv
import asyncio
from prompts import get_prompt

def gemma(prompt):
    load_dotenv()
    client = genai.Client(api_key = key)
    response = client.models.generate_content(
        model="gemma-3-12b-it",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0
        ),
    )
    return response.text

class Court_Detector():
    async def predict(self, prompt):
        loop = asyncio.get_running_loop()
        benign_setup = loop.run_in_executor(None, gemma, get_prompt(prompt_name = "defense_prompt", user_input = prompt))
        adversarial_setup = loop.run_in_executor(None, gemma, get_prompt(prompt_name = "prosecution_prompt", user_input = prompt))

        # Runs the formulation of the benign and adversarial prompts in parrallel to reduce latency
        benign, adversarial = await asyncio.gather(benign_setup, adversarial_setup)
        
        judgement = await loop.run_in_executor(None, gemma, get_prompt(prompt_name = "judge_prompt", user_input = prompt, benign = benign, adversarial = adversarial))
        verdict = await loop.run_in_executor(None, gemma, get_prompt(prompt_name = "verdict_prompt", verdict_info = judgement))
        return {"benign":benign, "adversarial":adversarial, "judgement":judgement, "verdict":verdict.lower()}

# Serves as comparison to the Court Detector
class Direct_Detector():
    def predict(self, prompt):
        direct_thought = gemma(get_prompt(prompt_name = "direct_prompt", user_input = prompt))
        verdict = gemma(get_prompt(prompt_name = "verdict_prompt", verdict_info = direct_thought))
        return {"direct_throught":direct_thought, "verdict":verdict.lower()}