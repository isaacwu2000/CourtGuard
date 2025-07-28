# CourtGuard
CourtGuard is a multiagent LLM-based system that can use local LLMs to detect prompt injections. First, two “attorney models”--a “defense” and “prosecution” model--construct arguments in parallel for why the prompt is and is not a prompt injection respectively. Then, a ‘judge model’ considers both arguments and gives a final verdict on whether the prompt is a prompt injection. This final verdict is then extracted by another model (because Gemma lacks the ability to do structured output) and is presented as the result. 

This repository is a copy of a private one that I used to evaluate CourtGuard on Llmail-Inject, NotInject, and Qualifire and store the results using Weights and Biases.

## Technical notes
This system uses the Gemini API to access Gemma-3-12b-it, the Llama API (Preview) to access Llama-3-8b, and the Weights and Biases Inference API to access Phi-4-mini-instruct. All of these models can be run locally or through other providers instead.

## Setup
First, run
```pip install -r requirements.txt```

To setup Gemma with the Gemini API, set the environmental variable
```GEMINI_API_KEY = your_key_here```

To setup Llama with the Llama API, set the environmental variable
```Llama_API_KEY = your_key_here```

To setup Phi with the W&B API (assuming you have credits), set the environmental variable
```OPENAI_API_KEY = your_wandb_key_here```

Also, make sure to replace the project name in ```phi.py``` with your actual project name.

Finally, when you want to test the Court Detector (CourtGuard) or the Direct Detector, change the file from which the they are imported from to the model you wish to use, and run ```main.py```.
