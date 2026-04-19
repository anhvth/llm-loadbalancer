from llm_utils import *

llm = LLM(8001, cache=False)
out = llm([
    turn('u', 'hi')    
])
print(out)