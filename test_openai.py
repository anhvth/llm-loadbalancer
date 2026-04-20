
import json
import os
import uuid

from llm_utils import LLM, turn


def assistant_turn(message):
    out = {"role": "assistant", "content": message.content}
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        out["reasoning"] = reasoning
    return out


llm = LLM(8001)
session_id = os.environ.get("LLM_SESSION_ID") or f"openai-demo-{uuid.uuid4()}"
metadata = {
    "user_id": json.dumps(
        {
            "device_id": "test_openai.py",
            "account_uuid": "",
            "session_id": session_id,
        }
    )
}

print(f"Session ID: {session_id}")

msg1 = [{"role": "user", "content": "hi openai"}]
result1 = llm(msg1, metadata=metadata, return_dict=True, cache=False)
ret = result1["message"]
ret_history = assistant_turn(ret)
print('Ret1')
print(ret)
print("Returned completion id:", result1["completion"].id)
print("Returned session id:", getattr(result1["completion"], "session_id", None))
# output = llm('hi openai')
result2 = llm([*msg1, ret_history, turn('u', 'how are you')], metadata=metadata, return_dict=True, cache=False)
ret2 = result2["message"]
print('Ret2\n---')
print(ret2)
print("Returned completion id:", result2["completion"].id)
print("Returned session id:", getattr(result2["completion"], "session_id", None))
print('History\n---')
llm.inspect_history(-1, 10)
