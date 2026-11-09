"""Part D: database-backed refund tools and bounded ReAct loop."""
from pathlib import Path
import pandas as pd
import json,sqlite3,time,uuid,csv
from datetime import datetime,timezone
import mock_llm
from policy_rag import PolicyRAG
ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/"data/instamart.db"; OUT=ROOT/"outputs"; STATE=ROOT/"state"
SCENARIO_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
rag=PolicyRAG()
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row=con.execute("select order_id,status,order_amount,delivery_minutes from orders where order_id=?",(order_id,)).fetchone()
    return {"error":"order not found"} if not row else dict(zip(["order_id","status","order_amount","delivery_minutes"],row))
def search_policy(query): return {"results":rag.retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {"refund_inr":round(amount,2)}
def _ledger(tool,order_id,refund_inr):
    OUT.mkdir(exist_ok=True); path=OUT/"refund_ledger.jsonl"
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"tool":tool,"order_id":order_id,"refund_inr":refund_inr})+"\n")
    return {"refund_inr":refund_inr}
def issue_refund(order_id,refund_inr): return _ledger("issue_refund",order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger("escalate_to_human",order_id,refund_inr)
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve an order's status, amount and delivery time using its order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique ID of the order."
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents and return the most relevant policy chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The policy question or search keywords."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount based on the order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number",
                    "description": "The total order amount in INR."
                },
                "delivery_minutes": {
                    "type": ["number", "null"],
                    "description": "The delivery time in minutes."
                }
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an approved refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique ID of the order."
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The approved refund amount in INR."
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request that requires human support approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique ID of the order."
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The refund amount requiring human approval in INR."
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

def _continue_agent(
    run_id,
    scenario_id,
    messages,
    tokens,
    steps,
    executed,
    llm_calls,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after=None
):
    start = time.monotonic()
    checkpoint_path = STATE / f"{run_id}.json"

    def fallback(status, summary):
        return {
            "status": status,
            "final_answer": summary,
            "steps": steps,
            "tools_executed": executed,
            "llm_calls": llm_calls,
            "tokens_used": tokens
        }

    def checkpoint(step):
        obj = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens,
            "step_number": step,
            "tools_executed": executed,
            "steps": steps,
            "llm_calls": llm_calls
        }

        checkpoint_path.write_text(
            json.dumps(obj, default=str),
            encoding="utf-8"
        )

    # max_iterations is the total allowed model calls,
    # including calls made before a crash.
    while llm_calls < max_iterations:

        if time.monotonic() - start >= time_limit_seconds:
            return fallback(
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Time limit reached. Hand off to a human agent."
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens += response.usage.total_tokens

        if tokens > token_limit:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Token limit exceeded. Hand off to a human agent."
            )

        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""

            return {
                "status": content.split(":", 1)[0].strip(),
                "final_answer": content,
                "steps": steps,
                "tools_executed": executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

        tc = msg.tool_calls[0]
        args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": args
            }]
        })

        observation = TOOL_MAP[tc.name](**args)
        executed.append(tc.name)

        steps.append({
            "tool": tc.name,
            "arguments": args,
            "observation": observation
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(observation)
        })

        # Save the checkpoint AFTER the tool observation is appended.
        checkpoint(len(steps))

        # Simulate a crash only after the checkpoint is safely saved.
        if crash_after is not None and len(executed) == crash_after:
            raise RuntimeError("Simulated crash after checkpoint")

    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Maximum iterations reached. Hand off to a human agent."
    )


def run_agent(
    user_message,
    scenario_id="demo",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None
):
    run_id = run_id or str(uuid.uuid4())
    STATE.mkdir(exist_ok=True)

    messages = [
        {
            "role": "system",
            "content": "AGENT: Use tools to verify orders and follow refund policy."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    return _continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        tokens=0,
        steps=[],
        executed=[],
        llm_calls=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after=crash_after
    )


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    checkpoint_path = STATE / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    saved = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )

    print(f"Resuming run: {run_id}")
    print(f"Previously executed tools: {saved['tools_executed']}")

    return _continue_agent(
        run_id=saved["run_id"],
        scenario_id=saved["scenario_id"],
        messages=saved["messages"],
        tokens=saved["tokens_used"],
        steps=saved["steps"],
        executed=saved["tools_executed"],
        llm_calls=saved["llm_calls"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )

def get_refund_amount(result):

    refund_amount = 0.0

    for step in result["steps"]:

        observation = step["observation"]

        if "refund_inr" in observation:

            refund_amount = float(
                observation["refund_inr"]
            )

    return round(refund_amount, 2)


# --------------------------------------------------
# 7. RUN ALL SEVEN SCENARIOS
# --------------------------------------------------

def run_all_scenarios():

    if not SCENARIO_FILE.exists():

        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIO_FILE}"
        )

    # Read CSV
    with SCENARIO_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        scenarios = list(reader)
        columns = reader.fieldnames or []

    print("\nScenario CSV columns:", columns)

    # Identify scenario ID column
    id_column = next(
        (
            col for col in
            ["scenario_id", "scenario", "id"]
            if col in columns
        ),
        None
    )

    # Identify user message column
    message_column = next(
        (
            col for col in
            [
                "user_message",
                "message",
                "prompt",
                "user_prompt",
                "user_request",
                "request"
            ]
            if col in columns
        ),
        None
    )

    if id_column is None or message_column is None:

        raise ValueError(
            "Could not identify scenario ID or user message column. "
            f"Available columns: {columns}"
        )

    # Find expected refund column if present
    expected_column = next(
        (
            col for col in
            ["expected_refund_inr", "expected_refund"]
            if col in columns
        ),
        None
    )

    OUT.mkdir(exist_ok=True)

    output_file = OUT / "agent_runs.jsonl"

    # Start a fresh JSONL file for this seven-scenario run
    if output_file.exists():
        output_file.unlink()

    results = []

    for row in scenarios:

        scenario_id = str(row[id_column]).strip()
        user_message = str(row[message_column]).strip()

        print("\n" + "=" * 55)
        print("Running scenario:", scenario_id)

        result = run_agent(
            user_message=user_message,
            scenario_id=scenario_id
        )

        refund_amount = get_refund_amount(result)

        # Validate refund against expected value, when available
        if expected_column:

            expected_value = str(row[expected_column]).strip()

            if expected_value:

                expected_refund = float(expected_value)

                if abs(refund_amount - expected_refund) > 0.01:

                    print(
                        "WARNING: Refund mismatch. "
                        f"Expected {expected_refund:.2f}, "
                        f"got {refund_amount:.2f}"
                    )

        # Record scenario result
        record = {
            "scenario_id": scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        # Append one JSON line per scenario
        with output_file.open(
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(record, default=str) + "\n"
            )

        results.append(record)

        print("Status:", result["status"])
        print("Final answer:", result["final_answer"])
        print("Refund amount:", refund_amount)
        print("Model calls:", result["llm_calls"])
        print("Tokens used:", result["tokens_used"])

    if len(results) != 7:

        print(
            f"WARNING: Expected 7 scenarios, "
            f"but found {len(results)}."
        )

    print("\nJSONL output saved to:", output_file)

    return results


# --------------------------------------------------
# 8. DISPLAY SCENARIO RESULTS
# --------------------------------------------------

def display_scenario_results(results):

    print("\n")
    print("=" * 85)
    print("                 REACT AGENT SCENARIO RESULTS")
    print("=" * 85)

    print(
        f"{'Scenario':<12}"
        f"{'Status':<28}"
        f"{'Refund (INR)':>15}"
        f"{'Calls':>10}"
        f"{'Tokens':>12}"
    )

    print("-" * 85)

    for result in results:

        refund_amount = get_refund_amount(result)

        print(
            f"{result['scenario_id']:<12}"
            f"{result['status']:<28}"
            f"{refund_amount:>15.2f}"
            f"{result['llm_calls']:>10}"
            f"{result['tokens_used']:>12}"
        )

    print("=" * 85)


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("========== TOOL SPECIFICATION ==========")

    for tool in TOOLS_SPEC:

        print("\nTool:", tool["name"])
        print("Description:", tool["description"])
        print("Parameters:", tool["parameters"])

    print("\n========== TOOL MAP ==========")

    for name, function in TOOL_MAP.items():

        print(name, "->", function.__name__)

    # Validate five tool definitions and mappings
    assert len(TOOLS_SPEC) == 5
    assert len(TOOL_MAP) == 5

    assert (
        set(tool["name"] for tool in TOOLS_SPEC)
        == set(TOOL_MAP.keys())
    )

    print("\nAll five tool definitions and mappings passed!")

    

    # Run all seven scenarios
    results = run_all_scenarios()

    # Display final results
    display_scenario_results(results)




    SCENARIO_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"

    scenarios = pd.read_csv(SCENARIO_FILE)

    s1_message = scenarios.loc[
        scenarios["scenario_id"] == "S1",
        "message"
    ].iloc[0]

    # DEMO 1: Token limit
    token_demo = run_agent(
        user_message=s1_message,
        scenario_id="S1_TOKEN_DEMO",
        token_limit=200
    )

    print("\n========== TOKEN LIMIT DEMO ==========")
    print("Status:", token_demo["status"])
    print("LLM calls:", token_demo["llm_calls"])
    print("Tokens used:", token_demo["tokens_used"])
    print("Final answer:", token_demo["final_answer"])

    # DEMO 2: Timeout
    timeout_demo = run_agent(
        user_message=s1_message,
        scenario_id="S1_TIMEOUT_DEMO",
        time_limit_seconds=0
    )

    print("\n========== TIMEOUT DEMO ==========")
    print("Status:", timeout_demo["status"])
    print("LLM calls:", timeout_demo["llm_calls"])
    print("Tokens used:", timeout_demo["tokens_used"])
    print("Final answer:", timeout_demo["final_answer"])

    
# DEMO: Crash and resume S1

resume_run_id = "S1_RESUME_DEMO"

print("\n========== CRASH AND RESUME DEMO ==========")

# First part: simulate a crash after the second tool.
try:
    run_agent(
        user_message=s1_message,
        scenario_id="S1",
        run_id=resume_run_id,
        crash_after=2
    )

except RuntimeError as e:
    print("Simulated crash:", e)

# Read the checkpoint saved before the crash.
crash_checkpoint = json.loads(
    (STATE / f"{resume_run_id}.json").read_text(
        encoding="utf-8"
    )
)

print("Tools executed before crash:")
print(crash_checkpoint["tools_executed"])

# Second part: resume from the saved checkpoint.
resumed_result = resume_agent(resume_run_id)

print("\nTools executed across both parts:")
print(resumed_result["tools_executed"])

print("Final status:", resumed_result["status"])
print("Final answer:", resumed_result["final_answer"])
print("LLM calls:", resumed_result["llm_calls"])
print("Tokens used:", resumed_result["tokens_used"])


