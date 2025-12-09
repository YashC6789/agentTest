from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import os, json

# Local Imports
from src.agent.setup import build_tool_agent
from src.config.prompts import SYSTEM_PROMPT_TEXT
from langchain_core.messages import SystemMessage, HumanMessage

from toolbench import api_json_to_openai_json, standardize

# Adjust if your tools live elsewhere
TOOL_ROOT_DIR = os.path.join("src", "data", "toolbench", "tools")


def load_all_tool_schemas(
    tool_root_dir: str,
    max_apis_per_tool: int = 99,  # effectively "all"
):
    tool_schemas = []

    for category in os.listdir(tool_root_dir):
        category_path = os.path.join(tool_root_dir, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if not fname.endswith(".json"):
                continue

            json_path = os.path.join(category_path, fname)
            with open(json_path, "r", encoding="utf-8") as f:
                tool_def = json.load(f)

            if "tool_name" not in tool_def:
                print(f"[WARN] Skipping {json_path}: no 'tool_name' field")
                continue

            standard_tool_name = standardize(tool_def["tool_name"])

            for raw_api in tool_def.get("api_list", [])[:max_apis_per_tool]:
                if "api_name" in raw_api:
                    api_json = raw_api
                else:
                    api_json = {
                        "api_name": raw_api.get("name", ""),
                        "api_description": raw_api.get("description", ""),
                        "required_parameters": raw_api.get("required_parameters", []),
                        "optional_parameters": raw_api.get("optional_parameters", []),
                        "category_name": tool_def.get("category_name", category),
                    }

                if not api_json["api_name"]:
                    print(f"[WARN] Skipping API with no name in file: {json_path}")
                    continue

                tool_schema, category_name, pure_api_name = api_json_to_openai_json(
                    api_json, standard_tool_name
                )
                tool_schemas.append(tool_schema)

    return tool_schemas


def load_filtered_tool_schemas(tool_root_dir, predicate):
    all_tools = load_all_tool_schemas(tool_root_dir)
    return [t for t in all_tools if predicate(t)]


def load_some_tool_schemas(
    tool_root_dir: str,
    max_tools: int = 5,
    max_apis_per_tool: int = 3,
):
    """
    Walks the ToolBench tools dir, reads JSONs, and uses api_json_to_openai_json
    to build OpenAI-style tool schemas usable with a tool-calling agent.

    Handles both:
      - ToolCommander-style flattened APIs with 'api_name', 'api_description'
      - Raw ToolBench-style APIs with 'name', 'description'
    """
    tool_schemas = []

    for category in os.listdir(tool_root_dir):
        category_path = os.path.join(tool_root_dir, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if not fname.endswith(".json"):
                continue

            json_path = os.path.join(category_path, fname)
            with open(json_path, "r", encoding="utf-8") as f:
                tool_def = json.load(f)

            # ToolBench tools typically have "tool_name" and "api_list"
            if "tool_name" not in tool_def:
                print(f"[WARN] Skipping {json_path}: no 'tool_name' field")
                continue

            standard_tool_name = standardize(tool_def["tool_name"])

            for raw_api in tool_def.get("api_list", [])[:max_apis_per_tool]:
                # --- Normalize to what api_json_to_openai_json expects ---

                # If it's already flattened (has api_name), just use it
                if "api_name" in raw_api:
                    api_json = raw_api
                else:
                    # Raw ToolBench format: use 'name' and 'description'
                    api_json = {
                        "api_name": raw_api.get("name", ""),
                        "api_description": raw_api.get("description", ""),
                        "required_parameters": raw_api.get("required_parameters", []),
                        "optional_parameters": raw_api.get("optional_parameters", []),
                        # api_json_to_openai_json returns category_name, so if we have it, keep it
                        "category_name": tool_def.get("category_name", category),
                    }

                # Skip if we still couldn't get a name
                if not api_json["api_name"]:
                    print(f"[WARN] Skipping API with no name in file: {json_path}")
                    continue

                tool_schema, category_name, pure_api_name = api_json_to_openai_json(
                    api_json, standard_tool_name
                )
                tool_schemas.append(tool_schema)

                if len(tool_schemas) >= max_tools:
                    return tool_schemas

    return tool_schemas


# ------------------------ TEST CASES ------------------------ #
"""
Now the tests are aligned with *actual* ToolBench domains:

- Amazon scrapers (amazon_data_scrapper / alkari_amazon_web_scraper)
- Taobao / Alibaba image search
- Translate API
- Oregon lottery tools (keno / mega_millions / megabucks)
- Etsy product API
- Baby pig pictures (fun / random content)
"""

TEST_CASES = [
    {
        "name": "Amazon product search",
        "prompt": (
            "Using your Amazon product search tools, find some options for wireless "
            "noise-cancelling headphones under $300."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "amazon",           # e.g. amazon_data_scrapper, alkari_amazon_web_scraper
            "search_results",   # get_amazon_search_results_for_...
        ],
    },
    {
        "name": "Amazon product details by ASIN",
        "prompt": (
            "Using your Amazon scraping APIs, look up detailed information, including "
            "price and ratings, for the product with ASIN B0B1234567."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "amazon",               # amazon-related tools
            "product_details",      # get_amazon_product_details_for_...
        ],
    },
    {
        "name": "Taobao / Alibaba image search",
        "prompt": (
            "Use your Taobao or Alibaba image search API to find visually similar "
            "products for a red summer dress. Just show me what tool you would call."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "image_search",         # image_search_for_taobao_alibaba1688_for_...
            "taobao",
            "alibaba",
        ],
    },
    {
        "name": "Translation API",
        "prompt": (
            "Using your translation API, translate the sentence "
            "'Hello, how are you?' into Spanish."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "translate",            # translator_for_translate
            "translator",
        ],
    },
    {
        "name": "Oregon lottery mega millions",
        "prompt": (
            "Using your Oregon lottery tools, get the most recent Mega Millions "
            "results and summarize them for me."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "mega_millions",        # mega_millions_for_oregon_lottery
            "oregon_lottery",
        ],
    },
    {
        "name": "Etsy product list",
        "prompt": (
            "Use your Etsy product API to list some products related to 'handmade mug'."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "etsy",                 # product_list_product_list_post_for_etsy_product
            "product_list",
        ],
    },
    {
        "name": "Baby pig pictures tool (fun)",
        "prompt": (
            "You have an API that returns random baby pig pictures. Use it to get "
            "some cute pig pictures."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "baby_pig",             # random_for_baby_pig_pictures
            "random_for_baby_pig",
        ],
    },
    {
        "name": "Simple chit-chat (no ToolBench APIs)",
        "prompt": "Tell me a short joke about programmers.",
        "expect_tool": False,
        "expected_name_substrings": [],
    },
]


def extract_tool_calls(messages):
    """
    Walk through the LangGraph state["messages"] list and collect all tool calls
    made by the agent (from AIMessage.tool_calls).
    """
    all_calls = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            all_calls.extend(tool_calls)
    return all_calls


def evaluate_tool_calls(test, tool_calls):
    """
    Given a test case + tool_calls list, return a dict:
      {
        "passed": bool,
        "reason": str,
        "tool_names": [...],
      }
    """
    expect_tool = test["expect_tool"]
    expected_substrings = [s.lower() for s in test["expected_name_substrings"]]

    tool_names = [tc["name"] for tc in tool_calls] if tool_calls else []

    # 1) Expected a tool but got none
    if expect_tool and not tool_calls:
        return {
            "passed": False,
            "reason": "Expected at least one tool call, but got none.",
            "tool_names": tool_names,
        }

    # 2) Did not expect a tool but got some
    if not expect_tool and tool_calls:
        return {
            "passed": False,
            "reason": f"Did not expect a tool call, but got: {tool_names}",
            "tool_names": tool_names,
        }

    # 3) If we expected a specific kind of tool, check for substrings
    if expect_tool and expected_substrings:
        lower_names = [n.lower() for n in tool_names]
        matches = []
        for sub in expected_substrings:
            for name in lower_names:
                if sub in name:
                    matches.append((sub, name))

        if not matches and tool_names:
            return {
                "passed": False,
                "reason": (
                    "Got tool calls, but none matched expected substrings "
                    f"{expected_substrings}. Tool names: {tool_names}"
                ),
                "tool_names": tool_names,
            }

    # If we made it here, behavior matches expectations
    return {
        "passed": True,
        "reason": "Tool usage matched expectations.",
        "tool_names": tool_names,
    }


def run_tool_tests():
    # 1) Load tools from ToolBench
    print(f"[INFO] Loading tools from: {TOOL_ROOT_DIR}")
    tools_exposed = load_all_tool_schemas(TOOL_ROOT_DIR)
    print(f"[INFO] Loaded {len(tools_exposed)} tool schemas.\n")

    # (Optional) comment this out if it's too spammy
    for t in tools_exposed:
        print(f"  - {t['name']}")
    print()

    # 2) Build your existing tool agent using your helper
    agent_executor = build_tool_agent(tools_exposed)

    results = []

    # 3) Run through test cases
    for idx, test in enumerate(TEST_CASES, start=1):
        print(f"================ TEST {idx}: {test['name']} ================")
        print(f"Prompt: {test['prompt']}\n")

        system_content = (
            SYSTEM_PROMPT_TEXT
            + "\n\nYou have access to many external APIs drawn from ToolBench, "
              "including Amazon scrapers, Taobao/Alibaba search, translation, "
              "lottery, and e-commerce APIs. "
              "Call these tools when they clearly apply to the user's request. "
              "For general chit-chat or simple reasoning (like jokes), "
              "answer directly without calling any tools."
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=test["prompt"]),
        ]

        # LangGraph-based agents take a dict, usually {"messages": [...]}
        state = agent_executor.invoke({"messages": messages})

        # state is a dict; the full conversation is state["messages"]
        msgs = state["messages"]
        final_msg = msgs[-1]

        print("Final message:")
        print(final_msg)
        print("\nFinal content:")
        print(repr(final_msg.content))

        # Extract all tool calls from the message history
        tool_calls = extract_tool_calls(msgs)
        if tool_calls:
            print("\nTool calls observed:")
            for tc in tool_calls:
                print(f"  - name: {tc['name']}, args: {tc['args']}")
        else:
            print("\nTool calls observed: NONE")

        eval_result = evaluate_tool_calls(test, tool_calls)
        print(f"\nResult: {'PASS' if eval_result['passed'] else 'FAIL'}")
        print(f"Reason: {eval_result['reason']}")
        print(f"Tool names: {eval_result['tool_names']}")
        print()

        results.append((test["name"], eval_result))

    # 4) Summary
    print("\n==================== SUMMARY ====================")
    passed = sum(1 for _, r in results if r["passed"])
    total = len(results)
    print(f"Passed {passed}/{total} tests.\n")

    for name, r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {name} -> {r['reason']}")


if __name__ == "__main__":
    run_tool_tests()