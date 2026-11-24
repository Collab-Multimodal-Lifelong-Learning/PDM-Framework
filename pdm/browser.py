"""Browser observations, actions and episode execution on the public REAL replicas."""

import ast
import base64
import io
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from .common import QWEN_MODEL


PRACTICE_URLS = {
    "opendining": "https://real-opendining.vercel.app",
    "udriver": "https://real-udriver.vercel.app",
}
TREE_LIMIT = 6000
SYSTEM = (
    "Use the screenshot and numbered page elements to complete the user's request. "
    "Return one command inside a fenced action block. Use click('id'), fill('id','text'), "
    "press('id','key'), select_option('id','value'), scroll(x,y), go_back(), or noop(milliseconds). "
    "Check what changed after each command. Finish with send_msg_to_user('answer') "
    "or report_infeasible('reason')."
)


def parse_action(response):
    """Accept one call with literal arguments and return canonical BrowserGym syntax."""
    signatures = {
        "click": (str,), "fill": (str, str), "press": (str, str), "select_option": (str, str),
        "scroll": (int, int), "go_back": (), "noop": (int,),
        "send_msg_to_user": (str,), "report_infeasible": (str,),
    }
    if not isinstance(response, str) or len(response) > 8192:
        raise ValueError("Expected a single browser command")
    command = response.strip()
    fenced = re.fullmatch(r"```(?:action|python)?\s*\n(.*?)\n```", command, re.DOTALL)
    if fenced:
        command = fenced[1].strip()
    try:
        call = ast.parse(command, mode="eval").body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
            raise ValueError
        name = call.func.id
        values = [ast.literal_eval(arg) for arg in call.args]
        if name == "noop" and not values:
            values = [1000]
        if tuple(type(v) for v in values) != signatures[name]:
            raise ValueError
        if name == "noop" and not 0 <= values[0] <= 5000:
            raise ValueError
        if name in {"click", "fill", "press", "select_option"} and not re.fullmatch(r"\w+", values[0]):
            raise ValueError
        if name == "scroll" and any(abs(v) > 10000 for v in values):
            raise ValueError
    except (KeyError, ValueError, SyntaxError, TypeError, RecursionError) as exc:
        raise ValueError("Expected one supported browser command with literal arguments") from exc
    return f"{name}({', '.join(repr(v) for v in values)})"


def make_env(task_id, goal=None, site=None):
    from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
    from agisdk.REAL.browsergym.core.env import BrowserEnv
    from agisdk.REAL.browsergym.core.task import OpenEndedTask
    from agisdk.REAL.browsergym.webclones.base import AbstractWebCloneTask
    from agisdk.REAL.browsergym.webclones.task_config import TASK_INDEX

    if goal is None:
        canonical = task_id if task_id.startswith("v2.") else f"v2.{task_id}"
        if canonical not in TASK_INDEX:
            raise ValueError(f"Unknown v2 benchmark task {task_id!r}")

        class ScoredTask(AbstractWebCloneTask):
            def setup(self, page):
                self.run_id = "0"  # Run locally, including when the SDK has a saved leaderboard ID.
                return super().setup(page)

            def validate(self, page, chat_messages):
                # Score once after the loop, including when the model runs out of steps.
                return 0.0, False, "", {}

        task_class = ScoredTask
        kwargs = {"task_id": canonical, "run_id": "0"}
    else:
        task_class = OpenEndedTask
        kwargs = {"start_url": PRACTICE_URLS[site], "goal": goal}
    actions = HighLevelActionSet(subsets=["bid", "chat", "infeas", "nav"], multiaction=False, strict=True)
    return BrowserEnv(task_entrypoint=task_class, task_kwargs=kwargs, headless=True, slow_mo=0,
                      action_mapping=actions.to_python_code)


def page_view(raw, history, start_url):
    from agisdk.REAL.browsergym.utils.obs import IGNORED_AXTREE_PROPERTIES, flatten_axtree_to_str

    if urlsplit(raw["url"]).netloc != urlsplit(start_url).netloc:
        raise ValueError("Browser left the selected replica")
    tree = flatten_axtree_to_str(raw["axtree_object"], raw["extra_element_properties"],
                                 with_clickable=True, skip_generic=False,
                                 ignored_roles=("LineBreak", "InlineTextBox"),
                                 ignored_properties=(*IGNORED_AXTREE_PROPERTIES, "url"))
    lines = [line.strip() for line in tree.splitlines()
             if not re.fullmatch(r"\s*(?:\[\w+\] )?generic", line)]
    tree = "\n".join(lines)
    last_actions = "\n".join(history[-8:]) or "(none)"
    prompt = (f"Request: {raw['goal']}\nPage: {raw['url']}\nRecent actions:\n{last_actions}\n"
              f"Page elements:\n{tree[:TREE_LIMIT]}")
    if raw["last_action_error"]:
        prompt += "\nPrevious action error: " + raw["last_action_error"][:500]
    jpeg = io.BytesIO()
    Image.fromarray(raw["screenshot"]).convert("RGB").save(jpeg, format="JPEG", quality=80)
    return tree, prompt, jpeg.getvalue()


def practice_state(env):
    """Read the simulator receipt in the same browser session as the action."""
    page = env.context.new_page()
    try:
        page.goto(env.task.start_url + "/finish")
        return json.loads(page.locator("pre").inner_text(timeout=15000))
    finally:
        page.close()


def request_action(endpoint, model, prompt, image):
    encoded = base64.b64encode(image).decode("ascii")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            {"type": "text", "text": prompt},
        ]},
    ]
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 128}
    request = Request(endpoint + "/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def score_official(task, state, answer):
    """Retain SDK script diagnostics so a broken evaluator cannot count as a failed task."""
    reward, _, _, details = task.evaluator.evaluate(state, answer)
    for _, criterion in details["results"]:
        if isinstance(criterion, dict) and ("error" in criterion or criterion.get("return_code", 0) != 0):
            raise RuntimeError(f"SDK scorer error: {criterion}")
    return float(reward), details


def run_episode(task_id, *, endpoint, out, max_steps=25, goal=None, site=None):
    """Run a policy in a fresh browser and record the evidence used for local scoring."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint.rstrip("/")
    official = goal is None
    site = task_id.removeprefix("v2.").split("-")[0] if official else site
    result = dict(task_id=task_id, site=site, status="error", steps=0, end="error", reward=None,
                  start_url=None, answer="", trace=str(out / "trace.jsonl"))
    env = None
    with (out / "trace.jsonl").open("x", encoding="utf-8") as trace:
        try:
            env = make_env(task_id, goal, site)
            raw, _ = env.reset(seed=0)
            result["start_url"] = env.task.url if official else env.task.start_url
            result["goal"] = raw["goal"]
            history = []
            result["end"] = "max_steps"
            for step in range(1, max_steps + 1):
                _, prompt, image = page_view(raw, history, result["start_url"])
                image_name = f"{step:02}.jpg"
                (out / image_name).write_bytes(image)
                reply = request_action(endpoint, QWEN_MODEL, prompt, image)
                row = dict(step=step, url=raw["url"], system=SYSTEM, prompt=prompt, image=image_name,
                           response=reply)
                result["steps"] = step
                try:
                    action = parse_action(reply)
                except ValueError as exc:
                    row["error"] = raw["last_action_error"] = str(exc)
                    trace.write(json.dumps(row) + "\n")
                    trace.flush()
                    continue
                row["action"] = action
                raw, _, done, truncated, _ = env.step(action)
                history.append(action)
                row["error"] = raw["last_action_error"]
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if raw["last_action_error"]:
                    if done or truncated:
                        result["end"] = "action_error"
                        break
                    continue
                if action.startswith(("send_msg_to_user(", "report_infeasible(")):
                    result["answer"] = ast.parse(action, mode="eval").body.args[0].value
                    result["end"] = "answered" if action.startswith("send_msg_to_user(") else "infeasible"
                    break
                if done or truncated:
                    result["end"] = "terminated"
                    break

            _, _, final_image = page_view(raw, history, result["start_url"])
            (out / "final.jpg").write_bytes(final_image)
            state = env.task.get_finish_json(timeout=15000) if official else practice_state(env)
            (out / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            if not isinstance(state.get("finalstate"), dict) or not state["finalstate"]:
                raise ValueError("The simulator returned no final state")
            if official:
                reward, details = score_official(env.task, state, result["answer"])
                (out / "score.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
                result.update(reward=reward, status="success" if reward > 0 else "fail")
            else:
                result["status"] = "completed" if result["end"] == "answered" else "fail"
        except Exception as exc:
            result.update(status="error", end="error", reward=None, error=f"{type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    if result["status"] != "error":
                        result.update(status="error", end="error", reward=None, error=f"Browser close: {exc}")
    return result
