"""Collect custom, browser-executed imitation examples on the official REAL replica sites."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


@dataclass(frozen=True)
class PracticeTask:
    episode: str
    domain: str
    goal: str
    pickup: str = ""
    destination: str = ""
    restaurant_index: int = 0
    car: str = "UdriverX"


def restaurant(index):
    ordinal = ("first", "second")[index]
    return PracticeTask(
        f"train-opendining-lunch-{ordinal}", "opendining",
        f"Reserve a table for two at the {ordinal} restaurant in the home page's "
        "list of available restaurants. Choose its first offered time today. "
        "Use phone 4155550146 and email diner@example.com.",
        restaurant_index=index,
    )


def ride(slug, pickup, destination, car):
    return PracticeTask(
        f"train-udriver-{slug}", "udriver",
        f"Book a {car} ride for me now from {pickup} to {destination}. Pay with cash.",
        pickup=pickup, destination=destination, car=car,
    )


TASKS = (
    restaurant(0), restaurant(1),
    ride("market-to-local", "Golden Farmer Market", "Local Edition", "UdriverX"),
    ride("local-to-market", "Local Edition", "Golden Farmer Market", "Comfort"),
)


def bid(tree, pattern):
    match = re.search(r"^\s*\[(\w+)\] " + pattern, tree, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"No browser element matches {pattern!r}")
    return match.group(1)


def click(tree, pattern):
    return f"click({bid(tree, pattern)!r})"


def fill(tree, field, text):
    return f"fill({bid(tree, field)!r}, {text!r})"


def restaurant_actions(task, chosen):
    tree = yield
    cards = re.split(r"^\s*\[\w+\] listitem '', clickable\s*$", tree, flags=re.MULTILINE)[1:]
    card = cards[task.restaurant_index]
    chosen["restaurant"] = re.search(r"link 'View details for ([^']+)'", card).group(1)
    chosen["time"] = re.search(r"link 'Book for ([^']+)'", card).group(1)
    tree = yield click(card, "link 'Book for ")
    field = r"textbox '' value='\+1'"
    tree = yield fill(tree, field, "4155550146")
    tree = yield fill(tree, "textbox 'Email address'", "diner@example.com")
    if "value='diner@example.com'" not in tree:
        raise ValueError("Email address was not accepted by the reservation form")
    tree = yield click(tree, "button 'Complete reservation'")
    for _ in range(3):
        if "StaticText 'Booking Complete'" in tree:
            break
        tree = yield "noop(1000)"
    if not all(f"StaticText '{value}'" in tree for value in
               ("Booking Complete", chosen["restaurant"], chosen["time"], "2 people")):
        raise ValueError("OpenDining did not confirm the requested table and time")
    message = f"Booked a table for two at {chosen['restaurant']} at {chosen['time']}."
    yield f"send_msg_to_user({message!r})"


def ride_actions(task):
    tree = yield
    if not re.search(r"\[\w+\] combobox 'Enter Location'", tree):
        tree = yield "noop(1000)"
    for field, place in (("Enter Location", task.pickup), ("Enter Destination", task.destination)):
        tree = yield fill(tree, f"combobox '{field}'", place)
        tree = yield click(tree, f"option '{re.escape(place)} -")
    tree = yield click(tree, "link 'See prices for your ride'")
    for _ in range(3):
        if "listitem 'UdriverX with " in tree:
            break
        tree = yield "noop(1500)"
    if task.car != "UdriverX":
        tree = yield click(tree, f"listitem '{task.car} with ")
    tree = yield click(tree, r"generic, clickable\n\s*StaticText 'Udriver Credits'")
    tree = yield click(tree, r"listitem '', clickable\n[^\n]*paragraph ''\n\s*StaticText 'Cash'")
    tree = yield click(tree, "button 'CONFIRM'")
    if "StaticText 'Cash'" not in tree:
        raise ValueError("UDriver did not select cash payment")
    tree = yield click(tree, f"button 'Request {task.car}'")
    for _ in range(4):
        if "heading 'Trip Completed'" in tree:
            break
        tree = yield "noop(4000)"
    if "heading 'Trip Completed'" not in tree:
        raise ValueError("The simulator did not complete the ride")
    message = f"Your {task.car} trip from {task.pickup} to {task.destination} is complete."
    yield f"send_msg_to_user({message!r})"


def verify_state(task, state, chosen):
    """Verify the completed task against the replica's /finish state independently of its UI."""
    final = state["finalstate"]
    if task.domain == "opendining":
        bookings = final["booking"]["bookingDetails"]
        for booking in bookings:
            if (booking["restaurant"]["name"] == chosen["restaurant"]
                    and booking["time"] == chosen["time"] and booking["guests"] == "2 people"
                    and re.sub(r"\D", "", booking["tel"]) == "14155550146"):
                return {**chosen, "guests": booking["guests"], "date": booking["date"],
                        "phone": booking["tel"]}
    else:
        # Search boxes can retain an unbooked route. Check the completed receipt itself.
        def trip_records(node):
            if isinstance(node, dict):
                if node.get("status") == "completed":
                    yield node
                else:
                    for value in node.values():
                        yield from trip_records(value)
            elif isinstance(node, list):
                for item in node:
                    yield from trip_records(item)

        for record in trip_records(final):
            if (record.get("pickup", {}).get("name") == task.pickup
                    and record.get("destination", {}).get("name") == task.destination
                    and record.get("car", {}).get("type") == task.car
                    and (record.get("paymentMethod") or {}).get("type") == "cash"):
                return record
    raise ValueError(f"No matching completed booking in the replica state for {task.episode}")


def collect_episode(task, path):
    from realeval.agent import LocalAgent
    from realeval.config import AgentConfig, ModelConfig
    from realeval.tasks import REAL_SITES, RealEvalEnvArgs

    config = AgentConfig(prompt_style="basic", axtree_max_chars=6000, parse_retries=0,
                         system_prompt_extra="Return only one browser action in a fenced action block.")
    agent = LocalAgent(ModelConfig(), config, max_steps=25)
    spec = dict(kind="custom", start_url=REAL_SITES[task.domain], goal=task.goal)
    env = RealEvalEnvArgs(
        task_name=task.episode, task_seed=0, max_steps=25, headless=True, slow_mo=0,
        viewport={"width": 1280, "height": 720}, task_spec=spec,
    ).make_env(agent.action_set.to_python_code, str(path))
    chosen = {}
    teacher = restaurant_actions(task, chosen) if task.domain == "opendining" else ride_actions(task)
    next(teacher)
    rows = []
    try:
        raw, _ = env.reset(seed=0)
        for step in range(1, 26):
            observation = agent.obs_preprocessor(raw)
            if urlsplit(observation["url"]).netloc != urlsplit(spec["start_url"]).netloc:
                raise ValueError("Browser left the practice replica")
            messages = agent._build_messages(observation, step, config.axtree_max_chars)
            tree = observation["axtree_txt"]
            action = teacher.send(tree)
            print(f"{task.episode} {step}: {action}", flush=True)
            target = re.match(r"(?:click|fill)\('(\w+)'", action)
            if target and f"[{target[1]}]" not in tree[:config.axtree_max_chars]:
                raise ValueError(f"Action target missing from the model's current observation: {action}")
            image_name = f"{task.episode}/{step:02}.jpg"
            (path / image_name).write_bytes(observation["screenshot_jpeg"])
            rows.append({
                "episode": task.episode, "domain": task.domain, "split": "train",
                "system": messages[0]["content"],
                "prompt": "\n\n".join(part["text"] for part in messages[1]["content"] if part["type"] == "text"),
                "image": image_name, "action": f"```action\n{action}\n```",
            })
            raw, _, done, truncated, info = env.step(action)
            if raw["last_action_error"]:
                raise ValueError(raw["last_action_error"])
            if done or truncated:
                if not done or not action.startswith("send_msg_to_user("):
                    raise ValueError(f"Episode ended before confirmation ({step} steps)")
                state = info["task_info"]["env_state"]
                evidence = verify_state(task, state, chosen)
                (path / task.episode / "outcome.json").write_text(json.dumps(
                    dict(episode=task.episode, goal=task.goal, url=spec["start_url"], verified=evidence),
                    indent=2, ensure_ascii=False))
                return rows
            # REAL's basic prompt includes only the sequence of canonical action strings.
            agent.history.append({"action": action})
        raise ValueError("Episode exceeded the step budget")
    finally:
        env.close()


def collect(output, tasks=TASKS):
    """Publish only a complete, verified collection. Existing output is preserved."""
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        run = Path(scratch) / "collection"
        run.mkdir()
        with (run / "manifest.jsonl").open("w", encoding="utf8") as manifest:
            for task in tasks:
                (run / task.episode).mkdir()
                rows = collect_episode(task, run)
                for row in rows:
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        run.rename(output)
    print(f"Verified demonstrations: {output / 'manifest.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/web-training-demo"))
    args = parser.parse_args()
    collect(args.output)


if __name__ == "__main__":
    main()
