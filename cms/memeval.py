"""Multi-turn agent memory eval (synthetic, seeded).

Each simulated customer has a long multi-session conversation. Early turns state
facts (city, contact preference, order id, diet, device, plan); some facts are
later UPDATED; the rest is distractor small talk and support chatter. At the end
the agent asks questions phrased differently from how the fact was stated.

A strategy returns k memory items; we score:
  hit@1 / hit@3  - the CURRENT value of the fact is in the top 1 / 3
  stale@1        - top-1 contains an OUTDATED value of the same slot
Strategies: last-N window (no retrieval), cosine retrieval, cosine + recency.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .memory import Event, MemoryStore

SLOTS = {
    "city": (["Pune", "Leeds", "Austin", "Lisbon", "Osaka", "Nairobi", "Denver", "Kraków"],
             ["I live in {v} now.", "We just relocated to {v}.", "My home address is in {v}."],
             ["Which city should the courier deliver to?", "Where is the customer based?"]),
    "contact": (["email", "SMS", "WhatsApp", "phone call"],
                ["Please reach me by {v} only.", "Contact preference: {v}.", "I'd rather get updates via {v}."],
                ["How does the customer prefer to be contacted?", "What channel should updates use?"]),
    "order": (["A-4417", "B-9032", "C-1185", "D-7720", "E-3391"],
              ["My order number is {v}.", "The order in question is {v}.", "Reference {v} is the order."],
              ["What is the order number we are discussing?", "Which order reference did they give?"]),
    "diet": (["vegan", "gluten-free", "halal", "nut allergy", "lactose intolerant"],
             ["I'm {v}, keep that in mind for meal kits.", "Dietary note: {v}.", "Remember I am {v}."],
             ["Any dietary restrictions for the meal kit?", "What food requirements does the customer have?"]),
    "device": (["Pixel 9", "iPhone 16", "Galaxy S25", "iPad Air"],
               ["The app crashes on my {v}.", "I'm using a {v}.", "My device is a {v}."],
               ["Which device is the app running on?", "What phone or tablet does the customer use?"]),
    "plan": (["Basic", "Plus", "Premium", "Family"],
             ["I'm on the {v} plan.", "I upgraded to the {v} plan.", "Subscription tier: {v}."],
             ["Which subscription plan is the customer on?", "What tier are they paying for?"]),
}
DISTRACTORS = [
    "Thanks, that helps.", "Can you check the status again?", "The weather has been awful lately.",
    "I'll wait for your reply.", "Is there a discount this month?", "My colleague recommended you.",
    "The delivery driver was very polite last time.", "How long does a refund usually take?",
    "I tried logging out and in again.", "Do you have a referral programme?", "OK, noted.",
    "Your website was slow this morning.", "I might order for a friend too.", "Is support open on Sunday?",
]


@dataclass
class Case:
    customer_id: str
    slot: str
    question: str
    current: str
    outdated: list


def generate(n_customers: int = 200, turns: int = 60, update_p: float = 0.4, seed: int = 0):
    rng = random.Random(seed)
    events, cases = [], []
    for c in range(n_customers):
        cid = f"cust-{c:04d}"
        slots = rng.sample(list(SLOTS), 4)
        plan = []  # (turn_index, slot, value)
        hist: dict[str, list] = {}
        for s in slots:
            vals = rng.sample(SLOTS[s][0], 2)
            t0 = rng.randrange(0, turns // 2)
            plan.append((t0, s, vals[0]))
            hist[s] = [vals[0]]
            if rng.random() < update_p:
                plan.append((rng.randrange(t0 + 5, turns), s, vals[1]))
                hist[s].append(vals[1])
        by_turn = {}
        for t, s, v in plan:
            by_turn.setdefault(t, []).append((s, v))
        ts = 1_700_000_000.0 + c
        for t in range(turns):
            ts += rng.uniform(30, 600)
            items = by_turn.get(t)
            if items:
                for s, v in items:
                    txt = rng.choice(SLOTS[s][1]).format(v=v)
                    events.append(Event(f"{cid}-{t}-{s}", cid, txt, ts, "fact"))
            else:
                events.append(Event(f"{cid}-{t}", cid, rng.choice(DISTRACTORS), ts, "chatter"))
        # later facts sometimes land on same turn order; final truth = last in time
        for s in slots:
            vs = [v for t, ss, v in sorted(plan) if ss == s]
            cases.append(Case(cid, s, rng.choice(SLOTS[s][2]), vs[-1], vs[:-1]))
    return events, cases


def _score(hits_texts: list[str], case: Case, k: int):
    top = hits_texts[:k]
    return any(case.current in t for t in top)


def run(n_customers=200, seed=0, k=3, recency_weight=0.15, window=5, embedder=None) -> dict:
    events, cases = generate(n_customers, seed=seed)
    store = MemoryStore(embedder=embedder, recency_weight=0.0, recency_tau=3 * 3600)
    for e in events:
        store.add(e)
    out = {}
    strategies = {
        f"last-{window}-turns (no retrieval)": lambda c: [e.text for e in reversed(store.recent(c.customer_id, window))],
        "cosine retrieval": lambda c: [e.text for e, _ in store.search(c.customer_id, c.question, k, recency_weight=0)],
        f"cosine + recency (w={recency_weight})": lambda c: [e.text for e, _ in store.search(
            c.customer_id, c.question, k, recency_weight=recency_weight)],
        "cosine top-10 -> newest among near-ties": lambda c: [e.text for e, _ in store.search_fresh(
            c.customer_id, c.question, k)],
    }
    upd = [c for c in cases if c.outdated]
    for name, fn in strategies.items():
        h1 = h3 = st = st_u = 0
        for c in cases:
            r = fn(c)
            h1 += _score(r, c, 1)
            h3 += _score(r, c, 3)
            stale = bool(r) and any(o in r[0] for o in c.outdated) and c.current not in r[0]
            st += stale
            st_u += stale and bool(c.outdated)
        out[name] = {"hit@1": h1 / len(cases), "hit@3": h3 / len(cases),
                     "stale@1_on_updated_facts": st_u / max(1, len(upd))}
    out["_meta"] = {"customers": n_customers, "events": len(events), "questions": len(cases),
                    "updated_facts": len(upd), "seed": seed, "embedder": store.embedder.name}
    return out
