"""Real inputs for NLP task-level trials and open-vocabulary detectors.

REAL_TEXT is a fixed natural-English passage bundled with the harness (no
network, no license issues -- written for this project). Tokenization happens
on the machine that has the model libraries, with each model's own tokenizer;
`token_ids_for` tiles/truncates to an exact length so bigbird stays on its
sparse path (needs > 704 tokens; use 1024).

COCO_CLASS_NAMES feeds the open-vocab detectors' task-level trials: feeding
random tokens would ask the model to find gibberish words in a real photo --
near-empty detections, vacuous task-level comparison.
"""

from __future__ import annotations

REAL_TEXT = """
The harbour town wakes slowly in winter. Before first light the fishing boats
knock against their moorings, and the smell of diesel and salt drifts up the
narrow streets to the bakery, where the ovens have been burning since four.
Gulls patrol the seawall in ragged lines, waiting for the first crates to come
ashore. By eight the fog usually lifts, and the water turns from grey to a
pale, glassy green that the painters who summer here never quite believe.

Marta has kept the harbour ledger for thirty-one years. She records each
landing in pencil first, then in ink once the weights are checked: cod, hake,
the occasional box of squid, and lately more of the warm-water species her
father never saw this far north. The ledger is a kind of history book, she
says, though nobody reads it but the insurance men. When her niece suggested a
spreadsheet, Marta laughed and said the sea does not care about spreadsheets,
and neither do the fishermen, and the matter was considered settled.

In spring the town changes tempo. Scaffolding climbs the hotel facades, the
ferry adds a second daily crossing, and the school lets out early on Fridays
so the children can help paint the hulls. The chandlery sells more rope in
April than in all the winter months together. Old rivalries between the north
quay and the south quay revive over the boat races, which have been held every
year since the war, except the year of the storm, which nobody discusses in
front of visitors.

The storm is its own chapter. It arrived on a Tuesday night in October with
almost no warning from the forecasts, and by morning the breakwater had a
forty-metre breach and three boats sat in the market square like toys left by
an enormous child. No one died, which the town attributes to the lighthouse
keeper's stubborn habit of ringing the old bell whenever his barometer
dropped faster than he liked, official instruments notwithstanding. The bell
has since been given a small plaque and an annual polishing.

Summer brings the visitors, and with them a version of the town that the
residents hardly recognise: postcards of boats that no longer sail, ice cream
in flavours no fisherman would touch, and guided walks that pause at the bell
for exactly ninety seconds. The guides tell the storm story well enough,
Marta admits, though they move the storm to November for dramatic effect and
double the size of the waves. She has stopped correcting them. Every ledger,
she says, has two editions: the one you write down and the one people prefer.

By late autumn the scaffolding comes down and the ferry returns to a single
crossing. The painters pack their easels, the gulls reclaim the seawall, and
the bakery goes back to opening an hour later, because the queue is short and
the mornings are long. On the last warm evening the whole town seems to be
out on the quay at once, mending nets that do not strictly need mending,
watching the light go, saying very little. Then winter closes over the harbour
again, patient and grey, and the ledger starts a new page.
""".strip()


# The 80 COCO class names, in canonical order.
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def token_ids_for(tokenizer, target_len, text=None):
    """Tokenize the bundled text to EXACTLY target_len ids (tile if short,
    truncate if long). `tokenizer` is any HF-style callable returning a dict
    with "input_ids" -- run this on the machine that has the model libs, and
    feed the SAME ids to reference and candidate.

        ids = token_ids_for(AutoTokenizer.from_pretrained(...), 1024)
        input_ids = np.array([ids], dtype=np.int64)
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    text = text if text is not None else REAL_TEXT
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("tokenizer produced no ids from the text")
    while len(ids) < target_len:
        ids = ids + ids
    return ids[:target_len]


def openvocab_queries():
    """Text queries for OWL-ViT's task-level trial (list-of-strings form)."""
    return list(COCO_CLASS_NAMES)


def groundingdino_prompt():
    """GroundingDINO's expected prompt format: lowercase, period-separated."""
    return ". ".join(COCO_CLASS_NAMES) + "."
