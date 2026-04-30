import random
from pathlib import Path

RNG = random.Random(42)
out_dir = Path("samples")


def sample_with_reuse(items, n, rng=RNG):
    """
    Return n items quickly, using each unique item once before reusing any.

    Some template pools are intentionally small.  For example, the MC section has
    70 possible unique lines, but the script asks for 75 test examples and 350
    train examples.  Sampling with reuse avoids an infinite uniqueness loop while
    still giving maximal variety before repeats.
    """
    items = list(items)
    if not items:
        return []

    out = []
    while len(out) < n:
        batch = items[:]
        rng.shuffle(batch)
        out.extend(batch[: n - len(out)])
    return out


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")

# =========================
# BASIC POOLS (different per split)
# =========================
def get_pools(split):
    if split == "train":
        return {
            "names": ["Sarah","Tom","Emily","Jake","Anna","Chris","Nina","Mark","Lisa","David"],
            "objects": ["book","keys","phone","wallet","hat","cup","ball","jacket","toy","laptop"],
            "places": ["kitchen","office","park","garage","library","school","beach","forest"],
            "colors": ["red","blue","green","yellow","black","white"],
            "foods": ["apple","sandwich","pizza","rice","soup","cookie"],
            "drinks": ["water","coffee","tea","juice"],
            "animals": ["cat","dog","rabbit","horse","bird","fish"],
            "comparisons": [
                ("heavier object", "book", "hat", "book"),
                ("brighter color", "yellow", "black", "yellow"),
                ("larger place", "school", "office", "school"),
                ("smaller object", "cup", "jacket", "cup"),
                ("better drink for waking up", "coffee", "water", "coffee"),
                ("common pet", "dog", "fish", "dog"),
                ("food usually eaten hot", "soup", "cookie", "soup"),
            ],
            "statements": [
                ("Sarah put the book in the kitchen", "True"),
                ("Tom drank a jacket", "False"),
                ("Emily saw a red cup", "True"),
                ("A laptop is a food", "False"),
                ("A bird is an animal", "True"),
                ("A garage is a drink", "False"),
                ("Rice is a food", "True"),
                ("A phone is a place", "False"),
            ],
        }
    elif split == "val":
        return {
            "names": ["Iris","Noah","Elena","Miles","Chloe"],
            "objects": ["map","scarf","camera","ring","glove"],
            "places": ["attic","lobby","studio","clinic","market"],
            "colors": ["teal","maroon","silver","gold"],
            "foods": ["muffin","noodles","pear","taco"],
            "drinks": ["lemonade","cocoa","smoothie"],
            "animals": ["fox","goose","goat","otter"],
            "comparisons": [
                ("warmer drink", "cocoa", "smoothie", "cocoa"),
                ("brighter color", "gold", "maroon", "gold"),
                ("larger object", "scarf", "ring", "scarf"),
                ("smaller food", "pear", "taco", "pear"),
                ("place for appointments", "clinic", "attic", "clinic"),
                ("object for pictures", "camera", "glove", "camera"),
                ("animal with feathers", "goose", "goat", "goose"),
            ],
            "statements": [
                ("Iris carried a map", "True"),
                ("Noah wore lemonade", "False"),
                ("Elena visited the clinic", "True"),
                ("A glove is a drink", "False"),
                ("A pear is a food", "True"),
                ("A studio is an animal", "False"),
                ("Gold is a color", "True"),
                ("A ring is a place", "False"),
            ],
        }
    else:
        return {
            "names": ["Bea","Caleb","Dina","Ethan","Freya"],
            "objects": ["badge","rope","helmet","violin","blanket"],
            "places": ["courtyard","bakery","workshop","pier","harbor"],
            "colors": ["indigo","cream","tan","magenta"],
            "foods": ["waffle","peach","dumpling","bagel"],
            "drinks": ["espresso","cider","milkshake"],
            "animals": ["raccoon","llama","seal","parrot"],
            "comparisons": [
                ("warmer drink", "espresso", "milkshake", "espresso"),
                ("brighter color", "magenta", "indigo", "magenta"),
                ("larger object", "blanket", "badge", "blanket"),
                ("smaller food", "peach", "bagel", "peach"),
                ("place near boats", "harbor", "bakery", "harbor"),
                ("object for music", "violin", "helmet", "violin"),
                ("animal with feathers", "parrot", "seal", "parrot"),
            ],
            "statements": [
                ("Bea found a badge", "True"),
                ("Caleb drank a rope", "False"),
                ("Dina walked to the pier", "True"),
                ("A helmet is a food", "False"),
                ("A waffle is a food", "True"),
                ("A workshop is a drink", "False"),
                ("Magenta is a color", "True"),
                ("A violin is a place", "False"),
            ],
        }

# =========================
# MULTIPLE CHOICE
# =========================
mc_templates = [
    "Pick the {attr}: {a} or {b}? Answer in one word: {ans}",
    "Between {a} and {b}, the {attr} is which? Answer in one word: {ans}",
    "Choose the {attr}: {a} vs {b}. Answer in one word: {ans}",
    "Identify the {attr}: {a} or {b}? Answer in one word: {ans}",
    "Out of {a} and {b}, select the {attr}. Answer in one word: {ans}",
]

def generate_mc(pools, n):
    lines = []
    for attr, a, b, ans in pools["comparisons"]:
        for template in mc_templates:
            lines.append(template.format(attr=attr, a=a, b=b, ans=ans))
            lines.append(template.format(attr=attr, a=b, b=a, ans=ans))
    return sample_with_reuse(lines, n)

# =========================
# TRUE / FALSE
# =========================
tf_templates = [
    "True or false: {stmt}? Answer in one word: {ans}",
    "{stmt} — True or false? Answer in one word: {ans}",
    "Answer True or False in one word: {stmt}. {ans}",
    "Is this statement true or false: {stmt}? Answer in one word: {ans}",
]

def generate_tf(pools, n):
    lines = [
        template.format(stmt=stmt, ans=ans)
        for stmt, ans in pools["statements"]
        for template in tf_templates
    ]
    return sample_with_reuse(lines, n)

# =========================
# CONTEXT (LONG, DISTRACTORS)
# =========================
def generate_context(pools, n):
    lines = []
    choice = RNG.choice
    randint = RNG.randint

    for _ in range(n):
        name = choice(pools["names"])
        t = randint(0, 5)

        if t == 0:
            obj = choice(pools["objects"])
            loc = choice(pools["places"])
            distract = choice(pools["objects"])
            line = f"{name} moved a {distract} before placing a {obj} in the {loc}. Answer in one word: where is the {obj}? {loc}"

        elif t == 1:
            obj = choice(pools["objects"])
            wrong = choice(pools["objects"])
            line = f"{name} saw a {wrong}, ignored it, and picked up a {obj}. Answer in one word: what did {name} pick up? {obj}"

        elif t == 2:
            color = choice(pools["colors"])
            wrong = choice(pools["colors"])
            obj = choice(pools["objects"])
            line = f"{name} noticed a {wrong} {obj} first, but later grabbed the {color} one. Answer in one word: what color was the {obj}? {color}"

        elif t == 3:
            animal = choice(pools["animals"])
            loc = choice(pools["places"])
            wrong = choice(pools["places"])
            line = f"The {animal} walked past the {wrong} and settled in the {loc}. Answer in one word: where is the {animal}? {loc}"

        elif t == 4:
            food = choice(pools["foods"])
            wrong = choice(pools["foods"])
            line = f"{name} thought about eating {wrong}, but chose {food} instead. Answer in one word: what did {name} eat? {food}"

        else:
            drink = choice(pools["drinks"])
            wrong = choice(pools["drinks"])
            line = f"{name} almost picked {wrong}, but ended up drinking {drink}. Answer in one word: what did {name} drink? {drink}"

        lines.append(line)

    return lines

def build_datasets():
    train_pools = get_pools("train")
    val_pools = get_pools("val")
    test_pools = get_pools("test")

    test = generate_mc(test_pools, 75) + generate_tf(test_pools, 75) + generate_context(test_pools, 50)
    train = generate_mc(train_pools, 350) + generate_tf(train_pools, 350) + generate_context(train_pools, 300)
    val = generate_mc(val_pools, 75) + generate_tf(val_pools, 75) + generate_context(val_pools, 50)

    RNG.shuffle(test)
    RNG.shuffle(train)
    RNG.shuffle(val)
    return train, val, test


def main():
    train, val, test = build_datasets()

    write_lines(out_dir / "questions_train.txt", train)
    write_lines(out_dir / "questions_val.txt", val)
    write_lines(out_dir / "questions_test.txt", test)

    write_lines(
        out_dir / "questions_all.txt",
        ["### TRAIN", *train, "", "### VAL", *val, "", "### TEST", *test],
    )

    print("Done.")


if __name__ == "__main__":
    main()
