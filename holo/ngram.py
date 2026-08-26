"""N-gram profiles: a trie/frequency-table's job done by one vector.

Each character trigram becomes a single phasor vector by binding
position-permuted letter codewords:

    g(c1 c2 c3) = rho^2(V(c1)) * rho(V(c2)) * V(c3)

Bundling every trigram of a text gives a fixed-size profile whose cosine
similarity approximates trigram-histogram overlap — the classic HDC
language-identification setup (Joshi et al.): one ~10k-d vector per
language replaces an explicit n-gram count table.
"""

import numpy as np

from .fhrr import FHRR, ItemMemory, Permutation


class NGramEncoder:
    def __init__(self, space, n=3, perm_seed=1):
        self.space = space
        self.n = n
        self.letters = ItemMemory(space, "letters")
        self.rho = Permutation(space, seed=perm_seed)

    def profile(self, text):
        text = "".join(c for c in text.lower() if c.isalpha() or c == " ")
        p = self.space.zeros()
        for i in range(len(text) - self.n + 1):
            g = None
            for j, c in enumerate(text[i:i + self.n]):
                v = self.rho(self.letters.get(c), power=self.n - 1 - j)
                g = v if g is None else FHRR.bind(g, v)
            p += g
        return p


TRAIN = {
    "english": ("the weather turned cold last week and the children stayed "
                "inside reading books by the window. a good meal with friends "
                "is one of the simple pleasures of life. the train arrives at "
                "the station every morning just before eight o'clock. she "
                "walked through the garden and noticed the flowers had begun "
                "to bloom."),
    "spanish": ("el tiempo se puso frio la semana pasada y los ninos se "
                "quedaron dentro leyendo libros junto a la ventana. una buena "
                "comida con amigos es uno de los placeres sencillos de la "
                "vida. el tren llega a la estacion cada manana justo antes de "
                "las ocho. camino por el jardin y noto que las flores habian "
                "empezado a florecer."),
    "german":  ("das wetter wurde letzte woche kalt und die kinder blieben "
                "drinnen und lasen buecher am fenster. ein gutes essen mit "
                "freunden ist eine der einfachen freuden des lebens. der zug "
                "kommt jeden morgen kurz vor acht am bahnhof an. sie ging "
                "durch den garten und bemerkte dass die blumen zu bluehen "
                "begonnen hatten."),
    "french":  ("le temps est devenu froid la semaine derniere et les enfants "
                "sont restes a l'interieur a lire des livres pres de la "
                "fenetre. un bon repas entre amis est l'un des plaisirs "
                "simples de la vie. le train arrive a la gare chaque matin "
                "juste avant huit heures. elle a marche dans le jardin et a "
                "remarque que les fleurs commencaient a fleurir."),
}

TEST = [
    ("english", "my brother brought fresh bread from the bakery this morning"),
    ("english", "we should leave early because the roads will be busy today"),
    ("spanish", "mi hermano trajo pan fresco de la panaderia esta manana"),
    ("spanish", "debemos salir temprano porque las calles estaran llenas hoy"),
    ("german",  "mein bruder brachte heute morgen frisches brot vom baecker"),
    ("german",  "wir sollten frueh losfahren weil die strassen heute voll sind"),
    ("french",  "mon frere a apporte du pain frais de la boulangerie ce matin"),
    ("french",  "nous devrions partir tot car les routes seront chargees"),
]


def demo(dim=4096, seed=0):
    print(f"== N-gram profiles: language ID in one vector each (d={dim}) ==")
    space = FHRR(dim, seed=seed)
    enc = NGramEncoder(space, n=3)
    profiles = {lang: enc.profile(text) for lang, text in TRAIN.items()}
    correct = 0
    for true_lang, sentence in TEST:
        scores = {lang: space.cos(enc.profile(sentence), p)
                  for lang, p in profiles.items()}
        guess = max(scores, key=scores.get)
        correct += guess == true_lang
        shown = ", ".join(f"{l[:2]}={s:.2f}" for l, s in scores.items())
        mark = "ok " if guess == true_lang else "ERR"
        print(f"  [{mark}] {true_lang:<8} -> {guess:<8} ({shown})")
    print(f"  accuracy: {correct}/{len(TEST)} "
          f"(4 language 'tables', each a single {dim}-d complex vector)")
    print()
