from dataclasses import dataclass


Occupations = tuple[int, int, int, int]


# TODO kw_only and slots are 3.10
@dataclass(frozen=True)
class Indices:
    indices_closed_act: list[tuple[int, int]]
    indices_closed_secondary: list[tuple[int, int]]
    indices_act_secondary: list[tuple[int, int]]


def form_indices_from_occupations(occupations: Occupations) -> Indices:
    assert len(occupations) == 4
    nocc_a, nvirt_a, nocc_b, nvirt_b = occupations
    assert (nocc_a + nvirt_a) == (nocc_b + nvirt_b)
    norb = nocc_a + nvirt_a
    nelec = nocc_a + nocc_b
    nact = abs(int(nocc_a - nocc_b))
    nclosed = (nelec - nact) // 2
    nsecondary = norb - (nclosed + nact)
    range_closed = list(range(0, nclosed))
    range_act = list(range(nclosed, nclosed + nact))
    range_secondary = list(range(nclosed + nact, nclosed + nact + nsecondary))

    # TODO unused
    # self.indices_rohf = (
    #     self.indices_closed_act + self.indices_closed_secondary + self.indices_act_secondary
    # )
    # self.indices_display_rohf = [(p + 1, q + 1) for (p, q) in self.indices_rohf]
    return Indices(
        indices_closed_act=[(i, t) for i in range_closed for t in range_act],
        indices_closed_secondary=[(i, a) for i in range_closed for a in range_secondary],
        indices_act_secondary=[(t, a) for t in range_act for a in range_secondary],
    )
