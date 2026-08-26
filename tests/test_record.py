"""Role-filler records and analogical queries."""

from holo import RecordSpace


def test_record_access_and_analogy(space):
    rs = RecordSpace(space)
    usa = rs.encode({"capital": "washington", "currency": "dollar"})
    mex = rs.encode({"capital": "cdmx", "currency": "peso"})
    assert rs.get(usa, "capital")[0] == "washington"
    assert rs.analogy(usa, mex, "dollar")[0] == "peso"
    assert rs.analogy(mex, usa, "cdmx")[0] == "washington"
