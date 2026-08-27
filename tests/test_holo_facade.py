"""holo is the implementation and SDK surface; hdc is the shim.

After the physical migration, these tests pin three things: the
charter-named facade modules re-export the implementation objects, the
hdc compatibility shims resolve to the SAME objects (so pre-migration
callers keep working), and no facade module declares a name it does not
actually export.
"""

import hdc
import holo


def test_hdc_shim_mirrors_holo():
    missing = [n for n in hdc.__all__ if not hasattr(holo, n)]
    assert missing == []
    assert hdc.__version__ == holo.__version__
    assert hdc.FHRR is holo.FHRR
    assert hdc.ORStrokeScene is holo.ORStrokeScene


def test_hdc_module_shims_resolve_to_holo_objects():
    from hdc import accel, fhrr, orset, phase, render
    assert fhrr.FHRR is holo.FHRR
    assert phase.quantize is holo.storage.quantize
    assert render.render_orthographic is holo.render_orthographic
    assert orset.ORStore is holo.ORStore
    assert accel.spectral_bundle is holo.backend.spectral_bundle
    from hdc.ngram import TEST, TRAIN
    from holo.ngram import TEST as HTEST
    from holo.ngram import TRAIN as HTRAIN
    assert TRAIN is HTRAIN and TEST is HTEST


def test_facade_names_are_the_implementation_objects():
    from holo import ngram, phase
    assert holo.core.FHRR is holo.FHRR
    assert holo.structures.HoloMap is holo.HoloMap
    assert holo.encode.GaussianSplatField is holo.GaussianSplatField
    assert holo.scene.ColorSplatField is holo.ColorSplatField
    assert holo.query.RecordSpace is holo.RecordSpace
    assert holo.render.render_orthographic is holo.render_orthographic
    assert holo.fit.HoloRegressor is holo.HoloRegressor
    assert holo.sync.ORStrokeScene is holo.ORStrokeScene
    assert holo.storage.quantize is phase.quantize
    assert ngram.NGramEncoder is holo.NGramEncoder


def test_every_facade_module_exports_what_it_declares():
    for mod in (holo.core, holo.encode, holo.structures, holo.scene,
                holo.query, holo.render, holo.fit, holo.sync,
                holo.storage, holo.backend):
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert missing == [], f"{mod.__name__}: {missing}"


def test_storage_facade_carries_all_three_codecs():
    # regression: pack_complex/pack_polar were reachable only via
    # holo.phase, not the holo.storage facade (caught by the peer
    # session mid-measurement)
    from holo import phase, storage
    assert storage.pack is phase.pack
    assert storage.pack_complex is phase.pack_complex
    assert storage.pack_polar is phase.pack_polar
    assert storage.unpack is phase.unpack
