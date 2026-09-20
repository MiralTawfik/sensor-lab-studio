"""Every term shown in the app has a description."""
import pytest

from core import glossary, registry
from core.glossary_data import ENTRIES
from core.help_ui import MODEL_TERMS


def test_unique_keys_and_fields():
    keys = [e[0] for e in ENTRIES]
    assert len(keys) == len(set(keys))
    for e in ENTRIES:
        assert len(e) == 7 and e[1] and e[3] and e[2] in glossary.CATEGORY_ORDER


def test_model_term_keys_exist():
    for sub, keys in MODEL_TERMS.items():
        for k in keys:
            assert k in glossary.TERMS, (sub, k)


@pytest.mark.parametrize("key", list(registry.MODELS))
def test_every_parameter_has_help(key):
    m = registry.MODELS[key]
    for q in m.params:
        assert q.help, (key, q.key)


@pytest.mark.parametrize("key", list(registry.MODELS))
def test_every_derived_quantity_and_constraint_described(key):
    m = registry.MODELS[key]
    p = {q.key: q.default for q in m.params}
    for label in m.derived(p, *m.x_default):
        assert glossary.short(label), (key, label)
    if m.constraint:
        assert glossary.short(m.constraint.name), m.constraint.name


def test_all_sub_families_have_terms():
    for m in registry.MODELS.values():
        assert m.subfamily in MODEL_TERMS


def test_search():
    assert any(t.key == "sensitivity" for t in glossary.search("slope"))
    assert glossary.search("zzzz-nothing") == []


def test_kpi_has_description():
    from core import branding
    html = branding.kpi("Sensitivity (model)", "1", "mV/g")
    assert "k-desc" in html
