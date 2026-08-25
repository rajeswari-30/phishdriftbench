"""Unit tests for the P1/P2/P3 feature-provenance taxonomy (provenance/taxonomy.py)."""
import pandas as pd
import pytest

from phishdriftbench.provenance import taxonomy


def test_classify_partitions_by_provenance():
    names = ["url_length", "domain_age_days", "tranco_rank", "totally_made_up_feature"]
    audit = taxonomy.classify(names)
    assert audit.p1 == ["url_length"]
    assert audit.p2 == ["domain_age_days"]
    assert audit.p3 == ["tranco_rank"]
    assert audit.unknown == ["totally_made_up_feature"]


def test_has_lookup_dependency_true_when_p2_or_p3_present():
    assert taxonomy.classify(["domain_age_days"]).has_lookup_dependency is True
    assert taxonomy.classify(["tranco_rank"]).has_lookup_dependency is True


def test_has_lookup_dependency_false_for_p1_only():
    assert taxonomy.classify(["url_length", "num_dots"]).has_lookup_dependency is False


def test_ablate_keeps_only_requested_provenance_columns():
    df = pd.DataFrame({
        "url_length": [1, 2],
        "domain_age_days": [10, 20],
        "tranco_rank": [100, 200],
    })
    p1_only = taxonomy.ablate(df, keep=taxonomy.Provenance.P1_STATIC_LEXICAL)
    assert list(p1_only.columns) == ["url_length"]

    p1_p2 = taxonomy.ablate(df, keep=(taxonomy.Provenance.P1_STATIC_LEXICAL, taxonomy.Provenance.P2_LOOKUP_AT_INFERENCE))
    assert set(p1_p2.columns) == {"url_length", "domain_age_days"}


def test_p1_features_match_lexical_field_names():
    from phishdriftbench.features.lexical import LexicalFeatures

    assert taxonomy.P1_FEATURES == set(LexicalFeatures.field_names())


@pytest.mark.parametrize("method", ["whois_lookup", "dns_lookup", "ssl_lookup"])
def test_live_lookup_refuses_without_confirmation_flag(method):
    """Every live-lookup method must refuse to run unless the caller explicitly
    confirms it understands this contacts external infrastructure -- this is a
    safety gate, not an implementation detail, so it must never be silently
    bypassable via a default argument."""
    client = taxonomy.LiveLookupClient()
    with pytest.raises(RuntimeError):
        getattr(client, method)("example.com")
