from nontag_pipeline.data import PUBMED_CLASSES, CORA_CLASSES
from nontag_pipeline.narratives import TOPIC_MAPS, STYLE_TEMPLATES, map_proportions_to_themes


def test_topic_map_covers_all_pubmed_classes():
    mapping = TOPIC_MAPS["pubmed"]
    for cls in PUBMED_CLASSES:
        assert cls in mapping, f"Missing pubmed class in topic map: {cls}"


def test_topic_map_covers_all_cora_classes():
    mapping = TOPIC_MAPS["cora"]
    for cls in CORA_CLASSES:
        assert cls in mapping, f"Missing cora class in topic map: {cls}"


def test_topics_are_mutually_distinct():
    # Separability is the property that drives accuracy — no class may share a topic.
    for dataset, mapping in TOPIC_MAPS.items():
        topics = list(mapping.values())
        assert len(topics) == len(set(topics)), f"Duplicate topic in {dataset} map"


def test_topic_map_is_independent_of_style():
    # The whole point: topics do not depend on style, so map_proportions_to_themes
    # takes no style argument and is identical regardless of which style runs.
    props = {"Diabetes_Mellitus_Type_1": 1.0}
    assert map_proportions_to_themes(props, "pubmed") == [("war", 1.0)]


def test_map_proportions_sorted_descending():
    props = {"Diabetes_Mellitus_Type_1": 0.7, "Diabetes_Mellitus_Experimental": 0.3}
    result = map_proportions_to_themes(props, "pubmed")
    assert result[0][1] >= result[1][1]


def test_map_proportions_dominant_theme_first():
    props = {"Diabetes_Mellitus_Type_1": 0.7, "Diabetes_Mellitus_Experimental": 0.3}
    result = map_proportions_to_themes(props, "pubmed")
    assert result[0][0] == "war"
    assert result[1][0] == "nature"


def test_map_proportions_returns_correct_length():
    props = {
        "Diabetes_Mellitus_Type_1": 0.5,
        "Diabetes_Mellitus_Experimental": 0.3,
        "Diabetes_Mellitus_Type_2": 0.2,
    }
    result = map_proportions_to_themes(props, "pubmed")
    assert len(result) == 3


def test_map_proportions_single_class():
    props = {"Diabetes_Mellitus_Type_2": 1.0}
    result = map_proportions_to_themes(props, "pubmed")
    assert result == [("love", 1.0)]


def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(ValueError, match="TOPIC_MAPS"):
        map_proportions_to_themes({"x": 1.0}, "imdb")
