from nontag_pipeline.data import PUBMED_CLASSES, CORA_CLASSES
from nontag_pipeline.narratives import LABEL_MAPS, map_proportions_to_themes


def test_label_map_covers_all_pubmed_classes():
    mapping = LABEL_MAPS[("pubmed", "poetry")]
    for cls in PUBMED_CLASSES:
        assert cls in mapping, f"Missing pubmed class in poetry map: {cls}"


def test_label_map_covers_all_cora_classes():
    mapping = LABEL_MAPS[("cora", "poetry")]
    for cls in CORA_CLASSES:
        assert cls in mapping, f"Missing cora class in poetry map: {cls}"


def test_map_proportions_sorted_descending():
    props = {"Diabetes_Mellitus_Type_1": 0.7, "Diabetes_Mellitus_Experimental": 0.3}
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert result[0][1] >= result[1][1]


def test_map_proportions_dominant_theme_first():
    props = {"Diabetes_Mellitus_Type_1": 0.7, "Diabetes_Mellitus_Experimental": 0.3}
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert result[0][0] == "the unbroken storm"
    assert result[1][0] == "fire and trial"


def test_map_proportions_returns_correct_length():
    props = {
        "Diabetes_Mellitus_Type_1": 0.5,
        "Diabetes_Mellitus_Experimental": 0.3,
        "Diabetes_Mellitus_Type_2": 0.2,
    }
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert len(result) == 3


def test_map_proportions_single_class():
    props = {"Diabetes_Mellitus_Type_2": 1.0}
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert result == [("slow tide", 1.0)]
