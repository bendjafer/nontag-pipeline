# run_textualize_demo.py
"""Smoke test: textualize 5 test-mask nodes, assert no leakage, save pseudo-TAG."""
import random
from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.textualize import generate_node_text
from nontag_pipeline.io import save_pseudo_tag


def main() -> None:
    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        config.DATASET, seed=config.SEED, root=config.DATA_ROOT
    )

    # Build visible_labels ONCE from train + val only
    visible_labels = {
        int(n): int(y[n])
        for n in G.nodes()
        if train_mask[n].item() or val_mask[n].item()
    }

    # Assert: no test node label is present in visible_labels (leakage check)
    test_nodes = [int(n) for n in G.nodes() if test_mask[n].item()]
    for n in test_nodes:
        assert n not in visible_labels, f"Leakage: test node {n} found in visible_labels"

    # Pick 5 test nodes deterministically
    rng = random.Random(config.SEED)
    sample = rng.sample(test_nodes, 5)

    records: list[dict] = []
    for node_id in sample:
        print(f"\n{'=' * 60}")
        print(f"Node {node_id}  (test, true label: {class_names[int(y[node_id])]})")

        result = generate_node_text(
            G, node_id, visible_labels, class_names,
            config.DATASET, config.STYLE, config.TARGET_LEN,
        )
        records.append(result)

        print(f"Status : {result['status']}")
        if result["status"] == "ok":
            print(f"Themes : {result['themes']}")
            print(f"\nGenerated text:\n{result['text']}")
        else:
            print("No visible-labeled neighbors — skipped LLM call.")

    print(f"\n{'=' * 60}")
    print(f"Leakage assertion passed for all {len(test_nodes)} test nodes.")
    print("Demo complete. Full pipeline run and save are separate steps.")


if __name__ == "__main__":
    main()
