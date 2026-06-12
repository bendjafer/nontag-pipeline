# run_textualize_full.py
"""Step 4 driver: textualize ALL nodes and save the pseudo-TAG to outputs/."""
from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.textualize import generate_node_text
from nontag_pipeline.io import save_pseudo_tag


def main() -> None:
    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        config.DATASET, seed=config.SEED, root=config.DATA_ROOT
    )

    # Build visible_labels ONCE from train + val only — the sole label source
    # passed to generation (test labels absent by construction).
    visible_labels = {
        int(n): int(y[n])
        for n in G.nodes()
        if train_mask[n].item() or val_mask[n].item()
    }

    records: list[dict] = []
    n_total = G.number_of_nodes()
    for i, node_id in enumerate(G.nodes()):
        records.append(
            generate_node_text(
                G, int(node_id), visible_labels, class_names,
                config.DATASET, config.STYLE, config.TARGET_LEN,
            )
        )
        if (i + 1) % 100 == 0 or i + 1 == n_total:
            print(f"{i + 1}/{n_total} nodes textualized", flush=True)

    n_no_signal = sum(1 for r in records if r["status"] == "no_signal")
    pt_path, json_path = save_pseudo_tag(
        G, y, train_mask, val_mask, test_mask, class_names, records,
        output_dir=config.OUTPUT_DIR, dataset=config.DATASET, style=config.STYLE,
    )
    print(f"Done: {n_total - n_no_signal} nodes textualized, {n_no_signal} no_signal.")
    print(f"Saved: {pt_path}\n       {json_path}")


if __name__ == "__main__":
    main()
