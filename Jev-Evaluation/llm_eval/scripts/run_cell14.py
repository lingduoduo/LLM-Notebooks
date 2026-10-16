"""Run only cell 14 of the notebook via nbclient (skip cells 1-13)."""
import sys, json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJ = Path(__file__).resolve().parents[1]
NB_PATH = PROJ / "notebooks" / "jevbench_intro.ipynb"
NB_OUT = PROJ / "notebooks" / "jevbench_intro_executed.ipynb"


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    target_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            src = c.source
            if "Cell 14 — 4 维横评" in src or "# Cell 14" in src[:60]:
                target_idx = i
                break
    if target_idx is None:
        print("cell 14 not found")
        return 1

    # nbclient has no per-cell execute API, so run cell 14 in a one-cell
    # notebook. Run from the project root: cell 1 (which normally cd's
    # there) is skipped, and cell 14 uses root-relative runs/ paths.
    nb_14 = nbformat.v4.new_notebook()
    nb_14.cells = [nb.cells[target_idx]]
    nb_14.metadata = nb.metadata

    client = NotebookClient(nb_14, kernel_name="python3", timeout=300,
                            resources={"metadata": {"path": str(PROJ)}})
    client.execute()

    print("--- cell 14 outputs ---")
    for cell_out in nb_14.cells[0].get("outputs", []):
        if cell_out.get("output_type") == "stream":
            print(cell_out.get("text", "")[:2000])
        elif cell_out.get("output_type") == "display_data":
            data = cell_out.get("data", {})
            if "image/png" in data:
                imgpath = PROJ / "runs" / "notebook-demo" / "multi" / "charts" / "14_real_compare.png"
                if imgpath.exists():
                    print(f"image saved: {imgpath}")
            if "text/plain" in data:
                print("---text/plain:")
                print(data["text/plain"][:3000])
        elif cell_out.get("output_type") == "error":
            print("ERROR:", cell_out.get("ename"), (cell_out.get("evalue") or "")[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
