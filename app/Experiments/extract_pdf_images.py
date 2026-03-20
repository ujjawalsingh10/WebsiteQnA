"""
extract_pdf_images.py
---------------------
Quick script to extract all images from your crawled PDFs.
Run this, look at what comes out, then decide image pipeline strategy.

Usage:
    python extract_pdf_images.py --pdfs ./output/pdfs --out ./output/pdf_images
"""

import argparse
import json
import sys
from pathlib import Path

def extract_images_from_pdf(pdf_path: Path, out_dir: Path) -> list[dict]:
    """Extract all images from a single PDF. Returns list of image records."""
    import fitz  # pymupdf

    records = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"  ERROR opening {pdf_path.name}: {e}")
        return records

    img_count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                img_ext    = base_image["ext"]          # jpg, png, etc.
                width      = base_image["width"]
                height     = base_image["height"]
                colorspace = base_image["colorspace"]

                # Skip tiny images (icons, bullets, decorative)
                if width < 200 or height < 200:
                    continue
                if len(img_bytes) < 2048:  # under 2KB
                    continue

                # Save image
                img_filename = f"{pdf_path.stem}_p{page_num+1}_img{img_index+1}.{img_ext}"
                img_path = out_dir / img_filename
                img_path.write_bytes(img_bytes)

                records.append({
                    "pdf":        pdf_path.name,
                    "page":       page_num + 1,
                    "img_index":  img_index + 1,
                    "file":       img_filename,
                    "width":      width,
                    "height":     height,
                    "size_kb":    round(len(img_bytes) / 1024, 1),
                    "format":     img_ext,
                    "colorspace": colorspace,
                })
                img_count += 1

            except Exception as e:
                print(f"  Could not extract image xref={xref}: {e}")

    page_count = len(doc)
    doc.close()
    print(f"  {pdf_path.name}: {img_count} images extracted from {page_count} pages")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", default="./output/pdfs",
                        help="Directory containing downloaded PDFs")
    parser.add_argument("--out",  default="./output/pdf_images",
                        help="Output directory for extracted images")
    args = parser.parse_args()

    pdfs_dir = Path(args.pdfs)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdfs_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdfs_dir}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDFs — extracting images...\n")

    all_records = []
    for pdf_path in pdf_files:
        records = extract_images_from_pdf(pdf_path, out_dir)
        all_records.extend(records)

    # Save summary JSON
    summary_path = out_dir / "extraction_summary.json"
    summary_path.write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Print summary table
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  PDFs processed : {len(pdf_files)}")
    print(f"  Images found   : {len(all_records)}")
    print(f"  Output folder  : {out_dir}")
    print(f"  Summary JSON   : {summary_path}")
    print(f"{'='*60}\n")

    if all_records:
        print("Sample images found:")
        for r in all_records[:10]:
            print(f"  [{r['pdf']}] page {r['page']} — "
                  f"{r['width']}x{r['height']}px  {r['size_kb']}KB  .{r['format']}")
        if len(all_records) > 10:
            print(f"  ... and {len(all_records)-10} more")

    print(f"Done ")

if __name__ == "__main__":
    main()