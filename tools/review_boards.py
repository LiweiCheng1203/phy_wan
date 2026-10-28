"""Build compact visual review boards without changing dataset acceptance."""
from PIL import Image, ImageDraw
from common import ROOT, read_json, write_json

qc = read_json(ROOT / "reports/qc_report.json")
episodes = qc["review_required"]
boards = []
for offset in range(0, len(episodes), 6):
    batch = episodes[offset:offset + 6]
    board = Image.new("RGB", (1280, len(batch) * 264))
    draw = ImageDraw.Draw(board)
    for row, ep in enumerate(batch):
        sheet = Image.open(ROOT / "reports" / f"{ep}_qc.jpg")
        record = next(r for r in qc["episodes"] if r["episode"] == ep)
        draw.text((5, row * 264 + 5), f"{ep} median={record['median_px']:.1f}px coverage={record['coverage']:.3f}", fill="white")
        for column, j in enumerate([0, 2, 5, 7]):
            x, y = j % 2 * 640, j // 2 * 504 + 24
            crop = sheet.crop((x, y, x + 640, y + 480)).resize((320, 240))
            board.paste(crop, (column * 320, row * 264 + 24))
    name = f"review_board_{offset // 6:02}.jpg"
    board.save(ROOT / "reports" / name, quality=95)
    boards.append({"image": name, "episodes": batch})
write_json(ROOT / "reports/review_boards.json", boards)
