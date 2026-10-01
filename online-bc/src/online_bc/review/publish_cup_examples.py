"""Publish ten real held-cup→release training segments on the video page."""

import json
import subprocess
import html
from pathlib import Path
import imageio_ffmpeg
from PIL import Image, ImageDraw

ROOT = Path("/home/aiden/Desktop/lab/robot/pan-skill-models-20261001/coffee-online-bc")
TRACK = Path("/home/aiden/Desktop/lab/robot/robocasa-docker/tracking")
SOURCE = TRACK / "media/preparecoffee-three-models-20261001"
OUT = TRACK / "media/pi05-cup-bc-20261001"
OUT.mkdir(exist_ok=True)
reports = [json.loads((ROOT / (m + "-cup-dataset.json")).read_text()) for m in ["pi05", "xiaomi"]]
rows = [r for d in reports for r in d["accepted"] if r["end"] - r["start"] >= 128][:10]
assert len(rows) == 10
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
cards = []
tiles = []
for i, r in enumerate(rows):
    dest = OUT / r["episode"]
    dest.mkdir(exist_ok=True)
    start = r["start"] / 40
    duration = (r["end"] - r["start"]) / 40 + 0.05
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(SOURCE / r["episode"] / "video.mp4"),
            "-t",
            str(duration),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest / "video.mp4"),
        ],
        check=True,
    )
    for name, sec in [("start.jpg", 0), ("end.jpg", max(0, duration - 0.11))]:
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-ss",
                str(sec),
                "-i",
                str(dest / "video.mp4"),
                "-frames:v",
                "1",
                str(dest / name),
            ],
            check=True,
        )
    r["url"] = "/media/pi05-cup-bc-20261001/" + r["episode"] + "/video.mp4"
    (dest / "example.json").write_text(json.dumps(r, indent=2))
    tile = Image.new("RGB", (600, 226), "white")
    d = ImageDraw.Draw(tile)
    d.text((5, 5), f"{i + 1}. {r['episode']} | actions [{r['start']},{r['end']})", fill="black")
    for x, name in [(0, "start.jpg"), (300, "end.jpg")]:
        im = Image.open(dest / name)
        im.thumbnail((300, 190))
        tile.paste(im, (x, 25))
    tiles.append(tile)
    url = "/media/pi05-cup-bc-20261001/" + r["episode"]
    cards.append(
        f'<details class="card" {"open" if i == 0 else ""}><summary><strong>{i + 1}. {r["source_model"]} · Seed {r["seed"]}</strong></summary><p>컵 잡기 이후~컵 놓기 성공 · 액션 [{r["start"]}, {r["end"]}) · {r["samples"]}개 학습 청크 · 버튼 동작 제외</p><video controls playsinline preload="none" src="{url}/video.mp4" poster="{url}/start.jpg" style="width:100%;max-width:1100px"></video><p><a href="{url}/example.json">구간 정보</a> · <a href="{url}/video.mp4">영상 저장</a></p></details>'
    )
montage = Image.new("RGB", (1200, 226 * 5), "#eee")
for i, t in enumerate(tiles):
    montage.paste(t, ((i % 2) * 600, (i // 2) * 226))
montage.save(OUT / "examples-contact-sheet.jpg")
manifest = dict(
    instruction=reports[0]["instruction"],
    episodes=sum(r["episodes"] for r in reports),
    samples=sum(r["samples"] for r in reports),
    examples=rows,
    labels="Actual successful policy rollouts; not human expert demonstrations",
    training_model="pi05",
    stage="cup_placement_only",
)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
intro = f'<p><strong>학습 대상: π₀.₅ · 컵 놓기만</strong></p><p>{html.escape(manifest["instruction"])}</p><p>컵을 잡고 초기 위치에서 10cm 이상 이동한 16스텝 청크 경계에서 시작합니다. 디스펜서 아래에 놓고 잡기·접촉을 해제한 뒤 5스텝 유지하면 종료합니다. 버튼을 먼저 누른 궤적은 제외했습니다.</p><p>초기 데이터: 성공 롤아웃 {manifest["episodes"]}개, 학습 청크 {manifest["samples"]}개 (π 5개 + Xiaomi 6개). 아래 10개는 실제 학습 구간의 예시이며 사람 전문가 시연은 아닙니다. 관측 이미지·상태는 행동 실행 직전, 액션은 RoboCasa 원래 12차원입니다.</p><p><a href="/media/pi05-cup-bc-20261001/index.html">예시 전용 페이지</a> · <a href="/media/pi05-cup-bc-20261001/manifest.json">데이터 요약</a></p>'
body = intro + "".join(cards)
(OUT / "index.html").write_text(
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>π 컵 놓기 Online BC 데이터</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:20px;background:#faf9f6}summary{cursor:pointer}.card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:14px 0}video{width:100%}</style><h1>π₀.₅ · 컵 놓기 학습 데이터 예시 10개</h1>'
    + body
)
begin = "<!-- PI-CUP-BC-BEGIN -->"
end = "<!-- PI-CUP-BC-END -->"
block = (
    begin
    + '<details id="pi05-cup-bc" class="card" open><summary><strong>π₀.₅ · 컵 놓기 Online BC · 학습 데이터 예시 10개</strong></summary>'
    + body
    + "</details>"
    + end
)
page = TRACK / "index.html"
s = page.read_text()
if begin in s:
    s = s[: s.index(begin)] + block + s[s.index(end) + len(end) :]
else:
    s = s.replace("<!-- VIDEO-POSTS -->", "<!-- VIDEO-POSTS -->" + block, 1)
assert begin in s
tmp = page.with_suffix(".cup.tmp")
tmp.write_text(s)
tmp.replace(page)
print(json.dumps(manifest))
