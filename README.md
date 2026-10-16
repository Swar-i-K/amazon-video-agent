# Amazon Ad Agent

Builds faceless product videos (MP4, **always under 9 minutes**) from the photos and clips *you* supply, with a natural script, voice-over,
five visual templates, and a review checklist. It runs on GitHub Actions, **never uploads anything**, and you review and post each video yourself.

> **No Amazon scraping.** Amazon's Associates licence prohibits bots/data-extraction tools, and without the Creators/PA API you may not pull
> its images, ratings, reviews or prices. So the agent uses footage and facts you put in the project folders.

## Three video types

| Type | Folder holds | Length (auto) | What you get |
|---|---|---|---|
| **Review** (one product) | `facts.yaml` + `assets/` | about 5 min (3.5-8) | hook, overview, features, in use, limits, who it suits, verdict |
| **Comparison** (2-3 products) | `compare.yaml` | about 7.5 min (5.5-8.8) | a section per product, head-to-head on points every product has facts for, who each suits |
| **Top-N list** (3-9 products) | `list.yaml` | about 7.5 min (5.5-8.8) | countdown with a big **#N** badge per item, "best for" labels, recap |

`length: short` in the workflow gives 1.5-4.8 minutes for any type. All presets are in `config.yaml` (`length_presets`).
**Max 2 videos per day** (`safety.max_videos_per_day`), a brake against mass-production.

## Five visual templates (workflow dropdown `template`, or `video.template` in `config.yaml`)

| Key | Look |
|---|---|
| `one` | Clean and minimal: white panels, blue accent, soft fades |
| `two` | Bold and vibrant: solid colour captions, vivid backdrops, push transitions |
| `three` | Dark cinematic: letterbox bars, film grain, outlined captions, dip-to-black |
| `four` | Modern glass: real frosted-glass panels, pastel bokeh |
| `best` | Modern default: dark pill captions, amber accent |

Run **mode: preview** to get `template_preview.png`, one sample picture per template on one of your own photos. Templates are plain settings in
`config.yaml` (`templates:`), so you can tweak colours, caption style, grade, letterbox and transitions.

## Where things go

```
projects/<ASIN>/            a product:  facts.yaml, assets/ (photos + clips), music/ (optional), script.md
projects/<NAME>/            a comparison: compare.yaml  (or a list: list.yaml), script.md
projects/_template*/        blank copies to start from
```
**Keywords:** add `keywords:` to `facts.yaml`, `compare.yaml` or `list.yaml`. They are worked naturally into the title, description and tags (never stuffed).

## Making a video

1. Product folder: copy `projects/_template/` to `projects/<ASIN>/`, fill `facts.yaml` (true notes; **3+ hands_on_notes** from real use), add 8+ photos and some clips to `assets/`.
2. For a comparison or list: copy `projects/_template_compare/` or `projects/_template_list/` to `projects/<NAME>/` and fill it in.
3. **Actions -> Make video -> Run workflow**: project name, **mode: script** (tick **force** only to overwrite an existing script). Edit `projects/<NAME>/script.md`.
4. Optional: **mode: preview** to compare the five templates.
5. **mode: render** with `preview_seconds: 60` and an `edge_*` voice first; then again with `0`. Download the zip from the run's **Artifacts**:
   the `.mp4`, `title.txt`, `description.txt`, `pinned_comment.txt` and `REVIEW_CHECKLIST.md`.
6. Watch the whole video with sound before posting.

You can also write your own `script.md` (one line per sentence: `[id] text | product: k | callout: ... | assets: ...`) and just render.

## AI ad clips (optional, free GPU, untested on a real GPU)

**Make video -> mode: scenes** plans `scenes.md` for one product: polished commercial-style clips animated from your own product photos
(slow orbit/push-in, light sweep, rim light; up to `scenes.max_product_scenes`, product scenes first) plus faceless mood scenes. Unsafe prompts
(faces, frost/steam/instant effects, brand names, prices) are dropped. Then run `notebooks/make_ai_clips.ipynb` in Kaggle or Colab; it saves
`ai_*.mp4` into `assets/`. Renders use them automatically, add an on-screen "AI-generated scene" tag, an AI note in the description, and a
reminder to answer YES to YouTube's synthetic-content question. Your own video clips are graded and styled by the template but are not
AI-converted. Always check AI clips for distorted products or hands.

## Where the affiliate disclosure goes (`disclosure: mode:` in `config.yaml`)
`full` = spoken line + on-screen banner + tag | `on_screen` = silent text near the start + end card | `description_only` = nothing in the video
(shipped setting). Every run writes `description.txt` (Amazon's sentence on its **first line**) and `pinned_comment.txt`. The FTC says a
disclosure inside the video is clearer than one in the description box; `on_screen` is the middle option. You are responsible for compliance.

## Voices
`edge_*` free (no commercial-licence statement), `eleven_*` ElevenLabs (free plan: about one video a month, non-commercial, attribution added
automatically), `openai_*` pay-as-you-go. List with `python -m adagent voices`.

## Setup
1. Private GitHub repo with this project. 2. Secrets: `GROQ_API_KEY` (AI), optional `ELEVENLABS_API_KEY`, `OPENAI_API_KEY`.
3. Settings -> Actions -> General -> Workflow permissions -> Read and write. 4. `.github/workflows/` must contain only `make-video.yml`.

## Guard-rails
Spoken/on-screen rules block prices, price wording, star ratings, review quotes, "best seller", other retailers and health claims. Facts are never
moved between products. Rankings are presented as **your** pick. A winner in a comparison needs 3+ hands_on_notes on every product. Footage is
screened for faces and price/Amazon UI. You still own accuracy, YouTube's rules, music and image rights.

## Test status
Offline (simulated AI/voice): script, comparison and list logic, templates (grade, transitions, 5 caption styles, preview), disclosure modes, scenes,
asset checks, full renders under every template. **Not verified:** live Groq / ElevenLabs / Edge calls, the real GitHub run, and the AI-clip notebook on a real GPU.
