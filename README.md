# Amazon Ad Agent

Builds a **faceless product-ad video (MP4, 5-9 minutes)** from the photos and clips *you* supply, with a natural-sounding
script and voice-over, an automatic affiliate disclosure, and a review checklist. It runs on GitHub Actions, **never
uploads anything**, and you review and post the video yourself.

> **No Amazon scraping.** Amazon's Associates licence prohibits bots/data-extraction tools, and without the
> Creators/PA API you may not pull its images, ratings, reviews or prices. So the agent uses footage and facts you put in
> the project folder. The ASIN is only used to name the project and the link. When your API access is approved, the
> data source can be swapped in (`adagent/assets.py` is the only place that reads footage).

## What it does

| Step | What happens |
|---|---|
| 1. Check assets | Claude looks at every photo/clip: describes it, **excludes anything with a human face** (even on packaging) or **showing prices / star ratings / Amazon UI**. |
| 2. Script | Written from `facts.yaml` **only**. A second AI pass fact-checks every line; regex rules block prices, ratings, review quotes, promo language, other retailers. Angle rotates so videos don't share a template; scripts are compared with your earlier ones. |
| 3. You review | `projects/<ASIN>/script.md` is plain text. Edit freely on GitHub. |
| 4. Voice | Per-line TTS (cached, so edits never re-bill you). Switch voices with one dropdown. |
| 5. Edit | Studio/blur backgrounds, smooth pan-zoom (different framing on every reuse), cross-fades, captions, feature callouts, vignette, optional ducked music, end card. |
| 6. Output | `*.mp4` (1080p, loudness-normalised), `description.txt` (disclosure + link + chapters), `REVIEW_CHECKLIST.md`. |

Length is enforced: the script is sized to 5-9 min before any voice is paid for, and the final file is rejected if it falls outside.
A daily cap (default 3 videos) is a brake against mass-producing.

## New: comparison videos (2 or 3 products)

Make one video that compares your own products side by side. Each product keeps its own `facts.yaml`, footage and affiliate link.
1. Make sure every product is already a normal project folder (`projects/<ASIN>/` with `facts.yaml` and `assets/`).
   Optional: add `short_name: YESLIFE` to each `facts.yaml` for tidy on-screen labels.
2. Copy `projects/_template_compare/` to `projects/<NAME>/` (any 3-40 letters/digits/-/_, for example `COMPARE-TOWELS`),
   and fill in `compare.yaml` (title, the 2-3 product folders, category, audience).
3. Run **Make video** with that **NAME** as the project, mode **script**; read/edit `projects/<NAME>/script.md`; then mode **render**.

What it does: an intro, a section per product, a head-to-head on points that **every** product has facts for, who each one suits, and a verdict.
Every line is tagged to a product (`| product: 1`), shows that product's own footage, and gets an "A | name" chip on screen.
The description lists each product's own link; chapters carry the product names.
Rules: facts are never moved between products; a point is compared only if every product has a fact for it; no price/rating/review
comparisons; "better/best/winner" or a personal pick only when **every** product has 3+ `hands_on_notes` (otherwise the verdict names who each suits).
The fact-check also flags claims attributed to the wrong product. AI scenes are made per product (`scenes` mode on each product);
`ai_*` clips in a product's `assets/` are used automatically.

## New: cinematic shots, short ads, AI clips

**Cinematic product shots (free, automatic).** Photos with a clean plain backdrop are cut out and shown on a styled studio backdrop with a soft
shadow, floor reflection, a light sweep and slow camera moves, like a TV product ad. About 60% of eligible photo shots use it
(`video.cinematic_share` in `config.yaml`). If a cut-out edge looks rough, list the photo under `assets_no_cutout:` in `facts.yaml`,
or set `cinematic: false` there to turn it off for the product.

**Short ads.** The workflow has a **length** dropdown: `standard` (5-9 min) or `short` (1.5-4.8 min). Same as `--length short`.

**AI-generated clips (free GPU, optional).**
1. Run **Make video** with **mode: scenes**. It writes `projects/<ASIN>/scenes.md`: faceless mood scenes plus at most 2 gentle product-photo scenes.
   Unsafe prompts (faces, frost/steam/instant effects, brand names, prices) are dropped automatically. Edit the file if you like.
2. Open `notebooks/make_ai_clips.ipynb` in Kaggle or Colab (free GPU), follow the instructions at its top, and run it. It makes
   `ai_s01.mp4`, `ai_s02.mp4` ... and saves them into `projects/<ASIN>/assets/`. **This notebook is untested on a real GPU.**
3. Run **render** as usual. Clips named `ai_*` get an on-screen "AI-generated scene" tag, an AI note in `description.txt`, and a
   reminder to answer YES to YouTube's altered/synthetic content question. AI clips are still screened for faces and prices.
Check every AI clip yourself: AI video can distort hands and objects, and a clip must never imply a product effect you cannot support.

## Setup (10 minutes)

1. Create a **private** GitHub repo and upload this folder. (Private matters: your footage lives in it.)
2. **Settings -> Secrets and variables -> Actions**, add:
   - `ANTHROPIC_API_KEY` (required: script, fact-check, image checks)
   - `ELEVENLABS_API_KEY` (only for `eleven_*` voices)
   - `OPENAI_API_KEY` (only for `openai_*` voices)
3. **Settings -> Actions -> General -> Workflow permissions -> Read and write** (so it can save `script.md`).
4. For each product: copy `projects/_template/` to `projects/<ASIN>/`, fill `facts.yaml`, drop footage into `assets/`
   (see `assets/README.md`), commit.

## Making a video

1. **Actions -> Make video -> Run workflow**: ASIN, mode **script**. Open `projects/<ASIN>/script.md`, read it, edit it.
2. Run again with mode **render**, pick a voice. Start with `preview_seconds = 60` and an `edge_*` voice to check the look cheaply.
3. Download the artifact (zip) from the run page, **watch the whole video**, then upload to YouTube yourself and paste `description.txt`.
   Re-running render after edits is free of voice cost for unchanged lines.

Mode **all** does both steps at once if you don't want a review pause (not recommended).

Same thing locally: `pip install -r requirements.txt`, install ffmpeg, then `python -m adagent script B0XXXXXXXX` and
`python -m adagent render B0XXXXXXXX --voice edge_male`.

## Voices (switch any time)

`python -m adagent voices` lists them; edit `config.yaml` to add more. `python -m adagent eleven-voices` lists what your ElevenLabs key can use.

| Key | Provider | Notes |
|---|---|---|
| `eleven_female`, `eleven_female_2`, `eleven_male`, `eleven_male_2` | ElevenLabs | Most natural. IDs are ElevenLabs' stock voices; verify with `eleven-voices`. |
| `edge_*` | Microsoft Edge neural voices (free) | Good for previews. Unofficial endpoint, no commercial-licence statement. |
| `openai_*` | OpenAI TTS | Pay-as-you-go. |

**ElevenLabs free-plan reality check:** the free plan gives 10,000 credits/month, roughly 10 minutes of audio, which is about **one 7-minute video per month**, and its output is **non-commercial with required attribution**. A monetised affiliate channel needs a paid plan (Starter, $5/month at time of writing; check current terms). The agent reads your plan: on free it **auto-adds the "Voice generated by ElevenLabs" attribution** and warns you; it also stops before spending if you lack credits. `eleven_flash_v2_5` costs half the credits but sounds less natural (set `credits_per_char: 0.5` in `config.yaml`).

## Compliance guard-rails (and what they don't cover)

Built in: spoken + on-screen "As an Amazon Associate I earn from qualifying purchases", affiliate-link-only description, no prices/ratings/review text/other retailers, no faces, no Amazon UI, attribution of brand claims, AI-voice note in the description, and a checklist.

Still on you: the **channel must be listed in your Associates account**; **accuracy** of what you put in `facts.yaml`; **YouTube's** rules (its upload form asks about altered/synthetic content; answer honestly for your AI voice); **music/image rights**; and not posting near-identical videos. If you only ever feed it thin facts and stock-looking footage, the result will look like mass-produced content to a viewer or to YouTube, whatever the tool does. Your own hands-on notes and clips are what prevent that.

## Test status (be aware)

Verified in a sandbox without network keys: the whole render path (asset indexing, shot planning, pan/zoom, crossfades, captions, disclosure banner, end card, music ducking, loudness, 5-9 min gate, 1080p output), the script pipeline logic using a **simulated** AI (policy flags, fact-check/revise loop, length retry/trim, markdown round-trip), and the workflow YAML syntax.

**Not verified:** live calls to Anthropic, ElevenLabs, Edge TTS and OpenAI, and the real GitHub Actions run. Do a first run with a preview before relying on it. Expect to tweak the prompt text in `adagent/scriptgen.py` once you hear real output.

Offline test: `python tests/make_fixture.py && python -m adagent render TEST000001 --voice mock --no-vision --width 640 --height 360 --fps 24`
