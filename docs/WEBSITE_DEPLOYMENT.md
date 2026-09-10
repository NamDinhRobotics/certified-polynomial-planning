# CP / PLANNING website deployment

## Current state — 10 September 2026

The actual website source, two original viewer datasets and public assets were merged into `main` in [PR #2](https://github.com/NamDinhRobotics/certified-polynomial-planning/pull/2), commit `6c226f528bc73037fa26b0b78bed9f2607889cf6`.

The [main-branch website workflow](https://github.com/NamDinhRobotics/certified-polynomial-planning/actions/runs/34431579310) passed its complete `validate` job, including the public-file manifest and **44/44 grouped browser checks**. Its `deploy` job stopped at first-time Pages activation with:

```text
Create Pages site failed. Error: Resource not accessible by integration
```

**GitHub Pages is not yet live.** Do not describe the intended URL below as an already successful deployment. The default Actions token cannot perform first-time repository administration for this repository.

## Enable the public website

1. Open [repository Pages settings](https://github.com/NamDinhRobotics/certified-polynomial-planning/settings/pages) as the repository owner. Under **Build and deployment → Source**, select **GitHub Actions**.
2. Open [the CP Planning website workflow](https://github.com/NamDinhRobotics/certified-polynomial-planning/actions/workflows/website-pages.yml), choose **Run workflow**, select `main`, and run it. Alternatively, re-run the failed deployment job while its validated Pages artifact remains available.

After the deployment job completes successfully, the expected address is:

<https://namdinhrobotics.github.io/certified-polynomial-planning/>

This is a GitHub Pages address, independent of `chatgpt.site`. Activation requirements are documented by [GitHub's Pages action](https://github.com/actions/configure-pages/blob/main/action.yml) and [Pages REST API](https://docs.github.com/en/rest/pages/pages#create-a-github-pages-site). No access token should be pasted into chat or committed to this repository.

## Source and serving

The publishable source/build is [`website/`](../website/). It is pure HTML/CSS/JavaScript: there is no React compilation, npm dependency or backend to run. From the repository root:

```sh
python3 -m http.server 8000 --directory website
```

Then open `http://localhost:8000/`. Do not open `index.html` with `file://`; the viewer uses ES modules and fetch. Only `website/` is sent to the hosting service, never the repository's full research dataset.

## What was preserved

The actual `app.js`, all Three.js modules, both original evidence JSON files and imported media were copied byte-for-byte from the deployed Site, using fixed SHA-256 hashes. The HTML came from the original HTTP response, not a browser-rendered DOM or textual reconstruction. Two hosting-only changes remove the original hosting layer's Cloudflare challenge and request the same font families through Google Fonts CSS. Font binaries are not bundled.

Both `v3_20260909` and `objective_repair_20260909` retain scenes **41000–41019**. The original historical failures remain visible. The viewer's 20 representative scenes per campaign display the conic arm, repetition 0; they do not represent a new 80-attempt experiment.

The browser suite checks 40 scene/campaign combinations, WebGL initialization, one canvas, controls, camera presets, telemetry plots, all three original H.264 videos actually decoding/playing, and a 390px mobile viewport. It found no JavaScript errors or missing site resources. This is website regression testing, **not** a rerun of scientific optimization, physics, tracking or exact certificate verification.

See [`SOURCE_PROVENANCE.json`](../website/SOURCE_PROVENANCE.json), [`README_EXPORT.md`](../website/README_EXPORT.md) and [`tools/smoke_website.py`](../tools/smoke_website.py). The test requires official Google Chrome for the H.264 media; install Chrome or run `python -m playwright install chrome` when it is absent.

No `.env`, credentials, `node_modules`, manuscripts, private evidence, browser extension content or research ZIP archives were added. Existing research data and verifier code were not modified by this migration.

## Alternative preview check

A commit-pinned third-party CDN preview was also tested over HTTP and returned 403 for its resources. It is **not a verified working public link** and is not used as the repository's website. Use the Pages activation steps above or serve the portable website folder on another authorized static host.
