# Putting CardioTwin on a website

Three ways, from easiest to most flexible:

| Option | Cost | Stays online | Effort |
|---|---|---|---|
| **A. Hugging Face Spaces** (recommended) | free | yes (sleeps after 48 h without visitors, wakes on the next visit) | 10 minutes, all in the browser |
| **B. Any Docker host** (Render, Railway, Fly.io, Google Cloud Run, a VPS) | paid; varies by host (a 1 GB instance is needed) | yes | 20 minutes |
| **C. Share from your laptop** (Cloudflare quick tunnel) | free | only while your laptop runs it | 2 minutes |

Before you publish, read [the checklist](#before-you-publish) at the bottom.

---

## A. Hugging Face Spaces (free, recommended)

Hugging Face hosts machine-learning demos for free (2 CPUs, 16 GB memory, plenty for CardioTwin).
Your site will live at `https://huggingface.co/spaces/<your-username>/cardiotwin`.

1. **Make an account** at <https://huggingface.co/join> (free).
2. **Create a Space:** click your profile picture -> **New Space** (or go to
   <https://huggingface.co/new-space>).
   * **Space name:** `cardiotwin`
   * **License:** pick one (e.g. `mit`) or leave it empty
   * **Select the Space SDK:** **Docker** -> template **Blank**
   * **Space hardware:** **CPU basic - FREE**
   * **Public**, then click **Create Space**.
3. **Add the Dockerfile:** in your new Space, open the **Files** tab -> **+ Add file** ->
   **Create a new file**. Name it `Dockerfile` (capital D, no extension) and paste the entire
   contents of [`deploy/huggingface/Dockerfile`](../deploy/huggingface/Dockerfile) from this
   repository. Click **Commit new file to main**.
4. *(Optional, makes the Space page nicer)* Open the Space's `README.md` -> **Edit** and replace
   it with [`deploy/huggingface/README.md`](../deploy/huggingface/README.md). Keep the part between
   the `---` lines: `sdk: docker` and `app_port: 7860` must stay.
5. **Wait for the build.** The Space shows **Building** (5-10 minutes the first time: it installs
   PyTorch and downloads CardioTwin from GitHub), then **Running**. Click the **Logs** button to
   watch. When it says `CardioTwin is ready`, your website is live.

Share either link:
* `https://huggingface.co/spaces/<your-username>/cardiotwin` (with the Hugging Face page around it)
* `https://<your-username>-cardiotwin.hf.space` (the app alone, full screen)

**Updating the website after you change the code:** push your changes to GitHub, then in the
Space go to **Settings** -> **Factory rebuild**. The Dockerfile always downloads the latest code
from the branch named on its `ARG BRANCH=` line. If you merge the code into `main`, change that
line to `ARG BRANCH=main`.

**If the build fails:** open **Logs**, copy the red error lines and ask for help. The most common
cause is a typo when pasting the Dockerfile.

**Keep the GitHub repository public.** The Space downloads the code from GitHub while it builds;
if the repository is private, that download fails. (Alternative for a private repository: use
option B, or upload the code to the Space itself.)

---

## B. Any Docker host (Render, Railway, Fly.io, Cloud Run, your own server)

The repository root contains a `Dockerfile` that packages the app with CPU-only PyTorch (image
about 1.5 GB, needs about **0.5 GB of memory**, so choose an instance with at least 1 GB; 512 MB
free tiers are too small). The container listens on the port given by the `PORT` environment
variable (default 8000), which is what these hosts expect.

Try it on your own computer first ([install Docker Desktop](https://www.docker.com/products/docker-desktop/)):
```bash
docker build -t cardiotwin .
docker run -p 8000:8000 cardiotwin
# open http://localhost:8000
```

**Render** (<https://render.com>): **New** -> **Web Service** -> connect your GitHub repository ->
it detects the Dockerfile -> choose an instance with at least 1 GB of memory -> **Create Web
Service**. Render sets `PORT` automatically and redeploys on every push.

**Railway** (<https://railway.app>): **New Project** -> **Deploy from GitHub repo** -> pick the
repository. Railway builds the Dockerfile and sets `PORT`. Then **Settings** -> **Networking** ->
**Generate Domain**.

**Google Cloud Run:** `gcloud run deploy cardiotwin --source . --memory 1Gi --allow-unauthenticated`
(needs a Google Cloud account with billing enabled; small sites usually stay in the free quota).

**Your own server (VPS):** install Docker, run the two commands above with `-p 80:8000`, and put
a reverse proxy with HTTPS (e.g. Caddy) in front of it.

Without Docker, on any Linux server:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
HOST=0.0.0.0 PORT=8000 cardiotwin serve --no-browser
```

---

## C. Share from your laptop for a few hours (Cloudflare quick tunnel)

Good for showing someone the app right now. Your laptop must stay on with CardioTwin running.

```bash
brew install cloudflared                 # Mac; other systems: see developers.cloudflare.com
cardiotwin serve                         # terminal 1
cloudflared tunnel --url http://localhost:8000   # terminal 2
```
`cloudflared` prints a link like `https://random-words.trycloudflare.com`. Anyone with the link
can use your copy until you press Ctrl+C. No account needed.

---

## Showing it on your own website

Once it runs on Hugging Face (A), you can embed it in any web page (Wix, Squarespace, WordPress,
Google Sites or plain HTML) with an iframe:

```html
<iframe src="https://<your-username>-cardiotwin.hf.space" width="100%" height="900"
        style="border:0; border-radius:16px" title="CardioTwin"></iframe>
```

Or simply link to it: "Try the CardioTwin demo".

---

## Before you publish

* **Only the bundled MUSIC models and demo ECGs.** Never put MIMIC data, MIMIC-trained models or
  anything derived from MIMIC on a public website: the PhysioNet data use agreement forbids
  sharing it. The Dockerfiles only include the `models/` folder; check that it holds only
  `music-xyz-*` and `encoders/` before you deploy from your own computer.
* **Keep the disclaimer.** The site says on every page that it is a research model and not a
  medical device. Don't remove that, and don't advertise it as a health test.
* **Attribution.** The footer credits the MUSIC dataset (ODbL) and the README credits PTB-XL
  (CC BY 4.0), as their licences require. Keep them.
* **Uploads.** Uploaded ECGs are processed in memory and deleted immediately; nothing is stored.
  Uploads are limited to 25 MB and 10 files, and WFDB headers can only refer to the uploaded
  files. Still, tell visitors not to upload ECGs that identify a real person.
* **It is a demo.** One small server handles a handful of visitors at a time. If you expect a
  crowd (a class, a science fair), pick a bigger instance or warn people it may be slow.
