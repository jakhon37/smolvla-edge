# Submitting the report to arXiv

## Before you start

1. **Final repo state.** Merge PRs #1, #2 and #3, so that the repository link in the paper shows what the paper describes.
2. **A citable release (optional).** Connect the repository to [Zenodo](https://zenodo.org) under GitHub → Settings → Integrations, then publish a GitHub release, for example `v0.1.0`. Zenodo archives it and mints a DOI that the paper or a later version can cite.
3. **An arXiv account** with a personal email address.
4. **Endorsement.** First-time submitters to cs.RO usually need one. When arXiv asks, it shows an endorsement code. Send the code to someone who has posted papers in cs.RO, such as a former professor or colleague; they enter it on arXiv's endorsement page. It doesn't affiliate you with them. If endorsement stalls, [TechRxiv](https://www.techrxiv.org), IEEE's preprint server, needs none.

## Build the upload

```bash
bash paper/make_arxiv.sh     # writes paper/dist/smolvla-edge-arxiv.tar.gz
```

It contains `main.tex`, `main.bbl`, `refs.bib` and the five figure PDFs. arXiv compiles with pdfLaTeX and doesn't run BibTeX, which is why `main.bbl` must be included; `refs.bib` comes along so the source also builds where BibTeX does run. Rebuild the package after any change to `main.tex` or `refs.bib`; the script refuses a stale `main.bbl`.

On arXiv's upload page, add the `.tar.gz` file whole; arXiv unpacks it. The script also writes the same files as `dist/smolvla-edge-source.zip`, for servers that take only zip, such as Preprints.org. Then check the PDF arXiv builds before you submit. It uses Times where the local tectonic build used TeX Gyre Termes, a Times clone, so the pages should match.

## The form

| Field | Value |
|---|---|
| Title | Faster but Not Different: Parity-Checked Deployment of the SmolVLA Robot Policy on a Jetson Xavier NX |
| Authors | Jakhongir Nodirov |
| Abstract | Below; plain text, 1159 characters (the limit is 1920) |
| Comments | 5 pages, 5 figures, 4 tables. Code, results and figures: https://github.com/jakhon37/smolvla-edge |
| Primary category | cs.RO (Robotics) |
| Cross-list | cs.LG (Machine Learning) |
| License | CC BY 4.0 |
| ACM class (optional) | I.2.9; C.3 |

**Abstract**, the same text as `main.tex`'s; update both if one changes:

```
Vision-language-action (VLA) policies are now small enough for embedded GPUs, but compiling one for such hardware can change its actions without raising any error. We deploy SmolVLA, a 450M-parameter VLA policy, on an NVIDIA Jetson Xavier NX and make agreement with the reference implementation a precondition for every speed-up. The policy is exported as three static ONNX graphs (the vision encoder, the VLM prefix and one step of the action expert), its flow-matching loop runs in a C++17 runtime, and every split point is compared with LeRobot's own fp32 tensors, frozen from real LIBERO frames. With TensorRT 8.5, fp32 engines match LeRobot to 1.6e-5 action standard deviations at 1.0 s per 50-action chunk. fp16 engines run 3.1 times faster, at 0.33 s, and stay within 0.060 standard deviations, inside the range of the bf16 precision LeRobot itself runs this checkpoint in. They do so only once every layer the graph types as fp32 is pinned to fp32: TensorRT 8.5's default fp16 build returns finite actions unrelated to the reference. We describe this and four further pitfalls the parity checks exposed, and release the code, measurements and figures.
```

## After it is announced

- Moderation usually takes one to two working days; the paper then gets an arXiv ID.
- Add the arXiv badge and ID to the repository README.
- Hugging Face's [Papers](https://huggingface.co/papers) page lists arXiv papers and lets you link the repository to yours.
- A new version, for example with closed-loop results, is uploaded to the same arXiv ID as v2.
