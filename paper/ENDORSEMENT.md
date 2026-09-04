# arXiv endorsement and submission package

Policy pages reviewed 2026-09-04. Documented facts below come from arXiv's
own pages; submission recommendations and inferences are labeled.

## Endorsement mechanics

Since 2026-01-21, an institutional address alone is insufficient for a new
submitter. Automatic endorsement requires both an academic/research address
and claimed authorship of an arXiv paper accepted in the endorsement domain
being entered. Anyone who does not meet both conditions must seek personal
endorsement from an established arXiv author in the same domain. Existing
category endorsements remain valid, and staff cannot waive the requirement
or personally endorse authors. Sources: [policy-change announcement](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/)
and [endorsement help](https://info.arxiv.org/help/endorsement.html).

Start a submission and select the intended category. arXiv emails a request
link; the prospective endorser ultimately uses a six-character alphanumeric
code on the endorsement form and may endorse, decline (a negative vote), or
abstain by doing nothing. At least one positive endorsement is required per
endorsement category. Decisions and optional comments are private between the
endorser and arXiv. Do not mass-mail endorsers or repeatedly contact one
person. Source: [endorsement help](https://info.arxiv.org/help/endorsement.html).

An eligible endorser has authored enough papers in the relevant endorsement
domain, with only papers submitted between three months and five years ago
counting; is registered as an author of those papers; and has an active
positive endorsement for the area. arXiv does not publish a fixed paper-count
threshold because it varies by subject area, and it may suspend privileges.
Confirm eligibility through the "Which authors of this paper are endorsers?"
link on a relevant abstract page.

Endorsement is not peer review. The endorser is asked to know the person or see
the intended paper and confirm that the work fits the subject, that the author
knows the field's basic facts, and that the work connects to current work.
They need not validate correctness or read in detail, and should keep a shared
draft confidential. Source: [endorsement help](https://info.arxiv.org/help/endorsement.html).

### Categories for this submission

- **Primary — cs.CV:** image processing, computer vision, pattern recognition,
  and scene understanding; ACM I.2.10, I.4, and I.5.
- **Cross-list — cs.NE:** neural networks, connectionism, genetic algorithms,
  artificial life, and adaptive behavior; parts of ACM C.1.3, I.2.6, and I.5.
- **Cross-list — cs.RO:** robotics; ACM I.2.9.

Source: [arXiv category taxonomy](https://arxiv.org/category_taxonomy).

**Inference to verify in the live flow:** arXiv describes endorsement domains
as high-level subject areas or related-category groups and names physics as
the exception that uses individual classes. Thus cs.CV, cs.NE, and cs.RO
appear to share the Computer Science endorsement domain. There is no public
category-by-category eligibility table. Start with cs.CV and check whether
either cross-list produces another endorsement request.

Cross-list only where directly relevant. arXiv says more than one or two
cross-lists is rarely appropriate, regards excessive or inappropriate
cross-listing as bad etiquette, and removes bad cross-lists. Moderators may
add or remove cross-lists or reclassify the primary category. Sources:
[cross-listing policy](https://info.arxiv.org/help/cross.html) and
[moderation policy](https://info.arxiv.org/help/moderation/index.html).

## Message to a prospective endorser

> Subject: arXiv endorsement request for cs.CV
>
> Hello [RECIPIENT],
>
> A 3D Gaussian splatting scene is a list of primitives: fast to rasterize,
> awkward to query, merge, or carry. We show that such a scene can instead be
> superposed into a single fixed-size complex vector, and that three
> capabilities then follow from one algebra rather than from three mechanisms.
>
> I have attached the paper PDF. Supporting material is the public gallery at
> [GALLERY URL], the reproducible benchmark in `bench/scene_bench.py`, the
> [hdc-holo package on PyPI](https://pypi.org/project/hdc-holo/), and archived
> software DOI [10.5281/zenodo.22116367](https://doi.org/10.5281/zenodo.22116367).
> The arXiv request is [ENDORSEMENT LINK OR CODE].
>
> Would you be willing to endorse this submission for cs.CV?

The descriptive sentences are copied from `paper/abstract.txt`.

## Pre-submission checklist

### Rights and license

- [ ] Confirm Squatch Stack holds all necessary rights and accept the arXiv
  submittal agreement.
- [ ] Choose **CC BY 4.0** for redistribution, adaptation, and commercial reuse
  with attribution if the target venue/funder permits it; choose the **arXiv
  perpetual, non-exclusive license** if reuse should be limited principally to
  arXiv distribution. The paper license is distinct from the repository's
  Apache-2.0 software license.
- [ ] Check journal and funder policies first. arXiv says a version's selected
  license is irrevocable, although later versions may differ. Source:
  [license information](https://info.arxiv.org/help/license/index.html).

### Account, categories, and metadata

- [ ] Select **cs.CV** primary and request **cs.NE** and **cs.RO** only when
  each cross-list is directly justified; complete live endorsement prompts.
- [ ] Title: **Hypervector Scene Memory: Gaussian Splats in Superposition**.
- [ ] Authors: **Squatch Stack**. Confirm this public author identity satisfies
  arXiv identity rules and is consistent in the account and manuscript.
- [ ] Link the account to **[ORCID iD: 0000-0000-0000-0000]** after replacing
  the placeholder. ORCID is linked through the account, not added to Authors.
  Source: [ORCID help](https://info.arxiv.org/help/orcid.html).
- [ ] Paste `paper/abstract.txt` as ASCII plain text without an "Abstract"
  heading and verify it stays under arXiv's 1920-character maximum. The file is
  currently within the limit, and a repository test enforces it.
- [ ] Fill title, authors, abstract, and category. Comments may give the
  page/figure count and public links. Leave journal reference and publication
  DOI blank unless there is a publication; put the software DOI in Comments.
- [ ] Leave **MSC-class** blank (mathematics archives only). **ACM-class** is
  optional for Computer Science; if used, choose only applicable taxonomy
  classes listed above. Source: [metadata help](https://info.arxiv.org/help/prep.html).

### Submission and ancillary files

- [ ] Upload the TeX/PDFLaTeX bundle assembled by
  `paper/make_arxiv_bundle.py`; preview arXiv's compiled PDF and inspect its
  log. Do not upload a PDF generated from available TeX source.
- [ ] Include every figure; external links do not replace required figures.
- [ ] Prefer stable external links for the maintained package, gallery,
  benchmark repository, and DOI. For a frozen benchmark copy, use top-level
  `anc/`; ancillary files require TeX source, are version-bound, and cannot be
  updated independently or search-indexed. Source:
  [ancillary-file help](https://info.arxiv.org/help/ancillary_files.html).
- [ ] Remove secrets, local paths, generated clutter, and unnecessary files;
  ensure file-name case exactly matches TeX references.

### Moderation review

- [ ] The article is self-contained, topical, refereeable original research
  with professional neutral prose and carefully prepared sections, figures,
  tables, and references.
- [ ] Claims are supported; data, affiliation, authorship, and content are not
  misrepresented; citations and reused text avoid plagiarism or excessive
  overlap; all included material can lawfully be submitted.
- [ ] Category fit is evident. The title **Hypervector Scene Memory: Gaussian
  Splats in Superposition** already distinguishes the work from wave-optics
  "holographic" work, handling the known naming collision.
- [ ] Cross-lists are few and directly justified; moderators may remove one or
  reclassify the primary category.
- [ ] This is a research article, not a proposal, course project, news item,
  political statement, or incomplete draft; it contains no offensive material
  or embedded JavaScript.

Sources: [submission guidelines](https://info.arxiv.org/help/submit/index.html)
and [moderation policy](https://info.arxiv.org/help/moderation/index.html).
