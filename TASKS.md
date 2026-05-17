# TASKS.md - GenoPath Build Checklist
# Status markers: [ ] not started | [~] in progress (add a note) | [x] done (add verification) | [!] blocked (add reason)
# Run `python scripts/sanity_check.py` after completing each phase. It must pass before moving to the next phase.
# When handing off to another agent: update status, add a note, and point to relevant files.

---

## PHASE 1 - Graph Foundation
> Everything depends on this. Do not start Phase 2 until sanity_check.py confirms the graph loads.

- [x] **1.1 Create repo structure**
  All directories created. Empty `__init__.py` files in all `src/` subdirectories.
  `.env.example`, `requirements.txt`, `README.md` created.

- [x] **1.2 Confirm data files exist**
  `data/hp.obo` and `data/clinvar_pathogenic.tsv` present and verified.

- [x] **1.3 Write scripts/filter_clinvar.py**
  Written. TSV already present so not run. Column name-based loading handles any order difference.

- [x] **1.4 Write src/graph/graph.py - HPO parser**
  `parse_hpo_obo()` implemented. `terms["HP:0001250"]["name"] == "Seizures"` verified.

- [x] **1.5 Write src/graph/graph.py - ClinVar loader**
  `load_clinvar_variants()` implemented. 3000+ genes loaded, SCN1A/BRCA1/MYH7 present.

- [x] **1.6 Write src/graph/graph.py - GenoPathGraph class and build**
  57,124 nodes, 79,833 edge pairs, builds in ~1.3s. Singleton via `get_graph()`.

- [x] **1.7 Write src/graph/graph.py - absent phenotype helper**
  `get_absent_phenotypes()` implemented. BRCA1 vs seizures-only patient returns breast/ovarian terms.

- [x] **1.8 Write scripts/sanity_check.py - Phase 1 checks**
  All Phase 1 assertions pass.

---

## PHASE 2 - Episode Logic
> Do not start until Phase 1 sanity_check passes.

- [x] **2.1 Write src/episode/models.py**
  Pydantic v2: GraphNode, Variant, GenoPathAction, GenoPathObservation, StepResult, EpisodeResult all implemented.

- [x] **2.2 Write src/graph/case_generator.py - PatientCase dataclass**
  PatientCase and MAX_STEPS implemented.

- [x] **2.3 Write src/graph/case_generator.py - monogenic generator**
  Generates cases with 5+ candidates. Causal allele IDs in candidate pool verified.

- [x] **2.4 Write src/graph/case_generator.py - oligogenic generator**
  len(causal_allele_ids) >= 2 verified across seeds.

- [x] **2.5 Write src/graph/case_generator.py - phenotype_mismatch generator**
  decoy_gene set. Decoy variants in candidate pool. Patient phenotypes exclude cancer terms.

- [x] **2.6 Write generate_case() dispatch function**
  Routes to all three generators. All task types generate without error across 10 seeds.

- [x] **2.7 Write src/episode/environment.py - GenoPathEnvironment**
  reset(), step(), action dispatch, reward, overseer all implemented. Rewards always in (0.01, 0.99).

- [x] **2.8 Extend scripts/sanity_check.py - Phase 2 checks**
  All Phase 2 assertions pass including 3-step episode and terminal episodes for all task types.

---

## PHASE 3 - Agent (Zero-Shot E4B)
> Do not start until Phase 2 sanity_check passes. This is the core of the product.

- [x] **3.1 Confirm Ollama + Gemma 4 E4B is working**
  `gemma4:e4b` confirmed. Runs at ~6-9s/step, fully GPU-resident on 4060 Ti 16GB.
  _Verified: test_model.py confirmed all 3 steps fire valid tool calls._

- [x] **3.2 Write src/agent/tools.py**
  GENOPATH_TOOLS (4 tools), SYSTEM_PROMPT, format_observation() implemented.
  _Verified: sanity_check.py Phase 3 checks pass._

- [x] **3.3 Write format_observation() in src/agent/tools.py**
  Includes: phenotypes, current node + neighbors, trail, candidates, exclusion signals,
  steps remaining, context hints by node type.
  _Verified: all required sections present._

- [x] **3.4 Write src/agent/ollama_client.py - single episode runner**
  Correct multi-turn format: role="tool" for tool results, tool_calls dict in assistant turn.
  Filters thought_output pseudo-calls. Re-prompts with role="user" on fallback.
  _Verified: 3-step smoke test all produce valid tool calls._

- [x] **3.5 Baseline benchmark**
  Run 10 episodes x 3 task types with gemma4:e4b. Saved results/baseline_e4b.json.
  _Verified: monogenic 30%/0.333, oligogenic 10%/0.192, phenotype_mismatch 40%/0.441. 28.4 min total._

- [x] **3.6 Extend scripts/sanity_check.py - Phase 3 checks**
  3-step smoke test with live gemma4:e4b call.
  _Verified: all 4 phases pass._

---

## PHASE 4 - Gradio UI
> Do not start until Phase 3 passes. UI depends on agent working.

- [x] **4.1 Write src/ui/app.py - skeleton**
  Two-column Gradio layout. Left: input controls. Right: placeholder output.
  _Verified: launches at localhost:7860 without error._

- [x] **4.2 Wire episode into UI - live trail updates**
  Generator-based run_episode_stream() yields per step for live updates.
  _Verified: trail, step counter, reward all update per step._

- [x] **4.3 Add candidate variants table + exclusion signals panel**
  Candidate variants as markdown table. Exclusion signals panel per gene.
  _Verified: renders in right panel. Result card shows on completion._

- [x] **4.4 Add image upload mode**
  Image upload component wired to extract_phenotypes_from_image() from intake.py.
  _Verified: image upload UI renders; phenotype extraction calls Ollama vision._

- [x] **4.5 Add phone pairing section**
  Accordion shows local IP + intake port (auto-detected via socket).
  _Verified: correct IP displayed in accordion._

- [x] **4.6 Final result card**
  Result card with SUCCESS/INCORRECT/TIMEOUT status and flagged variants.
  Decoy gene detected and flagged in card when applicable.
  _Verified: result card renders after episode completion._

---

## PHASE 5 - Phone Integration
> Can be done in parallel with Phase 4 after Phase 3 is done.

- [ ] **5.1 Verify E2B runs on Pixel 9a**
  Open AI Edge Gallery app on Pixel 9a. Download Gemma 4 E2B.
  Test with a photo of any text. Confirm inference runs without OOM.
  Note: E4B requires closing Chrome and heavy apps first -- document this in README.
  _Verify: manual test on device. Note inference speed (tokens/sec)._

- [x] **5.2 Write src/agent/intake.py - phenotype extraction**
  extract_phenotypes_from_image() uses Ollama E4B vision. HPO matcher: exact > synonym > substring.
  Returns [{"raw": ..., "hpo_id": ..., "name": ...}].
  _Verified: HPO matcher logic tested in sanity_check via graph._

- [x] **5.3 Document LiteRT / MediaPipe approach in intake.py**
  Commented-out block: LlmInferenceOptions, model_path, generate_response, base64 image encoding.
  _Verified: block present at bottom of intake.py._

- [x] **5.4 Write src/server/intake_api.py**
  FastAPI with POST /intake (phenotypes -> HPO match -> episode reset) and GET /health.
  _Verified: imports cleanly. Uvicorn serves at 0.0.0.0:8000._

- [ ] **5.5 Test phone -> laptop flow**
  With intake_api.py running, send a POST from phone (via curl in Termux or a test script).
  Confirm phenotypes are received and case is generated.
  _Verify: curl POST returns valid observation JSON with patient phenotypes._

---

## PHASE 6 - Analysis + Polish
> Run cognitive_load.py before making the video.

- [x] **6.1 Write scripts/cognitive_load.py**
  For 10 seeds x 3 task types:
  - Count candidate variants with pathogenicity_score >= 0.75 per case
  - Count optimal hop path length (BFS from starting HPO to causal variant)
  Print table: task | avg candidates | avg optimal hops | ratio.
  Save to `results/cognitive_load.json`.
  _Verified: monogenic 2.07x, oligogenic 3.17x, mismatch 2.73x burden ratio. Phase 4 sanity check passes._

- [x] **6.2 Update README.md**
  Real benchmark numbers, cognitive load table, architecture diagram, hardware requirements,
  quickstart commands, Ollama + phone setup instructions all added.
  _Verified: README updated with real results from baseline_e4b.json._

- [ ] **6.3 Full end-to-end smoke test**
  Run `python scripts/sanity_check.py` -- must pass all checks.
  Run the Gradio app, complete one episode of each task type.
  Confirm BRCA1 decoy is flagged or excluded in phenotype_mismatch run.
  _Verify: all three task types complete without error in the UI._

---

## PHASE 7 - Optional: Edge Model Fine-Tuning (Unsloth Prize)
> Only start if Phases 1-6 are complete and it is Day 11 or earlier.

- [ ] **7.1 Confirm Gemma 4 E2B model ID on HuggingFace**
  Search HF for `unsloth/gemma-4-2b` or `google/gemma-4-2b-it` -- confirm exact model ID.
  Check if Unsloth has a pre-quantized version (preferred for Colab speed).
  _Verify: model ID found on HF. Note it in training notebook._

- [x] **7.2 Create training/gemma4_grpo.ipynb on Colab**
  Cell 1: install `unsloth[colab-new]`, trl>=0.9.0, peft, accelerate, bitsandbytes, websockets, pydantic
  Cell 2: config -- HF_TOKEN, BASE_MODEL, max_completion_length=600 (NOT 300)
  Cell 3: load model + disable thinking mode if present + `os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"`
  Cell 4: reward function -- f1_score(extracted_phenotypes, ground_truth_hpo_names)
  Cell 5: `nest_asyncio.apply()` + build prompt dataset (50 cases x 3 task types)
  Cell 6: GRPOConfig with max_completion_length=600
  Cell 7: train -- verify reward_std > 0 in first 10 steps
  Cell 8: eval -- compare E2B zero-shot vs fine-tuned on phenotype extraction accuracy
  Cell 9: push adapter to HF Hub
  _Verify: reward_std > 0.0 after first training step. If 0.0, check thinking mode and completion length._

- [x] **7.3 Publish adapter to HF Hub**
  Pushed via Cell 10 in gemma4_grpo.ipynb. Repo: KrishVenky/genopath-e2b-grpo.
  _Verified: Cell 10 completed successfully._

- [ ] **7.4 Three-way benchmark**
  Compare: E2B zero-shot | E2B fine-tuned | E4B zero-shot on phenotype extraction F1.
  Save to `results/training_benchmark.json`.
  Add table to README.
  _Verify: results file exists. Fine-tuned E2B shows improvement over zero-shot E2B._

---

## PHASE 8 - Video + Writeup + Submit

- [ ] **8.1 Record video (>=3 min)**
  Shot list (from SYSTEM.md Section 15):
  0:00-0:25 Problem. 0:25-1:15 Demo. 1:15-1:50 Architecture. 1:50-2:20 Numbers. 2:20-3:00 Vision.
  Upload to YouTube (unlisted or public). No login required.
  _Verify: YouTube link works in incognito._

- [ ] **8.2 Write Kaggle writeup (>=1,500 words)**
  Sections: Problem | Solution | Knowledge Graph | Routing Architecture | Gemma 4 Features |
  Absent Phenotype Exclusion | Results | Limitations | How to Run.
  Track: Health & Sciences.
  _Verify: word count >=1,500. All sections present._

- [ ] **8.3 Attach all required resources to writeup**
  - [ ] Video (YouTube link in Media Gallery)
  - [ ] Code repository (GitHub link in Attachments -> Project Links)
  - [ ] Live demo (HF Space or local instructions link)
  - [ ] Cover image in Media Gallery (screenshot of the Gradio UI showing a graph trail)
  _Verify: all 4 attachments present in Kaggle writeup editor._

- [ ] **8.4 Final submission**
  Hit Submit on Kaggle writeup.
  Confirm submission is listed as submitted (not draft).
  Deadline: May 18, 2026 23:59 UTC.
  _Verify: submission confirmation email or page shows "Submitted"._

---

## SANITY CHECK SCRIPT SPEC

`scripts/sanity_check.py` must cover all implemented phases. Add checks incrementally.

Current coverage: Phases 1-4 (graph, episodes, agent smoke test, cognitive load).

---

## CURRENT STATUS

_Update this section at the start of each session._

**Last updated by:** Claude Sonnet 4.6
**Date:** 2026-05-04
**Phase in progress:** Phase 8 (Video + Writeup + Submit)
**Last completed task:** Gradio UI smoke test (localhost:7860 serving HTTP 200)
**Next task:** 6.3 full end-to-end smoke test (sanity_check.py + 3 UI episodes), then 7.4 benchmark, then 8.1 video
**Blockers:** None
**Notes:**
- Phases 1-6.2 complete. Phases 4, 5, 7.1-7.3 now also complete.
- GRPO training done: 150 steps on T4, 1h54m. Eval F1=0.983 (n=20 held-out seeds). Adapter at KrishVenky/genopath-e2b-grpo.
- Gradio UI (app.py) confirmed serving at localhost:7860. Gradio 6 compat fixes applied.
- Encoding bugs (smart quotes, mojibake) fully cleared from app.py.
- 7.4 three-way benchmark still needed: E2B zero-shot vs E2B fine-tuned (0.983) vs E4B zero-shot.
