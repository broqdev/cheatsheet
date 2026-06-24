# FlashAttention-4 CUTE Forward/Backward Completion

## Goal

Improve the FlashAttention-4 page's paired LaTeX and CUTE-style code so the forward and backward sections are source-shaped enough to teach the real SM100/SM110 implementation, not only the dedicated HD256 path. The implementation replaces vague or unused-looking helper names with compact, accurate context; adds comments where behavior comes from CUTLASS/CuTe internals rather than code shown in the snippet; and fixes wording that put host-dispatch facts in the wrong kernel-code context.

## Knowledge

- Primary local files updated: `src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py` and `src/features/cheat-sheet-view/data/subPages/flashAttention4.ts`.
- Upstream generic anchors used during the audit/implementation: `flash_attn/cute/flash_fwd_sm100.py`, `flash_bwd_sm100.py`, `flash_bwd_preprocess.py`, `flash_bwd_postprocess.py`, and `flash_attn/cute/interface.py` in `Dao-AILab/flash-attention`.
- Upstream HD256 anchors used during the audit/implementation: `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`, `sm100_hd256_2cta_fmha_backward.py`, `sm100_hd256_2cta_fmha_backward_dqkernel.py`, and `sm100_hd256_2cta_fmha_backward_dkdvkernel.py`.
- Completed forward shape: the page comments that `make_fragment_A/B/C` are CuTe/CUTLASS tiler internals, models generic `q_stage` as an M-stage pipeline, preserves the dedicated HD256 divergence, routes final O/LSE through a source-shaped `correction_epilogue`, exposes `softmax_stats_mbar_ptr` with distinct producer/consumer state for the `sScale` row-sum/row-max handoff, and shows HD256's separate `sSum` producer/consumer path as concrete calls instead of a string note.
- Completed backward shape: the page now separates generic SM100 backward from the HD256 wrapper. Generic rows point at preprocess/main/postprocess, `dQaccum`, semaphores, and MMA fragment construction. HD256 rows point at a wrapper plus dedicated dQ and dK/dV kernel sketches whose mainloops and epilogues now use source-shaped TMA atoms, shared-memory tiles, TMEM tiles, `cute.gemm`, fragment load/store helpers, and explicit TMA epilogue staging.
- Completed wording cleanup: the HD256 dQ path now says host dispatch passes final dQ through the historical `dQ_accum` ABI slot, while the dedicated dQ kernel stores dQ directly and skips generic postprocess/reduction. The DSMEM discussion is bounded as a paper/generic reduction note rather than current HD256 wrapper code.
- Content-system constraint: `defineAttentionContent(...)` code refs must match raw `# @ref` markers exactly, and nested refs are not supported.
- Final verification passed on 2026-06-24 after `R1A9390-012` and `R1A9390-013`: `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py`, Vite SSR/module-load for `flashAttention4.ts`, `git diff --check`, the no-string-placeholder `rg` scan, and `npm run build`.

## Worklog

### 2026-06-24 16:07:54 UTC - Created doc-only task from FA4 CUTE audit

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "9F60198F-78EA-4E14-9B9A-703A51A32445", "trigger": "revisit" }
```

Created this task doc from the user's `noisy:tmd --new` request. Read the local FA4 page code/prose and checked the upstream SM100/SM110 HD256 forward, backward wrapper, dedicated dQ kernel, dedicated dK/dV kernel, and interface dispatch. No source code or LaTeX was changed. The audit found two large completion tracks: forward fragment/pipeline/epilogue detail, and backward dedicated split-kernel detail. It also found wording cleanup work for comments that currently cite `Interface.py` inside code-pane context instead of describing the host-dispatch/ABI-slot behavior at the right layer.

### 2026-06-24 16:13:44 UTC - Broadened scope beyond HD256

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "235E4516-22DC-40CD-990A-A3B817FD5269", "trigger": "revisit" }
```

Corrected the task doc after the user pointed out that the audit should not be HD256-specific. Rechecked the upstream CUTE directory and recorded `flash_fwd_sm100.py`, `flash_bwd_sm100.py`, `flash_bwd_preprocess.py`, and `flash_bwd_postprocess.py` as broad forward/backward anchors. Kept the HD256 files as special-path references rather than the whole scope. No source code or LaTeX was changed.

### 2026-06-24 16:22:43 UTC - Completed audit and converted roadmap to fix tasks

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "07B19C2E-4A77-4D7A-B4B9-6E8AED8D3A6C", "trigger": "revisit" }
```

Completed the local/upstream audit before updating the roadmap. Checked the current upstream CUTE files for generic forward/backward and HD256-specific paths, then checked the local FA4 code and LaTeX refs for the mismatches listed in Knowledge. Also ran `python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py` on the current snippet asset; it passed. No source code or LaTeX was changed.

### 2026-06-24 16:54:54 UTC - Implemented FA4 CUTE completion tasks

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "0E7D633B-C986-4B55-A810-1C583DD59D57", "trigger": "roadmap" }
```

Completed `R1A9390-005` through `R1A9390-011` by local execution. Updated `flashAttention4CuteDsl.py` to make forward fragment creation, generic softmax stats, correction epilogue, generic backward, and dedicated HD256 dQ/dK/dV kernels source-shaped. Updated `flashAttention4.ts` so forward/backward rows and code refs point at the corrected generic and HD256-specific sections. Removed stale Knowledge that described current local mismatches as future work and replaced it with the completed state. Verification passed: `python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py`, Vite SSR/module-load for `flashAttention4.ts`, `git diff --check`, and `npm run build`.

### 2026-06-24 17:03:53 UTC - Added detailed-logic follow-up for HD256 sketches

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "6C77C573-95DF-4FD0-BE8E-F02DAA988C82", "trigger": "revisit" }
```

Updated the task doc only after user feedback that string pseudo-code such as `score_tmem = "S = Q @ K^T in TMEM"` is not acceptable. Added `R1A9390-012` to replace the remaining HD256 dedicated-kernel string placeholders with detailed logic. No source code or LaTeX was changed in this doc update.

### 2026-06-24 17:09:18 UTC - Added review findings as follow-up tasks

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "E3870114-7763-4A4C-8B60-0449E744F4CC", "trigger": "review" }
```

Reviewed the current diff against this task doc and found three remaining goal mismatches: HD256 dedicated kernels still use string pseudo-code, the forward softmax-stat handoff declares `softmax_stats_mbar_ptr` without showing a matching source-shaped pipeline, and the doc's completed-state wording could hide the reopened work. Updated Knowledge and Roadmap only; no source code or LaTeX was changed.

### 2026-06-24 17:21:32 UTC - Implemented detailed forward/backward follow-ups

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "DCE68553-8065-49F8-95B7-4F31C020213B", "trigger": "roadmap" }
```

Completed `R1A9390-012` and `R1A9390-013` by local execution. Rechecked upstream CUTE anchors for generic forward stats and dedicated HD256 backward kernels, then updated `flashAttention4CuteDsl.py` so generic forward uses a visible `softmax_stats_mbar_ptr`/`sScale` pipeline instead of a mismatched correction barrier, and so the dedicated HD256 dQ and dK/dV sketches replace string pseudo-code with source-shaped TMA/TMEM/SMEM mainloop and epilogue logic. Updated `flashAttention4.ts` so the LaTeX rows claim only the now-visible mechanisms. Verification is still tracked by `R1A9390-014`.

### 2026-06-24 17:23:08 UTC - Closed FA4 CUTE completion task

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "8E04E4EA-3284-455F-B8A9-0F22B5796D0D", "trigger": "roadmap" }
```

Completed `R1A9390-014` by reconciling this task doc and rerunning the final checks. Verification passed: `python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py`, Vite SSR/module-load for `flashAttention4.ts`, `git diff --check`, and `npm run build`. `npm run build` emitted the existing npm `min-release-age` config warning and Vite's large chunk warning, but exited successfully.

### 2026-06-24 17:26:34 UTC - Removed remaining visible string-note code

```jsonl
{ "agent": "codex-cli", "session_id": "codex-2026-06-24", "worklog_id": "38799A78-4BBF-482C-9A36-B5C6EE2142B0", "trigger": "roadmap" }
```

Tightened the final code pane after a stricter scan found two visible string-return notes. Replaced the HD256 forward `sSum` note with concrete producer/consumer calls and `store_hd256_sum_max` / `load_hd256_sum_max` helpers, replaced the paper/generic DSMEM return-string note with a bounded `generic_2cta_dsmem_repack` sketch, and changed the generic deterministic dQ semaphore from a descriptive string to an allocated semaphore tensor shape. Verification passed again: `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py`, Vite SSR/module-load for `flashAttention4.ts`, `git diff --check`, the no-string-placeholder `rg` scan, and `npm run build`. `npm run build` again emitted the existing npm `min-release-age` config warning and Vite's large chunk warning, but exited successfully.

## Archived

### R1A9390-001 - Complete the full forward code/prose path

```jsonl
{ "type": "roadmap", "id": "R1A9390-001", "deps": [], "parallels": [], "archived_at": "2026-06-24 16:22:43 UTC", "reason": "Superseded by the completed audit and split into concrete forward fix tasks.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Archived because this item was too broad and still carried audit-shaped work. Its actionable pieces are now covered by `R1A9390-005`, `R1A9390-006`, and `R1A9390-007`.

### R1A9390-002 - Complete generic and HD256 backward sketches

```jsonl
{ "type": "roadmap", "id": "R1A9390-002", "deps": [], "parallels": [], "archived_at": "2026-06-24 16:22:43 UTC", "reason": "Superseded by the completed audit and split into concrete backward fix tasks.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Archived because this item mixed generic SM100 and HD256 work in one oversized task. Its actionable pieces are now covered by `R1A9390-008`, `R1A9390-009`, and `R1A9390-010`.

### R1A9390-003 - Fix misleading comments and LaTeX wording

```jsonl
{ "type": "roadmap", "id": "R1A9390-003", "deps": ["R1A9390-001", "R1A9390-002"], "parallels": [], "archived_at": "2026-06-24 16:22:43 UTC", "reason": "Superseded by targeted wording and row-ref tasks.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Archived because the audit identified the exact wording layers to fix: CuTe/CUTLASS internals comments, HD256 host-dispatch wording, and placeholder-helper row refs. Those are now assigned to `R1A9390-005`, `R1A9390-009`, and `R1A9390-010`.

### R1A9390-004 - Verify code refs, parsing, and build output

```jsonl
{ "type": "roadmap", "id": "R1A9390-004", "deps": ["R1A9390-003"], "parallels": [], "archived_at": "2026-06-24 16:22:43 UTC", "reason": "Recreated as a final verification task after the concrete fix queue.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Archived because the old verification task depended on broad tasks that no longer exist. Verification is now represented by `R1A9390-011`.

### R1A9390-005 - Reconcile generic forward rows with source-shaped code

```jsonl
{ "type": "roadmap", "id": "R1A9390-005", "deps": [], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. The forward setup now explains CuTe/CUTLASS fragment creation, treats generic `q_stage` as an M-stage pipeline while preserving the HD256 divergence, and updates the LaTeX row refs around forward fragments.

### R1A9390-006 - Replace forward softmax and correction placeholders

```jsonl
{ "type": "roadmap", "id": "R1A9390-006", "deps": ["R1A9390-005"], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution, reopened by review, and completed again under `R1A9390-013`. The fake constant `load_softmax_stats()` path was replaced with an `sScale` row-sum/row-max handoff, and the final version uses visible `softmax_stats_mbar_ptr` producer/consumer state rather than overclaiming the correction barrier.

### R1A9390-007 - Make forward epilogue and HD256 forward divergence explicit

```jsonl
{ "type": "roadmap", "id": "R1A9390-007", "deps": ["R1A9390-006"], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. The compact output helpers were replaced with a source-shaped `correction_epilogue`, O tile store, LSE store, and an explicit HD256-forward stats note.

### R1A9390-008 - Add a separate generic SM100 backward section

```jsonl
{ "type": "roadmap", "id": "R1A9390-008", "deps": [], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. Added a generic SM100 backward map covering host dispatch, preprocess/main/postprocess, `dQaccum`, generic MMA atoms/fragments, dS, dK/dV, and dQ accumulation.

### R1A9390-009 - Fill dedicated HD256 dQ and dK/dV kernel details

```jsonl
{ "type": "roadmap", "id": "R1A9390-009", "deps": ["R1A9390-008"], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution, reopened by review, and completed again under `R1A9390-012`. The dedicated HD256 dQ and dK/dV sketches now build source-shaped TMA atoms, allocate shared/TMEM tiles, run explicit `cute.gemm` score/dP/dS/dQ and score/dP/dS/dK/dV flows, and drain accumulators through concrete epilogue staging helpers.

### R1A9390-010 - Rewrite host-dispatch and LaTeX wording for HD256

```jsonl
{ "type": "roadmap", "id": "R1A9390-010", "deps": ["R1A9390-009"], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. Reworded the HD256 dQ path around the historical `dQ_accum` ABI slot, removed `Interface.py`-style code-pane wording, rejected generic semaphore/postprocess implications, and moved DSMEM exchange into a paper/generic note.

### R1A9390-011 - Verify refs, parser behavior, and build after fixes

```jsonl
{ "type": "roadmap", "id": "R1A9390-011", "deps": ["R1A9390-007", "R1A9390-010"], "parallels": [], "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. Verification passed: `python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py`, Vite SSR/module-load for `flashAttention4.ts`, `git diff --check`, and `npm run build`.

### R1A9390-012 - Replace HD256 string pseudo-code with detailed kernel logic

```jsonl
{ "type": "roadmap", "id": "R1A9390-012", "created_at": "2026-06-24 17:03:53 UTC", "deps": [], "parallels": [], "archived_at": "2026-06-24 17:21:32 UTC", "reason": "Implemented source-shaped HD256 dedicated dQ and dK/dV kernel logic.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. The dedicated HD256 dQ and dK/dV sketches no longer use descriptive string assignments for TMA descriptors, TMEM products, or epilogue staging. They now expose source-shaped TMA copy atoms, shared-memory and TMEM tile allocation, `cute.gemm` mainloop products, dSoftmax helpers, TMA-backed epilogue staging/copy helpers, and a bounded generic DSMEM exchange sketch outside the current HD256 wrapper path.

### R1A9390-013 - Source-shape the forward softmax-stat handoff

```jsonl
{ "type": "roadmap", "id": "R1A9390-013", "created_at": "2026-06-24 17:09:18 UTC", "deps": [], "parallels": [], "archived_at": "2026-06-24 17:21:32 UTC", "reason": "Implemented a visible softmax-stat pipeline around mbar_softmax_stats and sScale.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. The generic forward sketch now allocates `softmax_stats_mbar_ptr` and `sScale`, creates distinct `softmax_stats_producer`/`softmax_stats_consumer` pipeline state, stores row sums/maxima through `store_softmax_stats(sScale, stage, ...)`, and loads them in correction through `load_softmax_stats(sScale, stage)`. The LaTeX row now claims the same `sScale` plus `mbar_softmax_stats` handoff shown in code, while HD256's separate `sSum` path remains distinct and is shown as explicit producer/consumer calls.

### R1A9390-014 - Reconcile completion state and rerun final verification

```jsonl
{ "type": "roadmap", "id": "R1A9390-014", "created_at": "2026-06-24 17:09:18 UTC", "deps": ["R1A9390-012", "R1A9390-013"], "parallels": [], "archived_at": "2026-06-24 17:23:08 UTC", "reason": "Reconciled task doc completion state and reran final verification.", "agent": "codex-cli", "session_id": "codex-2026-06-24" }
```

Completed by local execution. Knowledge now describes the final source state, reopened task notes point at their closing follow-ups, Roadmap has no active implementation tasks, and final verification passed: `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile src/features/cheat-sheet-view/data/subPages/code/flashAttention4CuteDsl.py`, Vite SSR/module-load for `flashAttention4.ts`, `git diff --check`, the no-string-placeholder `rg` scan, and `npm run build`.

## Roadmap

No active tasks. The FA4 CUTE forward/backward completion slice is implemented, verified, and archived.
