import type { AlgorithmLine, AttentionContent, LatexBlock } from '../model'

export type AlgorithmLineSpec = Omit<AlgorithmLine, 'codeLines'> & {
  codeRefs?: string[]
  codeLines?: number[]
}

export type LatexBlockSpec = Omit<LatexBlock, 'rows'> & {
  rows: AlgorithmLineSpec[]
}

export type AttentionContentSpec = Omit<AttentionContent, 'code' | 'rows' | 'prelude' | 'notes'> & {
  rawCode: string
  rows: AlgorithmLineSpec[]
  prelude?: LatexBlockSpec[]
  notes?: LatexBlockSpec[]
  ignoredUnusedRefs?: string[]
}

type ParsedCodeRefs = {
  code: string
  refIndex: Map<string, number[]>
}

const refStartPattern = /^\s*#\s*@ref\s+(.+?)\s*$/
const refEndPattern = /^\s*#\s*@end\s*$/

function parseRefNames(value: string) {
  return Array.from(new Set(value.split(/\s+/).filter(Boolean)))
}

function appendLine(refIndex: Map<string, number[]>, ref: string, lineNumber: number) {
  const lines = refIndex.get(ref) ?? []
  lines.push(lineNumber)
  refIndex.set(ref, lines)
}

function parseCodeRefs(rawCode: string): ParsedCodeRefs {
  const refIndex = new Map<string, number[]>()
  const codeLines: string[] = []
  let activeRefs: string[] | null = null
  const normalizedCode = rawCode.replace(/\r\n?/g, '\n').replace(/\n$/, '')

  for (const rawLine of normalizedCode.split('\n')) {
    const refStart = rawLine.match(refStartPattern)

    if (refStart) {
      if (activeRefs) {
        throw new Error(`Nested code refs are not supported near "${rawLine.trim()}".`)
      }

      const refs = parseRefNames(refStart[1] ?? '')

      if (refs.length === 0) {
        throw new Error(`Code ref marker is missing a ref name near "${rawLine.trim()}".`)
      }

      activeRefs = refs
      continue
    }

    if (refEndPattern.test(rawLine)) {
      if (!activeRefs) {
        throw new Error(`Found @end without an active code ref near "${rawLine.trim()}".`)
      }

      activeRefs = null
      continue
    }

    codeLines.push(rawLine)

    if (activeRefs) {
      const lineNumber = codeLines.length

      for (const ref of activeRefs) {
        appendLine(refIndex, ref, lineNumber)
      }
    }
  }

  if (activeRefs) {
    throw new Error(`Unclosed code ref block "${activeRefs.join(' ')}".`)
  }

  return {
    code: codeLines.join('\n'),
    refIndex,
  }
}

function uniqueLines(lines: number[]) {
  return Array.from(new Set(lines))
}

function resolveLine(
  line: AlgorithmLineSpec,
  refIndex: Map<string, number[]>,
  usedRefs: Set<string>
): AlgorithmLine {
  const refLines = (line.codeRefs ?? []).flatMap((ref) => {
    const lines = refIndex.get(ref)

    if (!lines?.length) {
      throw new Error(`Unknown code ref "${ref}" in row "${line.id}".`)
    }

    usedRefs.add(ref)
    return lines
  })

  return {
    ...line,
    codeLines: uniqueLines([...(line.codeLines ?? []), ...refLines]),
  }
}

function resolveNote(
  note: LatexBlockSpec,
  refIndex: Map<string, number[]>,
  usedRefs: Set<string>
): LatexBlock {
  return {
    ...note,
    rows: note.rows.map((line) => resolveLine(line, refIndex, usedRefs)),
  }
}

function warnUnusedRefs(
  refIndex: Map<string, number[]>,
  usedRefs: Set<string>,
  ignoredUnusedRefs: string[] = []
) {
  const ignoredRefs = new Set(ignoredUnusedRefs)
  const unusedRefs = Array.from(refIndex.keys()).filter(
    (ref) => !usedRefs.has(ref) && !ignoredRefs.has(ref)
  )

  if (unusedRefs.length > 0 && import.meta.env.DEV) {
    console.warn(`Unused code refs: ${unusedRefs.join(', ')}`)
  }
}

export function defineAttentionContent(spec: AttentionContentSpec): AttentionContent {
  const parsedCode = parseCodeRefs(spec.rawCode)
  const usedRefs = new Set<string>()
  const rows = spec.rows.map((line) => resolveLine(line, parsedCode.refIndex, usedRefs))
  const prelude = spec.prelude?.map((note) => resolveNote(note, parsedCode.refIndex, usedRefs))
  const notes = spec.notes?.map((note) => resolveNote(note, parsedCode.refIndex, usedRefs))

  warnUnusedRefs(parsedCode.refIndex, usedRefs, spec.ignoredUnusedRefs)

  return {
    code: parsedCode.code,
    require: spec.require,
    rows,
    prelude,
    notes,
  }
}
