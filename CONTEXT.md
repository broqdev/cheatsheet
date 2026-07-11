# Broq Cheatsheet

This context describes the study surface that keeps paper-style equations and their implementation path synchronized.

## Language

**Example**:
A named algorithm entry that supplies one or more equation-to-code documents.
_Avoid_: Page, demo

**Variant**:
A selectable change to an Example, such as causal masking, momentum, or weight decay.
_Avoid_: Toggle, option

**Equation-to-code document**:
A compiled pairing of algorithm blocks, rendered equations, and referenced code lines for one Example state.
_Avoid_: Content blob, page data

**Algorithm block**:
An ordered paper-style section with a requirement and equation rows.
_Avoid_: Card, panel

**Code reference**:
A named marker that links an equation row to lines in an implementation sketch.
_Avoid_: Anchor, line range

**Catalog**:
The ordered registration of Examples used for routing, search, and related-example navigation.
_Avoid_: Menu, registry

## Relationships

- The **Catalog** contains every **Example** exactly once.
- An **Example** has one base **Equation-to-code document** and may have multiple **Variants**.
- An **Equation-to-code document** contains one or more **Algorithm blocks**.
- An equation row uses zero or more **Code references**.

## Example dialogue

> **Dev:** "Should Nesterov be another Example in the Catalog?"
> **Domain expert:** "No. Nesterov is a Variant of the SGD Example, so it resolves a different Equation-to-code document without changing the Catalog entry."

## Flagged ambiguities

- "toggle" previously meant both a control and its algorithmic effect; use **Variant** for the effect and "control" only for the rendered input.
