<div align="center">
  <a>
    <img src="https://github.com/ChimeraMetta/Chimera/blob/main/assets/header.jpg?raw=true" alt="Logo">
  </a>

  <h2 align="center">Chimera Design</h2> <div style="height:30px"></div>
</div>

Chimera has been designed as a combination of 2 core ideas:

1. Reasoning and deduction are best done through symbolic logic (MeTTa)
2. Ideation and inference are best done through statistical methods (LLMs)

By leveraging each wherever it's best suited, we can create something that is more than the sum of its parts. Chimera is an attempt to do this specifically in the domain of programming

--

## Ontology

Reasoning about code is best done by understanding the relationship of components within it. Things like dependencies, imports and calls create a full ontology of a code base that can then be used to reason on it logically. Chimera utilises:

1. Some base ontology rules for how to deal with Python code
2. Ontologies that are generated from a specific codebase, the one you're working on

This allows MeTTa to get a "full picture" of its environment and then reason about it.

--

## Proofs

The problem with utilising LLMs for generating new candidates for functions or pieces of code is that LLMs are really bad at reasoning. We need a system that allows MeTTa to guide an LLM into figuring it if a particular candidate will work, and for this Chimera uses a proof system. Proofs for a function generally break down into:

1. Pre-conditions: Conditions required of the inputs
2. Post-conditions: Conditions required of the outputs
3. Loop invariants

For these we use a custom JSON intermediate representation of a function, which is then parsed through MeTTa and used to figure out if a particular candidate function matches the desired outcomes.

--

## Cybersecurity Domain Application

Chimera's hybrid reasoning architecture extends beyond code analysis to domain-specific knowledge systems. The cybersecurity query module demonstrates this by combining:

1. **MeTTa ontology as single source of truth** — threats, vulnerabilities, mitigations, attack vectors, and their relationships are all defined in `metta/cybersecurity_ontology.metta`. Adding a new entity to the `.metta` file makes it automatically discoverable.
2. **Sentence-transformer embeddings for NL understanding** — intent classification uses cosine similarity against pre-embedded exemplar phrases, handling arbitrary paraphrased queries without regex patterns.
3. **Deductive reasoning via MeTTa rules** — the ontology contains rules like `(= (mitigations-for $threat) ...)` that Chimera evaluates to answer multi-hop queries about threat relationships and mitigation chains.
