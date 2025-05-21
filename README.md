<div align="center">
  <a>
    <img src="https://github.com/ChimeraMetta/Chimera/blob/main/assets/header.jpg?raw=true" alt="Logo">
  </a>

  <h2 align="center">Chimera</h2> <div style="height:30px"></div>

  <p align="center">
Autonomous code evolution through symbolic reasoning and LLM inference
    <br />
    <br />
  </p>
</div>

Chimera is a research-driven software agent that autonomously evolves codebases using a hybrid reasoning engine, combining symbolic logic and statistical language models. Inspired by human-level programming workflows, Chimera can read, reason about, modify, and test software—learning and adapting iteratively.

This project aims to answer a bold question:

> Can a machine reason about and improve its own source code, autonomously, without predefined goals or human intervention?

Chimera bridges the gap between symbolic systems (using [MeTTa](https://metta-lang.dev/docs/learn/learn.html), theorem provers, and AST reasoning) and generative AI (e.g., GPT, Claude) to create a closed feedback loop for code understanding, editing, and validation.

--

## Key Features

Chimera intends to implement the following features:

- 🔄 **Autonomous Evolution Loop:** Reads code, hypothesizes changes, tests outcomes, and learns from failure.
- 🧩 **Symbolic Reasoning Layer:** Uses logic and pattern-matching to identify structure, invariants, and design flaws.
- 📚 **LLM-Augmented Inference:** Leverages language models to generate new code, fix errors, and explain intentions.
- 🧪 **Test-Aware Mutation:** Integrates with unit and integration tests to validate hypotheses and regressions.
- 🧠 **Memory and Self-Reflection:** Tracks past decisions and adapts future iterations accordingly.

--

## Current Features

For its current phase, Chimera is able to statically analyze your code symbolically. It can be invoked through 

```sh
chimera summary file_to_summarize.py
```

which will provide a full summary of the relational ontology of your files and folders

// insert image

You can also ask Chimera to improve your code by looking for complexity in the ontology and generate alternative 
candidates that simplify and improve its structure using

```sh
chimera analyze file_to_analyze.py --api_key=OPENAI_API_KEY
```

This feature uses an LLM to generate alternative candidates that preserve the structure of your functions and their intent.

// insert next image

-- 

## Next Phase

The next phase will be the implementation of the current features in an autonomous way to improve and heal code autonomously, in a live environment!