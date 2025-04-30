<div align="center">
  <a>
    <img src="https://github.com/ChimeraMetta/Chimera/blob/main/assets/header.jpg?raw=true" alt="Logo">
  </a>

  <h2 align="center">Chimera</h2> <div style="height:30px"></div>

  <p align="center">
Chimera is an agent designed to operate on its own, without human intervention required
    <br />
    <br />
  </p>
</div>

It's an evolutionary system that combines the symbolic reasoning of MeTTa with statistical LLM inference, producing a 
high performance, knowledgeable agent capable of providing deeper insights and more accurate changes to your context. The driving force 
of this project is to create this agent in such a way that it requires no human intervention and can make these changes completely autonomously.

This is *not* an agent that needs you to click "Accept" or approve changes.

## Phase 1

The first phase for Chimera is an ontology and proof system for codebases. This means it's able to read through your codebase and convert all Python 
code into a symbolic representation, including the relationships between your functions, classes and dependencies and how they interact. This allows it 
to achieve a compressed context of your code that goes beyond what an LLM is capable of on its own. 

It is also able to generate proofs for your functions, including pre-conditions, loop invariants and post-conditions. Chimera can read your function, 
understand the intent completely, and then make improvements, fixes and optimizations autonomously, ensuring that the changes constantly conform to what 
you intended as the developer