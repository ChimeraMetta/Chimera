# **Querying space content**

# [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/query_knowledge.html#querying-space-content)

## **Introduction**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/query_knowledge.html#introduction)

As a declarative language, MeTTa was designed for expressing complex relationships between entities of various types, performing computations on these relationships, and manipulating their structures. It allows programmers to specify AI algorithms and knowledge representations in a rich and flexible way. MeTTa code can be generated and processed in run-time by MeTTa programs themselves, which adds a lot of dynamism in working with complex data structures for AI tasks.

One of the main purposes of developing MeTTa was to operate over a knowledge metagraph called AtomSpace (or just Space), designed to store all sorts of knowledge, from raw sensory/motor data to linguistic and cultural knowledge, to abstract, mathematical, scientific or programming knowledge.

AtomSpace represents knowledge in the form of Atoms, the fundamental building block of all the data. Specifically, in the context of AI, an Atom can represent anything from objects, to concepts, to processes or relationships, to reasoning rules and algorithms.

While MeTTa may look like an ordinary language in certain aspects, it is built on top of operations over the knowledge metagraph, which is essential to understand how it works.

## **Knowledge declaration and matching query**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/query_knowledge.html#knowledge-declaration-and-matching-query)

Let us look at a basic example of specifying relations between concepts, e.g., family relationships. While there are different ways to do this, in MeTTa, one can simply put expressions like the following into the program

metta

(Parent Tom Bob)

This expression being put into the program space can be treated as the fact that Tom is Bob's parent. We start Parent with capital P to distinguish it from a function, which we would prefer to start with p in this case, although this naming convention is not mandatory.

One can add more such expressions to the program space. But what can we do with such expressions? The [tutorial](https://metta-lang.dev/docs/learn/tutorials/eval_intro/intro.html) overviewed the evaluation process of expressions, for which equalities are specified. But is there any use of expressions without equalities?

The core operation in MeTTa is matching. It searches for all declared atoms corresponding to the given pattern and produces the output pattern. The process is similar to the manner in which one can search text strings with regular expressions, but it is for searching for subgraphs in a metagraph.

We can compose a query for matching using the grounded function match. It expects three arguments:

* a grounded atom referencing a Space;  
* pattern atom to be matched;  
* output pattern typically containing variables from the input pattern.

## **Basic examples**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/query_knowledge.html#basic-examples)

Let us consider the following program

metta

(Parent Bob Ann)  
; This match will be successful  
\! (match \&self (Parent Bob Ann) (Bob is Ann\`s father))  
; The following line will return \[\]

\! (match \&self (Parent Bob Joe) (Bob is Joe\`s father))

Run

\&self is a reference to the current program Space. We can refer to other Atomspaces, but we will cover it later. The second argument in the first match expression (Parent Bob Ann) is an expression to be matched against atoms in the current Space, and the third argument (Bob is Ann's father) is the atom to be returned if matching succeeded.

The program above will return \[(Bob is Ann's father)\] and \[\], since when the desired expression pattern wasn't found match returns nothing.

We can construct more interesting queries using variables. Let us consider the program

metta

(Parent Tom Bob)  
(Parent Pam Bob)  
(Parent Tom Liz)  
(Parent Bob Ann)

\! (match \&self (Parent $x Bob) $x) ; \[Tom, Pam\]

Run

The pattern (Parent $x Bob), i.e. "Who are Bob's parents?", can be matched against two atoms (facts) in the Space, and corresponding bindings for $x will be used to produce the result of match. Here, we will get two matches \[Tom, Pam\], which can be viewed as a nondeterministic evaluation of match.

Please, note that match doesn't search in subexpressions. The following code will return \[Ann\] only:

metta

(Parent Bob Ann)  
(Parent Pam (Parent Bob Pat))

\! (match \&self (Parent Bob $x) $x) ; Ann

Run

We can make even broader queries: "Who is a parent of whom?", or "Find $x and $y such that $x is a parent of $y".

metta sandbox

(Parent Tom Bob)

(Parent Pam Bob)

(Parent Tom Liz)

(Parent Bob Ann)

(Parent Bob Pat)

(Parent Pat Pat)

\! (match \&self (Parent $x $y) ($x $y))

Run

The output should contain the following pairs (the order can be different due to MeTTa's nondeterminism) \[(Pat Bob), (Bob Ann), (Bob Pat), (Tom Bob), (Tom Liz), (Pat Pat)\]. Can you add the query in the above program to retrieve only parents and children with same names?

Pager

