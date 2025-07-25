# **Nested queries and recursive graph traversal**

# [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/nested_queries.html#nested-queries-and-recursive-graph-traversal)

## **Composite queries**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/nested_queries.html#composite-queries)

We've already seen queries for conclude and deduce, which result is another query. At the same time, chaining of queries can be done in a more functional style with equalities as it could be done for

metta

(\= (get-grand-parents $x)

   ((get-parents (get-parents $x))))

Keeping knowledge declarative can be useful for implementing reasoning over it.

Imagine that we add more info on people like (Female Pam) or (Male Tom) into the knowledge base, and want to define more relations such as sister. One can turn facts into equalities like (= (Female Pam) True) and use functional logic (as in the [frog example](https://metta-lang.dev/docs/learn/tutorials/eval_intro/recursion_vars.html#a-piece-of-logic)), but let us keep simple facts for now.

One can add more functions like get-parents. A function for female would be more convenient to represent as a filter, e.g. (= (female $x) (match \&self (Female $x) $x)), so it will be composable, e.g. (= (get-mother $x $y) (female (get-parents $x $y))).

One can do this by a composite query instead.

metta sandbox

(Female Pam)

(Male Tom)

(Male Bob)

(Female Liz)

(Female Pat)

(Female Ann)

(Male Jim)

(Parent Tom Bob)

(Parent Pam Bob)

(Parent Tom Liz)

(Parent Bob Ann)

(Parent Bob Pat)

(Parent Pat Jim)

(\= (get-sister $x)

   (match \&self

     (, (Parent $y $x)

        (Parent $y $z)

        (Female $z))

     $z

   )

)

\! (get-sister Bob)

Run

Composite queries contain a few patterns (united by , into one expression), which should be satisfied simultaneously. Such queries can be efficient if the Atomspace query engine efficiently processes joints. This can be important for large knowledge bases. Otherwise, it is necessary to be careful about the order of nested queries of filters. For example, having (Female $z) with free variable $z as the innermost functional call or (match \&self (Female $z) ...) as the outermost query in a nested sequence of queries will be highly inefficient, because it will first extract all the females from the knowledge base, and only then will narrow down the set of the results.

Notice that the above program is imprecise. How can the mistake be fixed (check the sister of Liz \- one option would be to introduce (different $x $y) as a filter)? You can try implementing other relations like uncle in the above program or rewrite it in a functional way. Typically, we would like to represent such concepts using other derived concepts rather than monolithic composite queries (e.g. "Uncle is a brother of a parent" rather than "Uncle is a male child of a parent of a parent, but not the parent").

## **Recursion for graph traversal**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/nested_queries.html#recursion-for-graph-traversal)

Let us define the predecessor relation:

For any \`$x\` and \`$z\`: \`$x\` is a predecessor of \`$z\`    
    if there is \`$y\` such that    
       \`$y\` is a parent of \`$z\` and    
       \`$x\` is a predecessor of \`$y\`  

Recursion is a convenient way to represent such relations.

metta

(Parent Tom Bob)  
(Parent Pam Bob)  
(Parent Tom Liz)  
(Parent Bob Ann)  
(Parent Bob Pat)  
(Parent Pat Jim)  
(Parent Jim Lil)

(\= (parent $x $y) (match \&self (Parent $x $y) $x))  
(\= (predecessor $x $z) (parent $x $z))  
(\= (predecessor $x $z) (predecessor $x (parent $y $z)))  
; Who are predecessors of Lil

\! (predecessor $x Lil)

Run
