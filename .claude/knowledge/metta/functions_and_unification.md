# **Functions and unification**

# [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/unify_func.html#functions-and-unification)

## **Function evaluation and matching**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/unify_func.html#function-evaluation-and-matching)

As discussed in the [tutorial](https://metta-lang.dev/docs/learn/tutorials/eval_intro/intro.html), evaluable expressions can contain variables, and they are pattern-matched against left-hand side of equalities. In fact, evaluation of expressions can be understood as recursively constructing queries for equalities. Consider this code as an example

metta

(\= (only-a A) (Input A is accepted))  
\! (only-a A)  
\! (only-a B)

\! (only-a $x)

Run

Evaluation of (only-a A) can be thought of as execution of query (match \&self (= (only-a A) $result) $result). $result will be bound with the right-hand side of the function case (body), if the left-hand side matches with the expression under evaluation. Does it work for (only-a B) and (only-a $x)?

Let us check that the following program produces the same result:

metta

(\= (only-a A) (Input A is accepted))  
\! (match \&self (\= (only-a A) $result) $result)  
\! (match \&self (\= (only-a B) $result) $result)

\! (match \&self (\= (only-a $x) $result) $result)

Run

There is one difference. match produces the empty result in the second case, while the interpreter keeps this expression unreduced. The interpreter is performing some additional processing on top of such equality queries.

While allowing the MeTTa interpreter to construct equality queries automatically for evaluating expressions like (only-a A) is very convenient for functional programming, using match directly allows for more compact knowledge representation and efficient queries glued together in a custom way.

It should also be noted that obtaining multiple results in queries to knowledge bases is very typical, and since the semantics of evaluating expressions in MeTTa is natively related to such queries, all evaluations in MeTTa are secretly or explicitly nondeterministic.

Let us analyze the following program:

metta

(Parent Tom Bob)  
(Parent Pam Bob)  
(Parent Tom Liz)  
(Parent Bob Ann)  
(\= (get-parent-entries $x $y)  
   (match \&self (Parent $x $y) (Parent $x $y)))  
(\= (get-parents $x)  
   (match \&self (Parent $y $x) $y))  
\! (get-parent-entries Tom $\_)

\! (get-parents Bob)

Run

* We can call match from an ordinary function, and we can still pass variable arguments to it, so (get-parent-entries Tom $\_) is equivalent to (match \&self (Parent Tom $y) (Parent Tom $y)).  
* The result \[(Parent Tom Liz), (Parent Tom Bob)\] is not reduced further. It is convenient, when we want to represent pieces of knowledge and process them.  
* (get-parents Bob) returns \[Tom, Pam\]. Executing match from functions allows creating convenient functional abstractions while still working with declarative knowledge.

For example, how would you write a function, which returns grandparents of a given person?

metta sandbox

(Parent Tom Bob)

(Parent Pam Bob)

(Parent Tom Liz)

(Parent Bob Ann)

(Parent Bob Pat)

(Parent Pat Jim)

(\= (get-parents $x)

   (match \&self (Parent $y $x) $y))

(\= (get-grand-parents $x)

   (...))

\! (get-grand-parents Pat)

Run

## **From facts to rules**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/unify_func.html#from-facts-to-rules)

One may notice that equality queries for functions suppose that there are free variables not only in the query, but also in the Atomspace entries. These entries can be not only function definitions, but other arbitrary expressions, which can be used to represent general knowledge or rules. For example, one can write

metta

(Parent Tom Bob)  
(Parent Bob Ann)  
(Implies (Parent $x $y) (Child $y $x))  
(\= (deduce $B)  
   (match \&self (Implies $A $B)  
          (match \&self $A $B))  
)  
(\= (conclude $A)  
   (match \&self (Implies $A $B)  
          (match \&self $A $B))  
)  
\! (deduce (Child $x Tom))    ; \[(Child Bob Tom)\]

\! (conclude (Parent Bob $y)) ; \[(Child Ann Bob)\]

Run

If Child and Parent were predicates returning True or False (as in the [frog example](https://metta-lang.dev/docs/learn/tutorials/eval_intro/recursion_vars.html#a-piece-of-logic)), we could somehow use \= instead of Implies. But here we don't evaluate the premise to True or False, but check that it is in the knowledge base. It makes inference better controllable. We can easily go from premises to conclusions with conclude, or to verify conclusions by searching for suitable premises with deduce.

We will discuss different ways of introducing reasoning in MeTTa in more detail later. What we want to focus on now is that in both cases a query with variables is constructed, say, (Implies (Parent Bob $y) $B) and it should be matched against some entry in the knowledge base with variables as well, namely, (Implies (Parent $x $y) (Child $y $x)) in our example. This operation is called unification, and it is available in MeTTa in addition to match.

## **Unification**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/unify_func.html#unification)

Function unify accepts two patterns to be unified (matched together in such the way that shared variables in them get most general non-contradictory substitutions). The function is evaluated to its third argument if unification is successful and to the fourth argument otherwise. The following program shows the basic example.

metta

\! (unify (parent $x Bob) ; the first pattern  
         (parent Tom $y) ; the second pattern  
         ($x $y) ; the output for successful unification

         Fail) ; fallback

Run

Here, we unify two expressions (parent $x Bob) and (parent Tom $y), and return a tuple ($x $y) if unification succeeded. The Fail atom will be returned if there are no matches. Note that (unify (A $x) ($x B) Yes No) will be reduced to No, because $x should have the same binding in both patterns (and it cannot be A and B simultaneously).

One of the first two arguments can be a reference to a Space as well. In this case, it will work like match but with an alternative option in the case of failed matching:

metta

(Parent Tom Bob)  
(Parent Bob Ann)

\! (unify \&self (Parent $x Bob) $x Fail) ; \[Tom\]

Run

Here, we pass a reference to the current Space as the first argument, so the second expression (parent $x Bob) is matched against the whole set of declared knowledge.

## **Chained unification**

## [​](https://metta-lang.dev/docs/learn/tutorials/ground_up/unify_func.html#chained-unification)

Let us analyze how (conclude (Parent Bob $y)) from the above example is evaluated.

* At first, (match \&self (= (Parent Bob $y) $result) $result) is executed to evaluate the subexpression. But this query returns no result, because equalities for Parent are not defined. Thus, (Parent Bob $y) remains unreduced.  
* Thus, the equality query for the whole expression (match \&self (= (conclude (Parent Bob $y)) $result) $result) is executed. The following two expressions (one is the query and another one is from Space) are unifiable:

metta

(\= (conclude (Parent Bob $y))  
   $result)  
(\= (conclude       $A)  
   (match \&self (Implies $A $B)

          (match \&self $A $B)))

$A will be bound to (Parent Bob $y), and $result will be

metta

(match \&self (Implies (Parent Bob $y) $B)

             (match \&self (Parent Bob $y) $B))

* match is executed directly as a grounded function (otherwise another equality query would be constructed) with (Implies (Parent Bob $y) $B) as a query. It unifies with the following entry in the Space:

metta

(Implies (Parent Bob $y)      $B      )

(Implies (Parent $x  $y) (Child $y $x))

One may notice that there could be some collisions of variable names, and the interpreter should deal with this. In overall, $x gets bound to Bob, and $B gets bound to (Child $y Bob). Since the output of this match is (match \&self (Parent Bob $y) $B), the expression for further evaluation becomes (match \&self (Parent Bob $y) (Child $y Bob)).

* (Parent Bob $y) unifies with (Parent Bob Ann) yielding (Child Ann Bob)  
* Query (= (Child Ann Bob) $result) finds no matches, so (Child Ann Bob) is the final result.

The overall chain of transformations in the course of interpretation can be viewed as:

metta

1\. (conclude (Parent Bob $y))  
2\. (match \&self (Implies (Parent Bob $y) $B)  
          (match \&self (Parent Bob $y) $B))  
3\. (match \&self (Parent Bob $y) (Child $y Bob))

4\. (Child Ann Bob)

These are not all the steps done by the interpreter, but they give the overall picture of what is really going on under the hood.

Pager

