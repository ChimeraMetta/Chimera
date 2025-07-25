# **Control flow**

# [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/control_flow.html#control-flow)

MeTTa has several specific constructs that allow a program to execute different parts of code based either on pattern matching or logical conditions.

## **if**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/control_flow.html#if)

if was already covered in [this tutorial](https://metta-lang.dev/docs/learn/tutorials/types_basics/metatypes.html#if-under-the-hood). But let us recap it as a part of stdlib.

The if statement implementation in MeTTa can be the following function

metta

(: if (\-\> Bool Atom Atom $t))  
(\= (if True $then $else) $then)

(\= (if False $then $else) $else)

Here, the first argument (condition) is Bool, which is evaluated before executing the equality-query for if.

The next two arguments are not evaluated and returned for the further evaluation depending on whether the first argument is matched with True or False.

The basic use of if in MeTTa is similar to that in other languages:

metta

(\= (foo $x)  
   (if (\>= $x 0)  
       (\+ $x 10)  
       (\* $x \-1)  
   )  
)  
\! (foo 1)  ; 11

\! (foo \-9) ; 9

Run

Here we have a function foo that adds 10 to the input value if it's grater or equal 0, and multiplies the input value by \-1 otherwise. The expression (\>= $x 0\) is the first argument of the if function, and it is evaluated to a Bool value. According to that value the expression (+ $x 10\) or (\* $x \-1) is returned for the final evaluation, and we get the result.

In contrast to other languages, one can pass a variable to if and it will be matched against equalities with both True and False. Consider the following example

metta

\! (if $x (\+ 6 1) (\- 7 2))  
(\= (foo $b $x)  
   (if $b  
       (\+ $x 10)  
       (\* $x \-1)  
   )  
)

\! ((foo $b 1) $b) ; \[(-1 False), (11 True)\]

Run

foo accepts the condition for if, and when we pass a variable, both branches are evaluated with the corresponding binding for $b.

if can also remain unreduced:

metta

\! (if (\> $x 0) (\+ $x 5) (\- $x 5))

Run

In this expression, (\> $x 0\) remains unreduced. Its overall type is Bool, but it can't be directly matched against neither True nor False. Thus, no equality is applied.

## **let**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/control_flow.html#let)

let has been briefly described [in another tutorial](https://metta-lang.dev/docs/learn/tutorials/types_basics/match_control.html#let-s-evaluate). Here, we will recap it.

The let function is utilized to establish temporary variable bindings within an expression. It allows introducing variables, assign values to them, and then use these values within the scope of the let block.

Once the let block has run, these variables cease to exist and any previous bindings are re-established. Depending on the interpreter version, let can be either a basic grounded function, or be implemented using other primitives. Let us consider its type

metta

\! (get-type let)

Run

The first argument of let is a pattern of Atom type, which is not evaluated. The second argument is the value, which is reduced before being passed to let The third parameter is an Atom again. An attempt to unify the first two arguments is performed. If it succeeds, the found bindings are substituted to the third argument, which is then evaluated. Otherwise, the empty result is returned.

Consider the following example:

metta

(\= (test 1) 1)  
(\= (test 1) 0)  
(\= (test 2) 2)

\! (let $W (test $X) (println\! ("test" $X \=\> $W)))

Run

The code above will print:

metta

("test" 1 \=\> 1)  
("test" 1 \=\> 0)

("test" 2 \=\> 2)

and return three unit results produced by println\!. It can be seen that variables from both the first and the second arguments can appear in the third argument.

The following example shows the difference between the first two arguments.

metta

(\= (test 1) 2)  
\! (let 2 (test 1) YES) ; YES

\! (let (test 1) 2 NO ) ; empty

Run

In case of (let 2 (test 1\) YES), (test 1\) is evaluated to 2, and it can be unified with the first argument, which is also 2. In case of (let (test 1\) 2 NO), (test 1\) is not reduced, and it cannot be unified (as a pattern) with 2, so the overall result is empty.

This example also shows that variables are not mandatory in let. What is needed is the possibility to unify the arguments. This allows using let for chaining operations, and this chaining can be conditional if the first operation returns some value, e.g.

metta

(\= (is-frog Sam) True)  
(\= (print-if-frog $x)  
   (let True (is-frog $x)  
     (println\! ($x is frog\!))))  
\! (print-if-frog Sam) ; ()

\! (print-if-frog Ben) ; empty

Run

Another basic use of let is to calculate values for passing them to functions accepting arguments of Atom type, for example:

metta

(Sam is 34 years old)  
\! (match \&self ($who is (\+ 20 14) years old) $who) ; empty  
\! (let $r (\+ 20 14)

   (match \&self ($who is $r years old) $who)) ; Sam

Run

Since the first argument can be not only a variable or a concrete value, but also an expression, let can be used for deconstructing expressions

metta

(\= (fact Sam) (age 34))  
(\= (fact Sam) (color green))  
(\= (fact Tom) (age 14))  
\! (let (age $r) (fact $who)

   ($who is $r)) ; \[(Tom is 14), (Sam is 34)\]

Run

The branches not corresponding to the (age $r) pattern are filtered out.

## **let\***

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/control_flow.html#let-1)

When several consecutive substitutions are required, let\* can be used for convenience. The first argument of let\* is Expression, which elements are the required substitutions, while the second argument is the resulting expression. In the following example, several values are subsequently calculated, and let\* allows making it more readable (notice also how pattern matching helps to calculate minimum and maximum values together with their absolute difference in one if).

metta

(Sam is 34)  
(Tom is 14)  
(\= (person-by-age $age)  
   (match \&self ($who is $age) $who))  
(\= (persons-of-age $a $b)  
   (let\* ((($age-min $age-max $diff)  
           (if (\< $a $b)  
               ($a $b (\- $b $a))  
               ($b $a (\- $a $b))))  
          ($younger (person-by-age $age-min))  
          ($older   (person-by-age $age-max))  
         )  
         ($younger is younger than $older by $diff years))  
)

\! (persons-of-age 34 14)

Run

Another case, for which let\* can be convenient, is the consequent execution of side-effect functions, e.g.

metta

(Sam is 34)  
(\= (age++ $who)  
   (let\* (($age (match \&self ($who is $a) $a))  
          ( ()  (println\! (WAS: ($who is $age))))  
          ( ()  (remove-atom \&self ($who is $age)))  
          ( ()  (add-reduct \&self ($who is (\+ $age 1))))  
          ($upd (match \&self ($who is $a) $a))  
          ( ()  (println\! (NOW: ($who is $upd)))))  
        $upd  
   )  
)

\! (age++ Sam) ; 35

Run

## **case**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/control_flow.html#case)

Another type of multiway control flow mechanism in MeTTa is the case function, which was briefly mentioned in [the tutorial](https://metta-lang.dev/docs/learn/tutorials/eval_intro/recursion.html#conditional-statements). It turns let around and subsequently tests multiple pattern-matching conditions for the given value. This value is provided by the first argument. While the formal argument type is Atom, it will be evaluated. The second argument is a tuple, which elements are pairs mapping condition patterns to results.

metta

(Sam is Frog)  
(Apple is Green)  
(\= (test $who)  
   (case (match \&self ($who is $x) $x)  
    (  
        (42   "The answer is 42\!")  
        (Frog "Do not ask me about frogs")  
        ($a   ($who is $a))  
    )))  
\! (test Sam) ; "Do not ask me about frogs"  
\! (test Apple) ; (Apple is Green)

\! (test Car) ; empty

Run

Cases are processed sequentially from the first to the last. In the example above, $a condition will always be matched, so it is put at the end, and the corresponding branch is triggered, when all the previous conditions are not met. Note, however, that $a is not matched against the empty result in the last case.

In order to handle such cases, one can use Empty symbol as a case pattern (in some versions of the interpreter, Empty is the dedicated symbol which (empty) is evaluated to). The following code should return "Input was really empty":

metta

\! (case (empty)  
    ((Empty "Input was really empty")  
    ($\_   "Should not be the case"))

  )

Run

In the current version of MeTTa (v0.1.12), Empty can be matched against a variable. This means that the result of (empty) can be matched against $*. In this situation, if we want to catch Empty case, we need to place it before $* case.

Let us consider the use of patterns in case on example of the rock-paper-scissors game. There are multiple ways of how to write a function, which will return the winner. The following function uses one case with five branches:

metta sandbox

(\= (rps-winner $x $y)

   (case ($x $y)

     (((Paper Rock) First)

      ((Scissors Paper) First)

      ((Rock Scissors) First)

      (($a $a) Draw)

      ($\_ Second))

   )

)

\! (rps-winner Paper Scissors) ; Second

\! (rps-winner Rock  Scissors) ; First

\! (rps-winner Paper Paper   ) ; Draw

Run

One could also write a function, which checks if the first player wins, and use it twice (for ($x $y) and ($y $x)). This could be more scalable for game extensions with additional gestures, and could be more robust to unexpected inputs (although this should be better handled with types). You can try experimenting with different approaches using the sandbox above.

