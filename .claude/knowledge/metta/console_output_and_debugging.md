# **Console output and debugging**

# [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/console_output.html#console-output-and-debugging)

All values obtained during evaluation of the MeTTa program or script are collected and returned. The whole program can be treated as a function. If a stand-alone program is executed via [a command-line runner](https://github.com/trueagi-io/hyperon-experimental/blob/main/python/hyperon/metta.py) or [REPL](https://github.com/trueagi-io/hyperon-experimental?tab=readme-ov-file#rust-library-and-repl) these results are printed at the end. This printing will not happen if MeTTa is used via [its API](https://metta-lang.dev/docs/learn/tutorials/python_use/intro.html).

However, MeTTa has two functions to send information to the console output: println\! and trace\!. They can be used by developers for displaying messages and logging information during the evaluation process, in particular, for debugging purposes.

## **Print a line**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/console_output.html#print-a-line)

The println\! function is used to print a line of text to the console. Its type signature is (-\> %Undefined% (-\>)).

The function accepts only a single argument, but multiple values can be printed by enclosing them within parentheses to form a single atom:

metta

\! (println\! "This is a string")

\! (println\! ($v1 "string" 5))

Run

Note that println\! returns the [unit](https://metta-lang.dev/docs/learn/tutorials/types_basics/match_control.html#unit-type) value (). Beside printing to stdout, the program will return two units due to println\! evaluation.

The argument of println\! is evaluated before println\! is called (its type is not Atom but %Undefined%), so the following code

metta

(Parent Bob Ann)  
\! (match \&self (Parent Bob Ann) (Ann is Bob\`s child))

\! (println\! (match \&self (Parent Bob Ann) (Bob is Ann\`s parent)))

Run

will print (Bob is Ann's parent) to stdout. Note that this result is printed before all the evaluation results (starting with the match expressions) are returned.

## **Trace log**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/console_output.html#trace-log)

trace\! accepts two arguments, the first is the atom to print, and the second is the atom to return. Both are evaluated before passing to trace\!, which type is (-\> %Undefined% $a $a), meaning that the reduced type of the whole trace\! expression is the same as the reduced type of the second argument:

metta

\! (get-type (trace\! (Expecting 3) (\+ 1 2))) ; Number

Run

trace\! can be considered as a syntactic sugar for the following construction using println\! and let (see [this section](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/let_case.html) of the tutorial for more detail):

metta

(: my-trace (\-\> %Undefined% $a $a))  
(\= (my-trace $out $res)  
   (let () (println\! $out) $res))

\! (my-trace (Expecting 3) (\+ 1 2))

Run

It can be used as a debugging tool that allows printing out a message to the terminal, along with valuating an atom.

metta

(Parent Bob Ann)  
\! (trace\! "Who is Anna\`s parent?" ; print this expression  
   (match \&self (Parent $x Ann)  
          ($x is Ann\`s parent)))  ; return the result of this expression  
\!(trace\! "Who is Bob\`s child?"  ; print this expression  
   (match \&self (Parent Bob $x)

          ($x is Bob\`s child))) ; return the result of this expression

Run

The first argument does not have to be a pure string, which makes trace\! work fine on its own

metta

(Parent Bob Ann)  
\! (trace\! ((Expected: (Bob is Ann\`s parent))  
           (Got: (match \&self (Parent $x Ann) ($x is Ann\`s parent)))  
          )

   ())

Run

## **Quote**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/console_output.html#quote)

Quotation was [already introduced](https://metta-lang.dev/docs/learn/tutorials/types_basics/metatypes.html#quoting-metta-code) as a tool for evaluation control. Let us recap that quote is just a symbol with (-\> Atom Atom) type without equalities (i.e., a constructor). In some versions of MeTTa and its stdlib, quote can be defined as (= (quote $atom) NotReducible), where the symbol NotReducible explicitly tells the interpreter that the expression should not be reduced.

The following is the basic example of the effect of quote:

metta

(Fruit apple)  
(\= (fruit $x)  
   (match \&self (Fruit $x) $x))  
\! (fruit $x) ; apple

\! (quote (fruit $x)) ; (quote (fruit $x))

Run

There is a useful combination of trace\!, quote, and let for printing an expression together with its evaluation result, which is then returned.

metta

(: trace-eval (\-\> Atom Atom))  
(\= (trace-eval $expr)  
   (let $result $expr  
        (trace\! (EVAL: (quote $expr) \--\> $result)  
                $result)))  
(Fruit apple)  
(\= (fruit $x)  
   (match \&self (Fruit $x) $x))  
; (EVAL: (quote (fruit $x)) \--\> apple) is printed to stdout

\! (Overall result is (trace-eval (fruit $x))) ; (Overall result is apple)

Run

In this code, trace-eval accepts $expr of Atom type, so it is not evaluated before getting to trace-eval. (let $result $expr ...) stores the result of evaluation of $expr into $result, and then prints both of them using trace\! ((quote $expr) is used to avoid reduction of $expr before passing to trace\!) and returns $result. The latter allows wrapping trace-eval into other expressions, which results in the behavior, which would take place without such wrapping, except for additional console output.

Another pattern of using trace\! with quote and let is to add tracing to the function itself. We first calculate the result (if needed), and then use trace\! to print some debugging information and return the result:

metta

(\= (add-bin $x)  
   (let $r (\+ $x 1)  
        (trace\! (quote ((add-bin $x) is $r))  
                $r)))  
(\= (add-bin $x)  
   (trace\! (quote ((add-bin $x) is $x))  
           $x))  
; (quote ((add-bin 1\) is 1)) and (quote ((add-bin 1\) is 2)) will be printed

\! (add-bin 1) ; \[1, 2\]

Run

Without quotation an atom such as (add-bin $x) evaluated from trace\! would result in an infinite loop, but quote prevents the wrapped atom from being interpreted.

In the following code (test 1\) would be evaluated from trace\! and would result in an infinite loop

metta

(\= (test 1) (trace\! (test 1) 1))  
(\= (test 1) (trace\! (test 0) 0))

\! (test 1)

## **Asserts**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/console_output.html#asserts)

MeTTa has a couple of assert operations that allow a program to check if a certain condition is true and return an error-expression if it is not.

assertEqual compares (sets of) results of evaluation of two expressions. Its type is (-\> Atom Atom Atom), so it interprets expressions internally and can compare erroneous expressions. If sets of results are equal, it outputs the unit value ().

metta

(Parent Bob Ann)  
\! (assertEqual  
   (match \&self (Parent $x Ann) $x)  
   (unify (Parent $x Ann) (Parent Bob $y) $x Failed)) ; ()  
\! (assertEqual (\+ 1 2) 3) ; ()

\! (assertEqual (\+ 1 2) (\+ 1 4)) ; Error-expression

Run

While assertEqual is convenient when we have two expressions to be reduced to the same result, it is quite common that we want to check if the evaluated expression has a very specific result. Imagine the situation when one wants to be sure that some expression, say (+ 1 x), is not reduced. It will make no sense to use (assertEqual (+ 1 x) (+ 1 x)).

Also, if the result of evaluation is nondeterministic, and the set of supposed outcomes is known, one would need to turn this set into a nondeterministic result as well in order to use assertEqual. It can be done with [superpose](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/superpose_collapse.html), but both issues are covered by the following assert function.

assertEqualToResult has the same type as assertEqual, namely (-\> Atom Atom Atom), and it evaluates the first expression. However, it doesn't evaluate the second expression, but considers it a set of expected results of the first expression.

metta

(Parent Bob Ann)  
(Parent Pam Ann)  
\! (assertEqualToResult  
    (match \&self (Parent $x Ann) $x)  
    (Bob Pam))  ; ()  
(\= (bin) 0)  
(\= (bin) 1)  
\! (assertEqualToResult (bin) (0 1)) ; ()  
\! (assertEqualToResult (\+ 1 2) (3)) ; ()  
\! (assertEqualToResult  
    (\+ 1 untyped-symbol)  
   ((\+ 1 untyped-symbol))) ; ()

\! (assertEqualToResult (\+ 1 2) ((\+ 1 2))) ; Error

Run

Let us notice a few things:

* We have to take the result into brackets, e.g., (assertEqualToResult (+ 1 2\) (3)) vs (assertEqual (+ 1 2\) 3\), because the second argument of assertEqualToResult is a set of results even if this set contains one element.  
* As a consequence, a non-reducible expression also gets additional brackets as the second argument, e.g., ((+ 1 untyped-symbol)). It is also a one-element set of the results.  
* The second argument is indeed not evaluated. The last assert yields an error, because (+ 1 2\) is reduced to 3. Notice 3 as what we got instead of expected (for the sake of the example) (+ 1 2\).

Pager
