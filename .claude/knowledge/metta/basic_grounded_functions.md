# **Basic grounded functions**

# [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/basic_grounded_functions.html#basic-grounded-functions)

## **Arithmetic operators and Number type**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/basic_grounded_functions.html#arithmetic-operators-and-number-type)

Arithmetic operations in MeTTa are grounded functions and use the prefix notation where the operator comes before the operands. MeTTa arithmetic works with atoms of Number type, which can store floating-point numbers as well as integers under the hood, and you can mix them in your calculations. The type of binary arithmetic operations is (-\> Number Number Number).

metta

; Addition  
\! (\+ 1 3) ; 4

; Subtraction  
\! (\- 6 2.2) ; 3.8

; Multiplication  
\! (\* 7.3 9) ; 65.7

; Division  
\! (/ 25 5) ; 5 or 5.0

; Modulus

\! (% 24 5) ; 4

Run

In the current implementation arithmetic operations support only two numerical arguments, expressions with more than two arguments like \!(+ 1 2 3 4\) will result in a type error (IncorrectNumberOfArguments). One should use an explicit nested expression in that case

metta

\! (\+ 1 (\+ 2 (\+ 3 4))) ; 10

\! (\- 8 (/ 6.4 4)) ; 6.4

Run

Numbers in MeTTa are presented as grounded atoms with the predefined Number type. Evaluation of ill-typed expressions produces an error expression. Notice, however, that arithmetic expressions with atoms of %Undefined% type will not be reduced.

metta

\! (\+ 2 S) ; (+ 2 S)

\! (\+ 2 "8") ; BadType

Run

Other common mathematical operations like sqr, sqrt, abs, pow, min, max, log2, ln, etc. are not included in the standard library as grounded symbols at the moment. But they can be [imported from Python directly](https://metta-lang.dev/docs/learn/tutorials/python_use/py_atom.html).

## **Comparison operations**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/basic_grounded_functions.html#comparison-operations)

Comparison operations implemented in stdlib are also grounded operations. There are four operations \<, \>, \<=, \>= of (-\> Number Number Bool) type.

metta

; Less than  
\! (\< 1 3)

; Greater than  
\! (\> 3 2)

; Less than or equal to  
\! (\<= 5 6.2)

; Greater than or equal to

\! (\>= 4 (\+ 2 (\* 3 5)))

Run

Once again, passing ordinary symbols to grounded operations will not cause errors, and the expression simply remains unreduced, if it type-checks. Thus, it is generally a good practice to ensure the types of atoms being compared are what the comparison operators expect to prevent unexpected results or errors.

metta

\! (\> $x (\+ 8 2)) ; Inner expression is reduced, but the outer is not  
\! (\>= 4 (\+ Q 2)) ; Reduction stops in the inner expression  
(: R CustomType)

\! (\>= 4 R) ; BadType

Run

The \== operation is implemented to work with both grounded and symbol atoms and expressions (while remaining a grounded operation). Its type is (-\> $t $t Bool). Its arguments are evaluated before executing the operation itself.

metta

\! (\== 4 (\+ 2 2)) ; True  
\! (\== "This is a string" "Just a string") ; False  
\! (\== (A B) (A B)) ; True

\! (\== (A B) (A (B C))) ; False

Run

Unlike \< or \>, \== will not remain unreduced if one of its arguments is grounded, while another is not. Instead, it will return False if the expression is well-typed.

metta

\! (\== 4 (\+ Q 2)) ; False  
(: R CustomType)

\! (\== 4 R) ; BadType

Run

## **Logical operations and Bool type**

## [​](https://metta-lang.dev/docs/learn/tutorials/stdlib_overview/basic_grounded_functions.html#logical-operations-and-bool-type)

Logical operations in MeTTa can be (and with some build options are) implemented purely symbolically. However, the Python version of stdlib contains their grounded implementation for better interoperability with Python. In particular, numeric comparison operations directly execute corresponding operations in Python and wrap the resulting bool value into a grounded atom. The grounded implementation is intended for subsymbolic and purely functional processing, while custom logic systems for reasoning are supposed to be implemented symbolically in MeTTa itself.

Logical operations in stdlib deal with True and False values of Bool type, and have signatures (-\> Bool Bool) and (-\> Bool Bool Bool) for unary and binary cases.

metta

; Test if both the given expressions are True  
\! (and (\> 4 2) (\== "This is a string" "Just a string")) ; False

; Test if any of the given expressions is True  
\! (or (\> 4 2) (\== "This is a string" "Just a string")) ; True

; Negates the result of a given Bool value  
\! (not (\== 5 5)) ; False

\! (not (and (\> 4 2) (\< 4 3))) ; True

Run

Pager

