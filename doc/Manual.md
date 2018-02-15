Manual {#manual}
======

- @subpage Introduction
- @subpage Model
- @subpage Solve
- @subpage ExtractProperties

@page Introduction Introduction

# Origin and scope {#OriginAndScope}
TBTK (Tight-Binding ToolKit) originated as a toolkit for solving tight-binding models.
However, the scope of the code has expanded beyond the area implied by its name, and is today best described as a library for building applications that solves second-quantized Hamiltonians with discrete indices  

<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}} + \sum_{\mathbf{i}\mathbf{j}\mathbf{k}\mathbf{l}}V_{\mathbf{i}\mathbf{j}\mathbf{k}\mathbf{l}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}^{\dagger}c_{\mathbf{k}}c_{\mathbf{l}}\f$.</center>

# Algorithms and data structures {#AlgorithmsAndDataStructures}

When writing software it is natural to think in terms of algorithms.
This is particularly true in scientific computation, where the objective most of the time is to carry out a set of well defined operations on an input state to arrive at a final answer.
Algorithm centered thinking is manifested in the imperative programing paradigm and is probably the best way to learn the basics of programming and to implement simple tasks.
However, while algorithms are of great importance, much of the success of todays computer software can be attributed to the development of powerful data structures.

Anyone who has writen software that is more than a few hundred lines of code knows that a major challange is to organize the code in such a way that the complexity do not scale up with the size of the project.
Otherwise, when e.g. coming back to a project after a few months, it may be difficult to make modifications to the code since you do not remember if or how it will affects other parts of the code.
The reason for this can largly be traced back to the lack of proper attention paid to data structures.
In particular, well designed data structures enables abstraction and encapsulation and is a core component of the object oriented programming paradigm.

Abstraction is the process of dividing code into logical units that aids the thinking process by allowing the programmer to think on a higher level.
Effective abstraction allows the programmer to forget about low level details and focus on the overarching problem at hands.
For some analogies, mathemtics and physics are rife with abstractions: derivatives are defined through limits but differential equations are written using derivative symbols, matrices are rectangles of numbers but we carry out much of the algebra manipulating letters representing whole matrices rather than individual matrix elements, etc.
Much of mathematics is ultimately just addition, subtraction, multiplication, and division of real numbers, but through abstraction problems of much greater complexity can be tackled than if everything was formulated with those four symbols alone.
Similarly programming is nothing but loops, conditional execution, assignment of values, function calls etc., but further abstraction allows the programmers mind to be freed from low level details to focus on the higher level aspects of much more complex problems.

While abstraction means dividing code into logical units that allows the programmer to think on a higher level, encapsulation means making those units largly independent of each other.
Different parts of a program should of course interact with each other, but low level details of a specific component should to an as large degree as posible be invisible to other components.
Instead of allowing (and requiring) other components of a code to manipulate its low level details, components should strive to present themself to other components through an easy to use interface.
The interface is provided through a so called application programming interface (API).
The API is essentially a contract between a component and the outside world, where the component specifies a promisse to solve a particular problem given a particular input.
Encapsulation makes it possible to update a piece of code without remembering what other parts of the code is doing, as long as the update respects the contract specified in the API, and is key to solving the scalability issue.
Developers mainly experienced with imperative programming likely recognize some of these concepts as being emboddied in the idea of dividing code into functions.
Object oriented programming greatly extends this powerful technique.

Scientific computing is often computationally intensive and much thought therefore goes into the development of different algorithms for solving the same problem.
Different algorithms may have their own strengths and weaknesses making them the preffered choice under different circumstances.
Often such algorithms are implemented in completely different software packages with little reuse of code, even though the code for the main algorithm may be a small part of the actual code.
This is a likely result when data structures are an afterthought and results in both replicated work and less reliable code since code that is reused in multiple projects is much more extensively tested.
A key to handling situations like this is called polymorphis and is a principle whereby different components essentially provides identical or partly identical contracts to the outside world, even though they internally may work very differently.
This allows for components to be used interchangeably with little changes to the rest of the code base.

TBTK is first and foremost a collection of data structures intended to enable the implementation of algorithms for solving quantum mechanical problems, but also implements several different algorithms for solving specific problems.

# c++11: Performance vs. user friendliness {#cpp11PerformanceVsUserFriendliness}

Scientific computations are often very demanding and high performance is therefore often a high priority.
However, while low level programming languages offer high performance, they also have a reputation of being relatively difficult to work with.
A comparatively good understanding of the low level details of how a computer works is usually required to write a program in languages such as c/c++ and FORTRAN compared to e.g. MATLAB and python.
However, while c++ provides the ability to work on a very low level, it also provides the tools necessary to abstract away much of these details.
A well writen library can alleviate many of these issues, such as for example putting little requirement on the user to manage memory (the main source of errors for many new c/c++ programmers).
Great progress in this direction was taken with the establishment of the c++11 standard.
The language of choice for TBTK has therefore been c++11, and much effort has gone into developing data structures that is as simple as possible to use.
Great care has also been taken to avoid having the program crash without giving error messages that provide information that helps the user to resolve the problem.

@page Model Model

@page Solve Solve

@page ExtractProperties Extract properties
