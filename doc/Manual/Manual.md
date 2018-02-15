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
Algorithm centric thinking is manifested in the imperative programing paradigm, which historically dominated computer programing, and probably still is the best way to learn the basics of programming and to implement simple tasks.
However, much of the success of todays computer software can be attributed to the development of powerful data structures.

Anyone who has writen software that is more than a few hundred lines of code knows that a major obstacle becomes to organize the code in such a way that the complexity do not increase with size.
When coming back to a project after a few months, it may be difficult to make modifications to one part of the code since you do not remember if or how it affects other parts of the code.
The reason for this can largly be traced back to the lack of proper attention paid to data structures.
In particular, well designed data structures enables abstraction and encapsulation and is a core motivation behind the object oriented programming paradigm.

Abstraction is the process of dividing code into logical units that aids the thinking process by allowing the programmer to think on a higher level.
Instead of 

@page Model Model

@page Solve Solve

@page ExtractProperties Extract properties
