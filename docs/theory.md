---
title: An Argument for the Necessity of Behavior Trees for LLM Systems
---

# An Argument for the Necessity of Behavior Trees for LLM Systems

We want to use LLMs and VLMs to build complex systems from simpler parts. Every sufficiently powerful formalism for building complex systems out of simpler parts has three mechanisms for achieving that goal: primitive elements, means of combination, and means of abstraction. The behavior tree formalism is no different. In our case, we have  

All sufficiently powerful formal systems have three common 

A language model \(\ell\) is a conditional probability distribution: given a sequence of tokens \(t_1, \ldots, t_n\) from a vocabulary \(V\),

1. LMs as CPDs
2. LMs as Context->Context mappings
3. Two issues with ad hoc solutions
  a. Semantics vs. Syntax: Contexts are syntactic objects
  b. Compositionality is unclear
  