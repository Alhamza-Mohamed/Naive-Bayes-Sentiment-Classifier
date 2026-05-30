# Intention of the project

## Bayes theorem

P(A|B)=\frac{P(B|A)P(A)}{P(B)}

law for a conditional property where P(A|B) is the probability of event A given that event B has occurred.
i.e. what is the probability of A to be true if the probability of B to be true (A depending on B).

---

In Naive Bayes we usually do not compute the exact posterior posterior probability.
instead, we compare relative scores


---

**What is posterior probability?**
It is the updated probability after observing evidence.
The word "posterior" means "after"

So $$P(A|B)$$ is called the posterior probability because:  

- before observing B, we had some belief about A.
- After observing B, we update that belief.

we don't need to compute the posterior probability in classification a because we compare P(positive ∣ words) vs P(negative ∣ words) and since both have the same denominator P(words) we just need to compare the numerator  
$$P(positive)×P(words∣positive)$$
vs
$$P(negative)×P(words∣negative)$$

This called proportional comparison.  
