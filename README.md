# A Deep Dive into Exotic Equity Derivatives Pricing

## Introduction
This is a personal project where I explore the mathematical/statistical concepts that are core to quantitative finance and implement them in the form of a web app.

## A Mathematical Deep Dive
While there are many ready-to-use packages and functions available online for implementation, it is important for one to understand the key concepts of why some of the algorithms are structured the way they are. This section explores the key derivations and findings on the underlying assumptions and mechanisms of options pricing.

To aid understanding, I have designed each of the sections to represent a building block that comes useful in the derivation of the Black-Scholes formula, one of the most important concepts in options pricing.

### 1. Brownian Motion
A key question one may have when one starts out to study stocks would be: how do I model the price of a stock? Indeed, while there are many nice statistical distributions out there, there seems to be no easy way to model how stock prices move - they all seem so random! The answer amongst many and likely the simplest to understand, is that we can model stock prices just by the idea that they "walk" randomly. In the following sections we will explore this in a more formal way.

#### Standard Brownian Motion - the building block of everything
We first need to understand the concept of random walk in the discrete space. A random walk is a simple process where we assume for $`n`$ intervals in a time period $`t`$, where $`t = n{\Delta}t`$, we have
```math
X(t) = Z_1 + Z_2 + ... + Z_n
```
and $`Z_i`$ takes value of $`1`$ or $`-1`$. For the case of symmetric random walk, the probability of $`Z_i`$'s value is exactly $`\frac{1}{2}`$.

A Brownian motion can be understood as the continuous extension of the discrete concept of random walk, where we assume a more sophisticated slicing of each time block $`{\Delta}t`$ into even smaller blocks $`\frac{{\Delta}t}{m}`$. Correspondingly, each increment of the random walk is transformed into
```math
W_m(s) = \frac{1}{\sqrt{m}}Z_{ms}, \space s \in [0, t]
```
and this is called a scaled random walk.

One can now quickly deduce that when the slicing parameter $`m \to \infty`$ and by the Central Limit Theorem, the Brownian motion $`W_m(t)`$ follows a normal distribution i.e. 
```math
W_m(t) \sim N(0, t)
```

#### Variations of Brownian Motion
There are many variations of the Brownian motion. One of them is the Brownian bridge and its variations where $`W(t)`$ starts at a certain level $`a`$ and stops at a certain level $`b`$. Another variation is the Brownian motion with drift, where we introduce two constant parameters $`\mu`$ which represents the "drift" and $`\sigma`$ which represents volatility, and the distribution now looks like this:
```math
X(t) = \mu t + \sigma W(t)
```
However, neither of these variations can effectively reflect the price of a stock in real-life - they all assume that the Brownian motion can be negative, whereas in reality stock prices do not go below 0.

#### Geometric Brownian Motion (GBM)
This leads to one of the most important Brownian motion variations in the context of finance: the Geometric Brownian Motion (GBM). Here we make the important assumption that the random variable can only be greater than $`0`$ and never negative, thus more accurately mimicking the behavior of stock prices in real life. Formally, given a BM with drift where the drift parameter is $`\mu`$ and volatility parameter is $`\sigma`$ i.e. $`X(t) = \mu t + \sigma W(t)`$, we have a GBM defined as:
```math
G(t) = G(0) \space \text{exp}[X(t)]
```
where $`t \ge 0`$ and $`G(t) > 0`$.

We will explain the significance of this GBM concept further in a later section.

### 2. Stochastic Calculus & Ito's Lemma
After GBM, the second building block to understand derivatives pricing is Ito's Integral and subsequently Ito's Lemma which are ideas extended from stochastic calculus.

#### The Stochastic Integral
In the calculus we've learnt in high school, when we take an integral of a function, that function is "deterministic" i.e. not random. An example of this would be integrating something like $`f(x) = x + 3`$, where this $`f(x)`$ is something that outputs a "set"/"deterministic" value for every single input. Say I input $`x = 1`$, then this $`f(x)`$ gives me a value of $`4`$ every single time I run this calculation. This is what we call a deterministic function.

But what about the Brownian motion $`W(t)`$ that we've talked about earlier? Every time we try to find "values" (or more accurately "draw samples") from this $`W(t)`$, the outcome would be random! It is a "random variable" as opposed to a "deterministic function". The second property of the Brownian motion is that it is continuous but NOT differentiable (proof I will put in the appendix). The natural question becomes, then how can we find the integral for this kind of process?

This is where stochastic integral comes in. We can view Ito's Integral as
```math
I(T) = \int_{0}^{T} g(u) \space dW(u)
```
where $`g(u)`$ is a random process and $`W(u)`$ are both random processes. In the context of a stock price movement, we can think of $`g(u)`$ as the number of shares we hold for that particular stock at that time instant, and $`W(t)`$ is the stock price.

### 3. Risk-Neutral Pricing & Martingale
To price derivatives fairly, we need to also understand risk-neutral pricing and the concept of a martingale. In a nutshell, there are two worlds:
- $`P`$ which represents the real-life probabilities of how a stock moves
- $`Q`$ which represents the "risk-neutral" probabilities of how a stock moves, under the assumption that there is no arbitrage

## The App
### Setup
Python version 3.9.6
```
pip install -r requirements.txt
```

###  Running the Program
```
cd src
streamlit run app.py
```