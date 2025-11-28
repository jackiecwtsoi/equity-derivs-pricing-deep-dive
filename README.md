# A Deep Dive into Exotic Equity Derivatives Pricing

## Introduction
This is a personal project where I explore the pricing of equity derivatives.

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

###  Features
1. Implied volatility surface
2. Selection of underlying
3. Simulated price of exotic option payoffs

## Key Mathematical Concepts
This app is an exploration of the mathematical/statistical concepts that are core to quantitative finance. While there are many ready-to-use packages and functions available for implementation, it is important for one to understand the key concepts of why some of the algorithms are structured the way they are. This section explores the key derivations and findings on the underlying assumptions and mechanisms of options pricing.

### 1. Brownian Motion
#### Standard Brownian Motion - the building block of everything
The first and foremost assumption is that prices of a stock are a Brownian motion. To understand this, we first need to understand the concept of random walk in the discrete space. 

A random walk is a simple process where we assume for $`n`$ intervals in a time period $`t`$, where $`t = n{\Delta}t`$, we have
```math
X(t) = Z_1 + Z_2 + ... + Z_n
```
and $`Z_i`$ takes value of $`1`$ or $`-1`$. For the case of symmetric random walk, the probability of $`Z_i`$'s value is exactly $`\frac{1}{2}`$.

A Brownian motion can be understood as the continuous extension of the discrete concept of random walk, where we assume a more sophisticated slicing of each time block $`{\Delta}t`$ into even smaller blocks $`\frac{{\Delta}t}{m}`$. Correspondingly, each increment of the random walk is transformed into
```math
W_m(s) = \frac{1}{\sqrt{m}}Z_{ms}, \space s \in [0, t]
```
and this is called a scaled random walk.

One can now quickly deduce that when the slicing parameter $`m \to \infin`$ and by the Central Limit Theorem, the Brownian motion $`W_m(t)`$ follows a normal distribution i.e. 
```math
W_m(t) \sim N(0, t)
```

#### Variations of Brownian Motion
There are many variations of the Brownian motion. One of them is the Brownian bridge and its variations where $`W(t)`$ starts at a certain level $`a`$ and sends at a certain level $`b`$. Another variation is the Brownian motion with drift, where we introduce two constant parameters $`\mu`$ which represents the "drift" and $`\sigma`$ which represents volatility, and the distribution now looks like this:
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

### 2. Risk-Neutral Pricing & Martingale
To price derivatives fairly, we need to also understand risk-neutral pricing and the concept of a martingale. In a nutshell, there are two worlds:
- $`P`$ which represents the real-life probabilities of how a stock moves
- $`Q`$ which represents the "risk-neutral" probabilities of how a stock moves, under the assumption that there is no arbitrage
