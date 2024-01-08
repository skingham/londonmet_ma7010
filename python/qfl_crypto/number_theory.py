
import numpy as np
import numba
import math
from typing import Generator, List, Set, Tuple


def sign(n: int):
    return 1 if n > 0 else -1

@numba.jit(nopython = True, parallel = False, fastmath = True, forceobj = False)
def gcd_recursion(a: int, b: int) -> int:
    """ Find GCD by recursing on a mod b """
    if(b == 0):
        return a if a > 0 else -a

    return gcd_recursion(b, a % b)

@numba.jit(nopython = True, parallel = False, fastmath = True, forceobj = False)
def gcd_euclidian(a: int, b: int) -> int:
    """ Euler algo """

    # a = q * b - r
    q = a // b
    r = a - q * b
    
    # Stopping condition: r = 0
    if r == 0:
        return b
    
    # next iteration b = r * q_2 + r
    return gcd_euclidian(b, r)

def gcd(a: int, b: int, fn = None) -> int:
    """  """
    # ensure a >= b
    a, b = (a, b) if a >= b else (b, a)

    fn = fn or gcd_recursion
    return fn(a, b)

def extended_euclidian(a: int, b: int, state:Tuple[int, int, int] = None) -> int:
    """ Euler algo """

    # ensure a >= b
    a, b = (a, b) if a >= b else (b, a)
    if state == None:
        state = (np.array([a, 1, 0]), np.array([b, 0, 1]))
    
    # Stopping condition: r = 0
    if state[1][0] <= 0:
        return tuple(v for v in state[0])
    
    # a = q * b - r
    q = state[0][0] // state[1][0]
    w = state[0] - q * state[1]
    state = (state[1], w)
    
    # next iteration b = r * q_2 + r
    return extended_euclidian(a, b, state)

def sieve_of_eratosthenes(n: int) -> Tuple[int]:
    """ Simple sieve to find primes up to n """
    n = n if n > 0 else -n
    if n < 2:
        return tuple()

    is_prime_sieve = np.full(n + 1, True)
    is_prime_sieve[0:2] = False
    is_prime_sieve[4::2] = False
    
    # Start with 2 and odd numbers from 3 to n
    sqrt_n = math.ceil(np.sqrt(n))
    for si in range(3, sqrt_n + 1, 2):
        if is_prime_sieve[si]:
            # Mark every multiple of si from si^2
            is_prime_sieve[si ** 2::si] = False
    return tuple(int(i) for i in np.flatnonzero(is_prime_sieve))

def sieve_of_eratosthenes_gen(n: int) -> Generator[int, int, None]:
    """ Simple sieve to find primes up to n """
    n = sign(n) * n
    if n < 2:
        return

    is_prime_sieve = np.full(n + 1, True, dtype=bool)
    is_prime_sieve[0:2] = False

    # Start with 2 and odd numbers from 3 to n
    yield 2
    is_prime_sieve[4::2] = False
    sqrt_n = math.ceil(np.sqrt(143))
    for si in range(3, sqrt_n + 1, 2):
        if is_prime_sieve[si]:
            # Mark every multiple of si from si^2
            yield si
            is_prime_sieve[si ** 2::si] = False
            
    # All primes have been marked, now just finish loop to yield them all
    for si in range(si + 2, n + 1, 2):
        if is_prime_sieve[si]:
            yield si
    return

#@numba.jit(nopython = True, parallel = True, fastmath = True, forceobj = False)
def sieve_of_eratosthenes_set(n: int) -> Set[int]:
    """ Simple sieve to find primes up to n """
    n = n if n > 0 else -n
    if n < 2:
        return

    is_prime_sieve = np.full(n + 1, True)
    is_prime_sieve[0:2] = False
    is_prime_sieve[4::2] = False
    
    # Start with 2 and odd numbers from 3 to n
    sqrt_n = math.ceil(np.sqrt(n))
    for si in range(3, sqrt_n + 1, 2):
        if is_prime_sieve[si]:
            # Mark every multiple of si from si^2
            is_prime_sieve[si ** 2::si] = False
    return set([int(i) for i in np.flatnonzero(is_prime_sieve)])

@numba.jit(nopython = True, parallel = True, fastmath = True, forceobj = False)
def numba_sieve_of_eratosthenes(n: int) -> np.ndarray:
    """ Simple sieve to find primes up to n """
    n = n if n > 0 else -n
    if n < 2:
        return

    is_prime_sieve = np.full(n + 1, True)
    is_prime_sieve[0:2] = False

    sqrt_n = math.ceil(np.sqrt(n))
    for si in numba.prange(2, sqrt_n + 1):
        if is_prime_sieve[si]:
            # Mark every multiple of si from si^2
            is_prime_sieve[si ** 2::si] = False
    return np.flatnonzero(is_prime_sieve) 

def primes_below(n: int) -> Generator[int, int, None]:
    """ """
    for p in sieve_of_eratosthenes(n - 1):
        yield p
    return

def next_prime(n: int):
    """ """
    for p in primes_below((math.trunc(np.sqrt(n)) + 1) ** 2):
        if p > n:
            return p
    return np.NaN

def is_prime(n: int) -> bool:
    """ check if is a prime by division: O(log(sqrt(n))) """
    for i in range(2, math.trunc(np.sqrt(n)) + 1):
        if n % i == 0:
            return  False
    return True

def ifactors(n: int) -> List[Tuple[int, int]]:
    """   """
    n = sign(n) * n
    factor_list = []
    for p in sieve_of_eratosthenes(n // 2):
        if n % p == 0:
            k = 1
            m = n // p
            while m % p == 0:
                m = m // p
                k += 1
            factor_list.append((p, k))
    
    return factor_list or [(n, 1)]

def divisors(n: int) -> Tuple[int, ...]:
    """ """
    factors = tuple(ifactors(n))
    factor_primes = np.array([f[0] for f in factors] + [0,])
    # add one to every power, as we generate powers as (i mod factor_powers[j])
    factor_powers = np.array([f[1] for f in factors] + [0,]) + 1
    
    factors_count = len(factors)
    divisors_count = np.prod(factor_powers)

    # calc product of array of each prime factor to some power, varying from 0 to the max given from ifactors fn
    ds = sorted(int(np.prod([factor_primes[j] ** (i // np.prod(factor_powers[j - factors_count:]) % factor_powers[j])
                             for j in range(factors_count)]))
                for i in range(divisors_count))
    return tuple(ds)

def co_primes(n: int):
    """ """
    return set([a for a in range(1, n) if gcd(a, n) == 1])

def totient(n: int) -> int:
    """ Euler's phi function is the number of values less than a that are co-prime with n """
    return len(co_primes(n))

def order_of_powers(g: int, n: int) -> List[int]:
    """ g ^ k % n for k being co-prime with phi(n) """

    # order_of_powers = sorted(set([pow(g, k, n) for k in co_primes(totient(n))]))
    # keep all calcs mod n to remove overflow errors
    ks = co_primes(totient(n))
    order_of_powers = set()
    g_k = 1
    for k in range(1, n):
        g_k = g_k * g  % n
        if k in ks:
            order_of_powers.add(g_k)
    return sorted(order_of_powers)

def order(a: int, n: int):
    """ Multiplicative order of a mod n is the smallest k for which a^k mod n is 1 """
    if a > n or gcd(a, n) != 1:
        return np.NaN

    a_k = 1
    for k in range(1, n):
        a_k = a_k * a  % n
        if a_k == 1:
            return k
    
    return np.NaN

def is_order_n(a: int, n: int):
    """ Multiplicative order of a mod n is the smallest k for which a^k mod n is 1 """
    if a > n or gcd(a, n) != 1:
        return np.NaN
    
    ord_n = totient(n)
    # we can do better than all k < n by only looking at divisors of totient(n)
    phi_n_divisors = divisors(ord_n)
    for k in phi_n_divisors:
        if pow(a, k, n) == 1:
            return k == ord_n
    
    return np.NaN

def cyclic_group(a, n, op):
    group = set([pow(a, k, n) for k in range(1, n)])
    return group

def primitive_roots(n):
    # g is a primitive root modulo n if for every integer a coprime to n, there is some integer k for which gk ≡ a (mod n)
    
    # check n is form 2, 4, p^s, 2p^s, where s is any positive integer and p is an odd prime
    factors = [f for f in ifactors(n)]
    if any((len(factors) < 1 or 2 < len(factors),  
            (len(factors) == 2 and (factors[0][0] != 2 or factors[0][1] > 1)),
            (len(factors) == 1 and factors[0][0] == 2 and factors[0][1] > 2),
            (len(factors) == 1 and factors[0][0] < 2))):
        return [] # Exception("No primitive roots exist")
    
    # find smallest  root
    ord_n = totient(n)
    g = None
    for a in co_primes(n):
        if order(a, n) == ord_n:
            g = a
            break
    
    # There are phi(phi(n)) roots: return all roots using factors co-prime with phi(n)
    prime_roots = order_of_powers(g, n)
     
    assert len(prime_roots) == totient(ord_n)
    assert all(pow(g, ord_n, n) == 1 for g in prime_roots)
    return prime_roots