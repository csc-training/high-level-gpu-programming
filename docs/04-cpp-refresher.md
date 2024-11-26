---
title:    C++ Refresher
subtitle: High-Level GPU Programming
author:   CSC Training
date:     2024-11
lang:     en
---

# Outline

- SYCL and Kokkos are modern C++ with classes, templates, lambdas, ...
- These constructions are reviewed


# Elements of a SYCL code

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename T>
void axpy(queue &q, const T &a, const std::vector<T> &x, std::vector<T> &y) {
  range<1> N{x.size()};
  buffer x_buf(x.data(), N); buffer y_buf(y.data(), N);

  q.submit([&](handler &h) {
    accessor x{x_buf, h, read_only};
    accessor y{y_buf, h, read_write};

    h.parallel_for(N, [=](id<1> i) {
      y[i] += a * x[i];
    });
  });
  q.wait_and_throw();
}
```

# Namespaces

- Namespace is a way of organizing variables, functions, classes, etc.

```cpp
// Fully qualified name
sycl::queue q{};

// Using names from the namespace
using namespace sycl;
queue q{};

```

# Placeholder type `auto`

- `auto` can be used in variable declaration if the compiler can deduce the type during compilation

```cpp
auto a = 5;

auto queue_event = queue.submit([&](handler& h) {...});

```

# Templates

- Templates allow writing generic functions and classes

```cpp
template <typename T>
T max(T a, T b) {
  return (a > b) ? a : b;
}

int a = 1, b = 2;
auto c = max(a, b);

double x = 3.4, y = 5.6;
auto z = max(x, y);

// Call int version explicitly
auto zi = max<int>(a, y);

```

# Abbreviated function templates with `auto`

- Since C++20, abbreviated function templates are possible

```cpp
auto max(auto a, auto b) {
  return (a > b) ? a : b;
}

int a = 1, b = 2;
auto c = max(a, b);

double x = 3.4, y = 5.6;
auto z = max(x, y);

auto zi = max(a, y);

```

# Pointers and references

- Raw pointer: Memory address of a variable (as in C)

```cpp
void foo1(int *a) { *a = 42; }

int x = 0;
int *x_ptr = &x;
foo1(x_ptr);
std::cout << x << std::endl;

```

- Reference: Alias of another variable

```cpp
void foo2(int &a) { a = 42; }


int y = 0;
foo2(y);
std::cout << y << std::endl;

```

# Containers

- C++ standard library provides generic containers for data
  - Follows [RAII](https://en.cppreference.com/w/cpp/language/raii) principle
  - Prefer over explicit memory management via raw pointers

```cpp
#include <vector>

std::vector<double> a(10);
a[0] = 5;
a[1] = 7;

// Raw pointer to the array; needed often for interoperability
double *a_ptr = a.data();

```

# Classes

- Composite data type grouping variables and functions

<div class="column">
```cpp
template <typename T>
class Particle {
private:
    T x, y;
public:
    Particle(T x, T y) : x(x), y(y) {}
    void move(T dx, T dy) {
        x += dx;
        y += dy;
    }
    void print() {
        std::cout << x << " " << y << std::endl;
    }
};

```
</div>

<div class="column">
```cpp
Particle<double> p{1.2, 3.4};
p.print();
p.move(5.6, 7.8);
p.print();

```
</div>


# Function objects

- Objects that behave like functions

```cpp
class Adder {
private:
    const int constant;
public:
    Adder(const int c) : constant{c} {}
    int operator()(const int a) const { return constant + a; }
};

Adder add{5};
int sum = add(2);
std::cout << "The sum is: " << sum << std::endl;

```


# Lambda expressions

- Unnamed function objects that can capture variables in scope
- Syntax: `[ captures ] (parameters) -> return-type { body }`

```cpp
int a = 1, b = 2, c = 3;

// Capture `a` by value
auto func1 = [a](int x) { return a + x; };
c = func1(4);  // 5

a = -1;
c = func1(4);  // 5

// Capture to a new variable
auto func2 = [d = 2*a](int x) { return d + x; };
c = func2(4);  // 2

```

# Lambda expressions cont'd

```cpp
...

// This will fail; `b` not captured
auto func3 = [a](int x) { return b + x; };

// Capture everything by value
auto func3 = [=](int x) { return b + x; };
c = func3(4);  // 6

```

# Lambda expressions cont'd

```cpp
...

// Capture `b` by reference
auto func4 = [&b](int x) { return b + x; };
c = func4(4);  // 6

b = -2;
c = func4(4);  // 2

// Capture everything by reference
auto func5 = [&](int x) { a = x; b = -x; };
func5(4);  // a = 4, b = -4

```

# Lambda expressions cont'd

```cpp
...

// Mix and match
auto func6 = [=,&b](int x) { return a + b + x; };
c = func6(4);  // 4

a = b = 0;
c = func6(4);  // 8

```

# Error handling

- Errors are handled via C++ exceptions

```cpp
int main() {
  int x, y;
  std::cout << "Enter two numbers: ";
  std::cin >> x >> y;

  try {
    if (y == 0) throw "Division by zero error";
    std::cout << "x / y = " << x / y << std::endl;
  } catch (const char* msg) {
    std::cerr << "Error: " << msg << std::endl;
  }
  return 0;
}

```

# Summary

- Modern C++ allows generic programming
- Classes, templates, lambdas, ...
- Reusable, expressive, and efficient code

